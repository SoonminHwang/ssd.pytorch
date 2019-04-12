from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss_PredictedAnchor as MultiBoxLoss
# from models.stairnet import build_stairnet
from models.ssd_multiscale_anchor_quantization import build_ssd
import os
import sys
import time
import torch
#from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import pprint

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--input_size', default=320, type=int,
                    help='Input size for training')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')

parser.add_argument('--alpha', default=1e-1, type=float,
                    help='Weight for prior loss')
parser.add_argument('--exp_name', default=None,
                    help='Specify experiment name')
parser.add_argument('--port', default=8821, type=int,
                    help='Tensorboard port')

args = parser.parse_args()


exp_name = args.exp_name


from datetime import datetime
exp_time        = datetime.now().strftime('%Y-%m-%d_%Hh%Mm')
exp_name        = '_' + exp_name if exp_name is not None else ''
jobs_dir        = os.path.join( 'jobs', exp_time + exp_name )
tensorboard_dir    = os.path.join( jobs_dir, 'tensorboardX' )
if not os.path.exists(jobs_dir):            os.makedirs(jobs_dir)
if not os.path.exists(tensorboard_dir):     os.makedirs(tensorboard_dir)



import tarfile, glob
tar = tarfile.open( os.path.join(jobs_dir, 'sources.tar'), 'w' )
for file in glob.glob('*.py') + glob.glob('data/*.py'):
    tar.add( file )
tar.add( 'layers' )
tar.add( 'models' )
tar.add( 'utils' )
tar.close()



import logging
import logging.handlers

fmt = logging.Formatter('[%(levelname)s][%(asctime)s][%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger('main')
logger.setLevel(logging.INFO)

h = logging.StreamHandler()
h.setFormatter(fmt)
logger.addHandler(h)

h = logging.FileHandler(os.path.join(jobs_dir, 'log_{:s}.txt'.format(exp_time)))
h.setFormatter(fmt)
logger.addHandler(h)


import subprocess, atexit

def run_tensorboard( jobs_dir, port=6006 ):
    pid = subprocess.Popen( ['tensorboard', '--logdir', jobs_dir, '--host', '0.0.0.0', '--port', str(port)] )    
    
    def cleanup():
        pid.kill()

    atexit.register( cleanup )

from tensorboardX import SummaryWriter
writer = SummaryWriter(os.path.join(jobs_dir, 'tensorboardX'))
run_tensorboard( jobs_dir, port=args.port )

def train():
    if args.dataset == 'VOC':
        cfg = voc_ssd_anchor_free[str(args.input_size)]
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        # net = ssd_net
        cudnn.benchmark = True

    if args.resume:
        logger.info('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load('weights/' + args.basenet)
        logger.info('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        logger.info('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        # ssd_net.topdown.apply(weights_init)
        ssd_net.extras.apply(weights_init)
        ssd_net.prior.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    nBins = int(ssd_net.prior[0].bias.size(0)/2)
    scale_prior = torch.zeros( (len(ssd_net.prior), nBins) ).cuda()
    
    scale_prior[0, int(nBins*0.15)] = 5.
    scale_prior[1, int(nBins*0.30)] = 5.
    scale_prior[2, int(nBins*0.45)] = 5.
    scale_prior[3, int(nBins*0.60)] = 5.
    scale_prior[4, int(nBins*0.75)] = 5.
    scale_prior[5, int(nBins*0.90)] = 5.


    # for ii in range(len(ssd_net.prior)):
    #     ssd_net.prior[ii].bias.data = torch.cat( [scale_prior[ii], scale_prior[ii]], 0 ).clone()
        
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, ssd_net.ref_vec, args.cuda)
    # criterion = MultiBoxLoss(cfg['num_classes'], 0.6, True, 0, True, 3, 0.5, False, ssd_net.ref_vec, args.cuda)

    net.train()
    # loss counters
    prior_loss = 0
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    logger.info('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    logger.info('Training SSD on: {}'.format(dataset.name))
    logger.info('Using the specified args:')
    # logger.info( pprint.pformat(args) )
    logger.info( args )

    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    loss_acc_sum = 0.0
    loss_acc_pri = 0.0
    loss_acc_loc = 0.0
    loss_acc_cls = 0.0

    num_pos = 0.
    num_neg = 0.

    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):

        try:
            # load train data
            images, targets = next(batch_iterator)

        except Exception:

            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
            
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        
        if args.cuda:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_p, loss_l, loss_c = criterion(out, targets)
        loss_p = args.alpha * loss_p
        loss = loss_l + loss_c + loss_p

        if torch.isnan(loss) or torch.isinf(loss):
            import pdb
            pdb.set_trace()

            criterion(out, targets)


        loss.backward()
        optimizer.step()
        t1 = time.time()

        num_pos += criterion.num_pos.item()
        num_neg += criterion.num_neg.item()

        prior_loss += loss_p.item()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        loss_acc_sum += loss.item()
        loss_acc_pri += loss_p.item()
        loss_acc_loc += loss_l.item()
        loss_acc_cls += loss_c.item()

        if (iteration+1) % 10 == 0:
            import pdb
            pdb.set_trace()
            
            logger.info('timer: %.4f sec.' % (t1 - t0))
            logger.info('iter %6d || Loss: %.4f = %.4f (prior) + %.4f (loc) + %.4f (cls) || Pos: %3d, Neg: %3d' 
                % (iteration, loss_acc_sum/10.0, loss_acc_pri/10.0, loss_acc_loc/10.0, loss_acc_cls/10.0, num_pos/10.0, num_neg/10.0))

            writer.add_scalars('loss', {'sum': loss_acc_sum/10.0, 'prior': loss_acc_pri/10.0, 'loc': loss_acc_loc/10.0, 'cls': loss_acc_cls/10.0}, iteration )
            writer.add_scalars('loss_p', {'prior': loss_acc_pri/10.0}, iteration )
            writer.add_scalars('Num samples', {'pos': num_pos/10.0, 'neg': num_neg/10.0}, iteration )

            num_pos = 0.
            num_neg = 0.

            loss_acc_sum = 0.0
            loss_acc_pri = 0.0
            loss_acc_loc = 0.0
            loss_acc_cls = 0.0

        if iteration != 0 and iteration % 5000 == 0:
            logger.info('Saving state, iter: {:}'.format(iteration))
            torch.save(ssd_net.state_dict(), os.path.join( jobs_dir, 'ssd300_{:s}_iter_{:06d}.pth'.format(args.dataset, iteration)))

    torch.save(ssd_net.state_dict(), os.path.join(jobs_dir, 'ssd300_{:s}_iter_{:06d}_final.pth'.format(args.dataset, iteration )))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == '__main__':
    train()
