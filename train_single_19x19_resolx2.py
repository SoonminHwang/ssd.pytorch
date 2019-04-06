from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd_single_19x19_resolx2 import build_ssd
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

parser.add_argument('--exp_name', default=None,
                    help='Specify experiment name')
parser.add_argument('--port', default=8803, type=int,
                    help='Tensorboard port')

args = parser.parse_args()


#if torch.cuda.is_available():
#    if args.cuda:
#        torch.set_default_tensor_type('torch.cuda.FloatTensor')
#    if not args.cuda:
#        print("WARNING: It looks like you have a CUDA device, but aren't " +
#              "using CUDA.\nRun with --cuda for optimal training speed.")
#        torch.set_default_tensor_type('torch.FloatTensor')
#else:
#    torch.set_default_tensor_type('torch.FloatTensor')


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
for file in glob.glob('*.py'):
    tar.add( file )
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

def run_tensorboard( jobs_dir, port ):
    pid = subprocess.Popen( ['tensorboard', '--logdir', jobs_dir, '--host', '0.0.0.0', '--port', str(port)] )    
    
    def cleanup():
        pid.kill()

    atexit.register( cleanup )

from tensorboardX import SummaryWriter
writer = SummaryWriter(os.path.join(jobs_dir, 'tensorboardX'))
run_tensorboard( jobs_dir, port=args.port )

def train():
    if args.dataset == 'VOC':
        cfg = voc_single_19x19_resolx2
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load('weights/' + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        #ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    logger.info('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    logger.info('Training SSD on: {}'.format(dataset.name))
    logger.info('Using the specified args:')
    logger.info(args)

    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    loss_acc_sum = 0.0
    loss_acc_loc = 0.0
    loss_acc_cls = 0.0

    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):

        try:
            # load train data
            images, targets = next(batch_iterator)

        except Exception:

            batch_iterator = iter(data_loader)
            
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
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        loss_acc_sum += loss.item()
        loss_acc_loc += loss_l.item()
        loss_acc_cls += loss_c.item()

        if (iteration+1) % 10 == 0:
            logger.info('timer: %.4f sec.' % (t1 - t0))
            logger.info('iter ' + repr(iteration) + ' || Loss: %.4f = %.4f (loc) + %.4f (cls)' % (loss_acc_sum/10.0, loss_acc_loc/10.0, loss_acc_cls/10.0))

            writer.add_scalars('loss', {'sum': loss_acc_sum/10.0, 'loc': loss_acc_loc/10.0, 'cls': loss_acc_cls/10.0}, iteration )
            loss_acc_sum = 0.0
            loss_acc_loc = 0.0
            loss_acc_cls = 0.0

        if iteration != 0 and iteration % 5000 == 0:
            logger.info('Saving state, iter:', iteration)
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


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

if __name__ == '__main__':
    train()