from data import *
from utils.augmentations import SSDAugmentation
# from layers.modules import MultiBoxLoss
# from models.stairnet import build_stairnet
from layers.modules import RefineDetMultiBoxLoss
from models.refinedet import build_refinedet
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
parser.add_argument('--input_size', default='300', choices=['300', '320', '512'],
                    type=str, help='RefineDet300 or RefineDet320 or RefineDet512')
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
parser.add_argument('--port', default=8821, type=int,
                    help='Tensorboard port')

args = parser.parse_args()


if torch.cuda.is_available():
   if args.cuda:
       torch.set_default_tensor_type('torch.cuda.FloatTensor')
   if not args.cuda:
       print("WARNING: It looks like you have a CUDA device, but aren't " +
             "using CUDA.\nRun with --cuda for optimal training speed.")
       torch.set_default_tensor_type('torch.FloatTensor')
else:
   torch.set_default_tensor_type('torch.FloatTensor')


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
        # cfg = voc_stairnet300
        cfg = voc_refinedet[args.input_size]
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    # ssd_net = build_stairnet('train', cfg['min_dim'], cfg['num_classes'])
    refinedet_net = build_refinedet('train', cfg['min_dim'], cfg['num_classes'])
    net = refinedet_net

    if args.cuda:
        # net = torch.nn.DataParallel(refinedet_net)
        net = refinedet_net
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        refinedet_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load('weights/' + args.basenet)
        print('Loading base network...')
        refinedet_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        refinedet_net.extras.apply(weights_init)
        refinedet_net.arm_loc.apply(weights_init)
        refinedet_net.arm_conf.apply(weights_init)
        refinedet_net.odm_loc.apply(weights_init)
        refinedet_net.odm_conf.apply(weights_init)
        #refinedet_net.tcb.apply(weights_init)
        refinedet_net.tcb0.apply(weights_init)
        refinedet_net.tcb1.apply(weights_init)
        refinedet_net.tcb2.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
    #                          False, args.cuda)
    arm_criterion = RefineDetMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    odm_criterion = RefineDetMultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda, use_ARM=True)

    net.train()
    # loss counters
    arm_loc_loss = 0
    arm_conf_loss = 0
    odm_loc_loss = 0
    odm_conf_loss = 0

    acc_arm_loc_loss = 0
    acc_arm_conf_loss = 0
    acc_odm_loc_loss = 0
    acc_odm_conf_loss = 0
    acc_loss = 0

    # loc_loss = 0
    # conf_loss = 0
    epoch = 0
    logger.info('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    logger.info('Training RefineDet on: {}'.format(dataset.name))
    logger.info('Using the specified args:')
    logger.info(args)

    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # # create batch iterator
    # loss_acc_sum = 0.0
    # loss_acc_loc = 0.0
    # loss_acc_cls = 0.0



    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):

        try:
            # load train data
            images, targets = next(batch_iterator)

        except StopIteration:

            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
            
            # reset epoch loss counters
            # loc_loss = 0
            # conf_loss = 0
            arm_loc_loss = 0
            arm_conf_loss = 0
            odm_loc_loss = 0
            odm_conf_loss = 0
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
        arm_loss_l, arm_loss_c = arm_criterion(out, targets)
        odm_loss_l, odm_loss_c = odm_criterion(out, targets)
        #input()
        arm_loss = arm_loss_l + arm_loss_c
        odm_loss = odm_loss_l + odm_loss_c
        loss = arm_loss + odm_loss
        loss.backward()
        optimizer.step()
        t1 = time.time()
        arm_loc_loss += arm_loss_l.item()
        arm_conf_loss += arm_loss_c.item()
        odm_loc_loss += odm_loss_l.item()
        odm_conf_loss += odm_loss_c.item()

        acc_arm_loc_loss += arm_loss_l.item()
        acc_arm_conf_loss += arm_loss_c.item()
        acc_odm_loc_loss += odm_loss_l.item()
        acc_odm_conf_loss += odm_loss_c.item()
        acc_loss += loss.item()

        if (iteration+1) % 10 == 0:
            logger.info('timer: %.4f sec.' % (t1 - t0))
            logger.info('iter ' + repr(iteration) + ' || ARM_L Loss: %.4f ARM_C Loss: %.4f ODM_L Loss: %.4f ODM_C Loss: %.4f ||' \
                % (acc_arm_loc_loss/10.0, acc_arm_conf_loss/10.0, acc_odm_loc_loss/10.0, acc_odm_conf_loss/10.0))

            writer.add_scalars('loss', {'sum': acc_loss/10.0, \
                'arm_loc': acc_arm_loc_loss/10.0, 'arm_cls': acc_arm_conf_loss/10.0, \
                'odm_loc': acc_odm_loc_loss/10.0, 'odm_cls': acc_odm_conf_loss/10.0, \
                }, iteration )

            acc_arm_loc_loss = 0
            acc_arm_conf_loss = 0
            acc_odm_loc_loss = 0
            acc_odm_conf_loss = 0
            acc_loss = 0
            # logger.info('iter ' + repr(iteration) + ' || Loss: %.4f = %.4f (loc) + %.4f (cls)' % (loss_acc_sum/10.0, loss_acc_loc/10.0, loss_acc_cls/10.0))

            # writer.add_scalars('loss', {'sum': loss_acc_sum/10.0, 'loc': loss_acc_loc/10.0, 'cls': loss_acc_cls/10.0}, iteration )
            # loss_acc_sum = 0.0
            # loss_acc_loc = 0.0
            # loss_acc_cls = 0.0

        if iteration != 0 and iteration % 5000 == 0:
            logger.info('Saving state, iter: {:}'.format(iteration))
            torch.save(refinedet_net.state_dict(), os.path.join( jobs_dir, 'refinedet_{:s}_iter_{:06d}.pth'.format(args.dataset, iteration)))

    torch.save(refinedet_net.state_dict(), os.path.join(jobs_dir, 'refinedet_{:s}_iter_{:06d}_final.pth'.format(args.dataset, iteration )))


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
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == '__main__':
    train()
