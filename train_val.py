import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import v2, v1, AnnotationTransform, VOCDetection, BaseTransform, detection_collate, VOCroot, VOC_CLASSES
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import numpy as np
import time

import utils.logutil as logutil
from datetime import datetime
import shutil

from data import VOC_CLASSES as labelmap
from data import VOCroot

from eval import voc_eval, Timer

import ipdb

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--version', default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet', default='weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int, help='Number of training iterations')
parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=True, action='store_true', help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--exp_name', type=str,  help='set if you want to use exp name', default=None)
parser.add_argument('--jobs_dir', default='jobs/', help='Location to save checkpoint models')
parser.add_argument('--voc_root', default=VOCroot, help='Location of VOC root directory')
args = parser.parse_args()

### Set job directory
current_time    = datetime.now()
exp_time        = current_time.strftime('%Y-%m-%d_%Hh%Mm')
args.exp_time   = exp_time

exp_name        = ('_' + args.exp_name) if args.exp_name is not None else ''
jobs_dir        = os.path.join( args.jobs_dir, exp_time + exp_name )
snapshot_dir    = os.path.join( jobs_dir, 'snapshots' )
if not os.path.exists(jobs_dir):            os.makedirs(jobs_dir)
if not os.path.exists(snapshot_dir):        os.makedirs(snapshot_dir)

### Backup important files
shutil.copy2( __file__, jobs_dir+'/' )
shutil.copy2( 'ssd.py', jobs_dir+'/' )
shutil.copy2( 'layers/modules/multibox_loss.py', jobs_dir+'/' )

### Logging experiment settings   
logger          = logutil.getLogger()
logutil.set_output_file( os.path.join(jobs_dir, 'log_%s.txt' % exp_time) )
logutil.logging_run_info( vars(args) )

if args.visdom:
    import visdom
    viz = visdom.Visdom(port=8098)


if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

cfg = (v1, v2)[args.version == 'v2']


train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
# train_sets = [('2007', 'trainval')]
# train_sets = [('2012', 'trainval')]
# train_sets = [('2012', 'train')]
val_sets = [('2007', 'test')]

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')


# train_sets = 'train'
ssd_dim = 300  # only support 300 now
means = (104, 117, 123)  # only support voc now
num_classes = len(VOC_CLASSES) + 1
batch_size = args.batch_size
accum_batch_size = 32
iter_size = accum_batch_size / batch_size
max_iter = 120000
weight_decay = 0.0005
stepvalues = (80000, 100000, 120000)
gamma = 0.1
momentum = 0.9

ssd_net = build_ssd('train', 300, num_classes)
net = ssd_net

if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    vgg_weights = torch.load(args.basenet)
    print('Loading base network...')
    ssd_net.vgg.load_state_dict(vgg_weights)

if args.cuda:
    net = net.cuda()


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
# optimizer = optim.Adam(net.parameters(), lr=args.lr)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)



def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        # filename = get_voc_results_file_template('test', cls)
        filename = 'tmp/det_test_%s.txt' % (cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

def do_python_eval(use_07=True):
    devkit_path = VOCroot + 'VOC2007'
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07    
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = 'tmp/det_test_%s.txt' % (cls)
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format('test'), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        # print('AP for {} = {:.4f}'.format(cls, ap))
        # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        #     pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    return np.mean(aps)
    

def validation( net, loader, dataset ):

    net.eval()    
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    num_images = len(loader)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    for i, (images, targets, heights, widths) in enumerate(loader):
        if args.cuda:
            images = Variable(images.cuda())
            # targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            # targets = [Variable(anno, volatile=True) for anno in targets]        

        h = heights[0]
        w = widths[0]

        _t['im_detect'].tic()
        detections = net(images)[0].data
        detect_time = _t['im_detect'].toc(average=True)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    net.train()    

    write_voc_results_file(all_boxes, dataset)

    return do_python_eval()

def trainval():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    dataset_test = VOCDetection(args.voc_root, val_sets, BaseTransform(300, means), AnnotationTransform())
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                   batch_size=1, 
                                   shuffle=False, num_workers=4)

    dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
        ssd_dim, means), AnnotationTransform())

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on', dataset.name)
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        epoch_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )

        epoch_map = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1,)).cpu(),
            opts=dict(
                xlabel='Epoch',
                ylabel='mAP',
                title='mAP of trained model',
                legend=['test mAP']
            )
        )

    batch_iterator = None
    data_loader = data.DataLoader(dataset, batch_size, num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    for iteration in range(args.start_iter, max_iter):
        if (not batch_iterator) or (iteration % epoch_size == 0):

            if epoch >= 10:
                # New epoch, validation
                ssd_net.set_phase('test')
                mAP = validation( net, test_loader, dataset_test )
                ssd_net.set_phase('train')

                viz.line(
                    X=torch.Tensor([epoch]).cpu(),
                    Y=torch.Tensor([mAP]).cpu(),
                    win=epoch_map,
                    update=True
                )

                if mAP > best_mAP:
                    print('Best mAP = {:.4f}'.format(mAP))
                    best_mAP = mAP

                    filename = os.path.join(jobs_dir, 'snapshots', 'ssd300_iter_{:07d}_mAP_{:.2f}.pth'.format(iteration, mAP))
                    print('Saving state, {:s}'.format(filename))
                    torch.save(ssd_net.state_dict(), filename)

            # create batch iterator
            batch_iterator = iter(data_loader)



        if iteration in stepvalues:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                    win=epoch_lot,
                    update='append'
                )
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        # images, targets = next(batch_iterator)
        images, targets, _, _ = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno, volatile=True) for anno in targets]
        # forward
        t0 = time.time()
        # out = net(images)
        out, sources = net(images)

        # for src in sources:

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        if iteration % 10 == 0:
            # print('Timer: %.4f sec.' % (t1 - t0))
            # print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')                        
            logger.info('[Iter {:5d}] loss: {:02.4f} = {:01.4f} (loc) + {:01.4f} (cls)\
                \t[time: {:.3f}sec]'.format( iteration, loss.data[0], loss_l.data[0], loss_c.data[0], t1-t0 ) )

            if args.visdom and args.send_images_to_visdom:
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())
        
        # if args.visdom and iteration % 10 == 0:
        if args.visdom and iteration > 10:
            viz.line(
                X=torch.ones((1, 3)).cpu() * iteration,
                Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                    loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                win=lot,
                update='append'
            )
            # hacky fencepost solution for 0th epoch plot
            if iteration == 0:
                viz.line(
                    X=torch.zeros((1, 3)).cpu(),
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu(),
                    win=epoch_lot,
                    update=True
                )
        # if iteration % 5000 == 0:
        #     filename = os.path.join(jobs_dir, 'snapshots', 'ssd300_iter_{:07d}.pth'.format(iteration))
        #     print('Saving state, {:s}'.format(filename))
        #     torch.save(ssd_net.state_dict(), filename)
    
    torch.save(ssd_net.state_dict(), 'weights/{:s}_final.pth'.format(args.exp_name))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    trainval()
