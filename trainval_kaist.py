import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from torch.autograd import Variable
from torch.optim import lr_scheduler

import utils.logutil as logutil
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd_kaist import build_ssd
from eval import voc_eval, Timer

import argparse
import numpy as np
import time
import shutil
from datetime import datetime
from collections import OrderedDict

# from data.voc0712 import AnnotationTransform, VOCDetection, detection_collate, VOCroot, VOC_CLASSES
# from data.voc0712 import VOC_CLASSES as labelmap
# from data.voc0712 import VOCroot

from data.kaist_rgbt_ped import AnnotationTransform, KAISTDetection, detection_collate
from data.kaist_rgbt_ped import KAIST_CLASSES as labelmap
from data.kaist_rgbt_ped import DBroot

from data import BaseTransform, v3, v2, v1

import ipdb

# parser.add_argument('--basenet',            default='weights/vgg16_bn-6c64b313.pth', help='pretrained base model')

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training on KAIST dataset')
parser.add_argument('--version',            default='v2', help='conv11_2(v2) or pool6(v1) as last layer')
parser.add_argument('--basenet',            default='weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold',  default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size',         default=16, type=int, help='Batch size for training')
parser.add_argument('--resume',             default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers',        default=8, type=int, help='Number of workers used in dataloading')
# parser.add_argument('--iterations',         default=120000, type=int, help='Number of training iterations')
parser.add_argument('--epochs',             default=1000, type=int, help='Number of training epochs')
parser.add_argument('--start_iter',         default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
parser.add_argument('--cuda',               default=True, action='store_true', help='Use cuda to train model')
parser.add_argument('--lr',                 default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum',           default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay',       default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma',              default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters',          default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom',             default=True, action='store_true', help='Use visdom to for loss visualization')
parser.add_argument('--images_on_visdom',   default=False, action='store_true', help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
parser.add_argument('--exp_name',           default=None, type=str,  help='set if you want to use exp name')
parser.add_argument('--jobs_dir',           default='jobs/', help='Location to save checkpoint models')
parser.add_argument('--db_root',            default=DBroot, help='Location of VOC root directory')
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
shutil.copy2( 'ssd_kaist.py', jobs_dir+'/' )
shutil.copy2( 'layers/modules/multibox_loss.py', jobs_dir+'/' )

### Logging experiment settings   
logger          = logutil.getLogger()
logutil.set_output_file( os.path.join(jobs_dir, 'log_%s.txt' % exp_time) )
logutil.logging_run_info( vars(args) )

def print(msg):
    logger.info(msg)

if args.visdom:
    import visdom
    viz = visdom.Visdom(port=8098)


if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# cfg = (v1, v2)[args.version == 'v2']


train_sets = [('04', 'train')]
# train_sets = [('20', 'train')]
val_sets = [('20', 'test')]

# annopath = os.path.join(args.db_root, 'VOC2007', 'Annotations', '%s.xml')
# imgpath = os.path.join(args.db_root, 'VOC2007', 'JPEGImages', '%s.jpg')
# imgsetpath = os.path.join(args.db_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
# annopath = os.path.join(args.db_root, 'annotations-xml', '%s.xml')
# imgpath = os.path.join(args.db_root, 'images', '%s.jpg')
# imgsetpath = os.path.join(args.db_root, 'imageSets', '{:s}.txt')


# train_sets = 'train'
# ssd_dim = 300  # only support 300 now
ssd_dim = (640, 512)  # only support 300 now
means = (104, 117, 123)  # only support voc now
# num_classes = len(labelmap) + 1
num_classes = 2
batch_size = args.batch_size
# accum_batch_size = 32
# iter_size = accum_batch_size / batch_size
# max_iter = 120000
# stepvalues = (80000, 100000, 120000)
# max_iter = 40000
# stepvalues = (20000, 30000)
weight_decay = args.weight_decay
gamma = args.gamma
momentum = args.momentum

ssd_net = build_ssd('train', ssd_dim if not isinstance(ssd_dim, tuple) else '{:d}x{:d}'.format(*ssd_dim), num_classes)
net = ssd_net

if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    # net = ssd_net
    cudnn.benchmark = True

if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
else:
    vgg_weights = torch.load(args.basenet)    
    print('Loading base network...')
    # ssd_net.vgg.load_state_dict(vgg_weights)

    # # Initialize additional tower for lwir
    # loaded_state = OrderedDict()
    # state = ssd_net.lwir.state_dict()
    # state_keys = state.keys()
    
    # vgg_weights['0.weight'] = torch.mean( vgg_weights['0.weight'], dim=1, keepdim=True )
    # for k in state_keys:
    #     loaded_state[k] = vgg_weights[k]    
    
    # state.update(loaded_state)    
    # ssd_net.lwir.load_state_dict(state)

if args.cuda:
    net = net.cuda()


def weights_init(m):
    # if isinstance(m, nn.Conv2d):
    #     init.xavier_uniform(m.weight.data)
    #     m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n))
        init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        # m.weight.data.normal_(0, 0.01)
        init.xavier_uniform(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

    ssd_net.lwir.apply(weights_init)
    ssd_net.vgg.apply(weights_init)

# optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                       momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
# optimizer = optim.Adam(net.parameters(), lr=args.lr)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)



def write_kaist_results_file(all_boxes, dataset, result_name):
    print('Writing KAIST result file')
    filename = os.path.join(jobs_dir, '{:s}.txt'.format(result_name))
    with open(filename, 'wt') as f:
        for ii, bbs in enumerate(all_boxes[1]):
            if bbs == []:
                continue
            for bb in bbs:                
                f.write('{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
                    ii+1, bb[0],bb[1],bb[2]-bb[0]+1,bb[3]-bb[1]+1,bb[4]))


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
           filename, annopath, imgsetpath.format('test20'), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        # print('AP for {} = {:.4f}'.format(cls, ap))
        # with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        #     pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    return np.mean(aps)
    

def validation( net, loader, dataset, result_name ):

    net.eval()    
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    num_images = len(loader)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    # all_boxes = [[[] for _ in range(num_images)]
    #              for _ in range(len(labelmap)+1)]
    all_boxes = [ [[] for _ in range(num_images)] for _ in range(num_classes) ]

    for i, (color, lwir, targets, heights, widths, index) in enumerate(loader):
        if args.cuda:
            color = Variable(color.cuda())
            lwir = Variable(lwir.cuda())
            # targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
        else:
            color = Variable(color)
            lwir = Variable(lwir)
            # targets = [Variable(anno, volatile=True) for anno in targets]        

        h = heights[0]
        w = widths[0]

        _t['im_detect'].tic()
        # detections = net(images)[0].data
        detections = net(color, lwir).data
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

            # ## DEBUG
            # if np.any(scores > 0.5):
            #     ipdb.set_trace()
            #     import cv2
            #     frame_id = dataset_train.ids[index.data.cpu().numpy()]
            #     set_id, vid_id, img_id = frame_id[-1]
            #     img = cv2.imread(dataset._imgpath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id ), cv2.IMREAD_COLOR )
            #     bbs = cls_dets.astype(np.uint16)
            #     for b in bbs:
            #         cv2.rectangle(img, (int(b[0]),int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2 )
            #     cv2.imwrite('result.jpg', img)

            all_boxes[j][i] = cls_dets            

        if i % 100 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s'.format(i+1, num_images, detect_time))

    net.train()    

    write_kaist_results_file(all_boxes, dataset, result_name)
    
    # return do_python_eval()
    return 1.0

def trainval():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    print('Loading Dataset...')

    dataset_test = KAISTDetection(args.db_root, val_sets, BaseTransform(ssd_dim, means), AnnotationTransform())
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, 
                                batch_size=1, num_workers=4,
                                shuffle=False)

    dataset_train = KAISTDetection(args.db_root, train_sets, SSDAugmentation(ssd_dim, means), AnnotationTransform())
    loader_train = torch.utils.data.DataLoader(dataset_train, 
                                batch_size=batch_size, num_workers=args.num_workers,
                                shuffle=True, collate_fn=detection_collate, pin_memory=True)

    print('Training SSD on {}'.format(dataset_train.name))
    
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            env=exp_time + exp_name,
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                ytype='log',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss'],
                width=800, height=500, size=30
            )
        )
        
        # epoch_map = viz.line(
        #     X=torch.zeros((1,)).cpu(),
        #     Y=torch.zeros((1,)).cpu(),
        #     env=exp_name,
        #     opts=dict(
        #         xlabel='Epoch',
        #         ylabel='mAP',
        #         title='mAP of trained model',
        #         legend=['test mAP'],
        #         width=800, height=500, size=30
        #     )
        # )

    # max_epoch = args.iterations // len(loader_train) + 1
    max_epoch = args.epochs
    logger.info('Max epoch: {}'.format(max_epoch))

    milestones = [ int(max_epoch*0.5), int(max_epoch*0.75) ]
    optim_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    logger.info('Milestones for LR schedulring: {}'.format(milestones))

    best_mAP = 0.0    
    iter_per_epoch = len(loader_train)
    for epoch in range(max_epoch):

        logger.info('\n')

        optim_scheduler.step()

        # import cv2
        # import sys
        # if sys.version_info[0] == 2:
        #     import xml.etree.cElementTree as ET
        # else:
        #     import xml.etree.ElementTree as ET
                
        # for ii in range(len(dataset_train.ids)):
        #     frame_id = dataset_train.ids[ii]

        #     blob = dataset_train.pull_item(ii)

        #     ipdb.set_trace()

        # try:
        #     target_transform = AnnotationTransform()
        #     for ii in range(len(dataset_train.ids)):
        #         frame_id = dataset_train.ids[ii]
        #         target = ET.parse(dataset_train._annopath % ( *frame_id[:-1], *frame_id[-1] ) ).getroot()
                
        #         set_id, vid_id, img_id = frame_id[-1]
        #         vis = cv2.imread(dataset_train._imgpath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id ), cv2.IMREAD_COLOR )
        #         lwir = cv2.imread(dataset_train._imgpath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id ), cv2.IMREAD_COLOR )
                
        #         # target = ET.parse(dataset_train._annopath % ( *frame_id[:-1], *frame_id[-1] ) ).getroot()
        #         height, width, channels = vis.shape                                           
        #         target = target_transform(target, width, height)

        #         target = np.array(target)

        #         if len(target) == 0:
        #             target = np.array([[-0.01, -0.01, -0.01, -0.01, -1]], dtype=np.float)
        #         else:
        #             valid = np.zeros( (len(target), 1) )
        #             for ii, bb in enumerate(target):
        #                 x1, y1, x2, y2, lbl, occ = bb
        #                 x1, y1, x2, y2 = x1*width, y1*height, x2*width, y2*height
        #                 w = x2 - x1 + 1
        #                 h = y2 - y1 + 1

        #                 if occ in dataset_train.cond['vRng'] and \
        #                     x1 >= dataset_train.cond['xRng'][0] and \
        #                     x2 >= dataset_train.cond['xRng'][0] and \
        #                     x1 <= dataset_train.cond['xRng'][1] and \
        #                     x2 <= dataset_train.cond['xRng'][1] and \
        #                     y1 >= dataset_train.cond['yRng'][0] and \
        #                     y2 >= dataset_train.cond['yRng'][0] and \
        #                     y1 <= dataset_train.cond['yRng'][1] and \
        #                     y2 <= dataset_train.cond['yRng'][1] and \
        #                     h >= dataset_train.cond['hRng'][0] and \
        #                     h <= dataset_train.cond['hRng'][1]:

        #                     valid[ii] = 1
        #                 else:
        #                     valid[ii] = 0
                                    
        #             target = target[np.where(valid)[0], :]
        # except:
        #     ipdb.set_trace()


        #     vis, boxes, labels = dataset_train.transform(vis, target[:, :4].copy(), target[:, 4].copy())
        #     lwir, _, _ = dataset_train.transform(lwir, target[:, :4].copy(), target[:, 4].copy())

        #     target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        #     target[np.where(labels == -1)[0], :-1] = 0.0

        #     if np.any( boxes > 10.0 ):
        #         ipdb.set_trace()


        #     # if len(target) > 2:
        #     #     vis, lwir = dataset_train.pull_image(ii)
        #     #     frame_id, gt, anno = dataset_train.pull_anno(ii)

        #     #     for b in gt:
        #     #         cv2.rectangle( vis, (int(b[0]),int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2 )        
        #     #     cv2.imwrite( 'vis.jpg', vis)

        #     #     ipdb.set_trace()            

        # aug = SSDAugmentation(ssd_dim, means)
        # vis_ = aug( vis, np.array(target)[:, :4], np.array(target)[:, 4] )


        for _iter, (color, lwir, targets, _, _, index) in enumerate(loader_train):            

            if args.cuda:
                # images = Variable(images.cuda())
                color = Variable(color.cuda())
                lwir = Variable(lwir.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                # images = Variable(images)
                color = Variable(color)
                lwir = Variable(lwir)
                targets = [Variable(anno, volatile=True) for anno in targets]

            # forward
            t0 = time.time()
            
            out = net(color, lwir)
            # out, sources = net(images)
            # for src in sources:

            # backprop
            optimizer.zero_grad()
            
            # if np.all(targets[:,-1] == -1):
            #     ipdb.set_trace()

            # ipdb.set_trace()
            

            # loss_l, loss_c, problem = criterion(out, targets)
            loss_l, loss_c = criterion(out, targets)

            # if problem:
            #     continue               


            if loss_l.data.cpu().numpy() > 100:            
                ipdb.set_trace()

            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.data[0]
            conf_loss += loss_c.data[0]
            
            iteration = epoch * len(loader_train) + _iter

            if iteration % 10 == 0:
                logger.info('[Epoch {:3d}] [Iter {:5d}/{:d}] loss: {:3.4f} = {:3.4f} (loc) + {:3.4f} (cls)\
                    \t[time: {:.3f}sec] [# of GT in minibatch: {:2d}]'.format( \
                        epoch, iteration, iter_per_epoch, 
                        loss.data[0], loss_l.data[0], loss_c.data[0], 
                        t1-t0, np.sum([np.sum(box.data.cpu().numpy()[:,-1] == 0) for box in targets]) ) )

                if args.visdom and args.images_on_visdom:
                    random_batch_index = np.random.randint(images.size(0))
                    viz.image(images.data[random_batch_index].cpu().numpy())
            
            if args.visdom and iteration % 10 == 0:
                                
                viz.line(
                    X=torch.ones((1, 3)).cpu() * iteration,
                    Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                        loss_l.data[0] + loss_c.data[0]]).unsqueeze(0).cpu(),
                    env=exp_time + exp_name,
                    win=lot,
                    update='append'
                )           


        if ( epoch > 0 and epoch <= 100 and epoch % 20 == 0) or ( epoch > 100 and epoch % 10 == 0 ):        
            # if epoch >= 0 and epoch % 1 == 0:
            # New epoch, validation
            ssd_net.set_phase('test')
            mAP = validation( net, loader_test, dataset_test, 'SSD300_{:s}_epoch_{:04d}'.format(exp_name, epoch) )
            ssd_net.set_phase('train')

            #ipdb.set_trace()

            # viz.line(
            #     X=torch.ones((1, )).cpu() * (epoch+1),
            #     Y=torch.Tensor([mAP]).cpu(),
            #     win=epoch_map,
            #     update=True
            # )

            # if mAP > best_mAP:
            #     print('Best mAP = {:.4f}'.format(mAP))
            #     best_mAP = mAP

            filename = os.path.join(jobs_dir, 'snapshots', 'ssd300_epoch_{:03d}_mAP_{:.4f}.pth'.format(epoch, mAP))
            print('Saving state, {:s}'.format(filename))
            torch.save(ssd_net.state_dict(), filename)

        filename = os.path.join(jobs_dir, 'snapshots', 'ssd300_epoch_{:03d}.pth'.format(epoch))
        print('Saving state, {:s}'.format(filename))
        torch.save(ssd_net.state_dict(), filename)
    # torch.save(ssd_net.state_dict(), 'weights/{:s}_final.pth'.format(exp_time + exp_name))


if __name__ == '__main__':
    trainval()


