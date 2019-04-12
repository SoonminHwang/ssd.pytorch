import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers import *
from layers.functions import Detect_AnchorFree
from layers.modules import L2Norm
# from data import voc, coco, voc_ssd_anchor_free
from data import coco, voc_ssd_anchor_free as voc
import os
import numpy as np
from itertools import product as product

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """    
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        if num_classes == 21:
        	self.cfg = voc[str(size)]            
        else:
            self.cfg = coco
        # self.cfg = (coco, voc)[num_classes == 21]
        # self.priorbox = PriorBox(self.cfg)
        #self.priors = Variable(self.priorbox.forward(), volatile=True)
        # self.priors = self.priorbox.forward().float().cuda()
        # self.priors_xy = self.priorbox.forward().float()[:,:2].cuda()	## Use only xy-coordinates

        image_size, feature_maps, steps, clip = self.cfg['min_dim'], self.cfg['feature_maps'], self.cfg['steps'], self.cfg['clip']
        self.priors_xy = self.get_prior_position(image_size, feature_maps, steps, clip).float().cuda()
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        ## Limit maximum scale depending on the level
        # self.ref_vec = torch.linspace(0,1,11,requires_grad=False).float().unsqueeze(0).cuda()
        self.ref_vec = torch.linspace(0,1,self.cfg['num_anchor_bins'],requires_grad=False).float().unsqueeze(0).cuda()
        ref_scales_mul = torch.ones( (self.priors_xy.size(0), 1), dtype=torch.float, requires_grad=False).cuda()
        ref_scales_add = torch.ones( (self.priors_xy.size(0), 1), dtype=torch.float, requires_grad=False).cuda()

        # nAnchors = [ b*f*f for b, f in zip(mbox[str(size)], self.cfg['feature_maps']) ]
        nAnchors = np.cumsum( [0] + [ f*f for f in self.cfg['feature_maps'] ] )        
        # max_scales = [ float(sz) / self.cfg['min_dim'] for sz in self.cfg['max_sizes'] ]

        for ii in range(len(nAnchors)-1):
            maxv, minv = float(self.cfg['max_sizes'][ii]), float(self.cfg['min_sizes'][ii])
            # ref_scales_mul[nAnchors[ii]:nAnchors[ii+1]] = (maxv - minv) / float(self.cfg['min_dim'])
            # ref_scales_add[nAnchors[ii]:nAnchors[ii+1]] = minv / float(self.cfg['min_dim'])
            
            ref_scales_mul[nAnchors[ii]:nAnchors[ii+1]] = 1.
            ref_scales_add[nAnchors[ii]:nAnchors[ii+1]] = 0.

            # ref_scales_mul[nAnchors[ii]:nAnchors[ii+1]] = .8
            # ref_scales_add[nAnchors[ii]:nAnchors[ii+1]] = .1

        # Broadcasting
        self.ref_vec = self.ref_vec * ref_scales_mul + ref_scales_add
        # self.ref_vec.unsqueeze_(0)

        self.prior = nn.ModuleList(head[0])
        self.loc = nn.ModuleList(head[1])
        self.conf = nn.ModuleList(head[2])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect_AnchorFree(self.cfg, num_classes, 0, 200, 0.01, 0.45, self.ref_vec)

    def __str__(self):
        return __class__.__name__ + str(self.size)

    def get_prior_position(self, image_size, feature_maps, steps, clip=True):
        mean = []
        for k, f in enumerate(feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = image_size / steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k                
                mean += [cx, cy]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 2)
        if clip:
            output.clamp_(max=1, min=0)
        return output

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        prior = list()
        loc = list()
        conf = list()

        if self.priors_xy.size(0) != x.size(0):
        	self.priors_xy = self.priors_xy.unsqueeze(0).repeat(x.size(0), 1, 1)

        device = x.get_device()
        self.priors_xy = self.priors_xy.to(device)

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, p, l, c) in zip(sources, self.prior, self.loc, self.conf):
            prior.append(p(x).permute(0, 2, 3, 1).contiguous())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())        
        
        loc = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in conf], 1)
        priors_wh = torch.cat([o.view(o.size(0), -1, 2*self.cfg['num_anchor_bins']) for o in prior], 1)        
        priors = torch.cat([self.priors_xy, priors_wh], dim=2)

        if self.phase == "test":
            output = self.detect(
                loc, 
                self.softmax(conf), 
                priors.type(type(x.data)),            	
            ).data
        else:
            output = (loc, conf, priors)
        
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, size, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v

    # SSD512 need add one more Conv layer(Conv12_2)
    if size == 512:
        layers += [nn.Conv2d(in_channels, 256, kernel_size=4, padding=1)]
    return layers


def multibox(vgg, extra_layers, num_classes, num_anchor_bins):
    prior_layers = []
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    # vgg_source = [24, -2] ??
    for k, v in enumerate(vgg_source):
        prior_layers += [nn.Conv2d(vgg[v].out_channels, 2*num_anchor_bins, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(vgg[v].out_channels, 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        prior_layers += [nn.Conv2d(v.out_channels, 2*num_anchor_bins, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(v.out_channels, 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (prior_layers, loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '320': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128],
}
# mbox = {
#     '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
#     '320': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
#     '512': [4, 6, 6, 6, 6, 4, 4],
# }


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    # if size != 300:
    #     print("ERROR: You specified size " + repr(size) + ". However, " +
    #           "currently only SSD300 (size=300) is supported!")
    #     return
    cfg = (coco, voc)[num_classes == 21][str(size)]

    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], size, 1024),
                                     num_classes, cfg['num_anchor_bins'])
    return SSD(phase, size, base_, extras_, head_, num_classes)
