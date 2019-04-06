import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['StairNet', 'stairnet300']

model_urls = {
    'stairnet300': ''
}

s_cfg = {
    '300': [38, 19, 10, 5, 3, 1],
    '512': [64, 32, 16, 8, 4, 2, 1],
}
    
ch_cfg = {
    '300': [512, 1024, 512 ,256, 256, 256],
    '512': [512, 1024, 512, 256, 256, 256, 256],
}

class StairNet(nn.Module):
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
        super(StairNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        # Use the same default anchors to original SSD300
        self.cfg = (coco, voc)[num_classes == 21]        
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = self.priorbox.forward().float().cuda()
        self.size = size

        # SSD-VGG16 network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        inter_planes = 256
        self.topdown = TopDownNet( reversed(s_cfg[str(size)]), reversed(ch_cfg[str(size)]), inter_planes)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def __str__(self):
        return __class__.__name__ + str(self.size)

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
        hs = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        hs.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        hs.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                hs.append(x)

        sources = self.topdown(hs)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            ).data
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

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


def add_extras(cfg, i, batch_norm=False):
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
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_stairnet(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return StairNet(phase, size, base_, extras_, head_, num_classes)

##########################################################################################
##########################################################################################

### Lateral module
class BasicConv(nn.Module):
    '''Basic convolution block for lateral connection'''
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

### Topdown module
class BasicTopDown(nn.Module):
    '''Basic convolution block for top-down connection'''
    def __init__(self, in_planes, out_planes, params):
        super(BasicTopDown, self).__init__()        

        locals().update(params)
        
        self.out_channels = out_planes
        
        if upsample_type.lower() == 'deconv':
            ''' params dictionary should contain the following keys: in_planes, out_planes, kernel_size, etc.'''                        
            self.upsample = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                                        dilation=dilation, groups=groups, bias=bias)
        elif upsample_type.lower() == 'bilinear':            
            self.upsample = nn.Upsample(size=size, scale_factor=scale_factor, mode=mode)
        elif upsample_type.lower() == 'bilinear+conv':
            self.upsample == nn.Sequential(
                nn.Upsample(size=size, mode='bilinear'),
                nn.Conv2d(inplanes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
            )

        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.upsample(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


### Fusion module: combine lateral and top-down connections
class Sum(nn.Module):
    def forward(self, x, y):
        return x + y

class Prod(nn.Module):
    def forward(self, x, y):
        return x * y

class BasicFusion(nn.Module):
    def __init__(self, fusion_type='sum'):
        super(BasicFusion, self).__init__()
        self.fusion_type = fusion_type.lower()

        if self.fusion_type == 'sum':
            self.module = Sum()
        elif self.fusion_type == 'prod':
            self.module = Prod()
        else:
            raise NotImplementedError

    def forward(self, h_top, h_cur):
        return self.module(h_top, h_cur)


class TopDownNet(nn.Module):

    def __init__(self, feats_spatial_resol, feats_channel, inter_planes):
        super(TopDownNet, self).__init__()
        
        num_hs = len(feats_spatial_resol)

        ##### 0. Projection: match channels (lateral)
        proj_params = {in_planes=inter_planes, out_planes=inter_planes, \
                        kernel_size=1, stride=1, padding=0, dilation=1, \
                        groups=1, relu=False, bn=True, bias=False}
        proj = list()
        for ii in range(num_hs):
            proj_params['in_planes'] = feats_channel[ii-1]
            proj.append( BasicConv(proj_params) )        
        # proj = [ BasicConv(proj_params) for _ in range(num_hs) ]

        ##### 1. Top-down
        # upsample_type: 'deconv'
        topdown_params = {upsample_type='deconv', in_planes=inter_planes, \
                        out_planes=inter_planes, kernel_size=3, stride=2, \
                        padding=1, output_padding=0, dilation=1, groups=1, relu=False}
        topdown = list()
        for ii in range(num_hs-1:)            
            if feats_spatial_resol[ii+1] % feats_spatial_resol[ii] == 0:                
                topdown_params['output_padding'] = 1
            else:
                topdown_params['output_padding'] = 0

            topdown.append( BasicTopDown(topdown_params) )

        # # # upsample_type: 'bilinear'
        # # topdown_params = {upsample_type='bilinear', out_planes, size, scale_factor=None, mode='bilinear', relu=True, bn=True}        
        # # # upsample_type: 'bilinear+conv'
        # # topdown_params = {upsample_type='bilinear+conv', size, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False}
        # topdown = [ BasicTopDown(topdown_params) for _ in range(num_hs) ]

        # ##### 2. Lateral
        # lateral_params = {in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False}
        # lateral = [ BasicConv(lateral_params) for _ in range(num_hs) ]

        ##### 3. Fusion (combining top-down and lateral paths)
        fusion_params = {fusion_type='sum'}
        # fusion_params = {fusion_type='prod'}
        # fusion_params = {fusion_type='max'}
        fusion = [ BasicFusion(fusion_params) for _ in range(num_hs) ]
        

        ##### 4. Extra conv. for multi-scale detection head
        extraconv_params = {in_planes=inter_planes, out_planes=inter_planes, \
                            kernel_size=3, stride=1, padding=1, dilation=1, \
                            groups=1, relu=True, bn=True, bias=False}
        extraconv = [ BasicConv(extraconv_params) for _ in range(num_hs) ]
        

        self.proj_layers = nn.ModuleList(proj)
        self.topdown_layers = nn.ModuleList(topdown)
        # self.lateral_layers = nn.ModuleList(lateral)
        self.fusion_layers = nn.ModuleList(fusion)
        self.extra_layers = nn.ModuleList(extraconv)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feats):

        sources = list()
        for ii, hs_cur in enumerate(reversed(feats)):
            
            from_cur = self.proj_layers[ii](hs_cur)

            if ii == 0:
                hs_top = from_cur
                continue

            from_top = self.topdown_layers[ii-1](hs_top)
            # from_cur = self.lateral_layers[ii-1](hs_cur)

            hs_top = self.fusion_layers[ii-1](from_top, from_cur)

            sources.insert( 0, self.extra_layers[ii-1](hs_top) )

        return sources


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


