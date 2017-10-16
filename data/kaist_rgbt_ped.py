"""
    KAIST Multispectral Pedestrian Dataset Classes
    Original author: Francisco Massa
    https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
    Updated by: Ellis Brown, Max deGroot, Soonmin Hwang
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

import ipdb


DBroot = os.path.join('data', 'kaist-rgbt')



# KAIST_CLASSES = (  # always index 0
#     'cyclist', 'person', 'people', 'person?')
KAIST_CLASSES = {  # always index 0
    'cyclist':0, 'person':0, 'people':-1, 'person?':-1}

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

# [vRng] 0: non-occlusion, 1: partial, 2: heavy
LOAD_CONDITIONS = {
    'Reasonable': {'hRng': (45, np.inf), 'vRng': (0, 1), 'xRng':(5, 635), 'yRng':(5, 475)},
    'Near': {'hRng': (115, np.inf), 'vRng': (0), 'xRng':(5, 635), 'yRng':(5, 475)}
}

class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        # self.class_to_ind = class_to_ind or dict(
        #     zip(KAIST_CLASSES, range(len(KAIST_CLASSES))))
        self.class_to_ind = class_to_ind or KAIST_CLASSES
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()            
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                # cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = float(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            bndbox.append( int(obj.find('occlusion').text) )
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind, occ]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class KAISTDetection(data.Dataset):
    """KAIST Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'KAIST')
        condition (string, optional): load condition
            (default: 'Reasonabel')
    """

    def __init__(self, root, image_sets, transform=None, target_transform=None,
                 dataset_name='KAIST', condition='Reasonable'):

        assert condition in LOAD_CONDITIONS

        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.cond = LOAD_CONDITIONS[condition]

        
        # {SET_ID}/{VID_ID}/{IMG_ID}.jpg
        self._annopath = os.path.join('%s', 'annotations-xml', '%s', '%s', '%s.xml')
        # {SET_ID}/{VID_ID}/{MODALITY}/{IMG_ID}.jpg
        self._imgpath = os.path.join('%s', 'images', '%s', '%s', '%s', '%s.jpg')  
        self.ids = list()

        for (skip, name) in image_sets:
            self.name += '_' + name + skip
            for line in open(os.path.join(self.root, 'imageSets', '{:s}{:02d}.txt'.format(name, int(skip)))):
                self.ids.append((self.root, line.strip().split('/')))

    def __getitem__(self, index):
        vis, lwir, gt, h, w = self.pull_item(index)
        return vis, lwir, gt, h, w, index

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        frame_id = self.ids[index]
        target = ET.parse(self._annopath % ( *frame_id[:-1], *frame_id[-1] ) ).getroot()
        
        set_id, vid_id, img_id = frame_id[-1]
        vis = cv2.imread(self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id ), cv2.IMREAD_COLOR )
        lwir = cv2.imread(self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id ), cv2.IMREAD_COLOR )
        
        assert vis.shape[:2] == lwir.shape[:2]

        height, width, channels = vis.shape
                       

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        try:
            if self.transform is not None:
                target = np.array(target)

                valid = np.zeros( (len(target)) )
                for ii, bb in enumerate(target):
                    x1, y1, x2, y2, lbl, occ = bb
                    x1, y1, x2, y2 = x1*width, y1*height, x2*width, y2*height
                    w = x2 - x1 + 1
                    h = y2 - y1 + 1

                    if occ in self.cond['vRng'] and \
                        x1 >= self.cond['xRng'][0] and \
                        x2 >= self.cond['xRng'][0] and \
                        x1 <= self.cond['xRng'][1] and \
                        x2 <= self.cond['xRng'][1] and \
                        y1 >= self.cond['yRng'][0] and \
                        y2 >= self.cond['yRng'][0] and \
                        y1 <= self.cond['yRng'][1] and \
                        y2 <= self.cond['yRng'][1] and \
                        h >= self.cond['hRng'][0] and \
                        h <= self.cond['hRng'][1]:

                        valid[ii] = 1
                    else:
                        valid[ii] = 0


                target = target[np.where(valid == 1)]


                if len(target) == 0:
                    target = np.array([[-0.01, -0.01, -0.01, -0.01, -1, 2]], dtype=np.float)

                vis, boxes, labels = self.transform(vis, target[:, :4].copy(), target[:, 4].copy())
                lwir, _, _ = self.transform(lwir, target[:, :4].copy(), target[:, 4].copy())
                # to rgb
                vis = vis[:, :, (2, 1, 0)]
                lwir = lwir[:, :, :1]
                # img = img.transpose(2, 0, 1)
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
                # target = target[:, np.where(labels != -1)[0]]
                target[np.where(labels == -1)[0], :-1] = 0.0
        except:
            ipdb.set_trace()


        return torch.from_numpy(vis).permute(2, 0, 1), torch.from_numpy(lwir).permute(2, 0, 1), target, height, width
        # lwir = lwir[:,:,np.newaxis]
        # return torch.from_numpy(vis).permute(2, 0, 1), torch.from_numpy(lwir).permute(2, 0, 1), target, height, width
        

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        frame_id = self.ids[index]
        set_id, vid_id, img_id = frame_id[-1]
        
        vis = cv2.imread(self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id ), cv2.IMREAD_COLOR )
        lwir = cv2.imread(self._imgpath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id ), cv2.IMREAD_GRAYSCALE )

        return vis, lwir

    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        # img_id = self.ids[index]
        # anno = ET.parse(self._annopath % img_id).getroot()
        frame_id = self.ids[index]
        anno = ET.parse(self._annopath % ( *frame_id[:-1], *frame_id[-1] ) ).getroot()

        gt = self.target_transform(anno, 1, 1)
        return frame_id, gt, anno

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        # return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
        vis, lwir = self.pull_image(index)
        return torch.Tensor(vis).unsqueeze_(0), torch.Tensor(lwir).unsqueeze_(0)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    vis = []
    lwir = []
    heights = []
    widths = []

    index = []
    for sample in batch:
        vis.append(sample[0])
        lwir.append(sample[1])
        targets.append(torch.FloatTensor(sample[2]))
        heights.append(sample[3])
        widths.append(sample[4])
        index.append(sample[5])
    
    return torch.stack(vis, 0), torch.stack(lwir, 0), targets, heights, widths, index