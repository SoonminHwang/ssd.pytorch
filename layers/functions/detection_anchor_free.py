import torch
from torch.autograd import Function
from ..box_utils import nms
from ..box_utils import decode
# from data import voc as cfg
import torch.nn.functional as F

class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, cfg, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, ref_vec):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']
        self.ref_vec = ref_vec

        self.num = cfg['num_anchor_bins']


    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        # num_priors = prior_data.size(0)
        num_priors = loc_data.size(1)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # reconstruct priors from prediction
        prior_w, prior_h = prior_data[:,:,2:].split(self.num, dim=2)    
        prior_w = (F.softmax(prior_w, dim=2) * self.ref_vec).sum(2, keepdim=True)
        prior_h = (F.softmax(prior_h, dim=2) * self.ref_vec).sum(2, keepdim=True)
        prior_data = torch.cat([prior_data[:,:,:2], prior_w, prior_h], dim=2)

        # bPredictedAnchor = (loc_data.dim() == prior_data.dim()).all()

        # Decode predictions into bboxes.
        for i in range(num):
            # decoded_boxes = decode(loc_data[i], prior_data, self.variance)        
            decoded_boxes = decode(loc_data[i], prior_data[i], self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output
