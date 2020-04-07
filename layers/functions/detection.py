import torch
from torch.autograd import Function
from ..box_utils import decode, decode_offset, decode_size, decode_four_corners, nms
from data import two_stage_end2end as cfg

import warnings
warnings.filterwarnings('ignore')


class Detect_offset(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data, has_lp_data, size_lp_data, offset_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors,4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch,num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors,4]
            has_lp_data: (tensor) Has LP preds from has_lp layers
                Shape: [batch,num_priors,1]
            size_lp_data: (tensor) Size LP preds from size_lp layers
                Shape: [batch,num_priors,2]
            offset_data: (tensor) Offset preds from offset layers
                Shape: [batch,num_priors,2]
        """

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 10)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)  # [batch,num_classes,num_priors]
        has_lp_preds = has_lp_data.view(num, num_priors, 1).transpose(2, 1)  # [batch,1,num_priors]

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)  # [num_priors,4]
            decoded_size_lp = decode_size(size_lp_data[i], prior_data, self.variance)  # [num_priors,2]
            decoded_offset = decode_offset(offset_data[i], prior_data, self.variance)  # [num_priors,2]
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()  # [num_classes,num_priors]
            has_lp_scores = has_lp_preds[i].clone()  # [1,num_priors]

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)  # [num_priors]
                scores = conf_scores[cl][c_mask]  # [number great than conf_threshold]
                scores_lp = has_lp_scores[0][c_mask]
                if scores.size(0) == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)  # [number great than conf_threshold, 4]
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_size_lp)
                size_lp = decoded_size_lp[l_mask].view(-1, 2)
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_offset)
                offset = decoded_offset[l_mask].view(-1, 2)

                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]], scores_lp[ids[:count]].unsqueeze(1),
                               size_lp[ids[:count]], offset[ids[:count]]), 1)

        flt = output.contiguous().view(num, -1, 10)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        return output


class Detect_four_corners(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']

    def forward(self, loc_data, conf_data, prior_data, four_corners_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors,4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch,num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors,4]
            four_corners_data: (tensor) Four corners preds from four_corners layers
                Shape: [batch,num_priors,8]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 13)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)  # [batch,num_classes,num_priors]

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)  # [num_priors,4]
            decoded_corners = decode_four_corners(four_corners_data[i], prior_data, self.variance)  # [num_priors,8]
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()  # [num_classes,num_priors]

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)  # [num_priors]
                scores = conf_scores[cl][c_mask]  # [number great than conf_threshold]
                if scores.size(0) == 0:
                    continue

                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)  # [number great than conf_threshold, 4]
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_corners)
                corners = decoded_corners[l_mask].view(-1, 8)  # [number great than conf_threshold, 8]
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]], corners[ids[:count]]), 1)

        flt = output.contiguous().view(num, -1, 13)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        return output
        