# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import two_stage_end2end as cfg
from ..box_utils import match_two_stage_end2end_offset, log_sum_exp


class ProposalTargetLayer_offset(nn.Module):
    def __init__(self):
        super(ProposalTargetLayer_offset, self).__init__()

    def forward(self, rpn_rois, targets, expand_num):
        num = rpn_rois.shape[0]
        rois_t = []

        for idx in range(num):
            truths = targets[idx].data
            defaults = rpn_rois[idx, 1, :, :]
            match_two_stage_end2end_offset(truths, defaults, rois_t, expand_num)

        return rois_t