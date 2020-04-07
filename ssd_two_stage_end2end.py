import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import two_stage_end2end, change_cfg_for_ssd512
import os
import numpy as np

from layers.modules import ProposalTargetLayer_offset
# https://github.com/longcw/RoIAlign.pytorch
from roi_align.crop_and_resize import CropAndResizeFunction


def to_varabile(tensor, requires_grad=False, is_cuda=True):
    if is_cuda:
        tensor = tensor.cuda()
    var = Variable(tensor, requires_grad=requires_grad)
    return var


def a_include_b(a_bbox, b_bbox):
    include_or_not = False
    a_xmin, a_ymin, a_xmax, a_ymax = a_bbox
    b_xmin, b_ymin, b_xmax, b_ymax = b_bbox
    if (b_xmin >= a_xmin).cpu().numpy() and (b_ymin >= a_ymin).cpu().numpy()\
            and (b_xmax <= a_xmax).cpu().numpy() and (b_ymax <= a_ymax).cpu().numpy():
        include_or_not = True

    return include_or_not


class SSD_two_stage_end2end(nn.Module):
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

    def __init__(self, phase, size, size_2, base, extras, head, base_2, head_2, num_classes, expand_num):
        super(SSD_two_stage_end2end, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = two_stage_end2end
        if size == 512:
            self.cfg = change_cfg_for_ssd512(self.cfg)
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.priorbox_2 = PriorBox_2(self.cfg)
        with torch.no_grad():
            self.priors_2 = Variable(self.priorbox_2.forward())
        self.size = size
        self.size_2 = size_2
        self.expand_num = expand_num

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.has_lp = nn.ModuleList(head[2])
        self.size_lp = nn.ModuleList(head[3])
        self.offset = nn.ModuleList(head[4])

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.detect = Detect_offset(num_classes, 0, 200, 0.01, 0.45)

        # SSD network
        self.vgg_2 = nn.ModuleList(base_2)

        self.loc_2 = nn.ModuleList(head_2[0])
        self.conf_2 = nn.ModuleList(head_2[1])
        self.four_corners_2 = nn.ModuleList(head_2[2])

        if phase == 'test':
            self.softmax_2 = nn.Softmax(dim=-1)
            self.detect_2 = Detect_four_corners(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x, targets):
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
        loc = list()
        conf = list()
        has_lp = list()
        size_lp = list()
        offset = list()

        sources_2 = list()
        loc_2 = list()
        conf_2 = list()
        four_corners_2 = list()

        # apply vgg up to conv1_1 relu
        for k in range(2):
            x = self.vgg[k](x)
            if k == 1:
                # conv1_1 feature relu
                conv1_1_feat = x

        # apply vgg up to conv4_3 relu
        for k in range(2, 23):
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
        for (x, l, c, h, s, o) in zip(sources, self.loc, self.conf, self.has_lp, self.size_lp, self.offset):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            has_lp.append(h(x).permute(0, 2, 3, 1).contiguous())
            size_lp.append(s(x).permute(0, 2, 3, 1).contiguous())
            offset.append(o(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        has_lp = torch.cat([o.view(o.size(0), -1) for o in has_lp], 1)
        size_lp = torch.cat([o.view(o.size(0), -1) for o in size_lp], 1)
        offset = torch.cat([o.view(o.size(0), -1) for o in offset], 1)

        # [num, num_classes, top_k, 10]
        rpn_rois = self.detect(
            loc.view(loc.size(0), -1, 4),  # loc preds
            self.softmax(conf.view(conf.size(0), -1,
                                   self.num_classes)),  # conf preds
            self.priors.cuda(),  # default boxes
            self.sigmoid(has_lp.view(has_lp.size(0), -1, 1)),
            size_lp.view(size_lp.size(0), -1, 2),
            offset.view(offset.size(0), -1, 2)
        )

        rpn_rois = rpn_rois.detach()

        # roi align or roi warping
        crop_height = self.size_2
        crop_width = self.size_2
        is_cuda = torch.cuda.is_available()

        if self.phase == 'test':
            has_lp_th = 0.5
            th = 0.6
            output = torch.zeros(1, 3, 200, 13)
            output[0, 1, :, :5] = rpn_rois[0, 1, :, :5]

            rois_idx = (rpn_rois[0, 1, :, 0] > th) & (rpn_rois[0, 1, :, 5] > has_lp_th)
            matches = rpn_rois[0, 1, rois_idx, :]
            if matches.shape[0] == 0:
                return output

            car_center = (matches[:, [1, 2]] + matches[:, [3, 4]]) / 2
            lp_center = car_center + matches[:, [8, 9]]
            lp_bbox_top_left = lp_center - matches[:, [6, 7]] / 2 * self.expand_num
            lp_bbox_bottom_right = lp_center + matches[:, [6, 7]] / 2 * self.expand_num
            lp_bbox = torch.cat((lp_bbox_top_left, lp_bbox_bottom_right), 1)
            lp_bbox = torch.max(lp_bbox, torch.zeros(lp_bbox.shape))
            lp_bbox = torch.min(lp_bbox, torch.ones(lp_bbox.shape))
            lp_bbox = torch.max(lp_bbox, matches[:, 1:3].repeat(1, 2))
            lp_bbox = torch.min(lp_bbox, matches[:, 3:5].repeat(1, 2))

            # [num_car, 4]
            rois_squeeze = lp_bbox

            # Define the boxes ( crops )
            # box = [y1/heigth , x1/width , y2/heigth , x2/width]
            boxes_data = torch.zeros(rois_squeeze.shape)
            boxes_data[:, 0] = rois_squeeze[:, 1]
            boxes_data[:, 1] = rois_squeeze[:, 0]
            boxes_data[:, 2] = rois_squeeze[:, 3]
            boxes_data[:, 3] = rois_squeeze[:, 2]

            # Create an index to indicate which box crops which image
            box_index_data = torch.IntTensor(range(boxes_data.shape[0]))

            image_data = conv1_1_feat.repeat(rois_squeeze.shape[0], 1, 1, 1)

            # Convert from numpy to Variables
            image_torch = to_varabile(image_data, is_cuda=is_cuda, requires_grad=False)
            boxes = to_varabile(boxes_data, is_cuda=is_cuda, requires_grad=False)
            box_index = to_varabile(box_index_data, is_cuda=is_cuda, requires_grad=False)

            # Crops and resize bbox1 from img1 and bbox2 from img2
            # n*64*crop_height*crop_width
            crops_torch = CropAndResizeFunction.apply(image_torch, boxes, box_index, crop_height, crop_width, 0)

            # second network
            x_2 = crops_torch

            for k in range(4):
                x_2 = self.vgg_2[k](x_2)
            sources_2.append(x_2)

            for k in range(4, 9):
                x_2 = self.vgg_2[k](x_2)
            sources_2.append(x_2)

            for k in range(9, 14):
                x_2 = self.vgg_2[k](x_2)
            sources_2.append(x_2)

            # apply multibox head to source layers
            for (x_2, l_2, c_2, f_2) in zip(sources_2, self.loc_2, self.conf_2, self.four_corners_2):
                loc_2.append(l_2(x_2).permute(0, 2, 3, 1).contiguous())
                conf_2.append(c_2(x_2).permute(0, 2, 3, 1).contiguous())
                four_corners_2.append(f_2(x_2).permute(0, 2, 3, 1).contiguous())

            loc_2 = torch.cat([o.view(o.size(0), -1) for o in loc_2], 1)
            conf_2 = torch.cat([o.view(o.size(0), -1) for o in conf_2], 1)
            four_corners_2 = torch.cat([o.view(o.size(0), -1) for o in four_corners_2], 1)

            output_2 = self.detect_2(
                loc_2.view(loc_2.size(0), -1, 4),
                self.softmax_2(conf_2.view(conf_2.size(0), -1,
                                            self.num_classes)),
                self.priors_2.cuda(),
                four_corners_2.view(four_corners_2.size(0), -1, 8)
            )
            output_2_pos = output_2[:, 1, 0, :]
            rois_size = rois_squeeze[:, 2:4] - rois_squeeze[:, :2]
            rois_top_left = rois_squeeze[:, :2]
            rois_size_expand = rois_size.repeat(1, 6)
            rois_top_left_expand = rois_top_left.repeat(1, 6)
            output_2_pos[:, 1:] = output_2_pos[:, 1:] * rois_size_expand + rois_top_left_expand
            num_car = output_2_pos.shape[0]
            output[0, 2, :num_car, :] = output_2_pos
            output[0, 1, :num_car, 5:9] = lp_bbox
            output[0, 1, :num_car, 9] = 1

            return output
        else:
            print("ERROR: Phase: " + self.phase + " not recognized")
            return

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


def vgg_2(cfg, i, batch_norm=False):
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
    # SSD512 need add two more Conv layer
    if size == 512:
        layers += [nn.Conv2d(in_channels, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1)]
    return layers


def multibox(vgg, extra_layers, cfg, num_classes, vgg_2, cfg_2):
    loc_layers = []
    conf_layers = []
    has_lp_layers = []
    size_lp_layers = []
    offset_layers = []
    vgg_source = [21, -2]

    loc_layers_2 = []
    conf_layers_2 = []
    four_corners_layers_2 = []
    vgg_source_2 = [2, 7, 12]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
        has_lp_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * 1, kernel_size=3, padding=1)]
        size_lp_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * 2, kernel_size=3, padding=1)]
        offset_layers += [nn.Conv2d(vgg[v].out_channels,
                                  cfg[k] * 2, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
        has_lp_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * 1, kernel_size=3, padding=1)]
        size_lp_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * 2, kernel_size=3, padding=1)]
        offset_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * 2, kernel_size=3, padding=1)]

    for k, v in enumerate(vgg_source_2):
        loc_layers_2 += [nn.Conv2d(vgg_2[v].out_channels,
                                 cfg_2[k] * 4, kernel_size=3, padding=1)]
        conf_layers_2 += [nn.Conv2d(vgg_2[v].out_channels,
                        cfg_2[k] * num_classes, kernel_size=3, padding=1)]
        four_corners_layers_2 += [nn.Conv2d(vgg_2[v].out_channels,
                                          cfg_2[k] * 8, kernel_size=3, padding=1)]

    return vgg, extra_layers, (loc_layers, conf_layers, has_lp_layers, size_lp_layers, offset_layers),\
           vgg_2, (loc_layers_2, conf_layers_2, four_corners_layers_2)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '56': [512, 512, 'M', 512, 512, 'M', 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],
    '56': [6, 6, 6],
    '512': [4, 6, 6, 6, 6, 4, 4],
}


def build_ssd(phase, size=300, size_2=56, num_classes=21, expand_num=3):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300 and size != 512:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 SSD512 (size=300 or size=512) is supported!")
        return
    base_, extras_, head_, base_2_, head_2_ = multibox(vgg(base[str(size)], 3),
                                                      add_extras(extras[str(size)], size, 1024),
                                                      mbox[str(size)],
                                                      num_classes,
                                                      vgg_2(base[str(size_2)], 64),
                                                      mbox[str(size_2)]
                                                      )
    return SSD_two_stage_end2end(phase, size, size_2, base_, extras_, head_, base_2_, head_2_, num_classes, expand_num)
