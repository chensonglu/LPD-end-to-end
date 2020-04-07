from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

CAR_CARPLATE_TWO_STAGE_END2END_CLASSES = (  # always index 0
    'car', 'carplate')


class CAR_CARPLATE_TWO_STAGE_END2ENDAnnotationTransform(object):
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
        self.class_to_ind = class_to_ind or dict(
            zip(CAR_CARPLATE_TWO_STAGE_END2END_CLASSES, range(len(CAR_CARPLATE_TWO_STAGE_END2END_CLASSES))))
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
            has_carplate = int(obj.find('has_carplate').text)

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            # has carplate
            bndbox.append(has_carplate)
            # offset, width and height of carplate
            offsets = ['width', 'height', 'x_offset', 'y_offset']
            for i, offset in enumerate(offsets):
                if has_carplate:
                    cur_offset = float(bbox.find(offset).text)
                    # scale height or width
                    cur_offset = cur_offset / width if i % 2 == 0 else cur_offset / height
                    bndbox.append(cur_offset)
                else:
                    bndbox.append(0)
            lp_pts = ['carplate_xmin', 'carplate_ymin', 'carplate_xmax', 'carplate_ymax',
                       'carplate_x_top_left', 'carplate_y_top_left', 'carplate_x_top_right', 'carplate_y_top_right',
                       'carplate_x_bottom_right', 'carplate_y_bottom_right', 'carplate_x_bottom_left', 'carplate_y_bottom_left']
            for i, lp_pt in enumerate(lp_pts):
                if has_carplate:
                    cur_lp_pt = float(bbox.find(lp_pt).text) - 1
                    # lp bbox or four corners
                    cur_lp_pt = cur_lp_pt / width if i % 2 == 0 else cur_lp_pt / height
                    bndbox.append(cur_lp_pt)
                else:
                    bndbox.append(0)
            # label -1
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            # if no carplate, has_carplate,size and offset are set to 0
            res += [bndbox]  # [xmin, ymin, xmax, ymax, has_carplate, width, height, x_offset, y_offset, carplate_bbox, carplate_four_points, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, has_carplate, width, height, x_offset, y_offset, carplate_bbox, carplate_four_points, label_ind], ... ]


class CAR_CARPLATE_TWO_STAGE_END2ENDDetection(data.Dataset):
    """VOC Detection Dataset Object

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
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=None,
                 transform=None, target_transform=CAR_CARPLATE_TWO_STAGE_END2ENDAnnotationTransform(keep_difficult=True),
                 dataset_name='trainval'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for line in open(osp.join(self.root, 'ImageSets', 'Main', self.name + '.txt')):
            self.ids.append((self.root, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :-1], target[:, -1])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        print(img_id)
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

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
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
