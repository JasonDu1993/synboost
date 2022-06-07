# -*- coding: utf-8 -*-
# @Time    : 2022/6/7 15:07
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : image_dataset.py
# @Software: PyCharm
import os
from natsort import natsorted
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
from image_dissimilarity.data.augmentations import get_transform


class ImageDataset(Dataset):

    def __init__(self, dataroot, preprocess_mode, input_shape, crop_size=512, aspect_ratio=0.5, flip=False, normalize=False,
                 prior=False, only_valid=False, roi=False, light_data=False, void=False, num_semantic_classes=19,
                 is_train=True, is_opencv=True):
        self.is_opencv = is_opencv

        self.original_paths = [os.path.join(dataroot, 'original', image)
                               for image in os.listdir(os.path.join(dataroot, 'original'))]

        if roi:
            self.label_paths = [os.path.join(dataroot, 'labels_with_ROI', image)
                                for image in os.listdir(os.path.join(dataroot, 'labels_with_ROI'))]
        elif void:
            self.label_paths = [os.path.join(dataroot, 'labels_with_void_no_ego', image)
                                for image in os.listdir(os.path.join(dataroot, 'labels_with_void_no_ego'))]
        else:
            self.label_paths = [os.path.join(dataroot, 'labels', image)
                                for image in os.listdir(os.path.join(dataroot, 'labels'))]

        # We need to sort the images to ensure all the pairs match with each other
        self.original_paths = natsorted(self.original_paths)
        self.label_paths = natsorted(self.label_paths)

        self.dataset_size = len(self.original_paths)
        self.preprocess_mode = preprocess_mode
        self.crop_size = crop_size
        self.aspect_ratio = aspect_ratio
        self.num_semantic_classes = num_semantic_classes
        self.is_train = is_train
        self.void = void
        self.flip = flip
        self.prior = prior
        self.normalize = normalize

        self.w = self.crop_size
        self.h = round(self.crop_size / self.aspect_ratio)
        self.img_c, self.img_h, self.img_w = input_shape

    def open_img(self, path, convert_to_rgb=False):
        img = cv2.imread(path)
        if convert_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
        else:
            img = img[:, :, 0][None, :, :]
        return img

    def to_tensor(self, img, is_norm=True):
        """将图片数据转换成0-1范围内的数据

        Args:
            img:
            is_norm:

        Returns:

        """
        img = img.float()
        if is_norm:
            img = img.div(255)
        return img

    def pytorch_resize_totensor(self, x, size=(256, 512), mul=1, interpolation=InterpolationMode.NEAREST,
                                totensor=True):
        if isinstance(x, Image.Image):
            x = np.array(x).transpose((2, 0, 1))
            x = torch.from_numpy(x)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        # x = x.type(torch.int)
        h, w = size

        if x.dim() < 3:
            x = x.unsqueeze(0)
        transform_semantic_resize = transforms.Resize(size=(h, w), interpolation=interpolation)
        x = transform_semantic_resize(x)
        if totensor:
            x = self.to_tensor(x)
        x = x * mul
        return x

    def img_preprocess(self, image, input_shape):
        if isinstance(image, str):
            image = cv2.imread(image)
        # image = Image.open(img_path)
        c, h, w = input_shape
        img_np = cv2.resize(image, (w, h))  # [h, w, c]  , 3 is bgr
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # 颜色转换
        img = torch.from_numpy(img_np)
        img = img.permute((2, 0, 1)).contiguous()  # 通道转换C,H,W
        return img

    def __getitem__(self, index):
        label_path = self.label_paths[index]
        label = self.open_img(label_path, convert_to_rgb=False)
        label = self.pytorch_resize_totensor(label, (self.h, self.w), mul=255, interpolation=InterpolationMode.NEAREST,
                                             totensor=True)

        image_path = self.original_paths[index]
        image = self.img_preprocess(image_path, (3, self.img_h, self.img_w))

        input_dict = {'label': label,
                      'original': image,
                      'label_path': label_path,
                      'original_path': image_path,
                      }

        return input_dict

    def __len__(self):
        return self.dataset_size
