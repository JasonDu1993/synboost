# Author: Xiangtai Li
# Email: lxtpku@pku.edu.cn

import os
import numpy as np
import random
import cv2
from torch.utils import data


class MapDataSet(data.Dataset):
    def __init__(self, root, split="train", max_iters=80000, crop_size=(321, 321), mean=(128, 128, 128), vars=(1, 1, 1),
                 scale=True,
                 mirror=True, ignore_label=255, RGB=False):
        self.root = root
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.vars = vars
        self.is_mirror = mirror
        self.rgb = RGB
        self.img_list, self.label_list = self._make_dataset(root, split)
        assert len(self.label_list) == len(self.img_list)
        print("Found dataset {} images".format(len(self.img_list)))
        if not max_iters == None:
            self.img_total = self.img_list * int(np.ceil(float(max_iters) / len(self.img_list)))
            self.label_total = self.label_list * int(np.ceil(float(max_iters) / len(self.label_list)))
        self.pair_list = []
        for i, img in enumerate(self.img_total):
            self.pair_list.append({
                "image": img,
                "label": self.label_total[i]
            })
        print('Total {} images are loaded!'.format(len(self.pair_list)))

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __len__(self):
        return len(self.pair_list)

    def generate_scale_label(self, image, label):
        f_scale = 0.7 + random.randint(0, 14) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def _make_dataset(self, root, split="train"):
        image_list = []
        label_list = []
        if split == "train":
            floder = os.path.join(root, "training")
            image_floder = os.path.join(floder, "images")
            label_floder = os.path.join(floder, "seg19_lbl")
            for sub_file in os.listdir(image_floder):
                image_list.append(os.path.join(image_floder, sub_file))
            for sub_file in os.listdir(label_floder):
                label_list.append(os.path.join(label_floder, sub_file))
        if split == "trainval":
            train_floder = os.path.join(root, "training")
            val_floder = os.path.join(root, "validation")

            image_floder = os.path.join(train_floder, "images")
            label_floder = os.path.join(train_floder, "seg19_lbl")
            for sub_file in os.listdir(image_floder):
                image_list.append(os.path.join(image_floder, sub_file))
            for sub_file in os.listdir(label_floder):
                label_list.append(os.path.join(label_floder, sub_file))

            image_floder = os.path.join(val_floder, "images")
            label_floder = os.path.join(val_floder, "seg19_lbl")
            for sub_file in os.listdir(image_floder):
                image_list.append(os.path.join(image_floder, sub_file))
            for sub_file in os.listdir(label_floder):
                label_list.append(os.path.join(label_floder, sub_file))

        image_list.sort()
        label_list.sort()
        return image_list, label_list

    def __getitem__(self, index):
        datafiles = self.pair_list[index]
        image = cv2.imread(datafiles["image"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label = self.id2trainId(label)
        size = image.shape

        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)

        if self.rgb:
            image = image[:, :, ::-1]  ## BGR -> RGB
            image /= 255  ## using pytorch pretrained models

        image -= self.mean
        image /= self.vars

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size)
