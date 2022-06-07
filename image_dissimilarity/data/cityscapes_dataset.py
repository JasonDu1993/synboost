import torch.utils.data as data
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from natsort import natsorted
from torchvision import transforms
import torch
import random
import cv2
from torchvision.transforms.functional import InterpolationMode

import sys

sys.path.append("..")
import image_dissimilarity.data.cityscapes_labels as cityscapes_labels
from image_dissimilarity.data.augmentations import get_transform

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid

# invalid frames are those where np.count_nonzero(labels_source) is 0 for Lost and Found Dataset
INVALID_LABELED_FRAMES = [17, 37, 55, 72, 91, 110, 129, 153, 174, 197, 218, 353, 490, 618, 686, 792, 793]


class CityscapesDataset(Dataset):

    def __init__(self, dataroot, preprocess_mode, crop_size=512, aspect_ratio=0.5, flip=False, normalize=False,
                 prior=False, only_valid=False, roi=False, light_data=False, void=False, num_semantic_classes=19,
                 is_train=True, is_opencv=False):
        self.is_opencv = is_opencv

        self.original_paths = [os.path.join(dataroot, 'original', image)
                               for image in os.listdir(os.path.join(dataroot, 'original'))]
        if light_data:
            self.semantic_paths = [os.path.join(dataroot, 'semantic_icnet', image)
                                   for image in os.listdir(os.path.join(dataroot, 'semantic_icnet'))]
            self.synthesis_paths = [os.path.join(dataroot, 'synthesis_spade', image)
                                    for image in os.listdir(os.path.join(dataroot, 'synthesis_spade'))]
        else:
            self.semantic_paths = [os.path.join(dataroot, 'semantic', image)
                                   for image in os.listdir(os.path.join(dataroot, 'semantic'))]
            self.synthesis_paths = [os.path.join(dataroot, 'synthesis', image)
                                    for image in os.listdir(os.path.join(dataroot, 'synthesis'))]
        if roi:
            self.label_paths = [os.path.join(dataroot, 'labels_with_ROI', image)
                                for image in os.listdir(os.path.join(dataroot, 'labels_with_ROI'))]
        elif void:
            self.label_paths = [os.path.join(dataroot, 'labels_with_void_no_ego', image)
                                for image in os.listdir(os.path.join(dataroot, 'labels_with_void_no_ego'))]
        else:
            self.label_paths = [os.path.join(dataroot, 'labels', image)
                                for image in os.listdir(os.path.join(dataroot, 'labels'))]
        if prior:
            if light_data:
                self.mae_features_paths = [os.path.join(dataroot, 'mae_features_spade', image)
                                           for image in os.listdir(os.path.join(dataroot, 'mae_features_spade'))]
                self.entropy_paths = [os.path.join(dataroot, 'entropy_icnet', image)
                                      for image in os.listdir(os.path.join(dataroot, 'entropy_icnet'))]
                self.logit_distance_paths = [os.path.join(dataroot, 'logit_distance_icnet', image)
                                             for image in os.listdir(os.path.join(dataroot, 'logit_distance_icnet'))]
            else:
                self.mae_features_paths = [os.path.join(dataroot, 'mae_features', image)
                                           for image in os.listdir(os.path.join(dataroot, 'mae_features'))]
                self.entropy_paths = [os.path.join(dataroot, 'entropy', image)
                                      for image in os.listdir(os.path.join(dataroot, 'entropy'))]
                self.logit_distance_paths = [os.path.join(dataroot, 'logit_distance', image)
                                             for image in os.listdir(os.path.join(dataroot, 'logit_distance'))]

        # We need to sort the images to ensure all the pairs match with each other
        self.original_paths = natsorted(self.original_paths)
        self.semantic_paths = natsorted(self.semantic_paths)
        self.synthesis_paths = natsorted(self.synthesis_paths)
        self.label_paths = natsorted(self.label_paths)
        if prior:
            self.mae_features_paths = natsorted(self.mae_features_paths)
            self.entropy_paths = natsorted(self.entropy_paths)
            self.logit_distance_paths = natsorted(self.logit_distance_paths)

        if only_valid:  # Only for Lost and Found
            self.original_paths = np.delete(self.original_paths, INVALID_LABELED_FRAMES)
            self.semantic_paths = np.delete(self.semantic_paths, INVALID_LABELED_FRAMES)
            self.synthesis_paths = np.delete(self.label_paths, INVALID_LABELED_FRAMES)
            self.label_paths = np.delete(self.label_paths, INVALID_LABELED_FRAMES)

        assert len(self.original_paths) == len(self.semantic_paths) == len(self.synthesis_paths) \
               == len(self.label_paths), \
            "Number of images in the dataset does not match with each other"
        "The #images in %s and %s do not match. Is there something wrong?"

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

    def one_hot_encoding(self, semantic, num_classes=20):
        if len(semantic.shape) == 4:
            B, C, H, W = semantic.shape
            semantic = semantic.squeeze(1)
            one_hot = torch.zeros(B, num_classes, H, W).to(semantic.device)
            for class_id in range(num_classes):
                one_hot[:, class_id, :, :] = (semantic == class_id)
            one_hot = one_hot[:, :num_classes - 1, :, :]
        elif len(semantic.shape) == 3:
            C, H, W = semantic.shape
            one_hot = torch.zeros(num_classes, H, W).to(semantic.device)
            for class_id in range(num_classes):
                one_hot[class_id, :, :] = (semantic == class_id)
            one_hot = one_hot[:num_classes - 1, :, :]
        return one_hot

    def open_img(self, path, convert_to_rgb=False):
        if self.is_opencv:
            img = cv2.imread(path)
            if convert_to_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.transpose(img, (2, 0, 1))
            else:
                img = img[:, :, 0][None, :, :]
        else:
            img = Image.open(path)
            if convert_to_rgb:
                img = img.convert('RGB')
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

    def __getitem__(self, index):

        # get and open all images
        label_path = self.label_paths[index]
        label = self.open_img(label_path)

        semantic_path = self.semantic_paths[index]
        semantic = self.open_img(semantic_path)

        image_path = self.original_paths[index]
        image = self.open_img(image_path, convert_to_rgb=True)

        syn_image_path = self.synthesis_paths[index]
        syn_image = self.open_img(syn_image_path, convert_to_rgb=True)

        if self.prior:
            mae_path = self.mae_features_paths[index]
            mae_image = self.open_img(mae_path)

            entropy_path = self.entropy_paths[index]
            entropy_image = self.open_img(entropy_path)

            distance_path = self.logit_distance_paths[index]
            distance_image = self.open_img(distance_path)

        # get input for transformations
        w = self.crop_size
        h = round(self.crop_size / self.aspect_ratio)
        image_size = (h, w)

        if self.flip:
            flip_ran = random.random() > 0.5
            label = _flip(label, flip_ran, self.is_opencv)
            semantic_tensor = _flip(semantic, flip_ran, self.is_opencv)
            image = _flip(image, flip_ran, self.is_opencv)
            syn_image = _flip(syn_image, flip_ran, self.is_opencv)
            if self.prior:
                mae_image = _flip(mae_image, flip_ran, self.is_opencv)
                entropy_image = _flip(entropy_image, flip_ran, self.is_opencv)
                distance_image = _flip(distance_image, flip_ran, self.is_opencv)

        # get augmentations
        base_transforms, augmentations = get_transform(image_size, self.preprocess_mode)
        if not self.is_opencv:
            # apply base transformations
            label = base_transforms(label) * 255
            semantic = base_transforms(semantic) * 255
            syn_image = base_transforms(syn_image)
            if self.prior:
                mae = base_transforms(mae_image)
                entropy = base_transforms(entropy_image)
                distance = base_transforms(distance_image)
            else:
                mae = []
                entropy = []
                distance = []

            if self.is_train and self.preprocess_mode != 'none':
                image = augmentations(image)
            else:
                image = base_transforms(image)

            if self.normalize:
                norm_transform = transforms.Compose(
                    [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # imageNet normamlization
                syn_image = norm_transform(syn_image)
                image = norm_transform(image)

            # post processing for semantic labels
            if self.num_semantic_classes == 19:
                semantic[semantic == 255] = self.num_semantic_classes + 1  # 'ignore label is 20'
            semantic = one_hot_encoding(semantic, self.num_semantic_classes + 1)
        else:
            label = self.pytorch_resize_totensor(label, (h, w), mul=255, interpolation=InterpolationMode.NEAREST,
                                                 totensor=True)
            semantic = self.pytorch_resize_totensor(semantic, (h, w), mul=255, interpolation=InterpolationMode.NEAREST,
                                                    totensor=True)
            syn_image = self.pytorch_resize_totensor(syn_image, (h, w), mul=1, interpolation=InterpolationMode.NEAREST,
                                                     totensor=True)
            if self.prior:
                mae = self.pytorch_resize_totensor(mae_image, (h, w), mul=1, interpolation=InterpolationMode.NEAREST,
                                                   totensor=True)
                entropy = self.pytorch_resize_totensor(entropy_image, (h, w), mul=1,
                                                       interpolation=InterpolationMode.NEAREST,
                                                       totensor=True)
                distance = self.pytorch_resize_totensor(distance_image, (h, w), mul=1,
                                                        interpolation=InterpolationMode.NEAREST,
                                                        totensor=True)
            else:
                mae = []
                entropy = []
                distance = []

            if self.is_train and self.preprocess_mode != 'none':
                image = augmentations(image)
            else:
                image = self.pytorch_resize_totensor(image, (h, w), mul=1, interpolation=InterpolationMode.NEAREST,
                                                     totensor=True)

            if self.normalize:
                norm_transform = transforms.Compose(
                    [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # imageNet normamlization
                syn_image = norm_transform(syn_image)
                image = norm_transform(image)

            # post processing for semantic labels
            if self.num_semantic_classes == 19:
                semantic[semantic == 255] = self.num_semantic_classes + 1  # 'ignore label is 20'
            semantic = self.one_hot_encoding(semantic, self.num_semantic_classes + 1)
        input_dict = {'label': label,
                      'original': image,
                      'semantic': semantic,
                      'synthesis': syn_image,
                      'label_path': label_path,
                      'original_path': image_path,
                      'semantic_path': semantic_path,
                      'syn_image_path': syn_image_path,
                      'entropy': entropy,
                      'mae': mae,
                      'distance': distance
                      }

        return input_dict

    def __len__(self):
        return self.dataset_size


def normalize():
    return


def _flip(img, flip, is_opencv=False):
    if flip:
        if is_opencv:
            return cv2.flip(img, 1)
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def one_hot_encoding(semantic, num_classes=20):
    one_hot = torch.zeros(num_classes, semantic.size(1), semantic.size(2))
    for class_id in range(num_classes):
        one_hot[class_id, :, :] = (semantic.squeeze(0) == class_id)
    one_hot = one_hot[:num_classes - 1, :, :]
    return one_hot


# ----------- FOR TESTING --------------

def test(dataset_args, dataloader_args, save_imgs=False, path='./visualization'):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    if save_imgs and not os.path.exists(path):
        os.makedirs(path)

    dataset = CityscapesDataset(**dataset_args)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    for counter, sample in enumerate(dataloader):
        print('Images Saved: ', sample['original'].shape[0] * counter)
        if save_imgs:
            transform = ToPILImage()
            decoder = DenormalizeImage(norm_mean, norm_std)
            for idx, (original, label, semantic, synthesis) in \
                    enumerate(zip(sample['original'], sample['label'], sample['semantic'], sample['synthesis'])):
                # get original image
                original = original.squeeze().cpu()
                original = decoder(original)
                original = np.asarray(transform(original))
                original = Image.fromarray(original)
                original.save(os.path.join(path, 'Original_%i_%i' % (counter, idx) + '.png'))

                # get label image
                label = label.squeeze().cpu().numpy()
                label = np.asarray(transform(label))
                label = Image.fromarray(label).convert('RGB')
                label.save(os.path.join(path, 'Label_%i_%i' % (counter, idx) + '.png'))

                # get semantic image
                semantic = semantic.squeeze().cpu().numpy()
                semantic = np.asarray(transform(semantic))
                semantic = visualization.colorize_mask(semantic)
                semantic = semantic.convert('RGB')
                semantic.save(os.path.join(path, 'Semantic_%i_%i' % (counter, idx) + '.png'))

                # get original image
                synthesis = synthesis.squeeze().cpu()
                synthesis = decoder(synthesis)
                synthesis = np.asarray(transform(synthesis))
                synthesis = Image.fromarray(synthesis)
                synthesis.save(os.path.join(path, 'Synthesis_%i_%i' % (counter, idx) + '.png'))


if __name__ == '__main__':
    from torchvision.transforms import ToPILImage
    import torch

    import sys

    sys.path.append("..")
    from image_dissimilarity.util.image_decoders import DenormalizeImage
    from image_dissimilarity.util import visualization

    dataset_args = {
        'dataroot': '/media/giancarlo/Samsung_T5/master_thesis/data/fs_static',
        'preprocess_mode': 'none',
        'crop_size': 512,
        'aspect_ratio': 2,
        'flip': True,
        'normalize': True,
        'void': False,
        'num_semantic_classes': 19,
        'is_train': False
    }

    dataloader_args = {
        'batch_size': 1,
        'num_workers': 1,
        'shuffle': False
    }

    test(dataset_args, dataloader_args, save_imgs=True)
