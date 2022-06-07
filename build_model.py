# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 19:57
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : build_model.py
# @Software: PyCharm
import argparse
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import yaml
import random
from options.config_class import Config
import cv2
import time as t
from torchvision.transforms.functional import InterpolationMode

import sys

sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(__file__), 'image_segmentation'))
from image_segmentation.datasets.cityscapes_labels import trainId2labelId
import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(__file__), 'image_synthesis'))
from image_synthesis.models.pix2pix_model import Pix2PixModel
from image_dissimilarity.models.dissimilarity_model import DissimNetPrior, DissimNet
from image_dissimilarity.models.vgg_features import VGG19_difference
from image_dissimilarity.data.cityscapes_dataset import one_hot_encoding

# sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(__file__), 'total_utils'))
from total_utils.seg_to_rect import postprocessing
from total_utils.draw_box_util import draw_total_box
import torch.nn as nn
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class RoadAnomalyDetector(nn.Module):
    def __init__(self, ours=True, input_shape=(3, 512, 1024), seed=0, fishyscapes_wrapper=True, vis=False):
        super().__init__()
        self.set_seeds(seed)

        # Common options for all models
        TestOptions = Config()
        self.opt = TestOptions
        torch.cuda.empty_cache()
        self.get_segmentation()
        self.get_synthesis()
        self.get_dissimilarity(ours)
        self.get_transformations()
        self.fishyscapes_wrapper = fishyscapes_wrapper
        self.c, self.h, self.w = input_shape
        # self.id_to_trainid = self.opt.dataset_cls.id_to_trainid
        self.id_to_trainid = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255,
                              11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7,
                              21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
                              31: 16, 32: 17, 33: 18, -1: -1}
        """产生下面的id_index
        a = list(np.arange(35).tolist())
        for trainid in trainId2labelId:
            labelid = trainId2labelId[trainid]
            a[trainid], a[labelid] = a[labelid], a[trainid]
        m = [0] * 35
        for i in range(len(a)):
            m[a[i]] = i
        print(m)
        """
        self.id_index = [20, 21, 24, 25, 26, 32, 19, 0, 1, 22, 23, 2, 3, 4, 27, 28, 31, 5, 33, 6, 7, 8, 9, 10, 11, 12,
                         13, 14, 15, 29, 30, 16, 17, 18, 34]
        self.vis = vis

    def valid(self, x, path="image.npy", thd=1e-3):
        image_f = np.load(path)
        image_f = torch.from_numpy(image_f).to(x.device)
        a = torch.abs(image_f - x)
        a[a < thd] = 0
        b = (a != 0).float()
        b = torch.sum(b)
        print(path, b, a.shape, b / a.numel(), torch.max(a))

    def one_hot_encoding(self, semantic, num_classes=20):
        B, C, H, W = semantic.shape
        semantic = semantic.squeeze(1)
        one_hot = torch.zeros(B, num_classes, H, W).to(semantic.device)
        for class_id in range(num_classes):
            one_hot[:, class_id, :, :] = (semantic == class_id)
        one_hot = one_hot[:, :num_classes - 1, :, :]
        return one_hot

    def pytorch_resize_totensor(self, x, size=(256, 512), mul=1, interpolation=InterpolationMode.NEAREST, totensor=True):
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

    def forward(self, img):
        """

        Args:
            img:shape[B, C, H, W], C is 3(RGB)

        Returns:

        """
        torch.cuda.synchronize()
        t0 = time()
        B, C, H, W = img.shape
        image_og_h, image_og_w = H, W
        # img = image.resize((self.w, self.h))
        # img_np = cv2.resize(image, (self.w, self.h))
        # img_np_chw = img_np[:, :, ::-1].transpose(2, 0, 1).copy()
        # 第一步：segnet
        # 0 segnet img origin
        # img_tensor = self.img_transform(img)
        # 0 segnet img pytorch
        # img = self.pytorch_resize_totensor(image, size=(self.h, self.w), mul=1, interpolation=InterpolationMode.NEAREST, totensor=False)
        # img = torch.from_numpy(img_np_chw).to("cuda:0")
        # img = img.squeeze()
        img_tensor = self.to_tensor(img)
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        seg_norm = transforms.Normalize(*mean_std)
        img_tensor = seg_norm(img_tensor)
        # self.valid(img_tensor, "image.npy")
        # np.save("image.npy", img_tensor.unsqueeze(0).cuda().cpu().numpy())
        torch.cuda.synchronize()
        t1 = time()
        print("img trans {} s".format(t1 - t0))
        # predict segmentation
        with torch.no_grad():
            seg_outs = self.seg_net(img_tensor)  # shape: [B, 19, H=512, W=1024]
        torch.cuda.synchronize()
        t2 = time()
        print("img seg_net {} s".format(t2 - t1))
        # np.save("seg_outs.npy", seg_outs.cpu().numpy())
        # 第一步：分割图后处理为了synnet和diss的输入
        torch.cuda.synchronize()
        a0 = time()
        seg_softmax_out = F.softmax(seg_outs, dim=1)  # shape: [B, 19, H=512, W=1024]
        torch.cuda.synchronize()
        a1 = time()
        print("syn_net preprocess softmax {} s".format(a1 - a0))
        # seg_final = np.argmax(seg_outs.cpu().numpy().squeeze(), axis=0)  # segmentation map shape: [H=1024, W=2048]
        _, seg_final = seg_softmax_out.max(dim=1)  # seg_final shape: [B, H=512, W=1024]
        seg_final = seg_final.float()
        torch.cuda.synchronize()
        a2 = time()
        print("syn_net preprocess argmax {} s".format(a2 - a1))
        # np.save("seg_final.npy", seg_final)

        # 第二步：synnet:label_img
        # get label map for synthesis model
        torch.cuda.synchronize()
        a4 = time()
        # seg_final = seg_final.cpu().numpy()
        # label_out = np.zeros_like(seg_final)
        # for label_id, train_id in self.opt.dataset_cls.id_to_trainid.items():
        #     a00 = time()
        #     label_out[np.where(seg_final == train_id)] = label_id
        #     a01 = time()
        #     # print("syn_net preprocess {} label deal {} s".format(label_id, a01 - a00))
        label_out = torch.zeros_like(seg_final).float()  # shape [B, H, W]

        for label_id, train_id in self.id_to_trainid.items():
            label_out[seg_final == train_id] = label_id
        torch.cuda.synchronize()
        a5 = time()
        print("syn_net preprocess label {} s".format(a5 - a4))
        # np.save("label_img.npy", np.array(label_out))

        # 1 label_img origin
        # label_img = Image.fromarray((label_out).astype(np.uint8))
        # prepare for synthesis
        # label_tensor = self.transform_semantic(label_img) * 255.0
        # 1 label_img pytorch
        label_out = label_out.unsqueeze(1)  # shape [B, 1, H, W]
        label_tensor = self.pytorch_resize_totensor(label_out, size=(256, 512), mul=255,
                                                    interpolation=InterpolationMode.NEAREST)  # shape [B, 1, 256, 512]
        torch.cuda.synchronize()
        a6 = time()
        print("syn_net preprocess label resize {} s".format(a6 - a5))
        # self.valid(label_tensor, "label_tensor.npy")

        label_tensor[label_tensor == 255] = 35  # 'unknown' is opt.label_nc shape [B, 1, 256, 512]
        torch.cuda.synchronize()
        a7 = time()
        print("syn_net preprocess label 255 {} s".format(a7 - a6))
        # 第二步：synnet: origin_img
        # 2 syn_img_input origin
        # image_tensor = self.transform_image_syn(img)
        # 2 syn_img_input pytorch
        image_tensor = self.pytorch_resize_totensor(img, size=(256, 512), mul=1,
                                                    interpolation=InterpolationMode.BILINEAR)  # shape [B, 3, 256, 512]
        syn_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image_tensor = syn_norm(image_tensor)
        # np.save("image_syn.npy", image_tensor.cpu().numpy())
        # self.valid(image_tensor, "image_syn.npy")
        torch.cuda.synchronize()
        a8 = time()
        print("syn_net preprocess img resize {} s".format(a8 - a7))

        # Get instance map in right format. Since prediction doesn't have instance map, we use semantic instead
        instance_tensor = label_tensor.clone()  # shape [B, 1, 256, 512]
        torch.cuda.synchronize()
        a9 = time()
        print("syn_net preprocess instance_tensor resize {} s".format(a9 - a8))
        print("syn_net preprocess total {} s".format(a9 - a0))
        torch.cuda.synchronize()
        t3 = time()
        print("img syn_net input preprocess {} s".format(t3 - t2))

        seg_softmax_out_mask = (seg_softmax_out == seg_softmax_out.max(dim=1, keepdim=True)[0]).float()
        mb, mc, mh, mw = seg_softmax_out_mask.shape
        mask = seg_softmax_out_mask.reshape(mb, mc, mh * mw)
        mask = F.pad(mask, [0, 0, 0, 16], mode="constant", value=0.0)
        mask = mask.reshape(mb, mc + 16, mh, mw)
        mask = mask[:, self.id_index, :, :]
        mask = self.pytorch_resize_totensor(mask, size=(256, 512), mul=1, interpolation=InterpolationMode.NEAREST,
                                            totensor=False)  # shape [B, 1, 256, 512]

        # run synthesis
        # syn_input = {'label': label_tensor, 'instance': instance_tensor,
        #              'image': image_tensor, "mask": mask}
        syn_input = {'instance': instance_tensor,
                     'image': image_tensor, "mask": mask}

        generated = self.syn_net(syn_input, mode='inference')  # shape [B, 3, 256, 512]

        # 第三步：vggdiff
        # np.save("generated.npy", generated.cpu().numpy())
        torch.cuda.synchronize()
        t4 = time()
        print("img syn_net {} s".format(t4 - t3))
        torch.cuda.synchronize()
        b0 = time()
        # get initial transformation
        # 第三步：vggdiff: 合成图，也是第四步的输入
        # image_numpy = ((generated + 1) / 2.0) * 255
        image_torch = ((generated + 1) / 2.0) * 255
        syn_image_tensor1 = self.pytorch_resize_totensor(image_torch, size=(256, 512), mul=1,
                                                         interpolation=InterpolationMode.NEAREST)  # shape [B, 3, 256, 512]
        # 4 diss
        # syn_image_tensor = self.norm_transform_diss(syn_image_tensor).unsqueeze(0).cuda()
        # 4 vgg_diff
        syn_image_tensor1 = self.norm_transform_diss(syn_image_tensor1)
        torch.cuda.synchronize()
        b1 = time()
        print("vgg_diff preprocess synimg resize norm {} s".format(b1 - b0))

        # 第三步：vggdiff: 输入原图，也是第四步的输入
        # 5 diss image origin origin
        # image_tensor = self.base_transforms_diss(img)
        # 5 diss image pytorch input12
        # img2 = self.pytorch_resize_totensor(img, size=(self.h, self.w), mul=1, interpolation=InterpolationMode.NEAREST,
        #                                     totensor=False)
        # img2 = img2.type(torch.uint8).cpu().numpy()
        image_tensor1 = self.pytorch_resize_totensor(img, size=(256, 512), mul=1,
                                                     interpolation=InterpolationMode.NEAREST)  # shape [B, 3, 256, 512]

        # 5 diss image pytorch input1
        # img1 = self.pytorch_resize_totensor(img, size=(self.h, self.w), mul=1, interpolation=InterpolationMode.NEAREST,
        #                                     totensor=False)
        # img1 = img1.permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        # img1 = Image.fromarray(img1.astype(np.uint8))
        # image_tensor2 = self.base_transforms_diss(img1)

        # 5 diss image pytorch input2
        # image_tensor3 = self.pytorch_resize_totensor(img, size=(256, 512), mul=1, interpolation=InterpolationMode.NEAREST)

        # 5 diss
        # image_tensor = self.norm_transform_diss(image_tensor).unsqueeze(0).cuda()
        # 5 vgg_diff
        image_tensor1 = self.norm_transform_diss(image_tensor1)
        torch.cuda.synchronize()
        b2 = time()
        print("vgg_diff preprocess inputimg resize norm {} s".format(b2 - b1))
        print("vgg_diff preprocess total {} s".format(b2 - b0))
        torch.cuda.synchronize()
        t5 = time()
        # np.save("vgg_diff_image_tensor.npy", image_tensor.cpu().numpy())
        # np.save("vgg_diff_syn_image_tensor.npy", syn_image_tensor.cpu().numpy())
        # self.valid(image_tensor1, "vgg_diff_image_tensor.npy")
        print("img vgg_diff preprocess {} s".format(t5 - t4))

        # get softmax difference
        perceptual_diff = self.vgg_diff(image_tensor1, syn_image_tensor1)  # shape [B, 1, 256, 512]
        # np.save("vgg_diff_perceptual_diff.npy", perceptual_diff.cpu().numpy())
        torch.cuda.synchronize()
        t6 = time()
        print("img vgg_diff {} s".format(t6 - t5))
        # 第四步：diss：perceptual
        # min_v = torch.min(perceptual_diff.squeeze())
        # max_v = torch.max(perceptual_diff.squeeze())
        # perceptual_diff = (perceptual_diff.squeeze() - min_v) / (max_v - min_v)
        # perceptual_diff *= 255
        perceptual_diff_r = perceptual_diff.reshape(B, -1)
        min_v = torch.min(perceptual_diff_r, dim=1)[0].reshape(-1, 1, 1, 1)
        max_v = torch.max(perceptual_diff_r, dim=1)[0].reshape(-1, 1, 1, 1)
        perceptual_diff = (perceptual_diff - min_v) / (max_v - min_v)  # shape [B, 1, 256, 512]
        perceptual_diff *= 255
        # 6 perceptual_diff origin
        # finish transformation
        # perceptual_diff_tensor = self.base_transforms_diss(perceptual_diff).unsqueeze(0).cuda()
        # 6 perceptual_diff pytorch
        perceptual_diff_tensor = self.pytorch_resize_totensor(perceptual_diff, size=(256, 512), mul=1,
                                                              interpolation=InterpolationMode.NEAREST)  # shape [B, 1, 256, 512]

        # 第四步：diss：entropy
        # get entropy
        entropy = torch.sum(-seg_softmax_out * torch.log(seg_softmax_out), dim=1)  # shape [B, H, W]
        entropy_r = entropy.reshape(B, -1)
        min_v1 = torch.min(entropy_r, dim=1)[0].reshape(-1, 1, 1)
        max_v1 = torch.max(entropy_r, dim=1)[0].reshape(-1, 1, 1)
        entropy = (entropy - min_v1) / max_v1  # shape [B, H, W]
        entropy *= 255  # for later use in the dissimilarity
        # 7 entropy origin
        # entropy_tensor = self.base_transforms_diss(entropy_img).unsqueeze(0).cuda()
        # 7 entropy pytorch
        entropy = entropy.unsqueeze(1)  # shape [B, 1, 256, 512]
        entropy_tensor = self.pytorch_resize_totensor(entropy, size=(256, 512), mul=1,
                                                      interpolation=InterpolationMode.NEAREST)  # shape [B, 1, 256, 512]
        # 第四步：diss：distance
        # get softmax distance
        distance, _ = torch.topk(seg_softmax_out, 2, dim=1)  # distance shape [B, 2, H, W]
        max_logit = distance[:, 0, :, :]  # shape [B, H, W]
        max2nd_logit = distance[:, 1, :, :]  # shape [B, H, W]
        result = max_logit - max2nd_logit  # shape [B, H, W]
        result_r = result.reshape(B, -1)  # shape [B, -1]
        min_v2 = torch.min(result_r, dim=1)[0].reshape(-1, 1, 1)
        max_v2 = torch.max(result_r, dim=1)[0].reshape(-1, 1, 1)
        distance = 1 - (result - min_v2) / max_v2
        distance *= 255  # for later use in the dissimilarity
        # 8 distance origin
        # distance_tensor = self.base_transforms_diss(distance_img).unsqueeze(0).cuda()
        # 9 entropy pytorch
        distance = distance.unsqueeze(1)  # shape [B, 1, 256, 512]
        distance_tensor = self.pytorch_resize_totensor(distance, size=(256, 512), mul=1,
                                                       interpolation=InterpolationMode.NEAREST)  # shape [B, 1, 256, 512]

        # 第四步：diss：分割图
        # 3 diss semantic origin
        # semantic_tensor = self.base_transforms_diss(semantic) * 255
        # 3 diss semantic pytorch
        # semantic_tensor = self.pytorch_resize_totensor(seg_final, size=(256, 512), mul=255,
        #                                                interpolation=InterpolationMode.NEAREST)  # shape [B, 256, 512]
        # 4 vgg_diff diss syn image origin
        # syn_image_tensor = self.base_transforms_diss(synthesis_final_img)
        # 4 vgg_diff diss syn image pytorch
        # hot encode semantic map
        # semantic_tensor[semantic_tensor == 255] = 20  # 'ignore label is 20'
        # semantic_tensor = semantic_tensor.unsqueeze(1)  # shape [B, 1, 256, 512]
        # semantic_tensor = self.one_hot_encoding(semantic_tensor, 20)  # shape [B, clsss_num=19, 256, 512]

        semantic_tensor = self.pytorch_resize_totensor(seg_softmax_out_mask, size=(256, 512), mul=1,
                                                       interpolation=InterpolationMode.NEAREST,
                                                       totensor=False)  # shape [B, 19, 256, 512]
        # print("seg_softmax_out_mask", torch.sum((seg_softmax_out_mask - semantic_tensor) > 1e-4))
        torch.cuda.synchronize()
        t7 = time()
        # 第四步：diss：三个不同的图
        print("img diss_model prerocess {} s".format(t7 - t6))
        # run dissimilarity
        # np.save("diss_model_semantic_tensor.npy", semantic_tensor.cpu().numpy())
        # np.save("diss_model_entropy_tensor.npy", entropy_tensor.cpu().numpy())
        # np.save("diss_model_perceptual_diff_tensor.npy", perceptual_diff_tensor.cpu().numpy())
        # np.save("diss_model_distance_tensor.npy", distance_tensor.cpu().numpy())
        with torch.no_grad():
            if self.prior:
                diss_pred = F.softmax(
                    self.diss_model(image_tensor1, syn_image_tensor1, semantic_tensor, entropy_tensor,
                                    perceptual_diff_tensor,
                                    distance_tensor), dim=1)
            else:
                diss_pred = F.softmax(self.diss_model(image_tensor1, syn_image_tensor1, semantic_tensor), dim=1)
        torch.cuda.synchronize()
        t8 = time()
        print("img diss_model {} s".format(t8 - t7))

        # do ensemble if necessary
        if self.ensemble:
            diss_pred = diss_pred[:, 1, :, :] * 0.75 + entropy_tensor[:, 0, :, :] * 0.25  # shape [B, 256,512]
        else:
            diss_pred = diss_pred[:, 1, :, :]
        diss_pred = diss_pred * 255
        torch.cuda.synchronize()
        t9 = time()
        # np.save("diss_pred.npy", diss_pred)
        print("total {} s".format(t9 - t0))
        if self.vis:
            out = {'anomaly_map': diss_pred, 'segmentation': seg_final, 'synthesis': generated,
                   'softmax_entropy': entropy_tensor, 'perceptual_diff': perceptual_diff_tensor,
                   'softmax_distance': distance_tensor}
            return seg_final, diss_pred, out
        else:
            return seg_final, diss_pred

    def set_seeds(self, seed=0):
        # set seeds for reproducibility
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    def get_segmentation(self):
        # Get Segmentation Net
        assert_and_infer_cfg(self.opt, train_mode=False)
        self.opt.dataset_cls = cityscapes
        net = network.get_net(self.opt, criterion=None)
        # net = torch.nn.DataParallel(net).cuda()
        print('Segmentation Net Built.')
        snapshot = os.path.join(os.getcwd(), os.path.dirname(__file__), self.opt.snapshot)
        self.seg_net, _ = restore_snapshot(net, optimizer=None, snapshot=snapshot,
                                           restore_optimizer_bool=False)
        self.seg_net.eval()
        print('Segmentation Net Restored.')

    def get_synthesis(self):
        # Get Synthesis Net
        print('Synthesis Net Built.')
        self.opt.checkpoints_dir = os.path.join(os.getcwd(), os.path.dirname(__file__), self.opt.checkpoints_dir)
        self.syn_net = Pix2PixModel(self.opt)
        self.syn_net.eval()
        print('Synthesis Net Restored')

    def get_dissimilarity(self, ours):
        # Get Dissimilarity Net
        if ours:
            config_diss = os.path.join(os.getcwd(), os.path.dirname(__file__),
                                       'image_dissimilarity/configs/test/ours_configuration.yaml')
        else:
            config_diss = os.path.join(os.getcwd(), os.path.dirname(__file__),
                                       'image_dissimilarity/configs/test/baseline_configuration.yaml')

        with open(config_diss, 'r') as stream:
            config_diss = yaml.load(stream, Loader=yaml.FullLoader)

        self.prior = config_diss['model']['prior']
        self.ensemble = config_diss['ensemble']

        if self.prior:
            self.diss_model = DissimNetPrior(**config_diss['model']).cuda()
        else:
            self.diss_model = DissimNet(**config_diss['model']).cuda()

        print('Dissimilarity Net Built.')
        save_folder = os.path.join(os.getcwd(), os.path.dirname(__file__), config_diss['save_folder'])
        if torch.__version__ <= "1.4.0":
            model_path = os.path.join(save_folder,
                                      '%s_net_%s.torch140.pth' % (
                                          config_diss['which_epoch'], config_diss['experiment_name']))
        else:
            model_path = os.path.join(save_folder,
                                      '%s_net_%s.pth' % (config_diss['which_epoch'], config_diss['experiment_name']))
        print("diss model path: {}".format(model_path))
        model_weights = torch.load(model_path)
        rt = self.diss_model.load_state_dict(model_weights, strict=False)
        if rt.missing_keys:
            print('Missing keys when loading states {}'.format(rt.missing_keys))
        if rt.unexpected_keys:
            print('Warning: Keys dismatch when loading backbone states:\n' + str(rt))
        self.diss_model.eval()
        print('Dissimilarity Net Restored')

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

    def get_transformations(self):
        # Transform images to Tensor based on ImageNet Mean and STD
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.img_transform = transforms.Compose([transforms.Normalize(*mean_std)])

        # synthesis necessary pre-process
        self.transform_semantic_resize = transforms.Compose(
            [transforms.Resize(size=(256, 512), interpolation=InterpolationMode.NEAREST)])
        # self.transform_semantic_resize = transforms.Compose(
        #     [transforms.Resize(size=(256, 512), interpolation=InterpolationMode.BILINEAR, antialias=True)])
        self.transform_image_syn = transforms.Compose(
            [transforms.Resize(size=(256, 512), interpolation=InterpolationMode.BICUBIC), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5),
                                  (0.5, 0.5, 0.5))])
        self.transform_image_syn_resize = transforms.Resize(size=(256, 512), interpolation=InterpolationMode.BICUBIC)
        # self.transform_image_syn_resize = transforms.Resize(size=(256, 512), interpolation=InterpolationMode.BILINEAR,
        #                                                     antialias=True)
        self.transform_image_syn_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # dissimilarity pre-process
        self.vgg_diff = VGG19_difference().cuda()
        self.base_transforms_diss = transforms.Compose(
            [transforms.Resize(size=(256, 512), interpolation=InterpolationMode.NEAREST), transforms.ToTensor()])
        self.base_transforms_diss_resize = transforms.Compose(
            [transforms.Resize(size=(256, 512), interpolation=InterpolationMode.NEAREST)])
        self.norm_transform_diss = transforms.Compose(
            [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # imageNet normamlization
        self.to_pil = ToPILImage()


palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


parser = argparse.ArgumentParser()
parser.add_argument('--demo_folder', type=str, default='./sample_images', help='Path to folder with images to be run.')
parser.add_argument('--save_folder', type=str, default='./results/tmp2', help='Folder to where to save the results')
opts = parser.parse_args()

demo_folder = opts.demo_folder
save_folder = opts.save_folder
# Save folders
semantic_path = os.path.join(save_folder, 'semantic')
anomaly_path = os.path.join(save_folder, 'anomaly')
synthesis_path = os.path.join(save_folder, 'synthesis')
entropy_path = os.path.join(save_folder, 'entropy')
distance_path = os.path.join(save_folder, 'distance')
perceptual_diff_path = os.path.join(save_folder, 'perceptual_diff')
box_path = os.path.join(save_folder, 'box')

os.makedirs(semantic_path, exist_ok=True)
os.makedirs(anomaly_path, exist_ok=True)
os.makedirs(synthesis_path, exist_ok=True)
os.makedirs(entropy_path, exist_ok=True)
os.makedirs(distance_path, exist_ok=True)
os.makedirs(perceptual_diff_path, exist_ok=True)
os.makedirs(box_path, exist_ok=True)


def img_preprocess(image, input_shape):
    # image = Image.open(img_path)
    c, h, w = input_shape
    img_np = cv2.resize(image, (w, h))  # [h, w, c]  , 3 is bgr
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # 颜色转换
    img = torch.from_numpy(img_np)
    img = img.permute((2, 0, 1)).contiguous()  # 通道转换C,H,W
    return img


if __name__ == '__main__':
    # input_shape = [3, 256, 512]
    input_shape = [3, 512, 1024]
    # input_shape = [3, 1024, 2048]
    root = "./sample_images"
    vis = True
    detector = RoadAnomalyDetector(True, input_shape, vis=vis)
    gpu = 0
    detector.to("cuda:{}".format(gpu))
    detector.eval()
    names = list(os.listdir(root))
    names = names
    for i, name in enumerate(names):
        t0 = time()
        # img_path = "./sample_images/road_anomaly_example.png"
        img_path = os.path.join(root, name)
        basename = os.path.basename(img_path).replace('.jpg', '.png')
        image = cv2.imread(img_path)
        image_og_h, image_og_w, _ = image.shape
        img = img_preprocess(image, input_shape)
        img = torch.stack([img, img], dim=0).cuda()
        # img = img.unsqueeze(0).cuda()  # 添加batch维度
        t1 = time()
        print("img to cuda {} s".format(t1 - t0))
        if vis:
            prediction, anomaly_score, outs = detector(img)
            # prediction = prediction.cpu().numpy()
            # anomaly_score = anomaly_score.cpu().numpy()
            # result = postprocessing(prediction, anomaly_score)
            print("spend total {} s".format(time() - t1))
            # draw_total_box(img_path, result[0], debug=True)

            # 可视化diss_pred
            diss_pred_img = anomaly_score[0].cpu().numpy().astype(np.uint8)
            diss_pred = Image.fromarray(diss_pred_img).resize((image_og_w, image_og_h))
            # np.save("diss_pred_resize.npy", np.array(diss_pred))
            # 可视化分割图
            seg_final_img = prediction[0].cpu().numpy().astype(np.uint8)
            semantic = Image.fromarray(seg_final_img)
            seg_img = semantic.resize((image_og_w, image_og_h))
            # np.save("seg_img.npy", np.array(seg_img))
            # 可视化合成图
            generated = outs["synthesis"]
            image_numpy = (np.transpose(generated[0].squeeze().cpu().numpy(), (1, 2, 0)) + 1) / 2.0 * 255
            synthesis_final_img = Image.fromarray(image_numpy.astype(np.uint8))
            synthesis = synthesis_final_img.resize((image_og_w, image_og_h))
            # 可视化entropy
            entropy = outs["softmax_entropy"]
            entropy = entropy[0].cpu().numpy()
            entropy_img = Image.fromarray(entropy.astype(np.uint8).squeeze())
            entropy = entropy_img.resize((image_og_w, image_og_h))
            # 可视化perceptual_diff
            perceptual_diff = outs["perceptual_diff"]
            perceptual_diff1 = perceptual_diff[0].squeeze().cpu().numpy()
            perceptual_diff = Image.fromarray(perceptual_diff1.astype(np.uint8))
            perceptual_diff = perceptual_diff.resize((image_og_w, image_og_h))
            # 可视化距离
            distance = outs["softmax_distance"]
            distance = distance[0].squeeze(0).cpu().numpy()
            distance_img = Image.fromarray(distance.astype(np.uint8).squeeze())
            distance = distance_img.resize((image_og_w, image_og_h))

            result = postprocessing(np.array(seg_img)[None, :], np.array(diss_pred)[None, :], image_og_h, image_og_w)
            img_box = draw_total_box(np.array(image), result[0], debug=True)
            results = {'anomaly_map': diss_pred, 'segmentation': seg_img, 'synthesis': synthesis,
                       'softmax_entropy': entropy, 'perceptual_diff': perceptual_diff, 'softmax_distance': distance,
                       "box": img_box}

            anomaly_map = results['anomaly_map'].convert('RGB')
            anomaly_map = Image.fromarray(
                np.concatenate([np.array(image)[:, :, ::-1], np.array(anomaly_map)], axis=1)
            )
            anomaly_map.save(os.path.join(anomaly_path, basename))

            semantic_map = colorize_mask(np.array(results['segmentation']))
            semantic_map.save(os.path.join(semantic_path, basename))

            synthesis = results['synthesis']
            synthesis.save(os.path.join(synthesis_path, basename))

            softmax_entropy = results['softmax_entropy'].convert('RGB')
            softmax_entropy.save(os.path.join(entropy_path, basename))

            softmax_distance = results['softmax_distance'].convert('RGB')
            softmax_distance.save(os.path.join(distance_path, basename))

            perceptual_diff = results['perceptual_diff'].convert('RGB')
            perceptual_diff.save(os.path.join(perceptual_diff_path, basename))

            img_box = results['box']
            cv2.imwrite(os.path.join(box_path, basename), img_box)
        else:
            prediction, anomaly_score = detector(img)

            # 可视化diss_pred
            diss_pred_img = anomaly_score[0].cpu().numpy()

            obs_h, obs_w = diss_pred_img.shape
            # 可视化分割图
            seg_final_img = prediction[0].cpu().numpy().astype(np.uint8)
            seg_img = cv2.resize(seg_final_img, (obs_w, obs_h))

            result = postprocessing(np.array(seg_img)[None, :], np.array(diss_pred_img)[None, :], image_og_h,
                                    image_og_w)
            img_box = draw_total_box(np.array(image), result[0], debug=True)
            print("spend total {} s".format(time() - t0))
            draw_total_box(img_path, result[0], debug=True)
            # break
            t.sleep(10)
