import os
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import yaml
import random
from options.config_class import Config

import sys

sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(__file__), 'image_segmentation'))
import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(__file__), 'image_synthesis'))
from image_synthesis.models.pix2pix_model import Pix2PixModel
from image_dissimilarity.models.dissimilarity_model import DissimNetPrior, DissimNet
from image_dissimilarity.models.vgg_features import VGG19_difference
from image_dissimilarity.data.cityscapes_dataset import one_hot_encoding

sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(__file__), 'utils'))
from seg_to_rect import postprocessing
from draw_box_util import draw_total_box
from time import time


class AnomalyDetector():
    def __init__(self, ours=True, input_shape=(3, 1024, 2048), seed=0, fishyscapes_wrapper=True):

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
        self.c, self.h, self.w, = input_shape

    def valid(self, x, path="image.npy", thd=1e-3):
        image_f = np.load(path)
        image_f = torch.from_numpy(image_f).to(x.device)
        a = torch.abs(image_f - x)
        a[a < thd] = 0
        b = (a != 0).float()
        b = torch.sum(b)
        print(path, b, a.shape, b / a.numel(), torch.max(a))

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

    def pytorch_resize_totensor(self, x, size=(256, 512), mul=1, interpolation=Image.NEAREST, totensor=True):
        if isinstance(x, Image.Image):
            x = np.array(x).transpose((2, 0, 1))
            x = torch.from_numpy(x)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.type(torch.uint8)
        h, w = size

        if x.dim() < 3:
            x = x.unsqueeze(0)
        transform_semantic_resize = transforms.Resize(size=(h, w), interpolation=interpolation)
        x = transform_semantic_resize(x)
        if totensor:
            x = self.to_tensor(x)
        x = x * mul
        return x

    def estimator_image(self, image):
        t0 = time()
        image_og_h, image_og_w, _ = image.shape
        # img = image.resize((self.w, self.h))
        img_np = cv2.resize(image, (self.w, self.h))
        img_np_chw = img_np[:, :, ::-1].transpose(2, 0, 1).copy()
        # 第一步：segnet
        # 0 segnet img origin
        # img_tensor = self.img_transform(img)
        # 0 segnet img pytorch
        # img = self.pytorch_resize_totensor(image, size=(self.h, self.w), mul=1, interpolation=Image.NEAREST, totensor=False)
        img = torch.from_numpy(img_np_chw).to("cuda:0")
        img_tensor = self.to_tensor(img)
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        seg_norm = transforms.Normalize(*mean_std)
        img_tensor = seg_norm(img_tensor)
        self.valid(img_tensor, "image.npy")
        # np.save("image.npy", img_tensor.unsqueeze(0).cuda().cpu().numpy())
        t1 = time()
        print("img trans {} s".format(t1 - t0))
        # predict segmentation
        with torch.no_grad():
            seg_outs = self.seg_net(img_tensor.unsqueeze(0).cuda())  # shape: [B, 19, H=1024, W=2048]
        t2 = time()
        print("img seg_net {} s".format(t2 - t1))
        # np.save("seg_outs.npy", seg_outs.cpu().numpy())
        # 第一步：分割图后处理为了synnet和diss的输入
        a0 = time()
        seg_softmax_out = F.softmax(seg_outs, dim=1)  # shape: [B, 19, H=1024, W=2048]
        a1 = time()
        print("syn_net preprocess softmax {} s".format(a1 - a0))
        seg_final = np.argmax(seg_outs.cpu().numpy().squeeze(), axis=0)  # segmentation map shape: [H=1024, W=2048]
        a2 = time()
        print("syn_net preprocess argmax {} s".format(a2 - a1))
        # np.save("seg_final.npy", seg_final)

        # 第二步：synnet:label_img
        # get label map for synthesis model
        a4 = time()
        label_out = np.zeros_like(seg_final)
        for label_id, train_id in self.opt.dataset_cls.id_to_trainid.items():
            a00 = time()
            label_out[np.where(seg_final == train_id)] = label_id
            a01 = time()
            # print("syn_net preprocess {} label deal {} s".format(label_id, a01 - a00))
        a5 = time()
        print("syn_net preprocess label {} s".format(a5 - a4))
        # np.save("label_img.npy", np.array(label_out))

        # 1 label_img origin
        # label_img = Image.fromarray((label_out).astype(np.uint8))
        # prepare for synthesis
        # label_tensor = self.transform_semantic(label_img) * 255.0
        # 1 label_img pytorch
        label_tensor = self.pytorch_resize_totensor(label_out, size=(256, 512), mul=255, interpolation=Image.NEAREST)
        a6 = time()
        print("syn_net preprocess label resize {} s".format(a6 - a5))
        # self.valid(label_tensor, "label_tensor.npy")

        label_tensor[label_tensor == 255] = 35  # 'unknown' is opt.label_nc
        a7 = time()
        print("syn_net preprocess label 255 {} s".format(a7 - a6))
        # 第二步：synnet: origin_img
        # 2 syn_img_input origin
        # image_tensor = self.transform_image_syn(img)
        # 2 syn_img_input pytorch
        image_tensor = self.pytorch_resize_totensor(img, size=(256, 512), mul=1, interpolation=Image.BICUBIC)
        syn_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image_tensor = syn_norm(image_tensor)
        # np.save("image_syn.npy", image_tensor.cpu().numpy())
        # self.valid(image_tensor, "image_syn.npy")
        a8 = time()
        print("syn_net preprocess img resize {} s".format(a8 - a7))

        # Get instance map in right format. Since prediction doesn't have instance map, we use semantic instead
        instance_tensor = label_tensor.clone()
        a9 = time()
        print("syn_net preprocess instance_tensor resize {} s".format(a9 - a8))
        print("syn_net preprocess total {} s".format(a9 - a0))
        t3 = time()
        print("img syn_net input preprocess {} s".format(t3 - t2))

        # run synthesis
        syn_input = {'label': label_tensor.unsqueeze(0), 'instance': instance_tensor.unsqueeze(0),
                     'image': image_tensor.unsqueeze(0)}

        generated = self.syn_net(syn_input, mode='inference')  # shape [B, 3, 256, 512]

        # 第三步：vggdiff
        # np.save("generated.npy", generated.cpu().numpy())
        t4 = time()
        print("img syn_net {} s".format(t4 - t3))
        b0 = time()

        # get initial transformation
        # 第三步：vggdiff: 合成图，也是第四步的输入
        image_numpy = ((generated.squeeze().cpu().numpy() + 1) / 2.0) * 255
        syn_image_tensor1 = self.pytorch_resize_totensor(image_numpy, size=(256, 512), mul=1,
                                                         interpolation=Image.NEAREST)
        # 4 diss
        # syn_image_tensor = self.norm_transform_diss(syn_image_tensor).unsqueeze(0).cuda()
        # 4 vgg_diff
        syn_image_tensor1 = self.norm_transform_diss(syn_image_tensor1).unsqueeze(0).cuda()
        b1 = time()
        print("vgg_diff preprocess synimg resize norm{} s".format(b1 - b0))

        # 第三步：vggdiff: 输入原图，也是第四步的输入
        # 5 diss image origin origin
        # image_tensor = self.base_transforms_diss(img)
        # 5 diss image pytorch input12
        # img2 = self.pytorch_resize_totensor(img, size=(self.h, self.w), mul=1, interpolation=Image.NEAREST,
        #                                     totensor=False)
        # img2 = img2.type(torch.uint8).cpu().numpy()
        image_tensor1 = self.pytorch_resize_totensor(img, size=(256, 512), mul=1, interpolation=Image.NEAREST)

        # 5 diss image pytorch input1
        # img1 = self.pytorch_resize_totensor(img, size=(self.h, self.w), mul=1, interpolation=Image.NEAREST,
        #                                     totensor=False)
        # img1 = img1.permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        # img1 = Image.fromarray(img1.astype(np.uint8))
        # image_tensor2 = self.base_transforms_diss(img1)

        # 5 diss image pytorch input2
        # image_tensor3 = self.pytorch_resize_totensor(img, size=(256, 512), mul=1, interpolation=Image.NEAREST)

        # 5 diss
        # image_tensor = self.norm_transform_diss(image_tensor).unsqueeze(0).cuda()
        # 5 vgg_diff
        image_tensor1 = self.norm_transform_diss(image_tensor1).unsqueeze(0).cuda()
        b2 = time()
        print("vgg_diff preprocess inputimg resize norm {} s".format(b2 - b1))
        print("vgg_diff preprocess total {} s".format(b2 - b0))
        t5 = time()
        # np.save("vgg_diff_image_tensor.npy", image_tensor.cpu().numpy())
        # np.save("vgg_diff_syn_image_tensor.npy", syn_image_tensor.cpu().numpy())
        # self.valid(image_tensor1, "vgg_diff_image_tensor.npy")
        print("img vgg_diff preprocess {} s".format(t5 - t4))

        # get softmax difference
        perceptual_diff = self.vgg_diff(image_tensor1, syn_image_tensor1)  # shape [B, 1, 256, 512]
        # np.save("vgg_diff_perceptual_diff.npy", perceptual_diff.cpu().numpy())
        t6 = time()
        print("img vgg_diff {} s".format(t6 - t5))
        # 第四步：diss：perceptual
        min_v = torch.min(perceptual_diff.squeeze())
        max_v = torch.max(perceptual_diff.squeeze())
        perceptual_diff = (perceptual_diff.squeeze() - min_v) / (max_v - min_v)
        perceptual_diff *= 255
        # 6 perceptual_diff origin
        # finish transformation
        # perceptual_diff_tensor = self.base_transforms_diss(perceptual_diff).unsqueeze(0).cuda()
        # 6 perceptual_diff pytorch
        perceptual_diff_tensor = self.pytorch_resize_totensor(perceptual_diff, size=(256, 512), mul=1,
                                                              interpolation=Image.NEAREST)
        perceptual_diff_tensor = perceptual_diff_tensor.unsqueeze(0).cuda()

        # 第四步：diss：entropy
        # get entropy
        entropy = torch.sum(-seg_softmax_out * torch.log(seg_softmax_out), dim=1)
        entropy = (entropy - entropy.min()) / entropy.max()
        entropy *= 255  # for later use in the dissimilarity
        # 7 entropy origin
        # entropy_tensor = self.base_transforms_diss(entropy_img).unsqueeze(0).cuda()
        # 7 entropy pytorch
        entropy_tensor = self.pytorch_resize_totensor(entropy, size=(256, 512), mul=1,
                                                      interpolation=Image.NEAREST)
        entropy_tensor = entropy_tensor.unsqueeze(0).cuda()
        # 第四步：diss：distance
        # get softmax distance
        distance, _ = torch.topk(seg_softmax_out, 2, dim=1)
        max_logit = distance[:, 0, :, :]
        max2nd_logit = distance[:, 1, :, :]
        result = max_logit - max2nd_logit
        distance = 1 - (result - result.min()) / result.max()
        distance *= 255  # for later use in the dissimilarity
        # 8 distance origin
        # distance_tensor = self.base_transforms_diss(distance_img).unsqueeze(0).cuda()
        # 9 entropy pytorch
        distance_tensor = self.pytorch_resize_totensor(distance, size=(256, 512), mul=1,
                                                       interpolation=Image.NEAREST)
        distance_tensor = distance_tensor.unsqueeze(0).cuda()

        # 第四步：diss：分割图
        # 3 diss semantic origin
        # semantic_tensor = self.base_transforms_diss(semantic) * 255
        # 3 diss semantic pytorch
        semantic_tensor = self.pytorch_resize_totensor(seg_final, size=(256, 512), mul=255, interpolation=Image.NEAREST)
        # 4 vgg_diff diss syn image origin
        # syn_image_tensor = self.base_transforms_diss(synthesis_final_img)
        # 4 vgg_diff diss syn image pytorch
        # hot encode semantic map
        semantic_tensor[semantic_tensor == 255] = 20  # 'ignore label is 20'
        semantic_tensor = one_hot_encoding(semantic_tensor, 20).unsqueeze(0).cuda()
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
        t8 = time()
        print("img diss_model {} s".format(t8 - t7))

        # do ensemble if necessary
        if self.ensemble:
            diss_pred = diss_pred[:, 1, :, :] * 0.75 + entropy_tensor * 0.25
        else:
            diss_pred = diss_pred[:, 1, :, :]
        diss_pred = diss_pred * 255
        t9 = time()
        # np.save("diss_pred.npy", diss_pred)
        print("total {} s".format(t9 - t0))

        # 可视化diss_pred
        diss_pred_img = diss_pred[0].squeeze(0).cpu().numpy().astype(np.uint8)
        diss_pred = Image.fromarray(diss_pred_img).resize((image_og_w, image_og_h))
        # np.save("diss_pred_resize.npy", np.array(diss_pred))
        # 可视化分割图
        semantic = Image.fromarray(seg_final.astype(np.uint8))
        seg_img = semantic.resize((image_og_w, image_og_h))
        # np.save("seg_img.npy", np.array(seg_img))
        # 可视化合成图
        image_numpy = (np.transpose(generated.squeeze().cpu().numpy(), (1, 2, 0)) + 1) / 2.0 * 255
        synthesis_final_img = Image.fromarray(image_numpy.astype(np.uint8))
        synthesis = synthesis_final_img.resize((image_og_w, image_og_h))
        # 可视化entropy
        entropy = entropy.cpu().numpy()
        entropy_img = Image.fromarray(entropy.astype(np.uint8).squeeze())
        entropy = entropy_img.resize((image_og_w, image_og_h))
        # 可视化perceptual_diff
        perceptual_diff1 = perceptual_diff.cpu().numpy()
        perceptual_diff = Image.fromarray(perceptual_diff1.astype(np.uint8))
        perceptual_diff = perceptual_diff.resize((image_og_w, image_og_h))
        # 可视化距离
        distance = distance.cpu().numpy()
        distance_img = Image.fromarray(distance.astype(np.uint8).squeeze())
        distance = distance_img.resize((image_og_w, image_og_h))

        result = postprocessing(np.array(seg_img)[None, :], np.array(diss_pred)[None, :], 19)
        img_box = draw_total_box(np.array(image), result[0], debug=True)
        out = {'anomaly_map': diss_pred, 'segmentation': seg_img, 'synthesis': synthesis,
               'softmax_entropy': entropy, 'perceptual_diff': perceptual_diff, 'softmax_distance': distance,
               "box": img_box}

        return out

    # Loop around all figures
    def estimator_worker(self, image):
        image_og_h = image.shape[0]
        image_og_w = image.shape[1]
        img = Image.fromarray(np.array(image)).convert('RGB').resize((2048, 1024))
        img_tensor = self.img_transform(img)

        # predict segmentation
        with torch.no_grad():
            seg_outs = self.seg_net(img_tensor.unsqueeze(0).cuda())

        seg_softmax_out = F.softmax(seg_outs, dim=1)
        seg_final = np.argmax(seg_outs.cpu().numpy().squeeze(), axis=0)  # segmentation map

        # get entropy
        entropy = torch.sum(-seg_softmax_out * torch.log(seg_softmax_out), dim=1)
        entropy = (entropy - entropy.min()) / entropy.max()
        entropy *= 255  # for later use in the dissimilarity

        # get softmax distance
        distance, _ = torch.topk(seg_softmax_out, 2, dim=1)
        max_logit = distance[:, 0, :, :]
        max2nd_logit = distance[:, 1, :, :]
        result = max_logit - max2nd_logit
        distance = 1 - (result - result.min()) / result.max()
        distance *= 255  # for later use in the dissimilarity

        # get label map for synthesis model
        label_out = np.zeros_like(seg_final)
        for label_id, train_id in self.opt.dataset_cls.id_to_trainid.items():
            label_out[np.where(seg_final == train_id)] = label_id
        label_img = Image.fromarray((label_out).astype(np.uint8))

        # prepare for synthesis
        label_tensor = self.transform_semantic(label_img) * 255.0
        label_tensor[label_tensor == 255] = 35  # 'unknown' is opt.label_nc
        image_tensor = self.transform_image_syn(img)
        # Get instance map in right format. Since prediction doesn't have instance map, we use semantic instead
        instance_tensor = label_tensor.clone()

        # run synthesis
        syn_input = {'label': label_tensor.unsqueeze(0), 'instance': instance_tensor.unsqueeze(0),
                     'image': image_tensor.unsqueeze(0)}
        generated = self.syn_net(syn_input, mode='inference')

        image_numpy = (np.transpose(generated.squeeze().cpu().numpy(), (1, 2, 0)) + 1) / 2.0
        synthesis_final_img = Image.fromarray((image_numpy * 255).astype(np.uint8))

        # prepare dissimilarity
        entropy = entropy.cpu().numpy()
        distance = distance.cpu().numpy()
        entropy_img = Image.fromarray(entropy.astype(np.uint8).squeeze())
        distance = Image.fromarray(distance.astype(np.uint8).squeeze())
        semantic = Image.fromarray((seg_final).astype(np.uint8))

        # get initial transformation
        semantic_tensor = self.base_transforms_diss(semantic) * 255
        syn_image_tensor = self.base_transforms_diss(synthesis_final_img)
        image_tensor = self.base_transforms_diss(img)
        syn_image_tensor = self.norm_transform_diss(syn_image_tensor).unsqueeze(0).cuda()
        image_tensor = self.norm_transform_diss(image_tensor).unsqueeze(0).cuda()

        # get softmax difference
        perceptual_diff = self.vgg_diff(image_tensor, syn_image_tensor)
        min_v = torch.min(perceptual_diff.squeeze())
        max_v = torch.max(perceptual_diff.squeeze())
        perceptual_diff = (perceptual_diff.squeeze() - min_v) / (max_v - min_v)
        perceptual_diff *= 255
        perceptual_diff = perceptual_diff.cpu().numpy()
        perceptual_diff = Image.fromarray(perceptual_diff.astype(np.uint8))

        # finish transformation
        perceptual_diff_tensor = self.base_transforms_diss(perceptual_diff).unsqueeze(0).cuda()
        entropy_tensor = self.base_transforms_diss(entropy_img).unsqueeze(0).cuda()
        distance_tensor = self.base_transforms_diss(distance).unsqueeze(0).cuda()

        # hot encode semantic map
        semantic_tensor[semantic_tensor == 255] = 20  # 'ignore label is 20'
        semantic_tensor = one_hot_encoding(semantic_tensor, 20).unsqueeze(0).cuda()

        # run dissimilarity
        with torch.no_grad():
            if self.prior:
                diss_pred = F.softmax(
                    self.diss_model(image_tensor, syn_image_tensor, semantic_tensor, entropy_tensor,
                                    perceptual_diff_tensor,
                                    distance_tensor), dim=1)
            else:
                diss_pred = F.softmax(self.diss_model(image_tensor, syn_image_tensor, semantic_tensor), dim=1)
        diss_pred = diss_pred.cpu().numpy()

        # do ensemble if necessary
        if self.ensemble:
            diss_pred = diss_pred[:, 1, :, :] * 0.75 + entropy_tensor.cpu().numpy() * 0.25
        else:
            diss_pred = diss_pred[:, 1, :, :]
        diss_pred = np.array(Image.fromarray(diss_pred.squeeze()).resize((image_og_w, image_og_h)))

        out = {'anomaly_score': torch.tensor(diss_pred), 'segmentation': torch.tensor(seg_final)}

        return out['anomaly_score']

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
        net = torch.nn.DataParallel(net).cuda()
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
        model_path = os.path.join(save_folder,
                                  '%s_net_%s.pth' % (config_diss['which_epoch'], config_diss['experiment_name']))
        model_weights = torch.load(model_path)
        self.diss_model.load_state_dict(model_weights)
        self.diss_model.eval()
        print('Dissimilarity Net Restored')

    def get_transformations(self):
        # Transform images to Tensor based on ImageNet Mean and STD
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])

        # synthesis necessary pre-process
        self.transform_semantic = transforms.Compose(
            [transforms.Resize(size=(256, 512), interpolation=Image.NEAREST), transforms.ToTensor()])

        self.transform_semantic1 = transforms.Resize(size=(256, 512), interpolation=Image.NEAREST)
        self.transform_semantic2 = transforms.ToTensor()

        self.transform_image_syn = transforms.Compose(
            [transforms.Resize(size=(256, 512), interpolation=Image.BICUBIC), transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5),
                                  (0.5, 0.5, 0.5))])

        # dissimilarity pre-process
        self.vgg_diff = VGG19_difference().cuda()
        self.base_transforms_diss = transforms.Compose(
            [transforms.Resize(size=(256, 512), interpolation=Image.NEAREST), transforms.ToTensor()])
        self.norm_transform_diss = transforms.Compose(
            [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # imageNet normamlization
        self.to_pil = ToPILImage()


if __name__ == '__main__':
    import bdlb

    # define fishyscapes test parameters
    fs = bdlb.load(benchmark="fishyscapes")
    # automatically downloads the dataset
    data = fs.get_dataset('Static')
    detector = AnomalyDetector(True)
    metrics = fs.evaluate(detector.estimator_worker, data)

    print('My method achieved {:.2f}% AP'.format(100 * metrics['AP']))
    print('My method achieved {:.2f}% FPR@95TPR'.format(100 * metrics['FPR@95%TPR']))
    print('My method achieved {:.2f}% auroc'.format(100 * metrics['auroc']))
