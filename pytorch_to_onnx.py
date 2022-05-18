# -*- coding: utf-8 -*-
# @Time    : 2022/5/12 15:16
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : pytorch_to_onnx.py
# @Software: PyCharm
import torch
import torch.nn
import os
import onnxruntime
import torchvision.transforms as standard_transforms
from PIL import Image
import numpy as np
from time import time

print(onnxruntime.get_device())


def get_onnx(save_onnx_path, input_shape):
    c, h, w = input_shape
    from build_model import RoadAnomalyDetector
    os.makedirs(os.path.dirname(save_onnx_path), exist_ok=True)
    model = RoadAnomalyDetector(True, input_shape=(c, h, w))
    # from t import A
    # model = A()

    input_names = ['data']
    output_names = ['output0', 'output1']

    x = torch.randn(1, c, h, w, requires_grad=False).cuda()

    torch.onnx.export(model, x, save_onnx_path, input_names=input_names, output_names=output_names, verbose=True,
                      opset_version=11)


def preprocess_image(image_path, input_shape, mean_std=None, is_numpy=True):
    img = Image.open(image_path)
    c, h, w = input_shape
    img = img.resize((w, h))  # [3, 2048, 1024]  , 3 is rgb
    img = torch.from_numpy(np.array(img))
    img = img.permute((2, 0, 1)).contiguous()

    img = img.unsqueeze(0).cuda()
    return img


def run_onnx(onnx_path, input_shape):
    model = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    print(model.get_providers())
    img_path = "./sample_images/road_anomaly_example.png"
    # img_path = "/zhoudu/workspaces/road_obstacles/sample_images/onnx_conversion/leftImg8bit/val/frankfurt/frankfurt_000001_083852_leftImg8bit.png"
    # mask_path = "/zhoudu/workspaces/road_obstacles/sample_images/onnx_conversion/gtFine/val/frankfurt/frankfurt_000001_083852_gtFine_labelIds.png"
    x = preprocess_image(img_path, input_shape)
    input_name = model.get_inputs()[0].name
    print("input_name:", input_name)
    outputs = model.run(None, {input_name: x})

    seg_final, diss_pred = outputs
    # main_out_np = main_out.cpu().numpy()
    # anomaly_score_np = anomaly_score.cpu().numpy()


if __name__ == '__main__':
    save_onnx_path = "./model/synboost_hw512x1024.onnx"
    input_shape = [3, 512, 1024]  # c, h, w
    if not os.path.exists(save_onnx_path):
        get_onnx(save_onnx_path, input_shape)
    # run_onnx(save_onnx_path, input_shape)
