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
import torch.onnx
import cv2
from onnx_conversion import remove_all_spectral_norm
from seg_to_rect import postprocessing
from draw_box_util import draw_total_box

print(onnxruntime.get_device())


def get_onnx(save_onnx_path, input_shape):
    print("save_onnx_path: {}".format(save_onnx_path))
    c, h, w = input_shape
    from build_model2 import RoadAnomalyDetector
    os.makedirs(os.path.dirname(save_onnx_path), exist_ok=True)
    model = RoadAnomalyDetector(True, input_shape=(c, h, w))
    remove_all_spectral_norm(model)
    # from t import A
    # model = A()

    input_names = ['data']
    output_names = ['output0', 'output1']

    x = torch.randn(1, c, h, w, requires_grad=False).cuda()

    torch.onnx.export(model, x, save_onnx_path, input_names=input_names, output_names=output_names, verbose=True,
                      opset_version=11)


def img_preprocess(image, input_shape):
    # image = Image.open(img_path)
    c, h, w = input_shape
    img_np = cv2.resize(image, (w, h))  # [h, w, c]  , 3 is bgr
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # 颜色转换
    # img = torch.from_numpy(img_np)
    img = img_np.transpose((2, 0, 1))[None, ...].astype(np.float32)
    return img


def run_onnx(onnx_path, input_shape):
    model = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    print(model.get_providers())
    img_path = "./sample_images/fs_static_example.png"
    # img_path = "/zhoudu/workspaces/road_obstacles/sample_images/onnx_conversion/leftImg8bit/val/frankfurt/frankfurt_000001_083852_leftImg8bit.png"
    # mask_path = "/zhoudu/workspaces/road_obstacles/sample_images/onnx_conversion/gtFine/val/frankfurt/frankfurt_000001_083852_gtFine_labelIds.png"
    image = cv2.imread(img_path)
    image_og_h, image_og_w, _ = image.shape
    x = img_preprocess(image, input_shape)
    input_name = model.get_inputs()[0].name
    print("input_name:", input_name, x.shape)
    t0 = time()
    outputs = model.run(None, {input_name: x})
    t1 = time()
    print("onnx run {} s".format(t1 - t0))
    seg_final, anomaly_score = outputs
    print(seg_final.shape, type(seg_final))
    print(anomaly_score.shape, type(anomaly_score))
    # 可视化diss_pred
    diss_pred_img = anomaly_score[0].astype(np.uint8)
    diss_pred = Image.fromarray(diss_pred_img).resize((image_og_w, image_og_h))
    # np.save("diss_pred_resize.npy", np.array(diss_pred))
    # 可视化分割图
    seg_final_img = seg_final[0].astype(np.uint8)
    semantic = Image.fromarray(seg_final_img)
    seg_img = semantic.resize((image_og_w, image_og_h))

    result = postprocessing(np.array(seg_img)[None, :], np.array(diss_pred)[None, :], 19)
    img_box = draw_total_box(np.array(image), result[0], debug=True)
    print("spend total {} s".format(time() - t0))


if __name__ == '__main__':
    save_onnx_path = "./model/synboost_hw512x1024.onnx"
    input_shape = [3, 512, 1024]  # c, h, w
    if not os.path.exists(save_onnx_path):
        get_onnx(save_onnx_path, input_shape)
    run_onnx(save_onnx_path, input_shape)
