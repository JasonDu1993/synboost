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
# import sys
# sys.path.append("utils")
from total_utils.seg_to_rect import postprocessing
from total_utils.draw_box_util import draw_total_box
import time as ti
from onnxsim import simplify
import onnx

print(onnxruntime.get_device())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_onnx(save_onnx_path, input_shape, isdynamic=True):
    print("save_onnx_path: {}".format(save_onnx_path))
    c, h, w = input_shape
    from build_model import RoadAnomalyDetector
    os.makedirs(os.path.dirname(save_onnx_path), exist_ok=True)
    model = RoadAnomalyDetector(True, input_shape=(c, h, w))
    remove_all_spectral_norm(model)
    # from t import A
    # model = A()

    input_names = ['data']
    output_names = ['output0', 'output1']

    x = torch.randn(1, c, h, w, requires_grad=False).cuda()
    if isdynamic:
        dynamic_axes = {
            # 'data': {0: 'batch_size', 2: 'height', 3: 'width'},
            # 'output0': {0: 'batch_size', 1: 'height', 2: 'width'},
            # 'output1': {0: 'batch_size'}

            'data': {0: 'batch_size'},
            'output0': {0: 'batch_size'},
            'output1': {0: 'batch_size'}
        }
    else:
        dynamic_axes = None
    torch.onnx.export(model, x, save_onnx_path, input_names=input_names, output_names=output_names, verbose=True,
                      opset_version=11, dynamic_axes=dynamic_axes, do_constant_folding=True)
    model_simp, check = simplify(save_onnx_path, dynamic_input_shape=True)
    save_onnx_path_sp = save_onnx_path.split(".")
    save_onnx_path_sp[-2] = save_onnx_path_sp[-2]+"_sim"
    output_path = ".".join(save_onnx_path_sp)
    print("sim onnx path: {}".format(output_path))
    onnx.save(model_simp, output_path)

def img_preprocess(image, input_shape):
    # image = Image.open(img_path)
    print("....input")
    c, h, w = input_shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 颜色转换
    img_resize = cv2.resize(img_rgb, (w, h))  # [h, w, c]  , 3 is bgr
    # img = torch.from_numpy(img_np)
    # img = img_resize.transpose((2, 0, 1))[None, ...].astype(np.float32)
    img = img_resize.transpose((2, 0, 1)).astype(np.float32)
    # img = np.stack([img, img])
    img = img[None, ...]
    print("img:", img.flatten()[:20])
    print("img end:", img.flatten()[-20:])
    return img


def run_onnx(onnx_path, input_shape):
    # sess_options = onnxruntime.SessionOptions()
    # print(sess_options.graph_optimization_level)

    # sess_options.intra_op_num_threads = 1

    # Set graph optimization level
    # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # To enable model serialization after graph optimization set this
    # sess_options.optimized_model_filepath = "./model/optimized_synboost_dynamic_batch_hw.onnx"
    sess_options = None
    model = onnxruntime.InferenceSession(onnx_path, sess_options=sess_options, providers=['CUDAExecutionProvider'])
    print("get_providers", model.get_providers())
    print("get_provider_options", model.get_provider_options())
    print("graph_optimization_level", model.get_session_options().graph_optimization_level)
    root = "./sample_images"
    names = list(os.listdir(root))
    names = names * 100
    for i, name in enumerate(names):
        img_path = os.path.join(root, name)
        print("img_path: {}".format(img_path))
        # img_path = "./sample_images/road_anomaly_example.png"
        # img_path = "/zhoudu/workspaces/road_obstacles/sample_images/onnx_conversion/leftImg8bit/val/frankfurt/frankfurt_000001_083852_leftImg8bit.png"
        # mask_path = "/zhoudu/workspaces/road_obstacles/sample_images/onnx_conversion/gtFine/val/frankfurt/frankfurt_000001_083852_gtFine_labelIds.png"
        image = cv2.imread(img_path)
        image_og_h, image_og_w, _ = image.shape
        x = img_preprocess(image, input_shape)
        input_name = model.get_inputs()[0].name
        print("input_name:", input_name, x.shape)
        # for i in range(100):
        # torch.cuda.synchronize()
        t0 = time()
        # print("i:", i)
        outputs = model.run(None, {input_name: x})
        # torch.cuda.synchronize()
        t1 = time()
        print("onnx run {} s".format(t1 - t0))
        seg_final, anomaly_score = outputs
        print("anomaly_score", anomaly_score.flatten()[:20])

        # 可视化diss_pred
        diss_pred_img = anomaly_score[0]

        obs_h, obs_w = diss_pred_img.shape
        # 可视化分割图
        seg_final_img = seg_final[0].astype(np.uint8)
        seg_img = cv2.resize(seg_final_img, (obs_w, obs_h))

        result = postprocessing(np.array(seg_img)[None, :], np.array(diss_pred_img)[None, :], image_og_h, image_og_w)
        img_box = draw_total_box(np.array(image), result[0], debug=True)
        print("spend total {} s".format(time() - t0))
        ti.sleep(10)


if __name__ == '__main__':
    # save_onnx_path = "./model/synboost_dynamic_batch_hw.onnx"
    # save_onnx_path = "./model/synboost_dynamic_batch_h256w512.onnx"
    # save_onnx_path = "./model/optimized_synboost_dynamic_batch_hw_1.onnx"
    # save_onnx_path = "./model/optimized_synboost_dynamic_batch_hw.onnx"
    # save_onnx_path = "./model/synboost_b2_h256w512.onnx"
    # save_onnx_path = "./model/synboost_dynamic_batch_h256w512_bool.onnx"
    # save_onnx_path = "optimized_model.onnx"
    save_onnx_path = "/zhoudu/workspaces/obstacle_det/actcommon/install/models/facerecog.road_obstacle_dect/obs_cuda0_b1h512w1024.onnx"
    isdynamic = False
    input_shape = [3, 512, 1024]  # c, h, w
    # input_shape = [3, 256, 512]  # c, h, w
    # if not os.path.exists(save_onnx_path):
    get_onnx(save_onnx_path, input_shape, isdynamic)
    # run_onnx(save_onnx_path, input_shape)
