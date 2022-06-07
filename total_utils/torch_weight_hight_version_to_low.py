# -*- coding: utf-8 -*-
# @Time    : 2021/8/11 12:05
# @Author  : 周渡
# @Email   : zhoudu@cloudwalk.com
# @File    : torch_weight_transform.py
# @Software: PyCharm
import torch


def high_version_to_low_version(path, save_path):
    state_dict = torch.load(path)
    torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    path = "/zhoudu/workspaces/synboost/../road_obstacles/models/image-dissimilarity/best_net_replicate_best_mult_3.pth"
    save_path = "/zhoudu/workspaces/synboost/../road_obstacles/models/image-dissimilarity/best_net_replicate_best_mult_3.torch140.pth"
    high_version_to_low_version(path, save_path)
    print("END!")
