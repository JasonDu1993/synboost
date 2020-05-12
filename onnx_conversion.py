import os
from PIL import Image
import numpy as np
import cv2
from collections import OrderedDict
import shutil
import torch
from torch.backends import cudnn
import torch.onnx
import torch.nn as nn
import torchvision.transforms as transforms
import onnx
import onnxruntime

from options.test_options import TestOptions


import sys
sys.path.insert(0, './image_segmentation')
import network
from optimizer import restore_snapshot
from datasets import cityscapes
from config import assert_and_infer_cfg

TestOptions = TestOptions()
opt = TestOptions.parse()

def convert_segmentation_model(model_name = 'segmentation.onnx'):


    assert_and_infer_cfg(opt, train_mode=False)
    cudnn.benchmark = False
    torch.cuda.empty_cache()

    # Get segmentation Net
    opt.dataset_cls = cityscapes
    net = network.get_net(opt, criterion=None)
    net = torch.nn.DataParallel(net).cuda()
    print('Segmentation Net built.')
    net, _ = restore_snapshot(net, optimizer=None, snapshot=opt.snapshot, restore_optimizer_bool=False)
    net.eval()
    print('Segmentation Net Restored.')

    # Input to the model
    batch_size = 1

    x = torch.randn(batch_size, 3, 512, 1024, requires_grad=True).cuda()
    torch_out = net(x)

    # Export the model
    torch.onnx.export(net.module,                # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      model_name,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                    'output' : {0 : 'batch_size'}})

    ort_session = onnxruntime.InferenceSession(model_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def convert_synthesis_model(dataroot = '/home/giancarlo/Desktop/images_temp', model_name = 'synthesis.onnx'):

    import sys
    sys.path.insert(0, './image_synthesis')
    from models.pix2pix_model import Pix2PixModel
    import data

    world_size = 1
    rank = 0

    # Corrects where dataset is in necesary format
    opt.dataroot = dataroot
    

    opt.world_size = world_size
    opt.gpu = 0
    opt.mpdist = False

    dataloader = data.create_dataloader(opt, world_size, rank)

    model = Pix2PixModel(opt)
    model.eval()
    
    def remove_all_spectral_norm(item):
        if isinstance(item, nn.Module):
            try:
                nn.utils.remove_spectral_norm(item)
            except Exception:
                pass
        
            for child in item.children():
                remove_all_spectral_norm(child)
    
        if isinstance(item, nn.ModuleList):
            for module in item:
                remove_all_spectral_norm(module)
    
        if isinstance(item, nn.Sequential):
            modules = item.children()
            for module in modules:
                remove_all_spectral_norm(module)

    remove_all_spectral_norm(model)
    # Input to the model
    batch_size = 1
    for i, data_i in enumerate(dataloader):
        label = data_i['label'].long()
        instance = data_i['instance']
        image = data_i['image']
        break

    # create one-hot label map
    label_map = label
    bs, _, h, w = label_map.size()
    nc = opt.label_nc + 1 if opt.contain_dontcare_label \
        else opt.label_nc
    FloatTensor = torch.FloatTensor
    
    input_label = FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    inst_map = instance
    instance_edge_map = get_edges(inst_map)
    input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

    torch_out = model(data_i, mode='inference')
    
    # Export the model
    #torch.onnx.export(model,               # model being run
    #                  (input_semantics, image),  # model input (or a tuple for multiple inputs)
    #                  model_name,   # where to save the model (can be a file or file-like object)
    #                  export_params=True,        # store the trained parameter weights inside the model file
    #                  opset_version=11,          # the ONNX version to export the model to
    #                  do_constant_folding=True, # whether to execute constant folding for optimization
    #                  verbose=True,
    #                  input_names = ['input'],   # the model's input names
    #                  output_names = ['output'], # the model's output names
    #                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
    #                               'output' : {0 : 'batch_size'}})

    ort_session = onnxruntime.InferenceSession(model_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    input_feeds = {}
    input_feeds[ort_session.get_inputs()[0].name] = to_numpy(input_semantics)
    input_feeds[ort_session.get_inputs()[1].name] = to_numpy(image)

    ort_outs = ort_session.run(None, input_feeds)
    
    imtype = np.uint8
    ort_outs = np.squeeze(ort_outs[0])
    image_numpy = (np.transpose(ort_outs, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy.astype(imtype)
    
    cv2.imwrite('onnx.png', image_numpy)

    torch_out = np.squeeze(to_numpy(torch_out))
    torch_out = (np.transpose(torch_out, (1, 2, 0)) + 1) / 2.0 * 255.0
    torch_out.astype(imtype)
    cv2.imwrite('torch.png', torch_out)
    
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(torch_out, ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    
    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        return images_np
    
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)
    
def get_edges(t):
    BoolTensor = torch.BoolTensor
    edge = BoolTensor(t.size()).zero_() # for PyTorch versions higher than 1.2.0, use BoolTensor instead of ByteTensor
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()

if __name__ == '__main__':
    #convert_segmentation_model()
    convert_synthesis_model()