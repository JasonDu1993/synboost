"""
Pytorch Optimizer and Scheduler Related Task
"""
import math
import logging
import torch
from torch import optim
from image_segmentation.config import cfg
from collections import OrderedDict
import torch.nn as nn


def get_optimizer(args, net):
    """
    Decide Optimizer (Adam or SGD)
    """
    param_groups = net.parameters()

    if args.sgd:
        optimizer = optim.SGD(param_groups,
                              lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum,
                              nesterov=False)
    elif args.adam:
        amsgrad = False
        if args.amsgrad:
            amsgrad = True
        optimizer = optim.Adam(param_groups,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               amsgrad=amsgrad
                               )
    else:
        raise ValueError('Not a valid optimizer')

    if args.lr_schedule == 'scl-poly':
        if cfg.REDUCE_BORDER_EPOCH == -1:
            raise ValueError('ERROR Cannot Do Scale Poly')

        rescale_thresh = cfg.REDUCE_BORDER_EPOCH
        scale_value = args.rescale
        lambda1 = lambda epoch: \
            math.pow(1 - epoch / args.max_epoch,
                     args.poly_exp) if epoch < rescale_thresh else scale_value * math.pow(
                1 - (epoch - rescale_thresh) / (args.max_epoch - rescale_thresh),
                args.repoly)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif args.lr_schedule == 'poly':
        lambda1 = lambda epoch: math.pow(1 - epoch / args.max_epoch, args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise ValueError('unknown lr schedule {}'.format(args.lr_schedule))

    return optimizer, scheduler


def load_weights(net, optimizer, snapshot_file, restore_optimizer_bool=False):
    """
    Load weights from snapshot file
    """
    logging.info("Loading weights from model %s", snapshot_file)
    net, optimizer = restore_snapshot(net, optimizer, snapshot_file, restore_optimizer_bool)
    return net, optimizer


def restore_snapshot(net, optimizer, snapshot, restore_optimizer_bool):
    """
    Restore weights and optimizer (if needed ) for resuming job.
    """
    checkpoint = torch.load(snapshot, map_location=torch.device('cpu'))
    logging.info("Checkpoint Load Compelete")
    if optimizer is not None and 'optimizer' in checkpoint and restore_optimizer_bool:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore(net, checkpoint)

    return net, optimizer


def forgiving_state_restore(net, loaded_dict):
    """
    Handle partial loading when some tensors don't match up in size.
    Because we want to use models that were trained off a different
    number of classes.
    """
    if isinstance(net, nn.DataParallel) or isinstance(net, nn.parallel.DistributedDataParallel) or \
            isinstance(net, nn.parallel.DataParallel):
        net = net.module
    new_loaded_dict = loaded_dict
    is_ddp = all([('module.' in k) for k in loaded_dict.keys()])
    if is_ddp:
        new_weights = OrderedDict()
        for k, v in loaded_dict.items():
            assert k.count('module.') == 1
            new_k = k.replace('module.', '')
            new_weights[new_k] = v
        new_loaded_dict = new_weights
    # net_state_dict.update(new_loaded_dict)
    rt = net.load_state_dict(new_loaded_dict, strict=False)
    if rt.missing_keys:
        print('Missing keys when loading states {}'.format(rt.missing_keys))
    if rt.unexpected_keys:
        print('Warning: Keys dismatch when loading backbone states:\n' + str(rt))
    return net
