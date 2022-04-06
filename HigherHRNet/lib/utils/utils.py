# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path, WindowsPath

import torch
import torch.optim as optim
import torch.nn as nn
def setup_logger(final_output_dir: WindowsPath, rank, phase):
    '''
    创建日志器
    Args:
        final_output_dir:日志生成路径
        rank:秩
        phase:阶段

    Returns:

    '''
    # 获取当前时间的年月日时分，并转换成指定类型的字符串
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_rank{}.log'.format(phase, time_str, rank)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    # 获取完整的日志记录文件
    final_log_file = final_output_dir / log_file
    # 定义日志器
    logger = logging.getLogger()
    # 定义输出控制器
    console_handler = logging.StreamHandler()
    # 定义文件控制器
    file_handler = logging.FileHandler(filename=str(final_log_file))

    # 定义等级：DEBUG < INFO < WARNING < ERROR < CRITICAL
    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)# 只将等级优于INFO的信息，打印在控制台
    file_handler.setLevel(logging.DEBUG)#将等级设置为DEBUG，可以所有信息都会打印到指定文件

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # 日志器添加控制器、
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, time_str


def create_logger(cfg, cfg_name, phase='train'):
    '''

    Args:
        cfg:配置对象
        cfg_name:配置文件名称
        phase:阶段

    Returns:
        返回
    '''
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # 当输出路径不存在时，创建对应的目录
    if not root_output_dir.exists() and cfg.RANK == 0:
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()
    else:
        # 确保输出路径存在：否则路径还在创建中，延时30秒等待创建完毕
        while not root_output_dir.exists():
            print('=> wait for {} created'.format(root_output_dir))
            time.sleep(30)

    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    # cfg_name = os.path.basename(cfg_name).split('.')[0]
    cfg_name = Path(cfg_name).stem # {str}w32_512_adam_lr1e-3
    # 输出根路径/数据集/模型/配置文件
    # final_output_dir = Path(os.path.join(root_output_dir,dataset,model,cfg_name))
    final_output_dir = root_output_dir / dataset / model / cfg_name

    if cfg.RANK == 0:
        print('=> creating {}'.format(final_output_dir))
        # 创建输出路径（日志、测试图片、模型）
        final_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        while not final_output_dir.exists():
            print('=> wait for {} created'.format(final_output_dir))
            time.sleep(5)

    logger, time_str = setup_logger(final_output_dir, cfg.RANK, phase)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / (cfg_name + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    # 创建tensorboard的日志保存路径
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    # 返回 日志器、输出路径、tensorboard日志路径
    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))

    if is_best and 'state_dict' in states:
        torch.save(
            states['best_state_dict'],
            os.path.join(output_dir, 'model_best.pth.tar')
        )


def get_model_summary(model, *input_tensors, item_length=26, verbose=True):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep
    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,}".format(flops_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
