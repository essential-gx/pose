# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter

import _init_paths
import models

from config import cfg
from config import update_config
# from config.models import MODEL_EXTRAS
# from config import MODEL_EXTRAS
from core.loss import MultiLossFactory
from core.trainer import do_train
from dataset import make_dataloader
from fp16_utils.fp16util import network_to_half
from fp16_utils.fp16_optimizer import FP16_Optimizer
from utils.utils import create_logger
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
# from utils.utils import setup_logger


def parse_args():

    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        # default='',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    # opts：记录命令行上剩余的参数。包括名称、值，依次排列组成列表，
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        # 所有剩余的参数，均转化为一个列表赋值给此项，通常用此方法来将剩余的参数传入另一个parser进行解析。
                        # 如果nargs没有定义，则可传入参数的数量由action决定，通常情况下为一个，并且不会生成长度为一的列表。
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        # default='0',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    cfg.defrost()
    cfg.RANK = args.rank
    cfg.freeze()
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg=cfg, cfg_name=args.cfg, phase='train'
    )
    # 注意：
    # 使用debug方式打印日志，不输出到控制台。
    # 使用info方式打印日志，输出到控制台
    logger.debug(pprint.pformat(args))
    logger.debug(cfg)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or cfg.MULTIPROCESSING_DISTRIBUTED
    # 可用GPU数量
    ngpus_per_node = torch.cuda.device_count()

    if cfg.MULTIPROCESSING_DISTRIBUTED:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, final_output_dir, tb_log_dir)
        )
    else:
        # Simply call main_worker function
        # 主线程训练
        main_worker(
            ','.join([str(i) for i in cfg.GPUS]),# GPU索引字符串
            ngpus_per_node,# 可用GPU数量
            args,# 命令行参数
            final_output_dir,# 输出文件保存路径（log日志、测试结果、模型）
            tb_log_dir,#tensorrboard日志保存路径
            logger# 日志对象
        )


def main_worker(gpu, ngpus_per_node, args, final_output_dir, tb_log_dir,logger):
    '''

    Args:
        gpu: GPU索引元组
        ngpus_per_node: 可用GPU数量
        args: 命令行参数
        final_output_dir: 输出文件保存路径（log日志、测试结果、模型）
        tb_log_dir: tensorrboard日志保存路径
        logger: 日志对象

    Returns:

    '''
    # cudnn related setting，使用pytorch提速
    cudnn.benchmark = cfg.CUDNN.BENCHMARK  # 为整个网络的每个卷积层搜索最适合它的卷积实现算法
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC  # True 固定cuda中的随机种子
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED  # cuda设置为使用使用非确定性算法

    if cfg.FP16.ENABLED:  # False：不采用混合精度运算
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if cfg.FP16.STATIC_LOSS_SCALE != 1.0:
        if not cfg.FP16.ENABLED:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    args.gpu = int(gpu)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:  # False：不启用分布式训练
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if cfg.MULTIPROCESSING_DISTRIBUTED:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.
              format(args.dist_url, args.world_size, args.rank))
        dist.init_process_group(
            backend=cfg.DIST_BACKEND,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

    update_config(cfg, args)

    # setup logger
    # logger, _ = setup_logger(final_output_dir, args.rank, 'train')
    # 创建HRNet对象
    # models.pose_higher_hrnet.get_pose_net(cfg,is_train=True)
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg,
        is_train=True
    )

    # copy model file
    if not cfg.MULTIPROCESSING_DISTRIBUTED or (
            cfg.MULTIPROCESSING_DISTRIBUTED
            and args.rank % ngpus_per_node == 0
    ):
        # 没有pytorch的分布式训练，复制模型文件
        this_dir = os.path.dirname(__file__)

        shutil.copy2(
            os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
            final_output_dir
        )

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if not cfg.MULTIPROCESSING_DISTRIBUTED or (cfg.MULTIPROCESSING_DISTRIBUTED and args.rank % ngpus_per_node == 0):
        # 当没有使用分布式训练，或者使用分布式训练，但是GPU数量为1
        dump_input = torch.rand(
            (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
        )
        writer_dict['writer'].add_graph(model, dump_input)
        # logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    if cfg.FP16.ENABLED:  # False
        model = network_to_half(model)

    if cfg.MODEL.SYNC_BN and not args.distributed:  # False
        print('Warning: Sync BatchNorm is only supported in distributed training.')

    if args.distributed:  # False：不使用分布式训练
        if cfg.MODEL.SYNC_BN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:# 有一个可用GPU，不可以分布式训练
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # 创建自定义损失函数
    loss_factory = MultiLossFactory(cfg).cuda()

    # Data loading code
    train_loader = make_dataloader(
        cfg, is_train=True, distributed=args.distributed
    )
    # logger.info(train_loader.dataset)

    best_perf = -1
    best_model = False
    last_epoch = -1
    # 获取优化器
    optimizer = get_optimizer(cfg, model)

    if cfg.FP16.ENABLED:
        optimizer = FP16_Optimizer(
            optimizer,
            static_loss_scale=cfg.FP16.STATIC_LOSS_SCALE,
            dynamic_loss_scale=cfg.FP16.DYNAMIC_LOSS_SCALE
        )

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth.tar')
    # 加载 预训练文件
    # if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
    #     logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    #     checkpoint = torch.load(checkpoint_file)
    #     begin_epoch = checkpoint['epoch']
    #     best_perf = checkpoint['perf']
    #     last_epoch = checkpoint['epoch']
    #     model.load_state_dict(checkpoint['state_dict'])
    #
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

    if cfg.FP16.ENABLED:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
            last_epoch=last_epoch
        )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # train one epoch
        do_train(cfg, model, train_loader, loss_factory, optimizer, epoch,
                 final_output_dir, tb_log_dir, writer_dict, fp16=cfg.FP16.ENABLED)

        # In PyTorch 1.1.0 and later, you should call `lr_scheduler.step()` after `optimizer.step()`.
        lr_scheduler.step()

        perf_indicator = epoch
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if not cfg.MULTIPROCESSING_DISTRIBUTED or (
                cfg.MULTIPROCESSING_DISTRIBUTED
                and args.rank == 0
        ):
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                # 'best_state_dict': model.module.state_dict(),
                'best_state_dict': model.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state{}.pth.tar'.format(gpu)
    )

    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


def load_model_parameter():
    pass


if __name__ == '__main__':
    # import os
    #
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
