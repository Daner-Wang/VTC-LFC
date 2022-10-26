# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import sys
sys.path.append('../VTC-LFC')

import argparse
from ast import arg
import datetime
from dis import dis
from operator import mod
import numpy as np
import time
from timm.data.auto_augment import color
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
# from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, accuracy

from datasets import build_dataset
from samplers import RASampler
import utils

from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torch import nn
import logging
import os
import sys
import torch.distributed as dist
import math
import copy

import models.deit.deit as deit
from pruning.tools.deit_pruning_tools import Pruning, count_channels, SubsetSampler
from engine import train_one_epoch, DistillationLoss, evaluate, param_groups_lrd
from optim_factory import create_optimizer

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--layer-decay', type=float, default=0.9, help='layer-wise lr decay from ELECTRA/BEiT')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'CIFAR10', 'IMNET', 'IMNETV2', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--num-classes', default=1000, type=int) 

    parser.add_argument('--output-dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # pruning parameters
    parser.add_argument('--prune-only', action='store_true', help='')
    parser.add_argument('--finetune-only', action='store_true', help='')
    parser.add_argument('--finetune-from-resume', action='store_true', help='')

    parser.add_argument('--prune', action='store_true', help='Perform pruning only')
    parser.add_argument('--prune-part', default='', type=str,
                        help='pruned parts of ViT, please split with ","')
    parser.add_argument('--prune-criterion', default='', type=str,
                        help='criterion used for pruning')
    parser.add_argument('--prune-block-id', default='', type=str,
                        help='pruned block id if ViT, please split with ","')
    parser.add_argument('--prune-mode', default='bcp', type=str, 
                        help="pruning pipeline")
    parser.add_argument('--keep-qk-scale', action='store_true', 
                        help='maintain value of the scale factor for qkv')
    parser.set_defaults(keep_qk_scale=True)
    parser.add_argument('--cutoff-channel', default=0.1, type=float, help='cutoff factor for channel pruning')
    parser.add_argument('--cutoff-token', default=0.85, type=float, help='cutoff factor for token pruning')
    parser.add_argument('--lfs-lambda', default=0.1, type=float, help='cutoff factor for token pruning')
    # BCP pipeline
    parser.add_argument('--num-samples', default=2000, type=int, help='')
    parser.add_argument('--allowable-drop', default=0., type=float, help='')
    parser.add_argument('--drop-for-token', default=0., type=float, help='')
    # global channel pruning
    parser.add_argument('--prune-rate', default=0.0, type=float,
                        help='pruning rate for out channels')
    return parser

def main(args):
# setting model.......................................................................................
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    torch.cuda.empty_cache()

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    args.normalize, args.for_train = True, True
    dataset_train, _ = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    args.for_train = False
    dataset_for_sample, _ =build_dataset(is_train=True, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)

    loss_scaler = NativeScaler()

    print(f"Creating model: {args.model}")
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https') or args.resume.startswith('http'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            qk_scale=64**-0.5 if args.keep_qk_scale else None,
            dim_cfg=checkpoint['dim_cfg'] if 'dim_cfg' in checkpoint else None,
            token_cfg=checkpoint['token_cfg'] if 'token_cfg' in checkpoint else None,
            cutoff=1-args.cutoff_token
        )
        # print(model)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model'].items():
            name = k[7:] if k.find('module.') > -1 else k # remove `module.`
            if k.find('head') > -1 and v.shape[0] != args.num_classes:
                continue
            new_state_dict[name] = v
        pretrained_dist = new_state_dict
        model_dist = model.state_dict()
        pretrained_dist = {k: v for k, v in pretrained_dist.items() if k in model_dist}
        model_dist.update(pretrained_dist)
        model.load_state_dict(model_dist) 

    else:
        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            token_cfg=None
        )

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        if args.teacher_path.startswith('https')  or args.teacher_path.startswith('http'):
            teacher_checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            teacher_checkpoint = torch.load(args.teacher_path, map_location='cpu')
        if args.teacher_model == 'regnety_160':
            teacher_model = create_model(
                args.teacher_model,
                pretrained=False,
                num_classes=args.num_classes,
                global_pool='avg',
            )
        else:
            teacher_model = create_model(
                args.teacher_model,
                pretrained=False,
                num_classes=args.num_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                qk_scale=None
            )
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in teacher_checkpoint['model'].items():
            name = k[7:] if k.find('module.') > -1 else k # remove `module.`
            if k.find('head') > -1 and v.shape[0] != args.num_classes:
                continue
            new_state_dict[name] = v
        pretrained_dist = new_state_dict
        model_dist = teacher_model.state_dict()
        pretrained_dist = {k: v for k, v in pretrained_dist.items() if k in model_dist}
        model_dist.update(pretrained_dist)
        teacher_model.load_state_dict(model_dist)
        teacher_model.to(device)
        teacher_model.eval()

    dense_model = create_model(
        args.model, pretrained=False, num_classes=args.num_classes, 
        drop_rate=args.drop, drop_path_rate=args.drop_path, drop_block_rate=None
    )
    flops = FlopCountAnalysis(dense_model, (data_loader_train.dataset[0][0].unsqueeze(0),)).total()
    n_parameters = sum(p.numel() for p in dense_model.parameters() if p.requires_grad)
    n_channel_org = count_channels(model=dense_model)
    
    model.to(device)
    model_without_ddp = model
    # testing model.......................................................................................  
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        print(f"Total FLOPs: {FlopCountAnalysis(model, (data_loader_train.dataset[0][0].unsqueeze(0).to(device),)).total()/1e9}G")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}M")
        return

    # fintuning only.......................................................................................
    if args.finetune_only:
        args.prune, args.prune_only = False, False
        i_mask = checkpoint['i_mask'] if args.resume and 'i_mask' in checkpoint else None
        o_mask = checkpoint['o_mask'] if args.resume and 'o_mask' in checkpoint else None
        sub_n_parameters, sub_flops = n_parameters, flops
        submodel = model
        submodel_without_ddp = model_without_ddp

        # 设置submodel的优化器
        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEma(
                submodel,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume='')

        if args.distributed:
            submodel = torch.nn.parallel.DistributedDataParallel(submodel, device_ids=[args.gpu])
            submodel_without_ddp = submodel.module

        if args.layer_decay < 1.:
            param_groups = param_groups_lrd(submodel_without_ddp, args.weight_decay,
                                            no_weight_decay_list=submodel_without_ddp.no_weight_decay(),
                                            layer_decay=args.layer_decay
                                            )
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, submodel_without_ddp, modelparam=param_groups if args.layer_decay < 1. else None)
        lr_scheduler, _ = create_scheduler(args, optimizer)

        criterion = LabelSmoothingCrossEntropy()
        if mixup_active:
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        criterion = DistillationLoss(criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau)

        if args.finetune_from_resume and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])

        test_stats = evaluate(data_loader_val, submodel, device)
        print(f"Accuracy after pruning on the {len(dataset_val)} test images: {test_stats['acc1']:.3f}%")

        n_channel_sub = count_channels(model=submodel_without_ddp)
        sub_n_parameters = sum(p.numel() for p in submodel_without_ddp.parameters() if p.requires_grad)
        sub_flops = FlopCountAnalysis(submodel_without_ddp, (data_loader_train.dataset[0][0].unsqueeze(0).to(device),)).total()

        channel_pr = 100. * (n_channel_org - n_channel_sub) / n_channel_org
        token_pr = (sum(dense_model.num_tokens) - sum(submodel_without_ddp.num_tokens)) / sum(dense_model.num_tokens) * 100.
        param_pr = (n_parameters - sub_n_parameters) / n_parameters * 100.
        flops_pr = (flops - sub_flops) / flops * 100.

        print(f'Channels drop: {channel_pr:.3f}%, Parameters drop: {param_pr:.3f}%, Flops drop: {flops_pr:.3f}%')
        print(f'pruned channels: {n_channel_org - n_channel_sub}')
        print(f'submodel (token-pr {token_pr}%) token-cfg: {submodel_without_ddp.num_tokens}')

    else:
        channel_pr, token_pr, param_pr, flops_pr = 0., 0., 0., 0.

    dim_cfg, i_mask, o_mask = model_without_ddp.dim_cfg, None, None
    args.num_heads = model_without_ddp.num_heads
    # pruning model.......................................................................................
    if args.prune:
        n_channel_org = count_channels(model=model_without_ddp)
        pruner_paths = os.path.join(output_dir, 'pruner.pth')
        pruner = Pruning(
            model=model_without_ddp, num_classes=args.num_classes, num_heads=args.num_heads, 
            prune_part=args.prune_part, prune_block_id=args.prune_block_id, prune_rate=args.prune_rate
            )
        pruner.dim_cfg = model_without_ddp.dim_cfg

        log_stats = {'args': str(args)}
        if args.output_dir and utils.is_main_process() and 1:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.prune_mode == 'bcp':
            block_id_list = [int(i) for i in args.prune_block_id.split(',')]
            total_token = sum(model_without_ddp.num_tokens)
            recorded_lr = args.lr
            linear_scaled_lr = 1e-5 * args.batch_size * utils.get_world_size() / 512.0
            args.lr = linear_scaled_lr

            sampler = SubsetSampler(np.random.choice(range(len(dataset_for_sample)),5000))
            data_loader = torch.utils.data.DataLoader(
                dataset_for_sample, sampler=sampler, 
                batch_size=int(1.5*args.batch_size), 
                num_workers=args.num_workers, 
                pin_memory=args.pin_mem, drop_last=False
                )
            
            base_scale = 2
            allowable_drop_grown = args.allowable_drop / len(block_id_list)
            token_cfg_tmp = None

            acc_list = []
            for block_id in block_id_list:
                # preparing pruning
                data_sampler = SubsetSampler(np.random.choice(range(len(dataset_for_sample)), args.num_samples))
                # data_sampler = SubsetSampler(list(range(0, len(dataset_for_sample), len(dataset_for_sample)//args.num_samples)))
                data_loader_for_prune = torch.utils.data.DataLoader(
                    dataset_for_sample, sampler=data_sampler, 
                    batch_size=50 if args.model.find('base') > -1 else 100, 
                    num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
                    )
                pruner.data_loader = data_loader_for_prune
                pruner.org_model.to(device)
                if args.distributed:
                    tmp_model = torch.nn.parallel.DistributedDataParallel(pruner.org_model, device_ids=[args.gpu])
                print('\nPreparing for searching...')
                for k, v in pruner.org_model.named_parameters():
                    v.requires_grad = True
                org_stats = evaluate(data_loader, tmp_model, device)
                
                # token pruning
                allowable_drop = allowable_drop_grown * args.drop_for_token
                pruned_i = 1
                pruned_last = 0
                acc1_drop = 0
                ntoken_i = pruner.org_model.num_tokens[block_id]
                ntoken_lw = 190 if block_id == 0 else 0
                token_cfg_tmp = pruner.org_model.num_tokens
                print('Start searching tokens...')
                pruner.prune_rate = 0.
                while token_cfg_tmp[block_id] > ntoken_lw:
                    token_cfg_tmp = np.array(token_cfg_tmp)
                    token_cfg_tmp[block_id:] = max(ntoken_i-pruned_i, ntoken_lw)
                    token_cfg_tmp = token_cfg_tmp.tolist()

                    print('token_cfg:\n', token_cfg_tmp)
                    submodel = create_model(
                        args.model,
                        pretrained=False,
                        num_classes=args.num_classes,
                        drop_rate=args.drop,
                        drop_path_rate=args.drop_path,
                        drop_block_rate=None,
                        qk_scale=64**-0.5 if args.keep_qk_scale else None,
                        dim_cfg=pruner.dim_cfg,
                        token_cfg=token_cfg_tmp,
                        cutoff=1-args.cutoff_token
                    )
                    # 从model向submodel载入参数
                    submodel.to(device)
                    pruner.load_subgraph_from_model(sub_grah=submodel)

                    submodel_without_ddp = submodel
                
                    print('Pruning token successfully, Start evaluating...')
                    test_stats = evaluate(data_loader, submodel, device)
                    acc1_drop = float(org_stats['acc1'] - test_stats['acc1'])
                    print(f"acc1: {test_stats['acc1']:.3f}% / {org_stats['acc1']:.3f}% | acc-drop-relative: {acc1_drop}% / {allowable_drop}%")
                    
                    if acc1_drop <= allowable_drop and token_cfg_tmp[block_id] > ntoken_lw:
                        pruned_last = pruned_i
                        pruned_i = min(math.ceil((base_scale-acc1_drop/allowable_drop)*pruned_i), math.ceil(pruned_i+token_cfg_tmp[block_id]/3))
                    else:
                        pruned_i = max(int((base_scale-acc1_drop/allowable_drop) * pruned_i), pruned_last)
                        print(f'pruned_i: {pruned_i}')
                        token_cfg_tmp = np.array(token_cfg_tmp)
                        token_cfg_tmp[block_id:] = max(ntoken_i-pruned_i, ntoken_lw)
                        token_cfg_tmp = token_cfg_tmp.tolist()

                        print('token_cfg:\n', token_cfg_tmp)
                        submodel = create_model(
                            args.model,
                            pretrained=False,
                            num_classes=args.num_classes,
                            drop_rate=args.drop,
                            drop_path_rate=args.drop_path,
                            drop_block_rate=None,
                            qk_scale=64**-0.5 if args.keep_qk_scale else None,
                            dim_cfg=pruner.dim_cfg,
                            token_cfg=token_cfg_tmp,
                            cutoff=1-args.cutoff_token
                        )
                        # 从model向submodel载入参数
                        submodel.to(device)
                        pruner.load_subgraph_from_model(sub_grah=submodel)
                        submodel_without_ddp = submodel
                        break
                
                #  channel pruning
                allowable_drop = allowable_drop_grown
                pruned_up = 0
                for k, m in pruner.org_model.named_modules():
                    if k.find('blocks.'+str(block_id)+'.') > -1 and isinstance(m, nn.Linear):
                        for k_part in pruner.prune_part:
                            pruned_up += m.weight.shape[0] if k_part in k else 0
                pruned_up -= 4
                pruned_i = round(pruned_up*0.01)
                pruned_last = 0
                n_channels = count_channels(model=pruner.org_model)
                acc1_drop = 0
                print('Start searching channels...')
                while acc1_drop < allowable_drop and pruned_i < pruned_up:
                    # 主进程采样submodel
                    if utils.is_main_process():
                        pruner.org_model.midfeature_id = [block_id]
                        pruner.org_model.output_midfeature = True
                        pruner.prune_block_id = [block_id]
                        pruner.prune_rate = pruned_i / n_channels
                        pruner.alpha = args.lfs_lambda
                        pruner.cutoff = args.cutoff_channel
                        print(f'Preparing to pruning {pruned_i}/{n_channels} channels in block-{block_id}...')
                        pruner.get_mask_and_newcfg(args.prune_criterion, None, device, args=args)
                        pruner.org_model.midfeature_id = None
                        pruner.org_model.output_midfeature = False
                        utils.save_on_master({'pruner': pruner}, pruner_paths)
                        if args.distributed:
                            dist.barrier()
                    else:
                        if args.distributed:
                            dist.barrier()
                    # 次进程获取submodel结构
                    pruner_point = torch.load(pruner_paths, map_location='cpu')
                    pruner = pruner_point['pruner']
                    print('pruner.dim_cfg:\n', pruner.dim_cfg)
                    submodel = create_model(
                        args.model,
                        pretrained=False,
                        num_classes=args.num_classes,
                        drop_rate=args.drop,
                        drop_path_rate=args.drop_path,
                        drop_block_rate=None,
                        qk_scale=64**-0.5 if args.keep_qk_scale else None,
                        dim_cfg=pruner.dim_cfg,
                        token_cfg=token_cfg_tmp,
                        cutoff=1-args.cutoff_token
                    )
                    # 从model向submodel载入参数
                    submodel.to(device)
                    pruner.org_model.to(device)
                    pruner.load_subgraph_from_model(sub_grah=submodel)
                    submodel_without_ddp = submodel

                    print('Pruning channel successfully, Start evaluating...')
                    test_stats = evaluate(data_loader, submodel, device)
                    acc1_drop = float(org_stats['acc1'] - test_stats['acc1'])
                    print(f"acc1: {test_stats['acc1']:.3f}% / {org_stats['acc1']:.3f}% | acc-drop-relative: {acc1_drop}% / {allowable_drop}%")
                    if acc1_drop <= allowable_drop:
                        pruned_last = pruned_i
                        pruned_i = min(math.ceil((base_scale-acc1_drop/allowable_drop)*pruned_i), math.ceil(pruned_i+(pruned_up-pruned_i)/4), pruned_up)
                    else:
                        pruned_i = max(int((base_scale-acc1_drop/allowable_drop) * pruned_i), pruned_last)
                        print(f'finish!!! pruned_i={pruned_i}')
                        if utils.is_main_process():
                            pruner.org_model.midfeature_id = [block_id]
                            pruner.org_model.output_midfeature = True
                            pruner.prune_block_id = [block_id]
                            pruner.prune_rate = pruned_i / n_channels
                            pruner.alpha = args.lfs_lambda
                            pruner.cutoff = args.cutoff_channel
                            print(f'Preparing to pruning {pruned_i}/{n_channels} channels in block-{block_id}...')
                            pruner.get_mask_and_newcfg(args.prune_criterion, None, device, args=args)
                            pruner.org_model.midfeature_id = None
                            pruner.org_model.output_midfeature = False
                            utils.save_on_master({'pruner': pruner}, pruner_paths)

                            if args.distributed:
                                dist.barrier()
                        else:
                            if args.distributed:
                                dist.barrier()
                        
                        pruner_point = torch.load(pruner_paths, map_location='cpu')
                        pruner = pruner_point['pruner']
                        print('pruner.dim_cfg:\n', pruner.dim_cfg)
                        submodel = create_model(
                            args.model,
                            pretrained=False,
                            num_classes=args.num_classes,
                            drop_rate=args.drop,
                            drop_path_rate=args.drop_path,
                            drop_block_rate=None,
                            qk_scale=64**-0.5 if args.keep_qk_scale else None,
                            dim_cfg=pruner.dim_cfg,
                            token_cfg=token_cfg_tmp,
                            cutoff=1-args.cutoff_token
                        )
                        submodel.to(device)
                        pruner.load_subgraph_from_model(sub_grah=submodel)
                        submodel_without_ddp = submodel
                        break

                if args.output_dir and 1:
                    checkpoint_paths = [output_dir / 'checkpoint_after_pruning_mid.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': submodel_without_ddp.state_dict(),
                            'dim_cfg': submodel_without_ddp.dim_cfg,
                            'token_cfg': submodel_without_ddp.num_tokens
                        }, checkpoint_path)
                acc_list.append(test_stats['acc1'])
                
                pruner.org_model = copy.deepcopy(submodel_without_ddp)
                n_channel_i = count_channels(model=submodel_without_ddp)
                log_stats = {'Block': block_id, 'Pruned channels': n_channel_org-n_channel_i, 'Max acc1 drop': acc1_drop, 'Acc1': str(acc_list)}    
                if args.output_dir and utils.is_main_process() and 1:
                    with (output_dir / "log.txt").open("a") as f:
                        f.write(json.dumps(log_stats) + "\n")

            print(f'token_cfg: {submodel_without_ddp.num_tokens}')
            submodel = create_model(
                    args.model,
                    pretrained=False,
                    num_classes=args.num_classes,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path,
                    drop_block_rate=None,
                    qk_scale=64**-0.5 if args.keep_qk_scale else None,
                    dim_cfg=submodel_without_ddp.dim_cfg,
                    token_cfg=submodel_without_ddp.num_tokens,
                    cutoff=1-args.cutoff_token
                )
            n_channel_sub = count_channels(model=submodel)
            channel_pr = 100.*(n_channel_org-n_channel_sub)/n_channel_org
            sub_n_parameters = sum(p.numel() for p in submodel.parameters() if p.requires_grad)
            param_pr = (n_parameters - sub_n_parameters) / n_parameters * 100.
            sub_flops = FlopCountAnalysis(submodel, (data_loader_train.dataset[0][0].unsqueeze(0),)).total()
            flops_pr = (flops - sub_flops) / flops * 100.
            dim_cfg, i_mask, o_mask = pruner.dim_cfg, None, None
            token_pr = (sum(model_without_ddp.num_tokens)-sum(submodel.num_tokens))/sum(model_without_ddp.num_tokens)*100.
            submodel.to(device)
            submodel.load_state_dict(submodel_without_ddp.state_dict())
            args.lr = recorded_lr

        else:
            # pruning channels--------------------
            data_sampler = SubsetSampler(np.random.choice(range(len(dataset_for_sample)), args.num_sample))
            # data_sampler = SubsetSampler(list(range(0, 1000000, 500)))
            data_loader_for_prune = torch.utils.data.DataLoader(
                dataset_for_sample, sampler=data_sampler, 
                batch_size=50 if args.model.find('base') > -1 else 100, 
                num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
            pruner.data_loader = data_loader_for_prune
            
            if utils.is_main_process():
                dataset_sample = dataset_val if args.debug else dataset_train
                pruner.n_samplers = len(dataset_val) if args.debug else args.num_samples
                pruner.org_model.midfeature_id = [12]
                pruner.org_model.output_midfeature = True
                if args.prune_rate > 0:
                    pruner.get_mask_and_newcfg(args.prune_criterion, dataset_sample, device, args=args)
                pruner.org_model.midfeature_id = None
                pruner.org_model.output_midfeature = False
                utils.save_on_master({'pruner': pruner}, pruner_paths)

                if args.distributed:
                    dist.barrier()
            else:
                if args.distributed:
                    dist.barrier()
            
            pruner_point = torch.load(pruner_paths, map_location='cpu')
            pruner = pruner_point['pruner']
            print('pruner.dim_cfg:\n', pruner.dim_cfg)
            submodel = create_model(
                args.model,
                pretrained=False,
                num_classes=args.num_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                qk_scale=64**-0.5 if args.keep_qk_scale else None,
                dim_cfg=pruner.dim_cfg,
                token_cfg=args.token_cfg
            )
            n_channel_sub = count_channels(model=submodel)
            channel_pr = 100.*(n_channel_org-n_channel_sub)/n_channel_org
            sub_n_parameters = sum(p.numel() for p in submodel.parameters() if p.requires_grad)
            param_pr = (n_parameters - sub_n_parameters) / n_parameters * 100.
            sub_flops = FlopCountAnalysis(submodel, (data_loader_train.dataset[0][0].unsqueeze(0),)).total()
            flops_pr = (flops - sub_flops) / flops * 100.
            dim_cfg, i_mask, o_mask = pruner.dim_cfg, pruner.i_mask, pruner.o_mask
            token_pr = (sum(model_without_ddp.num_tokens)-sum(submodel.num_tokens))/sum(model_without_ddp.num_tokens)*100.

            # 从model向submodel载入参数
            submodel.to(device)
            pruner.org_model.to(device)
            pruner.load_subgraph_from_model(sub_grah=submodel)
            pruner.org_model.cpu()

        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEma(
                submodel,
                decay=args.model_ema_decay,
                device='cpu' if args.model_ema_force_cpu else '',
                resume='')
        
        submodel_without_ddp = submodel
        if args.distributed:
            submodel = torch.nn.parallel.DistributedDataParallel(submodel, device_ids=[args.gpu])
            submodel_without_ddp = submodel.module
        
        # build optimizer with layer-wise lr decay (lrd)
        if args.layer_decay < 1.:
            param_groups = param_groups_lrd(submodel_without_ddp, args.weight_decay,
                                            no_weight_decay_list=submodel_without_ddp.no_weight_decay(),
                                            layer_decay=args.layer_decay
                                            )
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, submodel_without_ddp, modelparam=param_groups if args.layer_decay < 1. else None)
        lr_scheduler, _ = create_scheduler(args, optimizer)
        criterion = LabelSmoothingCrossEntropy()
        if mixup_active:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        criterion = DistillationLoss(criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau)

        test_stats = evaluate(data_loader_val, submodel, device)
        print(f"Accuracy after pruning on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        print(f'Channels drop: {channel_pr:.3f}%, Parameters drop: {param_pr:.3f}%, Flops drop: {flops_pr:.3f}%')
        print(f'pruned channels: {n_channel_org-n_channel_sub}')
        print(f'submodel (token-pr {token_pr}%) token-cfg: {submodel_without_ddp.num_tokens}')

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_after_pruning.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': submodel_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'dim_cfg': dim_cfg,
                    'i_mask': i_mask,
                    'o_mask': o_mask,
                    'token_cfg': submodel_without_ddp.num_tokens
                }, checkpoint_path)

# pruning only.......................................................................................
    if args.prune_only:
        test_stats = evaluate(data_loader_val, submodel, device)
        log_stats = {
            'acc1': test_stats['acc1'],
            'n_parameters': sub_n_parameters,
            'n_flops': int(sub_flops),
            'Parameters drop': param_pr,
            'Flops drop': flops_pr,
            'token_cfg': str(submodel_without_ddp.num_tokens)
            }
           
        if args.output_dir and utils.is_main_process() and 1:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        return

# finetuning model......................................................................................
    print(f"Start training for {args.epochs} epochs")
    for k, v in submodel_without_ddp.named_parameters():
        v.requires_grad = True

    start_time = time.time()
    max_accuracy = 0.0
    max_cdrop, max_pdrop, max_fdrop = 0., 0., 0.
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

            # 训练submodel
            epoch_time = time.time()
            train_stats = train_one_epoch(
                submodel, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn,
                set_training_mode=True
            )
            epoch_time = time.time() - epoch_time
            epoch_time = str(datetime.timedelta(seconds=int(epoch_time)))

            # 存储submodel并记录信息
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': submodel_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema),
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                        'dim_cfg': dim_cfg,
                        'i_mask': i_mask,
                        'o_mask': o_mask,
                        'token_cfg': submodel_without_ddp.num_tokens
                    }, checkpoint_path)

            test_stats = evaluate(data_loader_val, submodel, device)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.3f}%")
            
            if max_accuracy < test_stats["acc1"] and 1:
                max_accuracy = test_stats["acc1"]
                max_cdrop, max_pdrop, max_fdrop = channel_pr, param_pr, flops_pr
                if args.output_dir:
                    checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': submodel_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'model_ema': get_state_dict(model_ema),
                            'scaler': loss_scaler.state_dict(),
                            'args': args,
                            'dim_cfg': dim_cfg,
                            'i_mask': i_mask,
                            'o_mask': o_mask,
                            'token_cfg': submodel_without_ddp.num_tokens
                        }, checkpoint_path)
                
            print(f'Max accuracy: {max_accuracy:.3f}%, Parameters drop: {param_pr:.3f}%, Flops drop: {flops_pr:.3f}%')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': sub_n_parameters,
                        'n_flops': int(sub_flops),
                        'training_time': epoch_time,
                        'Parameters drop': param_pr,
                        'Flops drop': flops_pr
                        }
            
            if args.output_dir and utils.is_main_process() and 1:
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
        
        # 更新learning rate
        lr_scheduler.step(epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    log_stats = {
        'Max acc1': max_accuracy,
        'Parameters drop': max_pdrop,
        'Flops drop': max_fdrop,
        'n_parameters': sub_n_parameters,
        'n_flops': int(sub_flops),
        'token_cfg': str(submodel_without_ddp.num_tokens)
        }
            
    if args.output_dir and utils.is_main_process() and 1:
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
