import datetime
import os
import random
import subprocess
import time
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import torch.nn.functional as F
import re
import math
from collections.abc import Iterable
from mmcv.runner import get_dist_info

def set_grad_checkpoint(model, use_fp32_attention=False, gc_step=1):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.grad_checkpointing = True
        module.fp32_attention = use_fp32_attention
        module.grad_checkpointing_step = gc_step
    model.apply(set_attr)


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, 'grad_checkpointing', False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, **kwargs)
    return module(*args, **kwargs)


def checkpoint_sequential(functions, step, input, *args, **kwargs):

    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(kwargs))

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input, *args)
            return input
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

def save_checkpoint(work_dir,
                    epoch,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    keep_last=False,
                    step=None,
                    ):
    os.makedirs(work_dir, exist_ok=True)
    state_dict = dict(state_dict=model.state_dict())
    if model_ema is not None:
        state_dict['state_dict_ema'] = model_ema.state_dict()
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    if lr_scheduler is not None:
        state_dict['scheduler'] = lr_scheduler.state_dict()
    if epoch is not None:
        state_dict['epoch'] = epoch
        file_path = os.path.join(work_dir, f"epoch_{epoch}.pth")
        if step is not None:
            file_path = file_path.split('.pth')[0] + f"_step_{step}.pth"

    torch.save(state_dict, file_path)
    if keep_last:
        for i in range(epoch):
            previous_ckgt = file_path.format(i)
            if os.path.exists(previous_ckgt):
                os.remove(previous_ckgt)


def load_checkpoint(checkpoint,
                    model,
                    model_ema=None,
                    optimizer=None,
                    lr_scheduler=None,
                    load_ema=False,
                    resume_optimizer=True,
                    resume_lr_scheduler=True
                    ):
    assert isinstance(checkpoint, str)
    ckpt_file = checkpoint
    checkpoint = torch.load(ckpt_file, map_location="cpu")

    state_dict_keys = ['pos_embed', 'base_model.pos_embed', 'model.pos_embed']
    for key in state_dict_keys:
        if key in checkpoint['state_dict']:
            del checkpoint['state_dict'][key]
            if 'state_dict_ema' in checkpoint and key in checkpoint['state_dict_ema']:
                del checkpoint['state_dict_ema'][key]
            break

    if load_ema:
        state_dict = checkpoint['state_dict_ema']
    else:
        state_dict = checkpoint.get('state_dict', checkpoint)  # to be compatible with the official checkpoint
    # model.load_state_dict(state_dict)
    missing, unexpect = model.load_state_dict(state_dict, strict=False)
    if model_ema is not None:
        model_ema.load_state_dict(checkpoint['state_dict_ema'], strict=False)
    if optimizer is not None and resume_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if lr_scheduler is not None and resume_lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
    if optimizer is not None:
        epoch = checkpoint.get('epoch', re.match(r'.*epoch_(\d*).*.pth', ckpt_file).group()[0])
        print(f'Resume checkpoint of epoch {epoch} from {ckpt_file}. Load ema: {load_ema}, '
                    f'resume optimizer： {resume_optimizer}, resume lr scheduler: {resume_lr_scheduler}.')
        return epoch, missing, unexpect
    print(f'Load checkpoint from {ckpt_file}. Load ema: {load_ema}.')
    return missing, unexpect

def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    rank, world_size = get_dist_info()
    seed = np.random.randint(2 ** 31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False