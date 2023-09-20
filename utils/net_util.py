import math
import os
from typing import Dict

import torch
import torch.nn as nn


def save_model(config: Dict, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer, loss_scaler, path: str):
    state_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
    }

    if loss_scaler is not None:
        state_dict["loss_scaler"] = loss_scaler.state_dict()
    os.makedirs(path, exist_ok=True)
    torch.save(state_dict, os.path.join(path, "checkpoint.pth"))


def adjust_learning_rate(optimizer, epoch, num_epoch, lr, min_lr, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr * epoch / warmup_epochs
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epoch - warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
