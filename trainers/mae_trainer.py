import logging
import math
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch._six import inf
from torch.utils.data import DataLoader
from utils.mae_util import unpatchify
from utils.metric import correlation
from utils.net_util import adjust_learning_rate

logger = logging.getLogger(__name__)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type
        )
    return total_norm


def train_one_epoch(
    ddp_model: nn.parallel.DistributedDataParallel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.cuda.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    cfg: Dict,
) -> Tuple[float, float]:
    ddp_model.train()

    losses = []
    corrs = []
    accum_iter = cfg.accum_iter
    for i, (X, _) in enumerate(dataloader):
        if i % accum_iter == 0:
            logger.debug(f"Adjusting learning rate at iteration {i} epoch {epoch}")
            adjust_learning_rate(
                optimizer, i / len(dataloader) + epoch, cfg.num_epoch, cfg.lr, cfg.min_lr, cfg.warmup_epochs
            )

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            X = X.to(device).half()
            loss, pred, _ = ddp_model(X, mask_ratio=cfg.model.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training at step {i} epoch {epoch}")
            sys.exit(1)

        loss_scaler(loss, optimizer, parameters=ddp_model.parameters(), clip_grad=cfg.clip_grad)

        pred = pred.to("cpu").detach().float()
        pred = unpatchify(pred, cfg.model.patch_size)
        pred = pred[:, :, : cfg.dataset.org_num_voxels]
        X = X.to("cpu").detach().float()
        X = X[:, :, : cfg.dataset.org_num_voxels]

        cor = correlation(X, pred)
        losses.append(loss_value)
        corrs.append(cor)
        logger.debug(f"Epoch {epoch} | iteration {i} | loss {loss_value} | cor {cor}")

    return np.mean(corrs), np.mean(losses)
