import math
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
from utils.net_util import adjust_learning_rate
from torch._six import inf

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
    model: nn.Module,
    ddp_model: nn.parallel.DistributedDataParallel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    cfg: Dict
) -> Tuple[float, float]:
    ddp_model.train()
    optimizer.zero_grad()
    losses = []
    cors = []
    accum_iter = cfg.accum_iter
    for i, (X) in enumerate(dataloader):
        if i % accum_iter == 0:
            adjust_learning_rate(optimizer, i / len(dataloader) + epoch, cfg.num_epoch, cfg.lr, cfg.min_lr, cfg.warmup_epochs)

        X = X.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            loss, pred, _ = ddp_model(X, mask_ratio=cfg.model.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {i} epoch {epoch}")
            sys.exit(1)

        loss_scaler(loss, optimizer, parameters=ddp_model.parameters(), clip_grad=cfg.model.clip_grad)

        pred = pred.to("cpu").detach()
        X = X.to("cpu").detach()
        pred = model.unpatchify(pred)
        cor = torch.mean(
            torch.tensor([torch.corrcoef(torch.cat([p, s], axis=0))[0, 1] for p, s in zip(pred, X)])
        ).item()
        optimizer.zero_grad()

        losses.append(loss_value)
        cors.append(cor)

    return np.mean(cors), np.mean(losses)