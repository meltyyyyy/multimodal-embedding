import logging
import math
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch._six import inf
from torch.utils.data import DataLoader
from utils.cfip_util import contrastive_loss
from utils.mae_util import unpatchify
from utils.metric import correlation
from utils.net_util import adjust_learning_rate

logger = logging.getLogger(__name__)


def train_one_epoch(
    ddp_model: nn.parallel.DistributedDataParallel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.cuda.device,
    epoch: int,
    cfg: Dict,
) -> Tuple[float, float]:
    ddp_model.train()

    losses = []
    accum_iter = cfg.accum_iter
    for i, (resp, stim) in enumerate(dataloader):
        if i % accum_iter == 0:
            logger.debug(f"Adjusting learning rate at iteration {i} epoch {epoch}")
            adjust_learning_rate(
                optimizer, i / len(dataloader) + epoch, cfg.num_epoch, cfg.lr, cfg.min_lr, cfg.warmup_epochs
            )

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            stim = stim.to(device)
            resp = resp.to(device)
            logits_per_image, logits_per_fmri = ddp_model(stim, resp)

        loss = (contrastive_loss(logits_per_image) + contrastive_loss(logits_per_fmri)) / 2.0
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training at step {i} epoch {epoch}")
            sys.exit(1)

        loss.backward()

        if cfg.clip_grad is not None and isinstance(cfg.clip_grad, float) and cfg.clip_grad > 0:
            norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.clip_grad)
            logger.debug(f"Grad norm: {norm}")

        optimizer.step()

        losses.append(loss_value)
        logger.debug(f"Epoch {epoch} | iteration {i} | loss {loss_value}")

    return np.mean(losses)
