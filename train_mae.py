import logging
import os
import random
import sys
import time

import datasets
import hydra
import models
import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from trainers.mae_trainer import NativeScalerWithGradNormCount, train_one_epoch
from utils.mae_util import plot_reconstruction
from utils.net_util import save_model

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] : [%(funcName)s] %(message)s")
logger = logging.getLogger(__name__)
cfg = None


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(config):
    if not torch.cuda.is_available():
        logger.error("No GPU available")
        sys.exit(1)
    if not dist.is_available():
        logger.error("No distributed package available")
        sys.exit(1)

    global cfg
    cfg = config

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

    main_worker()


def main_worker():
    dist.init_process_group(backend="nccl", rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_id}")
    logger.info(f"Rank {rank} is using {device}")

    if rank == 0:
        wandb.init(
            project="multimodal-embedding",
            group="sc-mbm",
            config=cfg,
            dir=cfg.wandb.dir,
            mode=cfg.wandb.mode,
            reinit=True,
        )
        wandb.alert(title="Start training", text=f"Training started with parameters below.\n{cfg}")

    dataset = datasets.__dict__[cfg.dataset.name](debug=cfg.debug, **cfg.dataset)
    sampler = DistributedSampler(dataset, rank=rank)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, shuffle=False, pin_memory=True)
    logger.info(f"Rank {rank} has {len(dataset)} samples")
    logger.info(f"Dataset: {cfg.dataset.name}")
    logger.info(f"Number of voxels: {dataset.num_voxels}")

    model = models.__dict__[cfg.model.name](**cfg.model)
    if cfg.resume:
        assert os.path.exists(cfg.resume), f"Checkpoint not found: {cfg.resume}"
        model.load_state_dict(torch.load(cfg.resume, map_location="cpu")["model"])
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=cfg.find_unused_parameters,
    )
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    param_groups = optim_factory.add_weight_decay(ddp_model, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScalerWithGradNormCount()
    logger.info(f"Optimizer: AdamW")

    if rank == 0:
        wandb.watch(model, log="all", log_freq=1000)

    cors = []
    start_time = time.time()
    logger.info("Start Training")

    for epoch in range(cfg.num_epoch):
        sampler.set_epoch(epoch)
        mean_cor, mean_loss = train_one_epoch(ddp_model, dataloader, optimizer, device, epoch, loss_scaler, cfg)
        cors.append(mean_cor)

        lr = optimizer.param_groups[0]["lr"]

        if rank == 0:
            logger.info(f"Epoch {epoch} | lr {lr} | train loss {mean_loss} | cor {mean_cor}")
            wandb.log({"lr": lr}, step=epoch)
            wandb.log({"train loss": mean_loss}, step=epoch)
            wandb.log({"cor": mean_cor}, step=epoch)
            wandb.log({"time (min)": ((time.time() - start_time) / 60.0)}, step=epoch)

            if epoch % cfg.save_freq == 0 or epoch == cfg.num_epoch - 1:
                logger.info(f"Saving model at epoch {epoch}")
                save_model(cfg, epoch, model, optimizer, loss_scaler, cfg.run_base_dir)
                plot_reconstruction(
                    ddp_model, device, dataset, cfg.run_base_dir, cfg.model.mask_ratio, cfg.model.patch_size, epoch
                )

    logger.info(f"Training time {time.time() - start_time}")
    if rank == 0:
        wandb.log({"max cor": np.max(cors)}, step=cfg.num_epoch - 1)
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
