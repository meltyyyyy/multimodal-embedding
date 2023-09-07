import datetime
import os
import random
import time

import datasets
import hydra
import models
import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from trainers.mae_trainer import NativeScalerWithGradNormCount, train_one_epoch
from utils.net_util import save_model

cfg = None


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(config):
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
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda:" + os.environ["LOCAL_RANK"]) if torch.cuda.is_available() else torch.device("cpu")

    if int(os.environ["RANK"]) == 0:
        wandb.init(
            project="multimodal-embedding", group="sc-mbm", config=cfg.to_dict(), dir=cfg.wandb.dir, mode=cfg.wandb.mode
        )
        wandb.alert(title="Start training", text=f"Training started with parameters below.\n{cfg}")

    model = models.__dict__(cfg.model.name)(**cfg.model)
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[id for id in range(int(os.environ["LOCAL_WORLD_SIZE"]))],
        find_unused_parameters=True,
    )

    dataset = datasets.__dict__[cfg.dataset.name](**cfg.dataset)
    sampler = torch.utils.data.DistributedSampler(dataset, rank=int(os.environ["RANK"]))
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler, shuffle=cfg.shuffle, pin_memory=True)
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of voxels: {dataset.num_voxels}")

    param_groups = optim_factory.add_weight_decay(ddp_model, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScalerWithGradNormCount()
    print(optimizer)

    if int(os.environ["RANK"]) == 0:
        wandb.watch(model, log="all", log_freq=1000)

    cors = []
    start_time = time.time()
    print("Start Training the fmri MAE ...")

    for epoch in range(cfg.num_epoch):
        sampler.set_epoch(epoch)
        mean_cor, mean_loss = train_one_epoch(
            model,
            ddp_model,
            optimizer,
            dataloader,
            device,
            epoch,
            loss_scaler,
        )
        cors.append(mean_cor)

        lr = optimizer.param_groups[0]["lr"]
        wandb.log({"lr", lr}, step=epoch)
        wandb.log({"train loss": mean_loss}, step=epoch)
        wandb.log({"cor", mean_cor}, step=epoch)
        wandb.log("time (min)", (time.time() - start_time) / 60.0, step=epoch)

        if not cfg.debug and (epoch % cfg.save_freq == 0 or epoch == cfg.num_epoch - 1):
            save_model(cfg, epoch, model, optimizer, loss_scaler, cfg.save_dir)

    elapsed_time = time.time() - start_time
    print(f"Training time {elapsed_time}")
    if int(os.environ["RANK"]) == 0:
        wandb.log("max cor", np.max(cors), step=cfg.num_epoch - 1)
        wandb.finish()


if __name__ == "__main__":
    main()
