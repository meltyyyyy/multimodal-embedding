import datetime
import os
import time
import random
import numpy as np
import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import wandb
import hydra
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import datasets

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
    
    if int(os.environ["RANK"]) == 0:
        wandb.init(
            project="multimodal-embedding",
               config=cfg.to_dict(),
               dir=cfg.wandb.dir,
               mode=cfg.wandb.mode)
        wandb.alert(
            title="Start training",
            text=f"Training started with parameters below.\n{cfg}"
        )

    dataset = datasets.__dict__[cfg.dataset.name](**cfg.dataset)

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of voxels: {dataset.num_voxels}")
    sampler = torch.utils.data.DistributedSampler(dataset, rank=int(os.environ["RANK"]))

    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=sampler, shuffle=cfg.shuffle, pin_memory=True
    )

    model = MAEforFMRI(
        num_voxels=dataset_pretrain.num_voxels,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        decoder_embed_dim=config.decoder_embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        decoder_num_heads=config.decoder_num_heads,
        mlp_ratio=config.mlp_ratio,
        focus_range=config.focus_range,
        focus_rate=config.focus_rate,
        img_recon_weight=config.img_recon_weight,
        use_nature_img_loss=config.use_nature_img_loss,
    )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DistributedDataParallel(
        model,
        device_ids=[id for id in range(int(os.environ["LOCAL_WORLD_SIZE"]))],
        find_unused_parameters=True,
    )

    param_groups = optim_factory.add_weight_decay(model, cfg.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if int(os.environ["RANK"]) == 0:
        wandb.watch(model, log="all", log_freq=1000)

    cor_list = []
    start_time = time.time()
    print("Start Training the fmri MAE ...")

    for ep in range(cfg.n_epoch):
        sampler.set_epoch(ep) 
        cor = train_one_epoch(
            model,
            dataloader_hcp,
            optimizer,
            device,
            ep,
            loss_scaler,
            logger,
            config,
            start_time,
            model_without_ddp,
            img_feature_extractor,
            preprocess,
        )
        cor_list.append(cor)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    if int(os.environ["RANK"]) == 0:
        wandb.log("max cor", np.max(cor_list), step=cfg.num_epoch - 1)
        wandb.finish()


if __name__ == "__main__":
    main()
