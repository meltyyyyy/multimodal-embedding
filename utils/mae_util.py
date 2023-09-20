import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from utils.metric import correlation


def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(length, dtype=np.float32)

    grid_l = grid_l.reshape([1, length])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # cls token
        # height (== width) for the checkpoint position embedding
        orig_size = int(pos_embed_checkpoint.shape[-2] - num_extra_tokens)
        # height (== width) for the new position embedding
        new_size = int(num_patches)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %d to %d" % (orig_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, embedding_size).permute(0, 2, 1)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size))
            pos_tokens = pos_tokens.permute(0, 2, 1)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Transform images to patches.

    Args:
        imgs (torch.Tensor): Input images of shape (N, 1, num_voxels).
        patch_size (int): The size of each patch.

    Returns:
        torch.Tensor: Patches of shape (N, L, patch_size).
    """
    assert imgs.ndim == 3 and imgs.shape[2] % patch_size == 0

    batch_size = imgs.shape[0]
    num_patches = imgs.shape[2] // patch_size
    X = imgs.reshape(batch_size, num_patches, patch_size)
    return X


def unpatchify(X: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Transform patches back to images.

    Args:
        X (torch.Tensor): Patches of shape (N, L, patch_size).
        patch_size (int): The size of each patch.

    Returns:
        torch.Tensor: Images of shape (N, 1, num_voxels).
    """
    assert X.ndim == 3

    batch_size = X.shape[0]
    num_patches = X.shape[1]
    imgs = X.reshape(batch_size, 1, num_patches * patch_size)
    return imgs


@torch.no_grad()
def plot_reconstruction(
    ddp_model: DDP,
    device: torch.cuda.device,
    dataset: Dataset,
    output_path: str,
    mask_ratio: float,
    patch_size: int,
    epoch: int,
    num_figures=5,
):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    ddp_model.eval()
    fig, axs = plt.subplots(num_figures, 3, figsize=(30, 15))
    fig.tight_layout()
    axs[0, 0].set_title("Ground-truth")
    axs[0, 1].set_title("Masked Ground-truth")
    axs[0, 2].set_title("Reconstruction")

    for ax in axs:
        sample, _ = next(iter(dataloader))
        with torch.cuda.amp.autocast(enabled=True):
            sample = sample.to(device).half()
            _, pred, mask = ddp_model(sample, mask_ratio=mask_ratio)
        sample_with_mask = patchify(sample, patch_size).to("cpu").numpy().reshape(-1, patch_size)
        pred = unpatchify(pred, patch_size).to("cpu").numpy().reshape(-1)
        sample = sample.to("cpu").numpy().reshape(-1)
        mask = mask.to("cpu").numpy().reshape(-1)
        # cal the cor
        cor = np.corrcoef([pred, sample])[0, 1]

        x_axis = np.arange(0, sample.shape[-1])
        # groundtruth
        ax[0].plot(x_axis, sample)
        # groundtruth with mask
        s = 0
        for x, m in zip(sample_with_mask, mask):
            if m == 0:
                ax[1].plot(x_axis[s : s + len(x)], x, color="#1f77b4")
            s += len(x)
        # pred
        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel("cor: %.4f" % cor, weight="bold")
        ax[2].yaxis.set_label_position("right")

    os.makedirs(output_path, exist_ok=True)
    fig_name = "reconst-%s" % (datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_path, f"{fig_name}.png"))
    if os.environ["RANK"] == 0:
        wandb.log({"reconst": wandb.Image(fig)}, step=epoch)
    plt.close(fig)
