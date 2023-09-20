import math

import clip as CLIP
import numpy as np
import torch
from torch import nn


class CIFP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        brain_encoder: nn.Module,
        image_encoder: nn.Module,
        **kwargs,
    ):
        super().__init__()

        self.image_encoder = image_encoder
        self.fmri_encoder = brain_encoder

        self.fmri_seq_len = brain_encoder.num_patches
        self.fmri_latent_dim = brain_encoder.embed_dim
        self.channel_mapper = nn.Sequential(
            nn.Conv1d(self.fmri_seq_len, self.fmri_seq_len // 2, 1, bias=True, dtype=self.dtype),
            nn.Conv1d(self.fmri_seq_len // 2, 77, 1, bias=True, dtype=self.dtype),
            nn.Conv1d(77, 1, 1, bias=True, dtype=self.dtype),
        )
        self.fmri_projection = nn.Linear(brain_encoder.embed_dim, embed_dim, dtype=self.dtype)

        # Initialize channel_mapper
        for m in self.channel_mapper.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Initialize fmri_projection
        nn.init.kaiming_uniform_(self.fmri_projection.weight, a=math.sqrt(5))
        if self.fmri_projection.bias is not None:
            nn.init.zeros_(self.fmri_projection.bias)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return self.image_encoder.visual.conv1.weight.dtype

    def encode_images(self, images):
        return self.image_encoder(images.type(self.dtype))

    def encode_fmri(self, fmri):
        x = self.fmri_encoder(fmri).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = self.channel_mapper(x)
        x = self.fmri_projection(x)  # [batch_size, n_ctx, embed_dim]
        return x.squeeze(1)

    def forward(self, images, fmri):
        image_features = self.encode_images(images)
        fmri_features = self.encode_fmri(fmri)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        fmri_features = fmri_features / fmri_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ fmri_features.t()
        logits_per_fmri = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_fmri


class CLIPImageEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        clip, _ = CLIP.load("ViT-B/32", jit=False, download_root="./.cache/clip")

        for params in clip.transformer.parameters():
            params.requires_grad = False

        for params in clip.token_embedding.parameters():
            params.requires_grad = False

        for params in clip.ln_final.parameters():
            params.requires_grad = False

        self.visual = clip.visual

    def forward(self, image):
        return self.visual(image)
