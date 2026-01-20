from typing import Optional, Tuple, Union
from jaxtyping import Float
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class GaussianHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        # self.head = nn.Sequential(
        #     nn.Linear(in_features=in_channels, out_features=128),
        #     nn.GELU(),
        #     nn.Linear(in_features=128, out_features=128),
        #     nn.GELU(),
        #     nn.Linear(in_features=128, out_features=out_channels),
        # )
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, h: Float[torch.Tensor, "b c h w"]):
        return self.head(h)

class GaussianDecoder(nn.Module):
    def __init__(
        self,
        vae_decoder,
        train_size: Union[Tuple, list] = [1024, 1024],
        split_dim: tuple = (2, 1, 2, 3),
        num_gaussians_per_pixel: int = 1,
    ):
        super().__init__()
        gaussian_decoder = vae_decoder
        gaussian_decoder.conv_out = GaussianHead(
            vae_decoder.conv_out.in_channels,
            sum(split_dim) * num_gaussians_per_pixel,
        )
        self.decoder = gaussian_decoder
        self.train_size = train_size
        
        self.split_dim = split_dim
        self.num_gaussians_per_pixel = num_gaussians_per_pixel

    def forward(self, h: Float[torch.Tensor, "b c h w"]):
        h = self.decoder(h)
        h = rearrange(h, "b c h w -> b (h w) c")

        gaussians = h.chunk(self.num_gaussians_per_pixel, dim=2)
        gaussian_attributes = []

        for gaussian in gaussians:
            offset, rotation, scale, color = gaussian.split(self.split_dim, dim=2)
            gaussian_attributes.append(
                {
                    "offset": torch.tanh(offset),                   # [b, n, 2]
                    "rotation": torch.sigmoid(rotation) * 2 * math.pi,   # [b, n, 1]
                    "scale": F.softplus(scale),                     # [b, n, 2]
                    "color": color,                                 # [b, n, 3]
                }
            )
        
        return gaussian_attributes