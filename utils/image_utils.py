from typing import Union, Tuple
from jaxtyping import Float

import torch
import torch.nn.functional as F

import random

def random_crop_from_images(
    image1: Float[torch.Tensor, "b c h w"],
    image2: Float[torch.Tensor, "b c h w"],
    crop_size: Union[list, Tuple] = (1024, 1024),
    num_crops: int = 1,
):
    assert image1.shape == image2.shape
    b, c, h, w = image1.shape
    crop_w, crop_h = crop_size

    crops1, crops2 = [], []

    for i in range(b):
        for _ in range(num_crops):
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)

            crops1.append(image1[i: i+1, :, top: top+crop_h, left: left+crop_w])
            crops2.append(image2[i: i+1, :, top: top+crop_h, left: left+crop_w])
    
    crops1 = torch.cat(crops1, dim=0)
    crops2 = torch.cat(crops2, dim=0)
    
    return crops1, crops2