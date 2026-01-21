import os
import json
from PIL import Image
from PIL.ImageOps import exif_transpose

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import random

Image.MAX_IMAGE_PIXELS = None

class Diffusion4KDataset(Dataset):
    def __init__(
        self,
        data_root,
        size=(4096, 4096),
        center_crop=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.size = size
        self.center_crop = center_crop

        self.dataset = []
        with open(os.path.join(self.data_root, "metadata.jsonl"), "r") as json_file:
            json_list = list(json_file)

            for json_str in json_list:
                data = json.loads(json_str)
                self.dataset.append(data["file_name"])
        self.len = len(self.dataset)
        
        if center_crop:
            self.transforms = T.Compose(
                [
                    T.Resize(self.size, interpolation=T.InterpolationMode.BILINEAR),
                    T.CenterCrop(self.size),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ]
            )
        else:
            self.transforms = T.Compose(
                [
                    T.Resize(self.size, interpolation=T.InterpolationMode.BILINEAR),
                    T.RandomCrop(self.size),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                ]
            )
    
        

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # exit_transpose() throws error on the truncated iamges
        while True:
            try:
                data = self.dataset[index]

                image = Image.open(os.path.join(self.data_root, data))
                image = exif_transpose(image)
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                
                image_tensor = self.transforms(image)
                break

            except:
                index = random.randint(0, self.len - 1)

        return {
            "image": image_tensor,
        }
    