import os
import glob

from PIL import Image
from tqdm import tqdm
from PIL.ImageOps import exif_transpose

Image.MAX_IMAGE_PIXELS = None

data_path = "/data1/yhyun225/Diffusion4K/images"
checklist = "erorr_images.txt"

image_ext = ["*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.png", "*.webp"]
data_list = []
for ext in image_ext:
    data_list.extend(glob.glob(f"{data_path}/*/{ext}"))

with open(checklist, "w") as f:
    for data in tqdm(data_list, total=len(data_list)):
        image = Image.open(data)
        image = exif_transpose(image)
        # try:
        #     image = Image.open(data)
        # except:
        #     f.write(f"{data}\n")