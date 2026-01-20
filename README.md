# Env
(1) Torch: install compatible torch with your system.
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

(2) Requirements
```bash
pip install -r requirements.txt
```

(3) [DINOv3-ConvNeXT-Large](https://drive.google.com/file/d/1-kSZ2BfBJfO4DvEftju__XGT6Rsj596m/view?usp=sharing): Create the '/dinov3_weights' folder under the '/dinov3_gan' and download the weights into the folder.
```bash
mkdir /dinov3_gan/dinov3_weights
```

(4) [Image-GS](https://github.com/NYU-ICL/image-gs)
```bash
cd gmod
pip install -e ".[dev]" --no-build-isolation
```

