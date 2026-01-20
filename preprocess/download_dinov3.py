import gdown

file_id = "1-kSZ2BfBJfO4DvEftju__XGT6Rsj596m"
url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(url,output="dinov3_gan/dinov3_weights/", quiet=False)