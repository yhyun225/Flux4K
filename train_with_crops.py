import os
import argparse
from pathlib import Path
import math

import logging
from copy import deepcopy
import itertools

import cv2
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from einops import rearrange

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import AutoencoderKL
from diffusers.utils import is_wandb_available

from models import GaussianDecoder
from dinov3_gan.dinov3_convnext_dists import DINOv3ConvNeXtDISTS
from dinov3_gan.dinov3_convnext_disc import Dinov3ConvNeXtDiscriminator

from dataset import Diffusion4KDataset
from utils.render_utils import render_image_from_gaussians
from utils.image_utils import random_crop_from_images

logger = get_logger(__name__)

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="configs/config.yaml",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    return config

def main():
    config = load_config()
    
    output_dir = config.output_dir
    logging_dir = os.path.join(output_dir, "log")
    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    visualization_dir = os.path.join(output_dir, "visualization")
    
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(visualization_dir, exist_ok=True)

        # save configs
        configs_out = os.path.join(output_dir, "config.yaml")
        
        with open(configs_out, "w+") as f:
            OmegaConf.save(config=config, f=f)

        # code snapshot
        _temp_code_dir = os.path.join(output_dir, "code_tar")
        _code_snapshot_path = os.path.join(output_dir, "code_snapshot.tar")
        os.system(
            f"rsync --relative -arhvz --quiet --filter=':- .gitignore' --exclude '.git' . '{_temp_code_dir}'"
        )
        os.system(f"tar -cf {_code_snapshot_path} {_temp_code_dir}")
        os.system(f"rm -rf {_temp_code_dir}")
        logger.info(f"Code snapshot saved to: {_code_snapshot_path}")

    # set seed
    set_seed(config.seed + accelerator.process_index)

    # device & dtype
    device = accelerator.device
    if config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.mixed_precision == "fp32":
        weight_dtype = torch.float32
    
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    

    # ====================== Load model ======================
    # 1) FLUX vae
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_model_name_or_path, torch_dtype=weight_dtype, subfolder="vae"
    ).to(device)
    vae.requires_grad_(False)

    vae_decoder = deepcopy(vae.decoder)
    
    # 2) Gaussian decoder
    model = GaussianDecoder(
        vae_decoder,
        train_size=config.gaussian_decoder.train_size,
        split_dim=tuple(config.gaussian_decoder.split_dim),
        num_gaussians_per_pixel=config.gaussian_decoder.num_gaussians_per_pixel,
    ).to(device=device, dtype=weight_dtype)
    
    model.requires_grad_(True)
    model.train()

    # 3) DINOv3-ConvNeXT (for perceptual loss)
    net_dv3d = DINOv3ConvNeXtDISTS(dinov3_convnext_size="large")
    net_dv3d.to(device)

    # 4) DINOv3-ConvNeXT Discriminator (for GAN loss)
    net_disc = Dinov3ConvNeXtDiscriminator(dinov3_convnext_size="large", resolution=1024)
    net_disc.to(device)

    # ====================== loss ======================
    lambda_l1 = config.lambda_l1
    lambda_dv3d = config.lambda_dv3d
    lambda_gan = config.lambda_gan
    
    # ====================== optimizer ======================
    trainable_params_gs = list(filter(lambda p: p.requires_grad, model.parameters()))
    trainable_params_disc = list(filter(lambda p: p.requires_grad, net_disc.parameters()))

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    
    # Generator optimizer
    optimizer = optimizer_class(
        trainable_params_gs,
        lr=config.learning_rate,
        betas=config.adam_betas,
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    # Discriminator optimzer
    optimizer_d = optimizer_class(
        trainable_params_disc,
        lr=config.learning_rate,
        betas=config.adam_betas,
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )

    if accelerator.is_main_process:
        logger.info(f"Gaussian Decoder # parpams: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")
        logger.info(f"Deiscriminator # parpams: {sum([p.numel() for p in net_disc.parameters() if p.requires_grad])}")

    # ====================== compute batch size ======================
    assert config.batch_size is not None or config.batch_size_per_gpu is not None, \
        "either batch_size or batch_size_per_gpu should be specified"
    
    batch_size = config.batch_size
    batch_size_per_gpu = config.batch_size_per_gpu if config.batch_size_per_gpu is not None \
                                                    else batch_size // accelerator.num_processes
    

    # ====================== dataset ======================
    # TODO: write dataset & dataloader code!
    dataset = Diffusion4KDataset(
        data_root=config.dataset.data_root,
        size=config.dataset.size,
        center_crop=config.dataset.center_crop,
    )

    # ====================== dataloader ======================
    num_workers = config.dataloader.num_workers if config.dataloader.num_workers is not None \
                                                else int(np.ceil(os.cpu_count() / accelerator.num_processes))
    pin_memory = config.dataloader.pin_memory
    drop_last = config.dataloader.drop_last
    shuffle = config.dataloader.shuffle
    persistent_workers = config.dataloader.persistent_workers
    collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        shuffle=shuffle,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    # Prepare everything for training
    model, net_disc, optimizer, optimizer_d, data_loader = accelerator.prepare(
        model, net_disc, optimizer, optimizer_d, data_loader
    )

    global_step = 0

    num_update_steps_per_epoch = math.ceil(len(data_loader) / accelerator.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = batch_size_per_gpu * accelerator.num_processes * config.gradient_accumulation_steps

    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Num batches each epoch = {len(data_loader)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {batch_size_per_gpu}")
        logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps = {config.max_train_steps}")

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # ====================== Train! ======================
    # for debugging
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(num_train_epochs):
        for batch in data_loader:
            model.train()

            with accelerator.accumulate(*[model, net_disc]):
                image = batch["image"].to(device=device, dtype=weight_dtype)
                image_1k = F.interpolate(
                    image, (1024, 1024), mode="bilinear", align_corners=False,
                ) #.to(weight_dtype)

                with torch.no_grad():
                    latents = vae.encode(image_1k).latent_dist.mode()
                
                gaussians = model(latents)

                # first gaussians -> 1k image
                image_render_1k = render_image_from_gaussians(
                    gaussians[0], 1024, 1024,
                    render_h=1024, render_w=1024,
                    block_h=config.block_h, block_w=config.block_w,
                ).to(weight_dtype)
                
                # second gaussians -> diff 4k
                image_render_4k = render_image_from_gaussians(
                    gaussians[0], 1024, 1024,
                    render_h=4096, render_w=4096,
                    block_h=config.block_h, block_w=config.block_w,
                ).to(weight_dtype)

                diff_4k = render_image_from_gaussians(
                    gaussians[1], 1024, 1024,
                    render_h=4096, render_w=4096,
                    block_h=config.block_h, block_w=config.block_w,
                ).to(weight_dtype)
                
                image_render_hq_4k = image_render_4k + diff_4k

                # random crops from 4k images
                target_crops, render_crops = random_crop_from_images(
                    image, image_render_hq_4k, crop_size=(1024, 1024), num_crops=3
                )
                
                # pred & target
                image_target = torch.cat([image_1k, target_crops], dim=0)
                image_preds = torch.cat([image_render_1k, render_crops], dim=0)

                # update gaussian decoder weights
                loss_l1 = F.l1_loss(image_preds, image_target) * lambda_l1
                loss_dv3d = net_dv3d(image_preds, image_target) * lambda_dv3d
                loss_gan = net_disc(image_preds, for_G=True) * lambda_gan

                total_loss_G = loss_l1 + loss_dv3d + loss_gan
                accelerator.backward(total_loss_G)
                if accelerator.sync_gradients:
                    params_to_clip = trainable_params_gs
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()

                # update discriminator weights
                image_fake = image_preds.detach()
                loss_D_fake = net_disc(image_fake, for_real=False) * lambda_gan
                loss_D_real = net_disc(image_target, for_real=True) * lambda_gan

                total_loss_D = loss_D_fake + loss_D_real
                accelerator.backward(total_loss_D)
                if accelerator.sync_gradients:
                    params_to_clip = trainable_params_disc
                    accelerator.clip_grad_norm_(params_to_clip, config.max_grad_norm)
                
                optimizer_d.step()
                optimizer_d.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if (global_step % config.checkpoint_step == 0) and (global_step >= config.max_train_steps):
                    
                        checkpoint_path = f"{checkpoint_dir}/{global_step: 07d}.pt"
                        checkpoint = {
                            "model": model.module.state_dict() if accelerator.num_process > 1 else model.state_dict(),
                            "opt": optimizer.state_dict(),
                            "train_configs": config,
                            "steps": global_step,
                        }
                        
                        torch.save(checkpoint, checkpoint_path)

                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                        
                    if global_step % config.visualization_step == 0:
                        visualization_path = f"{visualization_dir}/{global_step: 07d}"
                        train_images_dir = os.path.join(visualization_path, "train")
                        os.makedirs(train_images_dir, exist_ok=True)

                        # TODO: save image_resize & image_render
                        for i in range(image_1k.shape[0]):
                            # rendering on 1k
                            torchvision.utils.save_image(
                                torch.clamp((image_render_1k[i: i+1] + 1) / 2, 0, 1),
                                os.path.join(train_images_dir, f"render_{i}.png")
                            )
                            torchvision.utils.save_image(
                                torch.clamp((image_1k[i: i+1] + 1) / 2, 0, 1),
                                os.path.join(train_images_dir, f"target_{i}.png")
                            )
                        
                        for i in range(target_crops.shape[0]):
                            # 1k crops from 4k image
                            torchvision.utils.save_image(
                                torch.clamp((render_crops[i: i+1] + 1) / 2, 0, 1),
                                os.path.join(train_images_dir, f"render_crops_{i}.png")
                            )
                            torchvision.utils.save_image(
                                torch.clamp((target_crops[i: i+1] + 1) / 2, 0, 1),
                                os.path.join(train_images_dir, f"target_crops_{i}.png")
                            )
                        
                        for i in range(image_render_4k.shape[0]):
                            # rendering on 4k
                            torchvision.utils.save_image(
                                torch.clamp((image_render_4k[i: i + 1] + 1) / 2, 0, 1),
                                os.path.join(train_images_dir, f"render_4k_{i}.png")
                            )

                            torchvision.utils.save_image(
                                torch.clamp((diff_4k[i: i + 1] + 1) / 2, 0, 1),
                                os.path.join(train_images_dir, f"diff_4k_{i}.png")
                            )

                            torchvision.utils.save_image(
                                torch.clamp((image_render_hq_4k[i: i + 1] + 1) / 2, 0, 1),
                                os.path.join(train_images_dir, f"render_4k_hq_{i}.png")
                            )

                            torchvision.utils.save_image(
                                torch.clamp((image[i: i + 1] + 1) / 2, 0, 1),
                                os.path.join(train_images_dir, f"image_4k_{i}.png")
                            )

                        
                    if global_step % config.log_step == 0:
                        logger.info(
                            f"[Step: {global_step: 5d}] (GS) Total_loss: {total_loss_G.detach().item():.4f}, L1_loss: {loss_l1.detach().item():.4f}, DV3D_loss: {loss_dv3d.detach().item():.4f} " \
                            f"|| (Disc) Total_loss: {total_loss_D.detach().item(): .4f}, D_fake_loss: {loss_D_fake.detach().item(): .4f}, D_real_loss: {loss_D_real.detach().item(): .4f}"
                        )

                    logs = {
                        "Total_loss_G": total_loss_G.detach().item(),
                        "L1_loss": loss_l1.detach().item(),
                        "DV3D_loss": loss_dv3d.detach().item(),
                        "GAN_loss": loss_gan.detach().item(),
                        "Total_loss_D": total_loss_D.detach().item(),
                        "D_fake_loss": loss_D_fake.detach().item(),
                        "D_real_loss":loss_D_real.detach().item(),
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                
                accelerator.wait_for_everyone()
            
            if global_step > config.max_train_steps:
                break
    
        if accelerator.is_main_process:
            logger.info(f"Completed epoch {epoch + 1}/{num_train_epochs}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Training completed!")
    
    accelerator.end_training()

if __name__ == "__main__":
    main()