import torch
from einops import rearrange

from gmod.gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gmod.gsplat.rasterize_sum import rasterize_gaussians_sum

def render_image_from_gaussians(
    gaussians: dict, 
    h: int = 1024, 
    w: int = 1024, 
    render_h: int = None, 
    render_w: int = None, 
    block_h: int = 16, 
    block_w: int = 16,
):
    if render_h is None:
        render_h = h
    if render_w is None:
        render_w = w
    
    upscale_factor = render_h / h

    # type casting to float32 for rendering
    offset = gaussians["offset"].float()
    rotation = gaussians["rotation"].float()
    scale = gaussians["scale"].float() * upscale_factor
    color = gaussians["color"].float()

    for i in range(offset.shape[0]):
        # image grid: centroid coordinates of each pixels in h x w image (normalized to [0, 1])
        ys, xs = torch.meshgrid(
            torch.arange(h),
            torch.arange(w),
            indexing="ij"
        )
        image_grid = torch.stack([(xs + 0.5) / w, (ys + 0.5) / h], dim=-1)      # [h, w, 2]

        pixel_size = 1. / torch.tensor((w, h), dtype=torch.float32, device=offset.device)
        xy_grid = rearrange(image_grid, "h w xy -> (h w) xy").to(offset.device)
        normalized_pos = xy_grid + offset[i] * pixel_size

        tile_bound = (
            (render_h + block_h - 1) // block_h,
            (render_w + block_w - 1) // block_w,
            1,
        )

        pos, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            normalized_pos,
            scale[i],
            rotation[i],
            render_h,
            render_w,
            tile_bound
        )

        render_image = rasterize_gaussians_sum(
            pos,
            radii,
            conics,
            num_tiles_hit,
            color[i],
            render_h,
            render_w,
            block_h,
            block_w
        )

    render_image = render_image.unsqueeze(0).permute(0, 3, 1, 2)    # [b, c, h, w]

    return render_image