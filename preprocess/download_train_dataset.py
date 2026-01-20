from huggingface_hub import snapshot_download

save_dir = "/data1/yhyun225/Diffusion4K"

snapshot_download(
    repo_id="zhang0jhon/Aesthetic-Train-V2",
    repo_type="dataset",
    local_dir=save_dir,
)