import pathlib
from typing import List, Tuple

import pytest
import torch
from diffusers import (
    AutoPipelineForText2Image,
    DDIMScheduler,
)
from diffusers.utils import make_image_grid
from diffusers.utils.torch_utils import randn_tensor

from tkg_dm.tkg_utils import apply_tkg_noise


@pytest.fixture
def save_dir(save_base_dir: pathlib.Path):
    save_dir = save_base_dir / "auto-pipeline"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


@pytest.mark.parametrize(
    argnames="model_name, model_id, target_shift",
    argvalues=(
        ("sdv1.5", "stable-diffusion-v1-5/stable-diffusion-v1-5", 0.07),
        ("sdxl", "stabilityai/stable-diffusion-xl-base-1.0", 0.11),
    ),
)
def test_pipeline(
    model_name: str,
    model_id: str,
    target_shift: float,
    method_and_prompts: Tuple[str, List[str]],
    torch_dtype: torch.dtype,
    device: torch.device,
    active_prompt: str,
    negative_prompt: str,
    seed: int,
    save_dir: pathlib.Path,
):
    # Unpack the method and prompts
    generation_method, prompts = method_and_prompts
    # Determine if TKG method is used
    use_tkg = generation_method == "tkg"

    # Load the model and scheduler
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    # Create latent variables
    num_prompts = len(prompts)
    latent_shape = (
        num_prompts,
        pipe.unet.config.in_channels,
        pipe.unet.config.sample_size,
        pipe.unet.config.sample_size,
    )

    # Define generator for reproducibility
    generator = torch.manual_seed(seed)

    # Generate random latents
    latents = randn_tensor(
        shape=latent_shape,
        generator=generator,
        device=device,
        dtype=torch_dtype,
    )

    # Apply TKG noise if applicable
    if use_tkg:
        latents = apply_tkg_noise(latents, target_shift)

    # Generate images based on the latents and prompts
    output = pipe(
        prompt=[f"{prompt}, {active_prompt}" for prompt in prompts],
        negative_prompt=[negative_prompt] * num_prompts,
        latents=latents,
    )
    images = output.images

    # Save images to disk
    for prompt, image in zip(prompts, images):
        image.save(save_dir / f"{model_name=}, {generation_method=}, {prompt=}.png")

    grid_image = make_image_grid(images=images, rows=1, cols=len(images))
    grid_image.save(save_dir / f"{model_name=}, {generation_method=}, grid.png")
