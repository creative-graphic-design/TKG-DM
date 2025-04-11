import pathlib
from typing import List

import pytest
import torch
from diffusers import DDIMScheduler

from tkg_dm.pipelines import TKGStableDiffusionPipeline


@pytest.fixture
def save_dir(save_base_dir: pathlib.Path):
    save_dir = save_base_dir / "tkg-sd-pipeline"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


@pytest.fixture
def model_id() -> str:
    return "stable-diffusion-v1-5/stable-diffusion-v1-5"


def test_pipeline(
    model_id: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    tkg_prompts: List[str],
    active_prompt: str,
    negative_prompt: str,
    seed: int,
    save_dir: pathlib.Path,
):
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
    )
    pipe = TKGStableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    output = pipe(
        prompt=[f"{prompt}, {active_prompt}" for prompt in tkg_prompts],
        negative_prompt=[negative_prompt] * len(tkg_prompts),
        generator=torch.manual_seed(seed),
    )
    images = output.images

    for prompt, image in zip(tkg_prompts, images):
        image.save(save_dir / f"{prompt=}.png")
