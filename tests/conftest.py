import pathlib
from typing import List, Tuple

import pytest
import torch


@pytest.fixture
def root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def save_base_dir(root_dir: pathlib.Path) -> pathlib.Path:
    save_base_dir = root_dir / "results"
    save_base_dir.mkdir(parents=True, exist_ok=True)
    return save_base_dir


@pytest.fixture
def seed() -> int:
    return 19950815


@pytest.fixture
def torch_dtype() -> torch.dtype:
    return torch.bfloat16


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def base_prompts() -> List[str]:
    return [
        "young woman with virtual reality glasses sitting in armchair",
        "yellow lemon and slice",
        "gray cat british short hair",
        "vintage golden trumpet making music concept",
        "set of many business people",
    ]


@pytest.fixture
def active_prompt() -> str:
    return "realistic, photo-realistic, 4K, high resolution, high quality"


@pytest.fixture
def negative_prompt() -> str:
    return "background, character, cartoon, anime, text, fail, low resolution"


@pytest.fixture
def green_back_prompt() -> str:
    return "isolated on a solid green background"


@pytest.fixture
def gbp_prompts(base_prompts: List[str], green_back_prompt: str) -> List[str]:
    return [f"{prompt}, {green_back_prompt}" for prompt in base_prompts]


@pytest.fixture
def tkg_prompts(base_prompts: List[str]) -> List[str]:
    return base_prompts


@pytest.fixture(
    params=[
        "gbp",  # Green Background Prompt (GBP)
        "tkg",  # TKG Prompt
    ]
)
def method_and_prompts(request) -> Tuple[str, List[str]]:
    if request.param == "gbp":
        return ("gbp", request.getfixturevalue("gbp_prompts"))
    elif request.param == "tkg":
        return ("tkg", request.getfixturevalue("tkg_prompts"))
    else:
        raise ValueError("Invalid method specified in the fixture.")
