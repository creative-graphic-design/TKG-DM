import torch
import torch.nn.functional as F


def calculate_positive_ratio(tensor: torch.Tensor) -> torch.Tensor:
    """Calculate the ratio of positive values in the tensor."""
    return (tensor > 0).float().mean()


def channel_mean_shift(z_T: torch.Tensor, target_shift: float) -> torch.Tensor:
    """Apply channel mean shift for color guidance.

    For channels 1 and 2, gradually shift the values until the target positive ratio is reached.
    """
    z_T_star = z_T.clone()
    for c in [1, 2]:
        channel = z_T[:, c, :, :]
        initial_ratio = calculate_positive_ratio(channel)
        target_ratio = initial_ratio + target_shift

        delta = 0.0
        while True:
            shifted = channel + delta
            current_ratio = calculate_positive_ratio(shifted)
            if current_ratio >= target_ratio:
                break
            delta += 0.01

        z_T_star[:, c, :, :] = shifted
    return z_T_star


def create_2d_gaussian(
    width: int,
    height: int,
    std_dev: float,
    device: torch.device,
    center_x: int = 0,
    center_y: int = 0,
) -> torch.Tensor:
    """Create a 2D Gaussian distribution where the center (center_x, center_y) is 1 and the periphery gradually decays towards 0.

    Coordinates are normalized in the range [-1, 1] using the specified standard deviation (std_dev).
    Returns a tensor with shape (1, 1, height//8, width//8).
    """

    y = torch.linspace(-1, 1, height // 8, device=device)
    x = torch.linspace(-1, 1, width // 8, device=device)
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

    x_grid = x_grid - center_x
    y_grid = y_grid - center_y

    gaussian = torch.exp(-((x_grid**2 + y_grid**2) / (2 * std_dev**2)))

    # shape: (h, w) -> (1, 1, h, w)
    gaussian = gaussian[None, None, :, :]

    return gaussian


def apply_tkg_noise(
    latents: torch.Tensor, target_shift: float, std_dev: float = 0.5
) -> torch.Tensor:
    """Apply noise processing to latent variables based on the tkg method."""
    B, C, H, W = latents.shape

    mask = create_2d_gaussian(
        height=H,
        width=W,
        std_dev=std_dev,
        device=latents.device,
    )

    mask = F.interpolate(
        mask,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )

    mask = mask.expand(B, C, -1, -1)

    mask = mask.to(
        device=latents.device,
        dtype=latents.dtype,
    )

    z_T_star = channel_mean_shift(latents, target_shift=target_shift)
    latents = mask * latents + (1 - mask) * z_T_star

    return latents
