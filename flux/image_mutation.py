"""Image augmentation utilities for on-the-fly training augmentation.

Pure PyTorch implementation for high-performance augmentation during training.
All operations work on normalized tensors in range [-1, 1].
"""

import torch
import random
from typing import Optional


def kelvin_to_rgb_torch(temperature: float) -> torch.Tensor:
    """
    Convert color temperature in Kelvin to RGB multipliers as torch tensor.

    Based on Tanner Helland's algorithm:
    https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html

    Args:
        temperature: Color temperature in Kelvin (1000-40000)

    Returns:
        Tensor of shape (3,) with (r, g, b) multipliers in range [0, 1]
    """
    # Clamp temperature to reasonable range
    temp = max(10, min(400, temperature / 100.0))

    # Calculate red
    if temp <= 66:
        red = 255.0
    else:
        red = temp - 60
        red = 329.698727446 * (red**-0.1332047592)
        red = max(0, min(255, red))

    # Calculate green
    if temp <= 66:
        green = temp
        green = 99.4708025861 * torch.log(torch.tensor(green)).item() - 161.1195681661
    else:
        green = temp - 60
        green = 288.1221695283 * (green**-0.0755148492)
    green = max(0, min(255, green))

    # Calculate blue
    if temp >= 66:
        blue = 255.0
    else:
        if temp <= 19:
            blue = 0.0
        else:
            blue = temp - 10
            blue = (
                138.5177312231 * torch.log(torch.tensor(blue)).item() - 305.0447927307
            )
            blue = max(0, min(255, blue))

    return torch.tensor([red / 255.0, green / 255.0, blue / 255.0], dtype=torch.float32)


def adjust_brightness_torch(img_tensor: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Adjust brightness of a normalized image tensor.

    Args:
        img_tensor: Image tensor of shape (C, H, W) or (B, C, H, W) in range [-1, 1]
        factor: Brightness factor. <1 darkens, >1 brightens. Range: 0.90-1.10

    Returns:
        Adjusted tensor clamped to [-1, 1]
    """
    # Convert from [-1, 1] to [0, 1] for proper brightness adjustment
    img_01 = (img_tensor + 1.0) / 2.0

    # Apply brightness factor
    img_01 = img_01 * factor

    # Clamp and convert back to [-1, 1]
    img_01 = torch.clamp(img_01, 0.0, 1.0)
    return img_01 * 2.0 - 1.0


def adjust_contrast_torch(img_tensor: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Adjust contrast of a normalized image tensor.

    Args:
        img_tensor: Image tensor of shape (C, H, W) or (B, C, H, W) in range [-1, 1]
        factor: Contrast factor. <1 reduces contrast, >1 increases. Range: 0.90-1.10

    Returns:
        Adjusted tensor clamped to [-1, 1]
    """
    # Calculate mean across spatial dimensions, keeping channel dimension
    if img_tensor.dim() == 3:  # (C, H, W)
        mean = img_tensor.mean(dim=[1, 2], keepdim=True)
    else:  # (B, C, H, W)
        mean = img_tensor.mean(dim=[2, 3], keepdim=True)

    # Apply contrast adjustment around the mean
    adjusted = (img_tensor - mean) * factor + mean
    return torch.clamp(adjusted, -1.0, 1.0)


def adjust_tint_torch(img_tensor: torch.Tensor, tint_shift: float) -> torch.Tensor:
    """
    Apply a color tint to the image along the magenta-green axis.

    Positive values add a magenta tint, negative values add a green tint.
    This simulates color cast from different light sources.

    Args:
        img_tensor: Image tensor of shape (C, H, W) or (B, C, H, W) in range [-1, 1]
        tint_shift: Tint shift amount. Range: -0.05 to +0.05
                   Positive = magenta, Negative = green

    Returns:
        Adjusted tensor clamped to [-1, 1]
    """
    adjusted = img_tensor.clone()

    if tint_shift > 0:  # Add magenta tint (boost red and blue channels)
        if img_tensor.dim() == 3:  # (C, H, W)
            adjusted[0] = adjusted[0] + tint_shift  # Red channel
            adjusted[2] = adjusted[2] + tint_shift  # Blue channel
        else:  # (B, C, H, W)
            adjusted[:, 0] = adjusted[:, 0] + tint_shift
            adjusted[:, 2] = adjusted[:, 2] + tint_shift
    else:  # Add green tint (boost green channel)
        tint_shift = abs(tint_shift)
        if img_tensor.dim() == 3:
            adjusted[1] = adjusted[1] + tint_shift  # Green channel
        else:
            adjusted[:, 1] = adjusted[:, 1] + tint_shift

    return torch.clamp(adjusted, -1.0, 1.0)


def adjust_white_balance_torch(
    img_tensor: torch.Tensor, temperature_shift: float
) -> torch.Tensor:
    """
    Adjust white balance by shifting color temperature.

    Args:
        img_tensor: Image tensor of shape (C, H, W) or (B, C, H, W) in range [-1, 1]
        temperature_shift: Temperature shift in Kelvin. Range: -800 to +800
                          Negative = cooler (more blue), Positive = warmer (more orange)

    Returns:
        Adjusted tensor clamped to [-1, 1]
    """
    # Assume neutral starting point of 6500K (daylight)
    base_temp = 6500
    # SUBTRACT temperature_shift because lower Kelvin = warmer (orange), higher Kelvin = cooler (blue)
    # So negative shift should increase Kelvin (cooler), positive shift should decrease Kelvin (warmer)
    target_temp = base_temp - temperature_shift

    # Get RGB multipliers
    rgb_multipliers = kelvin_to_rgb_torch(target_temp)

    # Move multipliers to same device as input tensor
    rgb_multipliers = rgb_multipliers.to(img_tensor.device)

    # Convert from [0, 1] multipliers to work with [-1, 1] range
    # First convert to [0, 1], apply multipliers, then back to [-1, 1]
    img_01 = (img_tensor + 1.0) / 2.0  # Convert to [0, 1]

    # Apply per-channel multipliers
    if img_tensor.dim() == 3:  # (C, H, W)
        for c in range(3):
            img_01[c] = img_01[c] * rgb_multipliers[c]
    else:  # (B, C, H, W)
        for c in range(3):
            img_01[:, c] = img_01[:, c] * rgb_multipliers[c]

    # Clamp and convert back to [-1, 1]
    img_01 = torch.clamp(img_01, 0.0, 1.0)
    return img_01 * 2.0 - 1.0


class RandomPhotoAugmentation:
    """
    Randomly apply photo augmentations to image tensors.

    This class applies 0-2 random augmentations from: brightness, contrast, tint, and white balance.
    All operations are performed on normalized tensors in range [-1, 1].

    Args:
        seed: Random seed for reproducibility (uses args.seed)
        apply_probability: Probability of applying each augmentation (default: 0.5)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        apply_probability: float = 0.5,
    ):
        self.apply_probability = apply_probability

        # own RNG; the global `random` must stay in sync across ranks
        self.rng = random.Random(seed)

       
        self.brightness_range = (0.80, 1.20)
        self.contrast_range = (0.80, 1.20)
        self.tint_range = (-0.10, 0.10)
        self.wb_range = (-1500, 1500)

        # Available augmentations
        self.augmentations = [
            self._apply_brightness,
            self._apply_contrast,
            self._apply_tint,
            self._apply_white_balance,
        ]

    def _apply_brightness(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random brightness adjustment."""
        factor = self.rng.uniform(*self.brightness_range)
        return adjust_brightness_torch(img, factor)

    def _apply_contrast(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random contrast adjustment."""
        factor = self.rng.uniform(*self.contrast_range)
        return adjust_contrast_torch(img, factor)

    def _apply_tint(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random tint adjustment."""
        shift = self.rng.uniform(*self.tint_range)
        return adjust_tint_torch(img, shift)

    def _apply_white_balance(self, img: torch.Tensor) -> torch.Tensor:
        """Apply random white balance adjustment."""
        shift = self.rng.uniform(*self.wb_range)
        return adjust_white_balance_torch(img, shift)

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to the input tensor.

        Args:
            img_tensor: Image tensor of shape (C, H, W) or (B, C, H, W) in range [-1, 1]

        Returns:
            Augmented tensor in range [-1, 1]
        """
        # Randomly select 0-2 augmentations to apply
        num_augs = self.rng.randint(0, 2)

        if num_augs == 0:
            return img_tensor

        # Randomly select which augmentations to apply
        selected_augs = self.rng.sample(self.augmentations, k=num_augs)

        # Apply selected augmentations sequentially
        result = img_tensor
        for aug_fn in selected_augs:
            if self.rng.random() < self.apply_probability:
                result = aug_fn(result)

        return result

    def __repr__(self):
        return (
            f"RandomPhotoAugmentation("
            f"apply_probability={self.apply_probability})"
        )
