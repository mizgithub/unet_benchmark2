import torch

def compute_depth_weight_map(img, alpha=0.005, normalize=True):
    """
    Compute depth-dependent weight map based on Lambert's attenuation law.

    Args:
        img: Input ultrasound image tensor, shape [B, 1, H, W]
        alpha: Attenuation coefficient (per pixel depth unit)
        normalize: Whether to normalize weights to [1, max]

    Returns:
        weight_map: Tensor of same shape [B, 1, H, W]
    """
    B, C, H, W = img.shape

    # Create depth map (z): increases with row index (depth)
    z = torch.linspace(0, 1, steps=H, device=img.device).view(1, 1, H, 1)
    # Lambert’s law: I(z) = I0 * exp(-2 * alpha * z)
    # Weight = inverse of attenuation effect -> exp(+2 * alpha * z)
    weight_map = torch.exp(2 * alpha * z).expand(B, 1, H, W)

    # Normalize to avoid large gradients if desired
    if normalize:
        weight_map = weight_map / torch.mean(weight_map)

    return weight_map
def compute_adaptive_depth_weight(img, alpha=0.005, beta=0.5, normalize=True):
    B, C, H, W = img.shape
    device = img.device

    # Depth coordinate (0 at top, 1 at bottom)
    z = torch.linspace(0, 1, steps=H, device=device).view(1, 1, H, 1)

    # Base attenuation compensation
    depth_weight = torch.exp(alpha * z)

    # Normalize intensity (avoid zero division)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Combine: darker & deeper pixels get higher weights
    weight_map = depth_weight * torch.exp(beta * (1 - img_norm))

    if normalize:
        weight_map = weight_map / torch.mean(weight_map)

    return weight_map
import torch
import torch.nn.functional as F

def compute_snsr_weight_map(img, beta=1.0, kernel_size=5, normalize=True):
    """
    Computes a loss weight map based on the local Speckle-Noise-to-Signal Ratio (SNSR).
    
    Higher weights are assigned to regions with low signal quality (high relative noise/CV),
    which is characteristic of attenuated areas in ultrasound images, regardless of depth.

    Args:
        img (torch.Tensor): The input ultrasound image tensor (B, C, H, W).
        beta (float): Hyperparameter controlling the severity of the penalty based on CV.
        kernel_size (int): Size of the kernel for local mean/variance calculation (must be odd).
        normalize (bool): If True, normalize the final weight map by its mean.

    Returns:
        torch.Tensor: The computed weight map (B, C, H, W).
    """
    B, C, H, W = img.shape
    device = img.device
    
    # 1. Input Validation and Setup
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Kernel for local averaging (Mean and E[I^2] calculation)
    padding = kernel_size // 2
    # Create a 2D convolution kernel for averaging
    # Shape is (C_out, C_in, H, W) -> (C, 1, kernel_size, kernel_size) with groups=C for channel independence
    kernel = torch.ones(C, 1, kernel_size, kernel_size, device=device) / (kernel_size**2)

    # 2. Calculate Local Mean (μ)
    # The convolution calculates the local sum, and the kernel normalizes it to the mean
    local_mean = F.conv2d(img, kernel, padding=padding, groups=C)
    # Add epsilon for numerical stability, especially in completely black regions
    local_mean_stable = torch.clamp(local_mean, min=1e-6)

    # 3. Calculate Local Variance (σ²)
    # Variance = E[I²] - (E[I])²
    
    # a) E[I²]: Local mean of the squared image
    local_sq_mean = F.conv2d(img ** 2, kernel, padding=padding, groups=C)
    
    # b) (E[I])²: Squared local mean
    local_mean_sq = local_mean ** 2
    
    # c) Variance (must be non-negative)
    local_variance = torch.clamp(local_sq_mean - local_mean_sq, min=1e-8)

    # 4. Calculate Coefficient of Variation (CV)
    # CV = StdDev / Mean = sqrt(Variance) / Mean
    local_std = torch.sqrt(local_variance)
    
    # This is the core physics-inspired metric: CV is the SNSR proxy
    local_cv = local_std / local_mean_stable

    # 5. Final Weight Map Construction
    # Use an exponential term to penalize high CV (low signal quality) more severely.
    # The weight increases exponentially as the CV increases.
    weight_map = torch.exp(beta * local_cv)

    # 6. Optional Normalization
    if normalize:
        # Normalize by the mean to keep the overall loss magnitude consistent
        weight_map = weight_map / torch.mean(weight_map)

    return weight_map