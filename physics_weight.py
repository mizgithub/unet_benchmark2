import torch

def compute_depth_weight_map(img, alpha=0.005, normalize=True):
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