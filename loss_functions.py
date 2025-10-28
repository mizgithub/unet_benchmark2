#binary cross entryphy + dice loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from physics_weight import compute_depth_weight_map, compute_adaptive_depth_weight

def bce_dice_loss(
    logits, targets, imgs=None, w_bce=0.5, w_dice=0.5, 
    weight_map=None, alpha=0.005, beta=0.5, use_adaptive_weight=False, eps=1e-7
):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    if weight_map is not None:
        bce = bce * weight_map
    bce = bce.mean()

    # --- Dice Loss ---
    probs = torch.sigmoid(logits)
    if weight_map is not None:
        intersection = torch.sum(weight_map * probs * targets)
        union = torch.sum(weight_map * probs) + torch.sum(weight_map * targets)
    else:
        intersection = torch.sum(probs * targets)
        union = torch.sum(probs) + torch.sum(targets)

    dice_loss = 1 - (2 * intersection + eps) / (union + eps)

    # --- Combined Loss ---
    total_loss = w_bce * bce + w_dice * dice_loss
    return total_loss

def dice_loss(logits, target, weight_map=None, smooth=1e-15):
    probs = torch.sigmoid(logits)

    # Ensure same shape
    target = target.float()
    if target.ndim == probs.ndim - 1:
        target = target.unsqueeze(1)

    # Flatten
    probs_flat = probs.contiguous().view(probs.shape[0], -1)
    target_flat = target.contiguous().view(target.shape[0], -1)

    if weight_map is not None:
        if weight_map.ndim == probs.ndim - 1:
            weight_map = weight_map.unsqueeze(1)
        weight_map = weight_map.contiguous().view(weight_map.shape[0], -1)
        intersection = torch.sum(weight_map * probs_flat * target_flat, dim=1)
        denominator = torch.sum(weight_map * (probs_flat + target_flat), dim=1)
    else:
        intersection = torch.sum(probs_flat * target_flat, dim=1)
        denominator = torch.sum(probs_flat + target_flat, dim=1)

    dice = (2. * intersection + smooth) / (denominator + smooth)
    loss = 1 - dice.mean()

    return loss

# ---------------------------------------------------------------------
# Edge alignment loss (Sobel-based, depth-weighted)
# ---------------------------------------------------------------------
def edge_l1_loss(pred, target, imgs=None, weight_map=None, use_adaptive_weight=False, alpha=0.005, beta=0.5):
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)

    # Gradient maps
    grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
    grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
    grad_tgt_x = F.conv2d(target, sobel_x, padding=1)
    grad_tgt_y = F.conv2d(target, sobel_y, padding=1)

    # Edge magnitude maps
    edge_pred = torch.sqrt(grad_pred_x ** 2 + grad_pred_y ** 2 + 1e-7)
    edge_tgt = torch.sqrt(grad_tgt_x ** 2 + grad_tgt_y ** 2 + 1e-7)

    # L1 difference (weighted)
    edge_diff = torch.abs(edge_pred - edge_tgt)
    if weight_map is not None:
        edge_diff = edge_diff * weight_map

    return edge_diff.mean()
def ssim_loss(logits, target, weight_map=None, window_size=11, window_sigma=1.5, smooth=1e-8):
    # Apply sigmoid to logits to convert to probability space
    probs = torch.sigmoid(logits)
    target = target.float()
    
    # Ensure same shape
    if target.ndim == probs.ndim - 1:
        target = target.unsqueeze(1)
    
    # Create Gaussian window for local statistics
    def gaussian_window(window_size, sigma):
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0) * g.unsqueeze(1)

    B, C, H, W = probs.shape
    window = gaussian_window(window_size, window_sigma).to(probs.device).unsqueeze(0).unsqueeze(0)
    
    # Compute local means
    mu_x = F.conv2d(probs, window, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=C)
    
    # Compute local variances and covariance
    sigma_x = F.conv2d(probs * probs, window, padding=window_size // 2, groups=C) - mu_x ** 2
    sigma_y = F.conv2d(target * target, window, padding=window_size // 2, groups=C) - mu_y ** 2
    sigma_xy = F.conv2d(probs * target, window, padding=window_size // 2, groups=C) - mu_x * mu_y
    
    # SSIM constants (for stability)
    L = 1  # pixel value range (since probs ∈ [0,1])
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # SSIM map
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    
    loss = 1 - ssim_map.mean()
    
    return loss