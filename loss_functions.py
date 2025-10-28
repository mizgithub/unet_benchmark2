#binary cross entryphy + dice loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from physics_weight import compute_depth_weight_map, compute_adaptive_depth_weight

def bce_dice_loss(
    logits, targets, imgs=None, w_bce=0.5, w_dice=0.5, 
    weight_map=None, alpha=0.005, beta=0.5, use_adaptive_weight=False, eps=1e-7
):
    """
    Combined BCE + Dice loss with optional adaptive depth-based weighting.

    Args:
        logits: Model raw outputs [B, 1, H, W]
        targets: Ground truth masks [B, 1, H, W]
        imgs: Input ultrasound images [B, 1, H, W] (required if use_adaptive_weight=True)
        w_bce: Weight of BCE component
        w_dice: Weight of Dice component
        weight_map: Optional precomputed weight map [B, 1, H, W]
        alpha: Depth attenuation coefficient
        beta: Intensity compensation factor
        use_adaptive_weight: Whether to auto-compute adaptive weights from imgs
        eps: Small constant for numerical stability
    """
    # Compute adaptive weights if requested
    # if use_adaptive_weight and imgs is not None:
    #     weight_map = compute_adaptive_depth_weight(imgs, alpha=alpha, beta=beta)

    # --- BCE Loss ---
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


# ---------------------------------------------------------------------
# Edge alignment loss (Sobel-based, depth-weighted)
# ---------------------------------------------------------------------
def edge_l1_loss(pred, target, imgs=None, weight_map=None, use_adaptive_weight=False, alpha=0.005, beta=0.5):
    """
    Edge alignment loss using Sobel gradients (L1), optionally weighted by adaptive depth map.

    Args:
        pred: Probability map (after sigmoid), shape [B, 1, H, W]
        target: Ground truth mask, shape [B, 1, H, W]
        imgs: Input ultrasound images [B, 1, H, W] (required if use_adaptive_weight=True)
        weight_map: Optional pixel-wise weights, shape [B, 1, H, W]
        use_adaptive_weight: Whether to compute adaptive weights
    """
    # Compute adaptive weights if requested
    # if use_adaptive_weight and imgs is not None:
    #     weight_map = compute_adaptive_depth_weight(imgs, alpha=alpha, beta=beta)

    # Sobel filters
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
