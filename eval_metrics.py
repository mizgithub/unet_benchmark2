import torch
def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    # pred and target are binary probs/tensors in {0..1}, shape (B,1,H,W)
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1) + eps
    return ((2. * inter + eps) / denom).mean()


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - inter
    return ( (inter + eps) / (union + eps) ).mean()