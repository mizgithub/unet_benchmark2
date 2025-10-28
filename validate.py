import numpy as np
import torch
from eval_metrics import dice_coeff, iou_score

def validate(model, dataloader, device):
    model.eval()
    dices = []
    ious = []
    with torch.no_grad():
        for img, mask in dataloader:
            img = img.to(device)
            mask = mask.to(device)

            logits, mask_prob, B_sim = model(img)

            d = float(dice_coeff(mask_prob, mask))
            i = float(iou_score(mask_prob, mask))

            dices.append(d)
            ious.append(i)

    return {
        'dice': np.mean(dices) if len(dices) > 0 else 0.0,
        'iou': np.mean(ious) if len(ious) > 0 else 0.0
    }
