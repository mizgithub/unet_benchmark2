import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from physics_weight import compute_depth_weight_map, compute_adaptive_depth_weight
from loss_functions import bce_dice_loss, edge_l1_loss, dice_loss, ssim_loss
import random
import time
import random
from tqdm import tqdm
import torch

def train_one_epoch(model, dataloader, optimizer, device, epoch, hyper):
    model.train()
    running = seg_running = edge_running = ssim_running = 0.0
    n = 0
    t0 = time.time()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

    for img, mask in pbar:
        img = img.to(device)
        mask = mask.to(device)

        # --- Random horizontal flip for augmentation ---
        if random.random() < 0.5:
            img = torch.flip(img, [-1])
            mask = torch.flip(mask, [-1])

        # --- Adaptive depth + intensity weighting ---
        # If you want the classic Lambert’s version, replace this call
        # with compute_depth_weight_map(img, alpha=hyper.get('alpha', 0.005))
        # weight_map = compute_adaptive_depth_weight(
        #     img,
        #     alpha=hyper.get('alpha', 0.005),
        #     beta=hyper.get('beta', 0.5),
        #     normalize=True
        # )
        weight_map = None
        # --- Forward pass ---
        logits, mask_prob, B_sim = model(img)

        # --- Segmentation loss (BCE + Dice) ---
        

        dice = dice_loss(
            logits,
            mask,
            weight_map=weight_map
        )

        ssim = ssim_loss(
            logits,
            mask,
        )
 

        # --- Edge consistency loss ---
        loss_edge = edge_l1_loss(mask_prob, mask, weight_map=weight_map)

        # --- Total loss ---
        loss = ssim + loss_edge + dice

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Tracking losses ---
        running += float(loss.item())
        seg_running += float(dice.item())
        edge_running += float(loss_edge.item())
        ssim_running +=float(ssim.item())
        n += 1

        pbar.set_postfix({
            'L_total': f'{running / n:.4f}',
            'L_seg': f'{seg_running / n:.4f}',
            'L_edge': f'{edge_running / n:.4f}',
            'L_ssim': f'{ssim_running / n:4f}'
        })

    elapsed = time.time() - t0
    pbar.close()

    # --- Epoch summary ---
    return {
        'epoch': epoch,
        'loss': running / max(1, n),
        'loss_dice': seg_running / max(1, n),
        'loss_edge': edge_running / max(1, n),
        'loss_ssim': ssim_running / max(1,n),
        'time_s': elapsed
    }
