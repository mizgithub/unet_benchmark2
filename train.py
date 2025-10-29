import os
import torch
from unet_model import UNet
from train_epoch import train_one_epoch
from data_fetcher import load_thyroid_datasets
from validate import validate
from viu_net import VIUNet
import pickle
def save_checkpoint(state: dict, path: str):
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device = None):
    ck = torch.load(path, map_location=device)
    return ck
def train():
    work_dir = './output_benchmark'
    os.makedirs(work_dir, exist_ok=True)
    train_loader, val_loader, test_loader = load_thyroid_datasets()
    device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    #preparing training log file, clear previous outputs
    with open("training.log", 'w') as f:
        f.write("")
    
    # Hyperparameters
    # ... (your existing hyperparameter setup) ...
    num_blocks = 4
    weight_decay = 1e-5
    embed_dim = 32
    lr=1e-4
    epochs = 1000
    w_bce = 1.0
    w_dice = 1.0
    w_edge = 1.0

    # Model and Optimizer setup
    # ... (your existing setup) ...
    model = UNet(n_channels=1, n_classes=1)
    # model = VIUNet()
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    best_dice = 0.0

    hyper = {
        'w_bce': w_bce,
        'w_dice': w_dice,
        'w_edge': w_edge,
    }

    history = {
        'epoch': [],
        'train_loss': [],
        'train_loss_dice': [],
        'train_loss_edge': [],
        'train_loss_ssim': [],
        'val_dice': [],
        'val_iou': [],
        'lr': []
    }

    # Main Epoch Loop
    #print("Starting training...")
    for epoch in range(start_epoch, epochs):
        info = train_one_epoch(model, train_loader, optimizer, device, epoch, hyper)
        scheduler.step()
        val_info = validate(model, val_loader, device)

        history['epoch'].append(epoch)
        history['train_loss'].append(info['loss'])
        history['train_loss_dice'].append(info['loss_dice'])
        history['train_loss_edge'].append(info['loss_edge'])
        history['train_loss_ssim'].append(info['loss_ssim'])
        history['val_dice'].append(val_info['dice'])
        history['val_iou'].append(val_info['iou'])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        is_best = val_info['dice'] > best_dice
        if is_best:
            best_dice = val_info['dice']

        # Save checkpoints
        ckpath = os.path.join(work_dir, f'ck_epoch_{epoch:03d}.pth')
    
        with open("training.log", "a") as f:
            f.write(f"Epoch {epoch} | train_loss {info['loss']:.4f} "
            f"dice {info['loss_dice']:.4f} edge {info['loss_edge']:.4f} | "
            f"ssim {info['loss_ssim']:.4f} | "
            f"val_dice {val_info['dice']:.4f} val_iou {val_info['iou']:.4f} | "
            f"best {best_dice:.4f}\n")  
    #print('Training finished. Best val dice:', best_dice)
    #saving the final model
    ckpath = os.path.join(work_dir, 'final_model_benchmark.pth')
    save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'optim': optimizer.state_dict(), 'best_dice': best_dice}, ckpath)
    history_file = os.path.join(work_dir, 'history_model_benchmark.pkl')

    # Save the history object
    with open(history_file, 'wb') as file:
        pickle.dump(history, file)
    #return model, history
if __name__ == "__main__":
    train()