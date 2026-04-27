r"""
This file contains the main execution code for the SETI data.

Use "
find . -type f -name "*.png" \
  | grep -v "^./abandoned/" \
  | grep -v "^./archived/" \
  | grep -v "^./data_process/post_process/" \
  | sort \
  | awk -F/ '
    {
      dir = $(1);
      for (i = 2; i < NF; ++i) dir = dir "/" $i;
      file_map[dir]++;
      if (file_map[dir] <= 3) print $0;
    }
  ' \
  | xargs git add -f
  " to sort and add pngs.

Use "
git diff --cached --name-only | grep '\.png$' | xargs git restore --staged
  “ to delete all pngs.

Make sure you do this before committing.

"""
import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from config.configs import load_config
from gen.SETIdataset import DynamicSpectrumDataset
from model.DetMSWNet import MSWNet
from model.utils.Regressor1D import FreqRegressionDetector
from utils.loss_func import DetectionCombinedLoss, MaskCombinedLoss
from utils.train_core import train_model
from utils.train_utils import safe_load_state_dict, load_optimizer_selectively

_config = load_config()

mode = _config["pmode"]
tchans = _config["tchans"]
fchans = _config["fchans"]
df = _config["df"]
dt = _config["dt"]
fch1 = _config["fch1"]
ascending = _config["ascending"]
drift_min = _config["drift_min"]
drift_max = _config["drift_max"]
drift_min_abs = _config["drift_min_abs"]
snr_min = _config["snr_min"]
snr_max = _config["snr_max"]
width_min = _config["width_min"]
width_max = _config["width_max"]
num_signals = _config["num_signals"]
noise_std_min = _config["noise_std_min"]
noise_std_max = _config["noise_std_max"]
noise_mean_min = _config["noise_mean_min"]
noise_mean_max = _config["noise_mean_max"]
nosie_type = _config["nosie_type"]
rfi_enhance = _config["rfi_enhance"]
use_fil = _config["use_fil"]
background_fil = _config["background_fil"]

batch_size = _config["batch_size"]
num_workers = _config["num_workers"]
num_epochs = _config["num_epochs"]
steps_per_epoch = _config["steps_per_epoch"]
valid_interval = _config["valid_interval"]
valid_steps = _config["valid_steps"]
log_interval = _config["log_interval"]
lr = _config["lr"]
weight_decay = _config["weight_decay"]
T_max = _config["T_max"]
eta_min = _config["eta_min"]
force_save_best = _config["force_save_best"]
freeze_backbone = _config["freeze_backbone"]
force_reconstruct = _config["force_reconstruct"]
mismatch_load = _config["mismatch_load"]
checkpoint_dir = _config["checkpoint_dir"]

dim = _config["dim"]
levels = _config["levels"]
dwtnet_args = _config["dwtnet_args"]
unet_args = _config["unet_args"]
detector = FreqRegressionDetector
detector_args = _config["detector_args"]
regress_loss_args = _config["regress_loss_args"]
mask_loss_args = _config["mask_loss_args"]
del _config


# Main function
def main():
    parser = argparse.ArgumentParser(description="Select additional options for training")
    parser.add_argument('-d', '--device',
                        type=int,
                        default=0,
                        help='CUDA device ID, default is 0')
    parser.add_argument('-l', '--load',
                        action='store_true',
                        help='Load best weights instead of checkpoint weights')
    args = parser.parse_args()
    cuda_id = args.device
    load_best = args.load

    # Set device
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_id}")
    else:
        device = torch.device("cpu")

    print(f"[\033[32mInfo\033[0m] Using device: {device}")

    # Create datasets
    train_dataset = DynamicSpectrumDataset(mode=mode, tchans=tchans, fchans=fchans, df=df, dt=dt, fch1=fch1,
                                           ascending=ascending, drift_min=drift_min, drift_max=drift_max,
                                           drift_min_abs=drift_min_abs, snr_min=snr_min, snr_max=snr_max,
                                           width_min=width_min, width_max=width_max, num_signals=num_signals,
                                           noise_std_min=noise_std_min, noise_std_max=noise_std_max,
                                           noise_mean_min=noise_mean_min, noise_mean_max=noise_mean_max,
                                           noise_type=nosie_type, rfi_enhance=rfi_enhance, use_fil=use_fil,
                                           background_fil=background_fil)

    valid_dataset = DynamicSpectrumDataset(mode=mode, tchans=tchans, fchans=fchans, df=df, dt=dt, fch1=fch1,
                                           ascending=ascending, drift_min=drift_min, drift_max=drift_max,
                                           drift_min_abs=drift_min_abs, snr_min=snr_min, snr_max=snr_max,
                                           width_min=width_min, width_max=width_max, num_signals=num_signals,
                                           noise_std_min=noise_std_min, noise_std_max=noise_std_max,
                                           noise_mean_min=noise_mean_min, noise_mean_max=noise_mean_max,
                                           noise_type=nosie_type, rfi_enhance=rfi_enhance, use_fil=use_fil,
                                           background_fil=background_fil)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Loss function and optimizer
    if mode == "detection":
        model = MSWNet(**dwtnet_args, **detector_args)
        # model = UNet(**unet_args)
        criterion = DetectionCombinedLoss(**regress_loss_args)
    else:  # "mask" as default
        model = MSWNet(**dwtnet_args)
        # model = UNet(**unet_args)
        criterion = MaskCombinedLoss(device, **mask_loss_args)

    summary(model, input_size=(1, 1, tchans, fchans))

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training log files
    step_log_file = Path(checkpoint_dir) / "training_log.csv"
    epoch_log_file = Path(checkpoint_dir) / "epoch_log.csv"
    best_weights_file = Path(checkpoint_dir) / "best_model.pth"

    # Initialize step log
    if os.path.exists(step_log_file):
        # Append to existing step log
        pass
    else:
        # Create new step log with header
        with open(step_log_file, 'w') as f:
            if mode == 'mask':
                f.write(
                    "epoch,global_step,total_loss,spectrum_loss,ssim_loss,rfi_loss,detection_loss,alpha,beta,gamma,delta\n")
            elif mode == 'detection':
                f.write(
                    "epoch,global_step,total_loss,detection_loss,denoise_loss,loc_loss,class_loss,conf_loss,n_matched,lambda_denoise\n")

    # Initialize epoch log
    if os.path.exists(epoch_log_file):
        # Load existing epoch log
        epoch_log = pd.read_csv(epoch_log_file).to_dict('records')
    else:
        epoch_log = []
        with open(epoch_log_file, 'w') as f:
            f.write("epoch,train_loss,valid_loss,epoch_time\n")

    # Determine the best validation loss from epoch log
    best_valid_loss = float('inf')
    start_epoch = 0
    resume_from = None
    checkpoint = None
    mismatched = False

    # Check for checkpoint to load from
    if load_best:
        if best_weights_file.exists():
            resume_from = best_weights_file
            print("[\033[32mInfo\033[0m] Loading best model for resume.")
        else:
            print("[\033[33mWarn\033[0m] Best model not found, starting from scratch.")
    else:
        checkpoint_files = list(Path(checkpoint_dir).glob("model_epoch_*.pth"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
            if latest_checkpoint.exists():
                resume_from = latest_checkpoint

    if resume_from:
        print(f"[\033[32mInfo\033[0m] Loading model from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        if mode == "detection":
            try:
                model.load_state_dict(state_dict, strict=not mismatch_load or force_reconstruct)
            except RuntimeError:
                if force_reconstruct:
                    print(
                        "[\033[33mWarn\033[0m] Detector state dict mismatch detected. Loading backbone weights and reinitializing detector.")
                    filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('detector.')}
                    model.load_state_dict(filtered_dict, strict=False)
                    model.detector = detector(**detector_args)
                    mismatched = True
                else:
                    mismatched = safe_load_state_dict(model, state_dict) if mismatch_load else model.load_state_dict(
                        state_dict, strict=True)

        else:
            try:
                model.load_state_dict(state_dict, strict=not mismatch_load)
            except RuntimeError:
                if mismatch_load:
                    mismatched = safe_load_state_dict(model, state_dict)
                else:
                    raise
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        # Load criterion state for mask mode
        if mode == 'mask':
            criterion.step = checkpoint['criterion_step']
            criterion.mse_moving_avg = checkpoint['mse_moving_avg']
        print(f"[\033[32mInfo\033[0m] Resumed at epoch {start_epoch}")
    else:
        print("[\033[32mInfo\033[0m] Starting training from scratch.")

    # Freeze backbone if enabled
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith('detector.'):
                param.requires_grad = False
        print("[\033[32mInfo\033[0m] Backbone frozen, training only detector head")

    # Moving to device
    model = model.to(device)
    criterion = criterion.to(device)

    # initialize optimizer and scheduler
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True,
    #                                                  min_lr=1e-9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    if resume_from:
        if 'optimizer_state_dict' not in checkpoint:
            print("[\033[33mWarn\033[0m] No optimizer state in checkpoint, using fresh optimizer.")
        elif mismatched:
            load_optimizer_selectively(optimizer, checkpoint['optimizer_state_dict'], device=device)
        else:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("[\033[32mInfo\033[0m] Optimizer state loaded.")

    # Handle force_save_best
    if resume_from and force_save_best:
        print("[\033[32mInfo\033[0m] Forcing best model save with reset validation loss.")
        best_valid_loss = float('inf')  # Reset the best validation loss
    elif epoch_log:
        valid_epochs = [log for log in epoch_log if pd.notna(log['valid_loss'])]
        if valid_epochs:
            best_log = min(valid_epochs, key=lambda x: x['valid_loss'])
            best_valid_loss = best_log['valid_loss']

    # Start training
    train_model(model=model, train_dataloader=train_dataloader, valid_dataloader=valid_dataloader, criterion=criterion,
                optimizer=optimizer, scheduler=scheduler, device=device, mode=mode, num_epochs=num_epochs,
                steps_per_epoch=steps_per_epoch, valid_interval=valid_interval, valid_steps=valid_steps,
                checkpoint_dir=checkpoint_dir, log_interval=log_interval, start_epoch=start_epoch,
                best_valid_loss=best_valid_loss, epoch_log=epoch_log)


if __name__ == "__main__":
    main()
