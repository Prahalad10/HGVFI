"""
train.py
--------
Training and validation loops for HintGuidedVFI.

Public API
----------
- train_epoch       : One training epoch over a DataLoader
- validate          : Evaluate PSNR/SSIM on raw image paths
- run_training      : Full training loop with checkpointing
- train_epoch_v2    : Variant that supports both hint and no-hint models
- validate_v2       : Validate a hint or no-hint model
- run_training_v2   : Full training loop for the ablation study
"""

import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from hgvfi.data import (
    Vimeo90KDataset,
    compress_frame_h264,
    downsample_hint,
    numpy_to_tensor,
    tensor_to_numpy,
)
from hgvfi.metrics import compute_metrics
from hgvfi.models import CharbonnierLoss


# ---------------------------------------------------------------------------
# Single-epoch helpers
# ---------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    Run one training epoch.

    Parameters
    ----------
    model : nn.Module
    dataloader : DataLoader
    optimizer : torch.optim.Optimizer
    loss_fn : nn.Module
    device : torch.device

    Returns
    -------
    avg_loss : float
    epoch_time : float  (seconds)
    """
    model.train()
    total_loss, n = 0, 0
    epoch_start = time.time()

    loop = tqdm(
        dataloader,
        desc="Training",
        leave=False,
        dynamic_ncols=True
    )

    for i, batch in enumerate(loop):
        batch_start = time.time()

        f0   = batch["frame0"].to(device)
        f2   = batch["frame2"].to(device)
        hint = batch["hint"].to(device)
        gt   = batch["gt"].to(device)

        out  = model(f0, f2, hint)
        loss = loss_fn(out, gt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_time = time.time() - batch_start

        total_loss += loss.item()
        n += 1

        # Update tqdm bar
        loop.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{(total_loss/n):.4f}",
            "batch_time": f"{batch_time:.2f}s"
        })

    epoch_time = time.time() - epoch_start
    return total_loss / n, epoch_time


def validate(model, val_paths, device, num_samples=100):
    """
    Evaluate PSNR and SSIM on raw image paths (no DataLoader).

    Each triplet is loaded, H.264-compressed, and passed through the model.
    Metrics are computed against the uncompressed ground-truth middle frame.

    Parameters
    ----------
    model : nn.Module
    val_paths : list of tuple
    device : torch.device
    num_samples : int

    Returns
    -------
    mean_psnr : float
    mean_ssim : float
    val_time : float  (seconds)
    """
    model.eval()
    psnr_list, ssim_list = [], []
    val_start = time.time()
    
    with torch.no_grad():
        for i, (im1p, im2p, im3p) in enumerate(val_paths[:num_samples]):
            f0  = np.array(Image.open(im1p)).astype(np.float32) / 255.0
            fgt = np.array(Image.open(im2p)).astype(np.float32) / 255.0
            f2  = np.array(Image.open(im3p)).astype(np.float32) / 255.0
            f0c = compress_frame_h264(f0,  crf=23)
            f2c = compress_frame_h264(f2,  crf=23)
            fgc = compress_frame_h264(fgt, crf=23)
            h   = downsample_hint(fgc, factor=4)

            out = model(numpy_to_tensor(f0c).to(device),
                        numpy_to_tensor(f2c).to(device),
                        numpy_to_tensor(h).to(device))
            psnr, ssim = compute_metrics(fgt, tensor_to_numpy(out))
            psnr_list.append(psnr)
            ssim_list.append(ssim)
    
    val_time = time.time() - val_start
    return np.mean(psnr_list), np.mean(ssim_list), val_time


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def run_training(
    train_paths,
    val_paths,
    model,
    device,
    project_root=".",
    model_path="hgvfi_model.pt",
    num_epochs=30,
    batch_size=4,
    lr=2e-4,
    limit=None,
):
    """
    Full training loop with cosine LR schedule and periodic checkpointing.

    Checkpoints are saved every 5 epochs to ``{project_root}/hgvfi_epoch_N.pt``
    and the final model to ``model_path``.

    Parameters
    ----------
    train_paths : list of tuple
    val_paths : list of tuple
    model : nn.Module
    device : torch.device
    project_root : str
    model_path : str
    num_epochs : int
    batch_size : int
    lr : float
    limit : int, optional
        Truncate both splits to this many samples (useful for quick runs).

    Returns
    -------
    history : dict
        Keys: ``train_loss``, ``val_psnr``, ``val_ssim``, ``epoch_times``,
        ``val_times``, ``total_time``.
    """
    training_start = time.time()
    
    if limit:
        train_paths = train_paths[:limit]
        val_paths   = val_paths[:limit]

    train_ds = Vimeo90KDataset(train_paths, crop_size=256, use_augmentation=True)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn   = CharbonnierLoss(eps=1e-6)

    history = {
        "train_loss": [], 
        "val_psnr": [], 
        "val_ssim": [],
        "epoch_times": [],
        "val_times": []
    }

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_time = train_epoch(model, loader, optimizer, loss_fn, device)
        scheduler.step()
        
        val_psnr, val_ssim, val_time = validate(
            model, val_paths, device, num_samples=min(200, len(val_paths))
        )
        
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        history["epoch_times"].append(train_time)
        history["val_times"].append(val_time)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"loss {train_loss:.4f} | "
            f"PSNR {val_psnr:.2f} | "
            f"SSIM {val_ssim:.4f} | "
            f"time {int(epoch_time)}s"
        )

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{project_root}/hgvfi_epoch_{epoch}.pt")

    total_training_time = time.time() - training_start
    
    torch.save(model.state_dict(), model_path)
    history["total_time"] = total_training_time

    print(f"\nDone in {int(total_training_time)}s | Final PSNR {history['val_psnr'][-1]:.2f}")

    return history


# ---------------------------------------------------------------------------
# Ablation variants (hint vs. no-hint)
# ---------------------------------------------------------------------------

def train_epoch_v2(model, dataloader, optimizer, loss_fn, device, use_hint=True):
    """
    Training epoch that supports both hint-conditioned and unconditional models.

    Parameters
    ----------
    use_hint : bool
        If True, pass ``batch["hint"]`` to the model.
    """
    model.train()
    total_loss, n = 0, 0
    for batch in dataloader:
        f0 = batch["frame0"].to(device)
        f2 = batch["frame2"].to(device)
        gt = batch["gt"].to(device)

        out  = model(f0, f2, batch["hint"].to(device)) if use_hint else model(f0, f2)
        loss = loss_fn(out, gt)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1
    return total_loss / n


def validate_v2(model, val_paths, device, num_samples=200, use_hint=True):
    """
    Validation that supports both hint-conditioned and unconditional models.

    Parameters
    ----------
    use_hint : bool
        If True, compress the GT frame and derive a hint for the model.
    """
    model.eval()
    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for im1p, im2p, im3p in val_paths[:num_samples]:
            f0  = np.array(Image.open(im1p)).astype(np.float32) / 255.0
            fgt = np.array(Image.open(im2p)).astype(np.float32) / 255.0
            f2  = np.array(Image.open(im3p)).astype(np.float32) / 255.0
            f0c = compress_frame_h264(f0,  crf=23)
            f2c = compress_frame_h264(f2,  crf=23)

            if use_hint:
                fgc = compress_frame_h264(fgt, crf=23)
                h   = downsample_hint(fgc, factor=4)
                out = model(numpy_to_tensor(f0c).to(device),
                            numpy_to_tensor(f2c).to(device),
                            numpy_to_tensor(h).to(device))
            else:
                out = model(numpy_to_tensor(f0c).to(device),
                            numpy_to_tensor(f2c).to(device))

            psnr, ssim = compute_metrics(fgt, tensor_to_numpy(out))
            psnr_list.append(psnr)
            ssim_list.append(ssim)

    return np.mean(psnr_list), np.mean(ssim_list)


def run_training_v2(train_paths, val_paths, model, device,
                    num_epochs=8, batch_size=4, lr=2e-4, limit=None, use_hint=True):
    """
    Full training loop for ablation experiments (hint vs. no-hint).

    Parameters
    ----------
    use_hint : bool
        Passed through to ``train_epoch_v2`` and ``validate_v2``.

    Returns
    -------
    history : dict
        Keys: ``train_loss``, ``val_psnr``, ``val_ssim``.
    """
    if limit:
        train_paths = train_paths[:limit]
        val_paths   = val_paths[:limit]

    train_ds = Vimeo90KDataset(train_paths, crop_size=256, use_augmentation=True)
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    loss_fn   = CharbonnierLoss(eps=1e-6)

    history = {"train_loss": [], "val_psnr": [], "val_ssim": []}

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch_v2(model, loader, optimizer, loss_fn, device, use_hint)
        scheduler.step()
        val_psnr, val_ssim = validate_v2(
            model, val_paths, device,
            num_samples=min(200, len(val_paths)), use_hint=use_hint
        )
        history["train_loss"].append(train_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"loss {train_loss:.4f} | "
            f"PSNR {val_psnr:.2f} | "
            f"SSIM {val_ssim:.4f}"
        )

    return history
