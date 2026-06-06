"""
evaluate.py
-----------
Benchmark and qualitative evaluation utilities for HGVFI.

Public API
----------
- SimpleLinearInterpolation  : Baseline — average of two reference frames
- BicubicInterpolation       : Baseline — downsample-upsample average
- BicubicHintInterpolation   : Baseline — bicubic upscale of the hint frame
- run_benchmark              : Compare all methods on a test split
- interpolate_all_methods    : Run all methods on a single triplet
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hgvfi.data import (
    compress_frame_h264,
    downsample_hint,
    numpy_to_tensor,
    tensor_to_numpy,
    upsample_hint_bicubic,
)
from hgvfi.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Baseline methods
# ---------------------------------------------------------------------------

class SimpleLinearInterpolation:
    """Baseline: pixel-wise average of the two reference frames."""
    def __call__(self, f0, f2):
        return (f0 + f2) / 2.0


class BicubicInterpolation:
    """Baseline: downsample the averaged frame and bicubic-upsample back."""
    def __call__(self, f0, f2):
        avg = (f0 + f2) / 2.0
        t   = torch.from_numpy(avg.transpose(2, 0, 1)).unsqueeze(0).float()
        h, w = t.shape[2:]
        down = F.interpolate(t, scale_factor=0.5, mode="bicubic", align_corners=False)
        up   = F.interpolate(down, size=(h, w),   mode="bicubic", align_corners=False)
        return up.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)


class BicubicHintInterpolation:
    """Baseline: bicubic upscale of the downsampled hint frame (4× → full res)."""
    def __call__(self, hint):
        return upsample_hint_bicubic(hint, scale=4)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(model, dataset, device, num_samples=200):
    """
    Evaluate PSNR and SSIM for all baselines and HGVFI on a test split.

    Parameters
    ----------
    model : nn.Module
        A trained HintGuidedVFI instance.
    dataset : list of tuple
        List of ``(im1_path, im2_path, im3_path)`` triplets.
    device : torch.device
    num_samples : int
        Number of triplets to evaluate.

    Returns
    -------
    summary : dict
        Keyed by method name (``"Linear"``, ``"Bicubic"``, ``"BicubicHint"``,
        ``"HGVFI"``).  Each value is a dict with:
        ``psnr_mean``, ``psnr_std``, ``ssim_mean``, ``ssim_std``,
        ``psnr_scores``, ``ssim_scores``.
    """
    linear_m  = SimpleLinearInterpolation()
    bicubic_m = BicubicInterpolation()
    bichint_m = BicubicHintInterpolation()

    results = {k: {"psnr": [], "ssim": []} for k in ["Linear", "Bicubic", "BicubicHint", "HGVFI"]}
    model.eval()

    for idx, (im1p, im2p, im3p) in enumerate(dataset[:num_samples]):
        f0  = np.array(Image.open(im1p)).astype(np.float32) / 255.0
        fgt = np.array(Image.open(im2p)).astype(np.float32) / 255.0
        f2  = np.array(Image.open(im3p)).astype(np.float32) / 255.0
        f0c = compress_frame_h264(f0,  crf=23)
        f2c = compress_frame_h264(f2,  crf=23)
        fgc = compress_frame_h264(fgt, crf=23)
        h   = downsample_hint(fgc, factor=4)

        preds = {
            "Linear":      linear_m(f0c, f2c),
            "Bicubic":     bicubic_m(f0c, f2c),
            "BicubicHint": bichint_m(h),
        }
        with torch.no_grad():
            out = model(numpy_to_tensor(f0c).to(device),
                        numpy_to_tensor(f2c).to(device),
                        numpy_to_tensor(h).to(device))
        preds["HGVFI"] = tensor_to_numpy(out)

        for name, pred in preds.items():
            psnr, ssim = compute_metrics(fgt, pred)
            results[name]["psnr"].append(psnr)
            results[name]["ssim"].append(ssim)

        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{num_samples}")

    summary = {}
    for name, vals in results.items():
        summary[name] = {
            "psnr_mean": float(np.mean(vals["psnr"])),
            "psnr_std":  float(np.std(vals["psnr"])),
            "ssim_mean": float(np.mean(vals["ssim"])),
            "ssim_std":  float(np.std(vals["ssim"])),
            "psnr_scores": vals["psnr"],
            "ssim_scores": vals["ssim"],
        }
    return summary


# ---------------------------------------------------------------------------
# Per-sample qualitative helper
# ---------------------------------------------------------------------------

def interpolate_all_methods(model, im1p, im2p, im3p, device):
    """
    Run every method on a single triplet and return predictions + metrics.

    Parameters
    ----------
    model : nn.Module
    im1p, im2p, im3p : str
        Paths to the three frames.
    device : torch.device

    Returns
    -------
    preds : dict
        Method name → np.ndarray (H, W, 3), float32.
        Keys: ``"GT (compressed)"``, ``"Linear"``, ``"BicubicHint"``, ``"HGVFI"``.
    metrics : dict
        Method name (excluding GT) → ``(psnr, ssim)``.
    fgc : np.ndarray
        H.264-compressed GT frame used as the evaluation reference.
    """
    f0  = np.array(Image.open(im1p)).astype(np.float32) / 255.0
    fgt = np.array(Image.open(im2p)).astype(np.float32) / 255.0
    f2  = np.array(Image.open(im3p)).astype(np.float32) / 255.0
    f0c = compress_frame_h264(f0,  crf=23)
    f2c = compress_frame_h264(f2,  crf=23)
    fgc = compress_frame_h264(fgt, crf=23)
    h   = downsample_hint(fgc, factor=4)

    preds = {
        "GT (compressed)": fgc,
        "Linear":          (f0c + f2c) / 2.0,
        "BicubicHint":     upsample_hint_bicubic(h, scale=4),
    }
    model.eval()
    with torch.no_grad():
        out = model(numpy_to_tensor(f0c).to(device),
                    numpy_to_tensor(f2c).to(device),
                    numpy_to_tensor(h).to(device))
    preds["HGVFI"] = tensor_to_numpy(out)

    metrics = {}
    for name, pred in preds.items():
        if name != "GT (compressed)":
            p, s = compute_metrics(fgc, pred)
            metrics[name] = (p, s)
    return preds, metrics, fgc
