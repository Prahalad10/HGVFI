"""
metrics.py
----------
Evaluation metrics for video frame interpolation.

Public API
----------
- compute_metrics : PSNR and SSIM between a ground-truth and predicted frame
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_metrics(gt, pred):
    """
    Compute PSNR and SSIM between two float32 images.

    Both inputs are clipped to [0, 1] and converted to uint8 before scoring,
    matching the evaluation protocol used in the HGVFI paper.

    Parameters
    ----------
    gt : np.ndarray, shape (H, W, 3), dtype float32, range [0, 1]
        Ground-truth frame.
    pred : np.ndarray, shape (H, W, 3), dtype float32, range [0, 1]
        Predicted frame.

    Returns
    -------
    psnr : float
        Peak Signal-to-Noise Ratio in dB.
    ssim : float
        Structural Similarity Index in [0, 1].
    """
    gt   = np.clip(gt,   0, 1)
    pred = np.clip(pred, 0, 1)
    gt_u8   = (gt   * 255).astype(np.uint8)
    pred_u8 = (pred * 255).astype(np.uint8)
    psnr = peak_signal_noise_ratio(gt_u8, pred_u8, data_range=255)
    ssim = structural_similarity(gt_u8, pred_u8, data_range=255, channel_axis=2)
    return psnr, ssim
