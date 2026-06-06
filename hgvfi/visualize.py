"""
visualize.py
------------
Plotting helpers for dataset inspection, training diagnostics, and result analysis.

All functions return the matplotlib Figure (and secondary figures where applicable)
so callers can save or display them as needed.

Public API
----------
- plot_dataset_samples         : Grid of random Vimeo-90K triplets
- plot_crf_comparison          : Side-by-side compressed frames + residuals across CRF values
- plot_crf_psnr_curve          : PSNR vs CRF line chart
- plot_hint_pipeline           : Four-stage hint frame pipeline
- plot_augmentation_pipeline   : Data augmentation visualisation for one sample
- plot_training_history        : Loss / PSNR / SSIM curves (single model)
- plot_ablation_history        : Loss / PSNR / SSIM curves for hint vs. no-hint
- plot_param_distribution      : Bar chart of parameter counts per submodule
- plot_benchmark_bars          : PSNR and SSIM bar charts for all methods
- plot_psnr_boxplot            : Box-and-whisker PSNR distribution
- plot_visual_comparison       : Multi-row qualitative comparison grid
- plot_zoomed_detail           : Cropped detail + GT with crop region overlay
- plot_residual_heatmaps       : Prediction vs GT residual heatmaps
- plot_scatter_hgvfi_vs_bicubic: Per-sample PSNR scatter (HGVFI vs BicubicHint)
- plot_crf_robustness          : PSNR vs CRF robustness curve
"""

import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from hgvfi.data import compress_frame_h264, downsample_hint, upsample_hint_bicubic
from hgvfi.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

_COLORS = {
    "blue":   "#2563eb",
    "green":  "#10b981",
    "purple": "#8b5cf6",
    "red":    "#dc2626",
    "slate":  "#94a3b8",
    "dark":   "#64748b",
}


# ---------------------------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------------------------

def plot_dataset_samples(test_paths, n=6):
    """
    Display a random selection of triplets from the test split.

    Parameters
    ----------
    test_paths : list of tuple
    n : int
        Number of rows (triplets) to show.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    random_samples = random.sample(test_paths, n)

    fig, axes = plt.subplots(n, 3, figsize=(12, 18))

    for row, (im1p, im2p, im3p) in enumerate(random_samples):
        for col, (path, title) in enumerate([
            (im1p, "Frame 0 (ref)"),
            (im2p, "Frame 1 (GT)"),
            (im3p, "Frame 2 (ref)")
        ]):
            img = np.array(Image.open(path))
            axes[row, col].imshow(img)
            axes[row, col].set_title(title if row == 0 else "")
            axes[row, col].axis("off")

    fig.suptitle("Random Vimeo90K Triplets", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Compression analysis
# ---------------------------------------------------------------------------

def plot_crf_comparison(test_paths, crf_values):
    """
    Side-by-side compressed frames and their residuals across CRF values.

    Parameters
    ----------
    test_paths : list of tuple
    crf_values : list of int

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, len(crf_values), figsize=(18, 6))

    _, im2p, _ = random.choice(test_paths)
    ref_frame = np.array(Image.open(im2p)).astype(np.float32) / 255.0

    for col, crf in enumerate(crf_values):
        comp = compress_frame_h264(ref_frame, crf=crf)
        psnr, _ = compute_metrics(ref_frame, comp)

        axes[0, col].imshow(comp.clip(0, 1))
        axes[0, col].set_title(f"CRF={crf}\n{psnr:.1f} dB")
        axes[0, col].axis("off")

        residual = np.abs(ref_frame - comp) * 10
        axes[1, col].imshow(residual.clip(0, 1), cmap="hot")
        axes[1, col].set_title("Residual ×10")
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Compressed", fontsize=10)
    axes[1, 0].set_ylabel("Residual (hot)", fontsize=10)

    fig.suptitle("H.264 Compression at Different Quality Levels (Random Frame)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_crf_psnr_curve(crf_values, psnr_by_crf):
    """
    PSNR vs CRF line chart with a vertical marker at the paper's default CRF.

    Parameters
    ----------
    crf_values : list of int
    psnr_by_crf : dict
        Mapping from CRF value to PSNR in dB.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(crf_values, [psnr_by_crf[c] for c in crf_values], "o-",
            color=_COLORS["blue"], lw=2)
    ax.axvline(23, color=_COLORS["red"], ls="--", lw=1.5, label="Paper default (CRF=23)")
    ax.set_xlabel("CRF (lower = better quality)")
    ax.set_ylabel("PSNR vs Original (dB)")
    ax.set_title("Quality vs Compression: PSNR–CRF Tradeoff")
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hint pipeline
# ---------------------------------------------------------------------------

def plot_hint_pipeline(test_paths):
    """
    Visualise the four-stage hint frame pipeline for a single test triplet.

    Parameters
    ----------
    test_paths : list of tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    gt_frame    = np.array(Image.open(test_paths[0][1])).astype(np.float32) / 255.0
    compressed  = compress_frame_h264(gt_frame, crf=23)
    hint        = downsample_hint(compressed, factor=4)
    hint_up     = upsample_hint_bicubic(hint, scale=4)

    psnr_comp, ssim_comp   = compute_metrics(gt_frame, compressed)
    psnr_up,   ssim_up     = compute_metrics(gt_frame, hint_up)
    print(f"Compressed  → PSNR {psnr_comp:.2f} dB, SSIM {ssim_comp:.4f}")
    print(f"BicubicHint → PSNR {psnr_up:.2f} dB,  SSIM {ssim_up:.4f}")

    stages = [
        (gt_frame,   f"Original\n{gt_frame.shape[1]}×{gt_frame.shape[0]}"),
        (compressed, f"H.264 Compressed (CRF=23)\n{psnr_comp:.2f} dB"),
        (hint,       f"Hint (4× down)\n{hint.shape[1]}×{hint.shape[0]}"),
        (hint_up,    f"BicubicHint (4× up)\n{psnr_up:.2f} dB"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (img, title) in zip(axes, stages):
        ax.imshow(img.clip(0, 1))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle("Hint Frame Pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def plot_augmentation_pipeline(test_paths):
    """
    Visualise the data augmentation pipeline for one random test triplet.

    Parameters
    ----------
    test_paths : list of tuple

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    im1p, im2p, im3p = random.choice(test_paths)

    raw_f0  = np.array(Image.open(im1p)).astype(np.float32) / 255.0
    raw_gt  = np.array(Image.open(im2p)).astype(np.float32) / 255.0
    raw_f2  = np.array(Image.open(im3p)).astype(np.float32) / 255.0

    comp_f0 = compress_frame_h264(raw_f0, crf=23)
    comp_gt = compress_frame_h264(raw_gt, crf=23)
    comp_f2 = compress_frame_h264(raw_f2, crf=23)
    hint_sm = downsample_hint(comp_gt, factor=4)

    y, x, s = 50, 100, 256
    crop_f0 = comp_f0[y:y+s, x:x+s]
    crop_gt = comp_gt[y:y+s, x:x+s]

    flip_f0 = crop_f0[:, ::-1]

    stages = [
        (raw_gt,   "Raw GT (im2)"),
        (comp_gt,  "H.264 Compressed (CRF=23)"),
        (hint_sm,  f"Hint ({hint_sm.shape[1]}×{hint_sm.shape[0]})"),
        (crop_f0,  f"Cropped f0 ({s}×{s})"),
        (crop_gt,  f"Cropped GT ({s}×{s})"),
        (flip_f0,  "Cropped + H-flip"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    for i, (img, title) in enumerate(stages):
        r, c = divmod(i, 3)
        axes[r, c].imshow(img.clip(0, 1))
        axes[r, c].set_title(title, fontsize=9)
        axes[r, c].axis("off")

    fig.suptitle("Data Augmentation Pipeline (Random Sample)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_training_history(history):
    """
    Plot training loss, validation PSNR and SSIM for a single run.

    Parameters
    ----------
    history : dict
        Output of ``run_training``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    epochs     = list(range(1, len(history["train_loss"]) + 1))
    train_loss = history["train_loss"]
    val_psnr   = history["val_psnr"]
    val_ssim   = history["val_ssim"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(epochs, train_loss, "o-", color=_COLORS["blue"], lw=2, ms=6)
    for e, v in zip(epochs, train_loss):
        ax.annotate(f"{v:.4f}", (e, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Charbonnier Loss (log)")
    ax.set_title("Training Loss")
    ax.set_xticks(epochs)

    ax = axes[1]
    ax.plot(epochs, val_psnr, "o-", color=_COLORS["green"], lw=2, ms=6)
    ax.axhline(38.69, ls="--", color=_COLORS["red"], lw=1.5, label="Paper target 38.69 dB")
    for e, v in zip(epochs, val_psnr):
        ax.annotate(f"{v:.2f}", (e, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Validation PSNR")
    ax.set_xticks(epochs)
    ax.legend(fontsize=9)

    ax = axes[2]
    ax.plot(epochs, val_ssim, "o-", color=_COLORS["purple"], lw=2, ms=6)
    for e, v in zip(epochs, val_ssim):
        ax.annotate(f"{v:.4f}", (e, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SSIM")
    ax.set_title("Validation SSIM")
    ax.set_xticks(epochs)

    fig.suptitle("HGVFI Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_ablation_history(history, history_nohint):
    """
    Overlay training curves for the hint vs. no-hint ablation.

    Parameters
    ----------
    history : dict
        Training history for the full HGVFI model.
    history_nohint : dict
        Training history for the no-hint ablation.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"],        "o-", color=_COLORS["blue"], lw=2, label="HGVFI (with hint)")
    ax.plot(epochs, history_nohint["train_loss"], "o-", color=_COLORS["red"],  lw=2, label="No Hint")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Charbonnier Loss (log)")
    ax.set_title("Training Loss"); ax.legend(); ax.set_xticks(epochs)

    ax = axes[1]
    ax.plot(epochs, history["val_psnr"],        "o-", color=_COLORS["blue"], lw=2, label="HGVFI (with hint)")
    ax.plot(epochs, history_nohint["val_psnr"], "o-", color=_COLORS["red"],  lw=2, label="No Hint")
    ax.axhline(38.69, ls="--", color="#6b7280", lw=1.5, label="Paper target 38.69 dB")
    ax.set_xlabel("Epoch"); ax.set_ylabel("PSNR (dB)")
    ax.set_title("Validation PSNR"); ax.legend(); ax.set_xticks(epochs)

    ax = axes[2]
    ax.plot(epochs, history["val_ssim"],        "o-", color=_COLORS["blue"], lw=2, label="HGVFI (with hint)")
    ax.plot(epochs, history_nohint["val_ssim"], "o-", color=_COLORS["red"],  lw=2, label="No Hint")
    ax.set_xlabel("Epoch"); ax.set_ylabel("SSIM")
    ax.set_title("Validation SSIM"); ax.legend(); ax.set_xticks(epochs)

    fig.suptitle("Ablation: Hint Guidance vs No Hint", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Model analysis
# ---------------------------------------------------------------------------

def plot_param_distribution(model):
    """
    Bar chart of parameter counts per major submodule.

    Parameters
    ----------
    model : HintGuidedVFI

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    labels  = ["HintBranch", "EMAVFIBackbone", "RefineNet"]
    counts  = [count_params(model.hint_branch),
               count_params(model.backbone),
               count_params(model.refine)]
    colors  = [_COLORS["blue"], _COLORS["purple"], _COLORS["green"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, [c/1e6 for c in counts], color=colors, width=0.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{c/1e6:.2f}M", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Parameters (millions)")
    ax.set_title("HGVFI Parameter Distribution")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Benchmark visualisation
# ---------------------------------------------------------------------------

def plot_benchmark_bars(benchmark):
    """
    Side-by-side PSNR and SSIM bar charts comparing all methods.

    Parameters
    ----------
    benchmark : dict
        Output of ``run_benchmark``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    methods    = list(benchmark.keys())
    psnr_means = [benchmark[m]["psnr_mean"] for m in methods]
    psnr_stds  = [benchmark[m]["psnr_std"]  for m in methods]
    ssim_means = [benchmark[m]["ssim_mean"] for m in methods]
    ssim_stds  = [benchmark[m]["ssim_std"]  for m in methods]
    colors     = [_COLORS["slate"], _COLORS["dark"], _COLORS["blue"], _COLORS["green"]]
    colors     = (colors * 4)[:len(methods)]  # extend if ablation methods present

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bars = ax.bar(methods, psnr_means, yerr=psnr_stds, color=colors,
                  capsize=4, width=0.55, error_kw={"lw": 1.5})
    for bar, v in zip(bars, psnr_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Method Comparison — PSNR")
    ax.set_ylim(bottom=max(0, min(psnr_means) - 3))

    ax = axes[1]
    bars = ax.bar(methods, ssim_means, yerr=ssim_stds, color=colors,
                  capsize=4, width=0.55, error_kw={"lw": 1.5})
    for bar, v in zip(bars, ssim_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("SSIM")
    ax.set_title("Method Comparison — SSIM")
    ax.set_ylim(bottom=max(0, min(ssim_means) - 0.1))

    fig.suptitle("HGVFI vs Baselines", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_psnr_boxplot(benchmark):
    """
    Box-and-whisker plot of per-sample PSNR for every method.

    Parameters
    ----------
    benchmark : dict
        Output of ``run_benchmark``.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
    """
    methods = list(benchmark.keys())
    if not isinstance(benchmark.get("Linear", {}).get("psnr_scores"), list):
        return None

    colors = [_COLORS["slate"], _COLORS["dark"], _COLORS["blue"], _COLORS["green"]]
    colors = (colors * 4)[:len(methods)]

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [benchmark[m]["psnr_scores"] for m in methods]
    bp = ax.boxplot(data, labels=methods, patch_artist=True, notch=True,
                    medianprops={"color": "white", "lw": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Per-Sample PSNR Distribution")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Qualitative comparison
# ---------------------------------------------------------------------------

def plot_visual_comparison(model, test_paths, device, sample_idxs=None):
    """
    Grid of qualitative comparisons across multiple test samples.

    Parameters
    ----------
    model : nn.Module
    test_paths : list of tuple
    device : torch.device
    sample_idxs : list of int, optional
        Indices into ``test_paths`` to show.  Defaults to [0, 150, 500].

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from hgvfi.evaluate import interpolate_all_methods

    if sample_idxs is None:
        sample_idxs = [0, 150, 500]

    fig, axes = plt.subplots(len(sample_idxs), 4, figsize=(18, 4.5 * len(sample_idxs)))

    for row, idx in enumerate(sample_idxs):
        preds, metrics, fgc = interpolate_all_methods(model, *test_paths[idx], device)

        for col, (name, img) in enumerate(preds.items()):
            ax = axes[row, col]
            ax.imshow(img.clip(0, 1))
            title = name
            if name in metrics:
                p, s = metrics[name]
                title += f"\nPSNR {p:.2f} dB  SSIM {s:.4f}"
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        axes[row, 0].set_ylabel(f"Sample {idx}", fontsize=10, rotation=0,
                                labelpad=60, va="center")

    fig.suptitle("Visual Comparison: GT / Linear / BicubicHint / HGVFI",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_zoomed_detail(model, test_paths, device,
                       zoom_idx=0, crop_y=60, crop_x=160, crop_h=100, crop_w=140):
    """
    Cropped detail comparison across all methods plus a GT overview with the crop box.

    Parameters
    ----------
    model : nn.Module
    test_paths : list of tuple
    device : torch.device
    zoom_idx : int
    crop_y, crop_x, crop_h, crop_w : int

    Returns
    -------
    fig : matplotlib.figure.Figure   (detail panel)
    fig2 : matplotlib.figure.Figure  (GT overview)
    """
    import matplotlib.patches as mpatches
    from hgvfi.evaluate import interpolate_all_methods

    preds, metrics, fgc = interpolate_all_methods(model, *test_paths[zoom_idx], device)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (name, img) in zip(axes, preds.items()):
        crop = img.clip(0, 1)[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        ax.imshow(crop)
        title = name
        if name in metrics:
            title += f"\n{metrics[name][0]:.2f} dB"
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.imshow(fgc.clip(0, 1))
    rect = plt.Rectangle((crop_x, crop_y), crop_w, crop_h,
                          linewidth=2, edgecolor=_COLORS["red"], facecolor="none")
    ax2.add_patch(rect)
    ax2.set_title("GT with crop region (red box)", fontsize=10)
    ax2.axis("off")

    fig.suptitle("Zoomed Detail Comparison", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig2.tight_layout()
    return fig, fig2


def plot_residual_heatmaps(model, test_paths, device, idx=0):
    """
    Prediction vs GT residual heatmaps for all non-GT methods.

    Parameters
    ----------
    model : nn.Module
    test_paths : list of tuple
    device : torch.device
    idx : int
        Index into ``test_paths``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from hgvfi.evaluate import interpolate_all_methods

    preds, metrics, fgc = interpolate_all_methods(model, *test_paths[idx], device)
    residual_methods = {k: v for k, v in preds.items() if k != "GT (compressed)"}

    fig, axes = plt.subplots(2, len(residual_methods), figsize=(15, 7))

    for col, (name, pred) in enumerate(residual_methods.items()):
        p, s = metrics[name]

        axes[0, col].imshow(pred.clip(0, 1))
        axes[0, col].set_title(f"{name}\n{p:.2f} dB", fontsize=10)
        axes[0, col].axis("off")

        residual = np.abs(fgc - pred).mean(axis=2)  # collapse to luminance diff
        im = axes[1, col].imshow(residual, cmap="hot", vmin=0, vmax=0.15)
        axes[1, col].set_title("Residual (hot)", fontsize=9)
        axes[1, col].axis("off")
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    fig.suptitle("Prediction vs Ground Truth — Residual Heatmaps",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Robustness analysis
# ---------------------------------------------------------------------------

def plot_scatter_hgvfi_vs_bicubic(model, test_paths, device, n=50):
    """
    Per-sample PSNR scatter plot: HGVFI vs BicubicHint.

    Parameters
    ----------
    model : nn.Module
    test_paths : list of tuple
    device : torch.device
    n : int
        Number of test samples to evaluate.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import torch
    from hgvfi.data import numpy_to_tensor, tensor_to_numpy

    psnr_bichint, psnr_hgvfi = [], []

    model.eval()
    for im1p, im2p, im3p in test_paths[:n]:
        f0c = compress_frame_h264(np.array(Image.open(im1p)).astype(np.float32)/255.0, crf=23)
        f2c = compress_frame_h264(np.array(Image.open(im3p)).astype(np.float32)/255.0, crf=23)
        fgc = compress_frame_h264(np.array(Image.open(im2p)).astype(np.float32)/255.0, crf=23)
        h   = downsample_hint(fgc, factor=4)

        p_bh, _ = compute_metrics(fgc, upsample_hint_bicubic(h, scale=4))
        with torch.no_grad():
            out = model(numpy_to_tensor(f0c).to(device),
                        numpy_to_tensor(f2c).to(device),
                        numpy_to_tensor(h).to(device))
        p_hg, _ = compute_metrics(fgc, tensor_to_numpy(out))

        psnr_bichint.append(p_bh)
        psnr_hgvfi.append(p_hg)

    fig, ax = plt.subplots(figsize=(6, 6))
    lo = min(min(psnr_bichint), min(psnr_hgvfi)) - 1
    hi = max(max(psnr_bichint), max(psnr_hgvfi)) + 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Equal performance")
    ax.scatter(psnr_bichint, psnr_hgvfi, alpha=0.7, color=_COLORS["green"],
               edgecolors="white", s=50)
    ax.set_xlabel("BicubicHint PSNR (dB)")
    ax.set_ylabel("HGVFI PSNR (dB)")
    ax.set_title(f"Per-Sample PSNR: HGVFI vs BicubicHint (n={n})")
    ax.legend()

    wins = sum(h > b for h, b in zip(psnr_hgvfi, psnr_bichint))
    ax.text(0.05, 0.95, f"HGVFI wins: {wins}/{n}",
            transform=ax.transAxes, fontsize=11, color=_COLORS["green"], va="top")
    plt.tight_layout()
    return fig


def plot_crf_robustness(model, test_paths, device, crf_test=None, sample_idx=5):
    """
    PSNR vs CRF robustness curve comparing HGVFI and BicubicHint.

    Parameters
    ----------
    model : nn.Module
    test_paths : list of tuple
    device : torch.device
    crf_test : list of int, optional
        CRF values to test.  Defaults to [10, 18, 23, 30, 40].
    sample_idx : int
        Index of the test triplet to use.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import torch
    from hgvfi.data import numpy_to_tensor, tensor_to_numpy

    if crf_test is None:
        crf_test = [10, 18, 23, 30, 40]

    sample = test_paths[sample_idx]
    im1p, im2p, im3p = sample

    raw_f0  = np.array(Image.open(im1p)).astype(np.float32) / 255.0
    raw_fgt = np.array(Image.open(im2p)).astype(np.float32) / 255.0
    raw_f2  = np.array(Image.open(im3p)).astype(np.float32) / 255.0

    psnr_hgvfi_crf, psnr_bichint_crf = [], []

    model.eval()
    for crf in crf_test:
        f0c = compress_frame_h264(raw_f0,  crf=crf)
        f2c = compress_frame_h264(raw_f2,  crf=crf)
        fgc = compress_frame_h264(raw_fgt, crf=crf)
        h   = downsample_hint(fgc, factor=4)

        p_bh, _ = compute_metrics(fgc, upsample_hint_bicubic(h, scale=4))
        with torch.no_grad():
            out = model(numpy_to_tensor(f0c).to(device),
                        numpy_to_tensor(f2c).to(device),
                        numpy_to_tensor(h).to(device))
        p_hg, _ = compute_metrics(fgc, tensor_to_numpy(out))

        psnr_hgvfi_crf.append(p_hg)
        psnr_bichint_crf.append(p_bh)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(crf_test, psnr_bichint_crf, "o--", color=_COLORS["blue"], lw=2, label="BicubicHint")
    ax.plot(crf_test, psnr_hgvfi_crf,   "o-",  color=_COLORS["green"], lw=2, label="HGVFI")
    ax.axvline(23, color="#6b7280", ls=":", lw=1, label="Training CRF=23")
    ax.fill_between(crf_test,
                    psnr_bichint_crf, psnr_hgvfi_crf,
                    alpha=0.15, color=_COLORS["green"], label="HGVFI gain")
    ax.set_xlabel("CRF (higher = more compression)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("HGVFI vs BicubicHint Across Compression Levels")
    ax.legend()
    plt.tight_layout()
    return fig
