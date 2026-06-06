"""
scripts/evaluate_hgvfi.py
-------------------------
Entry-point script: run the full benchmark and ablation study for HGVFI.

Usage
-----
    python scripts/evaluate_hgvfi.py [--samples 400] [--ablation-limit 100]

Loads a trained checkpoint from MODEL_PATH, runs the four-method benchmark
(Linear, Bicubic, BicubicHint, HGVFI), trains a brief no-hint ablation,
and prints a summary table.
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image

from hgvfi.data import (
    compress_frame_h264,
    downsample_hint,
    load_vimeo_triplet,
    numpy_to_tensor,
    tensor_to_numpy,
)
from hgvfi.evaluate import run_benchmark
from hgvfi.metrics import compute_metrics
from hgvfi.models import EMAVFINoHint, HintGuidedVFI
from hgvfi.train import run_training_v2
from hgvfi.visualize import (
    plot_ablation_history,
    plot_benchmark_bars,
    plot_crf_robustness,
    plot_psnr_boxplot,
    plot_residual_heatmaps,
    plot_scatter_hgvfi_vs_bicubic,
    plot_visual_comparison,
    plot_zoomed_detail,
)

DATASET_ROOT = "vimeo_triplet/"
MODEL_PATH   = DATASET_ROOT + "hgvfi_model.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HintGuidedVFI")
    parser.add_argument("--samples",        type=int, default=400, help="Benchmark sample count")
    parser.add_argument("--ablation-limit", type=int, default=100, help="Ablation dataset limit")
    parser.add_argument("--dataset",        type=str, default=DATASET_ROOT)
    parser.add_argument("--model-path",     type=str, default=MODEL_PATH)
    return parser.parse_args()


def print_summary(benchmark):
    print(f"\n{'Method':<15} {'PSNR (dB)':<25} {'SSIM'}")
    print("-" * 60)
    for method, m in benchmark.items():
        print(f"{method:<15} {m['psnr_mean']:.2f} ± {m['psnr_std']:.2f}{'':8}"
              f" {m['ssim_mean']:.4f} ± {m['ssim_std']:.4f}")


def main():
    args = parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    train_list  = os.path.join(args.dataset, "tri_trainlist.txt")
    test_list   = os.path.join(args.dataset, "tri_testlist.txt")
    train_paths = load_vimeo_triplet(args.dataset, train_list)
    test_paths  = load_vimeo_triplet(args.dataset, test_list)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    model = HintGuidedVFI().to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded weights from {args.model_path}")
    else:
        print("No checkpoint found, using random weights")

    # -----------------------------------------------------------------------
    # Main benchmark
    # -----------------------------------------------------------------------
    print(f"\nRunning benchmark on {args.samples} samples …")
    benchmark = run_benchmark(model, test_paths, device, num_samples=args.samples)
    print_summary(benchmark)

    # -----------------------------------------------------------------------
    # Ablation: no-hint model
    # -----------------------------------------------------------------------
    print(f"\nTraining no-hint ablation (limit={args.ablation_limit}) …")
    model_nohint = EMAVFINoHint().to(device)
    history_nohint = run_training_v2(
        train_paths, test_paths, model_nohint, device,
        num_epochs=12, limit=args.ablation_limit, use_hint=False
    )

    # Add no-hint scores to benchmark dict
    nohint_psnr, nohint_ssim = [], []
    model_nohint.eval()
    with torch.no_grad():
        for im1p, im2p, im3p in test_paths[:100]:
            f0  = np.array(Image.open(im1p)).astype(np.float32) / 255.0
            fgt = np.array(Image.open(im2p)).astype(np.float32) / 255.0
            f2  = np.array(Image.open(im3p)).astype(np.float32) / 255.0
            f0c = compress_frame_h264(f0, crf=23)
            f2c = compress_frame_h264(f2, crf=23)
            out = model_nohint(numpy_to_tensor(f0c).to(device),
                               numpy_to_tensor(f2c).to(device))
            p, s = compute_metrics(fgt, tensor_to_numpy(out))
            nohint_psnr.append(p)
            nohint_ssim.append(s)

    benchmark["NoHint"] = {
        "psnr_mean":   float(np.mean(nohint_psnr)),
        "psnr_std":    float(np.std(nohint_psnr)),
        "ssim_mean":   float(np.mean(nohint_ssim)),
        "ssim_std":    float(np.std(nohint_ssim)),
        "psnr_scores": nohint_psnr,
        "ssim_scores": nohint_ssim,
    }

    print("\nFull ablation table:")
    print_summary(benchmark)

    # -----------------------------------------------------------------------
    # Save plots
    # -----------------------------------------------------------------------
    plot_benchmark_bars(benchmark).savefig("benchmark_bars.png", bbox_inches="tight")
    plot_psnr_boxplot(benchmark).savefig("psnr_boxplot.png",  bbox_inches="tight")
    plot_scatter_hgvfi_vs_bicubic(model, test_paths, device).savefig(
        "scatter_psnr.png", bbox_inches="tight")
    plot_crf_robustness(model, test_paths, device).savefig(
        "crf_robustness.png", bbox_inches="tight")
    plot_residual_heatmaps(model, test_paths, device).savefig(
        "residual_heatmaps.png", bbox_inches="tight")
    plot_visual_comparison(model, test_paths, device).savefig(
        "visual_comparison.png", bbox_inches="tight")

    print("\nPlots saved.")


if __name__ == "__main__":
    main()
