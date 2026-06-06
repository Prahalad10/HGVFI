"""
scripts/explore_dataset.py
--------------------------
Quick dataset statistics and visualisation for the Vimeo-90K Triplet dataset.

Usage
-----
    python scripts/explore_dataset.py

Prints split sizes, frame properties, and storage estimates, then saves
sample visualisations and the hint pipeline figure.
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from hgvfi.data import compress_frame_h264, load_vimeo_triplet
from hgvfi.metrics import compute_metrics
from hgvfi.visualize import (
    plot_augmentation_pipeline,
    plot_crf_comparison,
    plot_crf_psnr_curve,
    plot_dataset_samples,
    plot_hint_pipeline,
)

DATASET_ROOT = "vimeo_triplet/"
TRAIN_LIST   = DATASET_ROOT + "tri_trainlist.txt"
TEST_LIST    = DATASET_ROOT + "tri_testlist.txt"

CRF_VALUES = [0, 10, 18, 23, 30, 45]


def print_dataset_stats(train_paths, test_paths):
    sample_img = np.array(Image.open(train_paths[0][0]))
    h, w, c = sample_img.shape

    print("Dataset Size:")
    print(f"  Training triplets : {len(train_paths):>8,}")
    print(f"  Test triplets     : {len(test_paths):>8,}")
    print(f"  Total triplets    : {len(train_paths) + len(test_paths):>8,}")

    print(f"\nFrame Properties:")
    print(f"  Resolution        : {w}×{h} pixels (W×H)")
    print(f"  Channels          : {c} (RGB)")
    print(f"  Hint resolution   : {w//4}×{h//4} pixels (1/16 original size)")
    print(f"  Total pixels/frame: {h * w:,}")
    print(f"  Hint pixels       : {(h//4) * (w//4):,} ({100*(h//4)*(w//4)/(h*w):.1f}% of original)")

    bytes_per_frame = os.path.getsize(train_paths[0][0])
    total_frames = (len(train_paths) + len(test_paths)) * 3
    total_size_mb = (bytes_per_frame * total_frames) / (1024 ** 2)
    total_size_gb = total_size_mb / 1024

    print(f"\nStorage:")
    print(f"  Avg frame size    : {bytes_per_frame / 1024:.1f} KB")
    print(f"  Total frames      : {total_frames:,} (3 per triplet)")
    print(f"  Dataset size      : {total_size_gb:.2f} GB")

    train_pct = 100 * len(train_paths) / (len(train_paths) + len(test_paths))
    test_pct = 100 - train_pct
    print(f"\nTrain/Test Split:")
    print(f"  Training          : {train_pct:.1f}%")
    print(f"  Test              : {test_pct:.1f}%")


def compute_crf_psnr(test_paths):
    ref_frame = np.array(Image.open(test_paths[0][1])).astype(np.float32) / 255.0
    psnr_by_crf = {}
    for crf in CRF_VALUES:
        comp = compress_frame_h264(ref_frame, crf=crf)
        psnr, _ = compute_metrics(ref_frame, comp)
        psnr_by_crf[crf] = psnr
        print(f"CRF {crf:2d} → PSNR {psnr:.2f} dB")
    return psnr_by_crf


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    train_paths = load_vimeo_triplet(DATASET_ROOT, TRAIN_LIST)
    test_paths  = load_vimeo_triplet(DATASET_ROOT, TEST_LIST)

    print_dataset_stats(train_paths, test_paths)

    print("\nComputing PSNR by CRF …")
    psnr_by_crf = compute_crf_psnr(test_paths)

    print("\nGenerating figures …")
    plot_dataset_samples(test_paths, n=6).savefig("dataset_samples.png", bbox_inches="tight")
    plot_crf_comparison(test_paths, CRF_VALUES).savefig("crf_comparison.png", bbox_inches="tight")
    plot_crf_psnr_curve(CRF_VALUES, psnr_by_crf).savefig("crf_psnr_curve.png", bbox_inches="tight")
    plot_hint_pipeline(test_paths).savefig("hint_pipeline.png", bbox_inches="tight")
    plot_augmentation_pipeline(test_paths).savefig("augmentation_pipeline.png", bbox_inches="tight")

    print("Figures saved.")


if __name__ == "__main__":
    main()
