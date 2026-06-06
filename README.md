# Hint-Guided Video Frame Interpolation for Video Compression

This repository contains an independent PyTorch reimplementation of the HGVFI framework proposed in the paper above, along with experiments that replicate and extend the paper's core findings on the Vimeo-90K dataset.

---

## What the paper proposes

Traditional video codecs (H.264, HEVC, VVC) are highly optimised but hit diminishing returns. Neural video codecs learn better representations but require separate models for each quality level, making deployment costly. The paper's key insight is a third path:

> **Transmit only keyframes via a traditional codec. Reconstruct intermediate frames with a neural interpolation network that is guided by a cheap, low-resolution compressed "hint" of the target frame.**

The hint is simply the target frame compressed with H.264, then bicubic-downsampled to ¼ spatial resolution (1/16 pixel count). Despite its tiny size, it carries enough structural and motion information to resolve the two hardest failure modes of standard VFI — **occlusion** and **large motion** — without requiring explicit optical flow.

### Architecture summary

```
frame₀ ──┐
          ├── EMA-VFI backbone (encoder → 8× CrossFrameAttention → decoder)
frame₂ ──┘         ↑ hint injected at every encoder & decoder level
                    │
hint ──── HintBranch (PixelShuffle × 2 + Residual Attention Blocks → s1, s2, s3)
                    │
                    └──► RefineNet (residual correction on coarse output)
```

The VFI backbone is a modified compact EMA-VFI. The hint branch uses PixelShuffle with Residual Attention Blocks (RABs) to progressively upsample the hint to three feature scales, which are injected as U-Net-style skip connections into both the encoder and decoder. RefineNet fuses the coarse prediction with the finest hint scale for a residual correction.

---

## Repository structure

```
├── hgvfi/
│   ├── __init__.py
│   ├── models.py        — all architectures (HintGuidedVFI, EMAVFIBackbone,
│   │                       HintBranch, CrossFrameAttention, RefineNet, EMAVFINoHint, …)
│   ├── data.py          — Vimeo-90K loading, H.264 compression simulation,
│   │                       hint generation, Vimeo90KDataset
│   ├── metrics.py       — PSNR / SSIM evaluation
│   ├── train.py         — training and validation loops (standard + ablation variants)
│   ├── evaluate.py      — baselines, run_benchmark, per-sample qualitative tools
│   └── visualize.py     — all plotting utilities
├── scripts/
│   ├── train_hgvfi.py       — CLI entry point for training
│   ├── evaluate_hgvfi.py    — benchmark + ablation study
│   └── explore_dataset.py   — dataset statistics and visualisation
├── requirements.txt
└── README.md
```

---

## Dataset

Training and evaluation use the **Vimeo-90K Triplet** dataset ([Xue et al., 2017](http://toflow.csail.mit.edu/)).

| Split    | Triplets  | Resolution |
|----------|-----------|------------|
| Training | 51,312    | 448 × 256  |
| Test     | 3,782     | 448 × 256  |

Each triplet contains three consecutive frames (`im1.png`, `im2.png`, `im3.png`). The model is given `im1` and `im3` as reference frames and must reconstruct `im2`. All frames are passed through H.264 at **CRF = 23** before being fed to the network — matching the paper's training protocol and simulating real-world compressed inputs.

The hint is derived from the compressed ground-truth middle frame bicubic-downsampled by **4×** in each spatial dimension (i.e., 112 × 64 pixels), representing just **6.25%** of the original pixel count.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
```

FFmpeg with `libx264` must be available on `PATH` for H.264 compression simulation.

Download the Vimeo-90K dataset and place it as:

```
vimeo_triplet/
├── sequences/
├── tri_trainlist.txt
└── tri_testlist.txt
```

---

## Usage

### Explore the dataset
```bash
python scripts/explore_dataset.py
```

### Train
```bash
python scripts/train_hgvfi.py --epochs 30 --batch-size 4
```

Quick sanity-check run:
```bash
python scripts/train_hgvfi.py --epochs 5 --limit 500
```

### Evaluate
```bash
python scripts/evaluate_hgvfi.py --samples 400
```

---

## Implementation details

| Detail | Value / Choice |
|---|---|
| VFI backbone | EMA-VFI (compact variant) extended with a third hint input |
| Hint generation | H.264 CRF=23 → bicubic 4× downsample |
| Hint upsampling | Learnable: Conv → RAB → PixelShuffle × 2 → RAB |
| Hint injection | U-Net skip connections into encoder levels 1–3 and decoder levels 1–3 |
| Cross-frame attention | 8 blocks at 1/8 scale, Q = f₀ features, K/V = f₂ features |
| Flow estimation | Lightweight CNN predicting bidirectional flow at 1/8 scale, upsampled and used to warp frames toward t = 0.5 |
| Loss function | Charbonnier loss, ε = 10⁻⁶ |
| Optimiser | AdamW, lr = 2 × 10⁻⁴, weight decay = 10⁻⁴ |
| LR schedule | Cosine annealing |
| Crop size | 256 × 256 (random crop) |
| Augmentation | Random horizontal flip (50%), temporal reversal / frame swap (50%) |
| Batch size | 4 |
| Gradient clipping | max norm = 1.0 |

### Parameter counts (this implementation)

| Submodule | Parameters |
|---|---|
| HintBranch | ~0.27 M |
| EMAVFIBackbone | ~15.8 M |
| RefineNet | ~0.24 M |
| **Total** | **~16.3 M** |

---

## Compression quality: CRF effect

A key design choice in the paper is using **CRF = 23** as the training compression level, the FFmpeg H.264 default. The experiment below (run on a single test frame) shows why:

| CRF | PSNR vs Original |
|---|---|
| 0 (lossless) | ∞ dB |
| 10 | ~44 dB |
| 18 | ~38 dB |
| **23 (default)** | **~35 dB** |
| 30 | ~31 dB |
| 45 | ~26 dB |

CRF 23 sits in the sweet spot — enough compression to simulate real streaming conditions while retaining sufficient structural information in the hint for the model to learn from.

---

## Results

### Single-frame interpolation on Vimeo-90K (paper results)

| Method | PSNR (dB) | SSIM |
|---|---|---|
| ToFlow | 33.73 | 0.9682 |
| SepConv | 33.79 | 0.9702 |
| CAIN | 34.65 | 0.9730 |
| DAIN | 34.71 | 0.9756 |
| XVFI | 35.07 | 0.9760 |
| IFRNet | 35.80 | 0.9794 |
| EMA-VFI (small) | 36.07 | 0.9797 |
| EMA-VFI | 36.64 | 0.9819 |
| **HGVFI (paper)** | **38.69** | **0.9885** |

The paper's model achieves **+2.05 dB** over EMA-VFI (the backbone it extends) on Vimeo-90K, and substantially larger gains on harder SNU-FILM subsets (Hard: +3.23 dB, Extreme: +7.68 dB). The gains are largest precisely where interpolation is hardest — confirming that the hint's motion cues are most valuable under occlusion and large temporal distance.

### Baselines compared in this replication

| Method | Description |
|---|---|
| Linear | Pixel-wise average of the two reference frames |
| Bicubic | Downsample the averaged frame and bicubic-upsample back |
| BicubicHint | Bicubic 4× upscale of the compressed hint frame |
| **HGVFI (ours)** | Full model |
| NoHint (ablation) | Same backbone without any hint conditioning |

---

## Ablation: does the hint actually help?

An `EMAVFINoHint` variant was trained with the identical backbone architecture but with all hint injection removed. Training curves and final metrics consistently show the hint-guided model outperforming the hint-free variant, particularly in PSNR and SSIM, confirming the paper's claim that the hint is the primary source of improvement and not simply model capacity.

---

## Key findings from experiments

**1. The hint provides critical motion cues under compression.** Even at CRF 23, the compressed hint retains enough structure for the model to resolve occlusions and large displacements that cause standard VFI to blur or hallucinate.

**2. CRF robustness.** When evaluated across CRF values from 10 to 40 (the model was trained only at CRF 23), HGVFI consistently outperforms BicubicHint at all compression levels, showing that the learned upsampling generalises well beyond the training distribution.

**3. Hint > bicubic upsampling.** Even pure bicubic upsampling of the hint outperforms linear frame averaging and bicubic interpolation of the averaged frame, showing that the hint's content information (not just the model's capacity) drives performance.

**4. Scatter analysis.** HGVFI wins the per-sample PSNR comparison against BicubicHint on the large majority of test triplets, with the biggest gains on hard cases (large motion, occlusion) — exactly the failure modes the paper targets.

---

## Differences from the paper

| Aspect | Paper | This implementation |
|---|---|---|
| Training set size | Full 51,312 triplets | Subset (4,000 in experiments) |
| Training epochs | Not specified | 12 epochs in ablation runs |
| Cross-frame attention | 8 blocks (compact EMA-VFI) | 8 blocks ✓ |
| Flow warping | Bidirectional, toward t = 0.5 | ✓ |
| Evaluation dataset | Vimeo-90K + SNU-FILM + UVG | Vimeo-90K |
| Video compression eval | H.264/HEVC rate-distortion curves | Not included |

The architecture faithfully replicates the paper's description. The main practical difference is training at reduced scale due to compute constraints.

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=9.0.0
scikit-image>=0.20.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

Plus: `ffmpeg` with `libx264` on `PATH`.

---

## Citation

```bibtex
@inproceedings{tan2025hgvfi,
  title     = {Hint-Guided Video Frame Interpolation for Video Compression},
  author    = {Tan, Pan and Feng, Wu-chi},
  booktitle = {ACM Multimedia Asia (MMAsia '25)},
  year      = {2025},
  month     = {December},
  address   = {Kuala Lumpur, Malaysia},
  doi       = {10.1145/3743093.3771081}
}
```
