"""
data.py
-------
Dataset loading and preprocessing utilities for the Vimeo-90K Triplet dataset.

Public API
----------
- load_vimeo_triplet     : Parse a Vimeo-90K list file and return (im1, im2, im3) path triples
- compress_frame_h264    : Encode a float32 frame to H.264 and decode back (quality simulation)
- downsample_hint        : Bicubic downscale of a frame to produce the hint
- upsample_hint_bicubic  : Bicubic upscale of the downsampled hint (baseline)
- numpy_to_tensor        : HWC numpy float32 → (1, C, H, W) torch tensor
- tensor_to_numpy        : (1, C, H, W) torch tensor → HWC numpy float32
- Vimeo90KDataset        : PyTorch Dataset with optional augmentation
"""

import os
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_vimeo_triplet(dataset_root, list_file, limit=None):
    """
    Parse a Vimeo-90K triplet list file.

    Parameters
    ----------
    dataset_root : str
        Root directory of the Vimeo-90K dataset.
    list_file : str
        Path to ``tri_trainlist.txt`` or ``tri_testlist.txt``.
    limit : int, optional
        If set, stop after this many entries.

    Returns
    -------
    list of tuple
        Each element is ``(im1_path, im2_path, im3_path)`` for one triplet.
        Only triplets where all three images exist on disk are included.
    """
    data = []
    with open(list_file) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            triplet_path = line.strip()
            seq_dir = os.path.join(dataset_root, "sequences", triplet_path)
            im1 = os.path.join(seq_dir, "im1.png")
            im2 = os.path.join(seq_dir, "im2.png")
            im3 = os.path.join(seq_dir, "im3.png")
            if os.path.exists(im1) and os.path.exists(im2) and os.path.exists(im3):
                data.append((im1, im2, im3))
    return data


# ---------------------------------------------------------------------------
# Compression simulation
# ---------------------------------------------------------------------------

def compress_frame_h264(frame_np, crf=23):
    """
    Simulate H.264 compression on a float32 frame using ffmpeg.

    Encodes the raw frame to an in-memory MP4 fragment with libx264 at the
    requested CRF, then decodes back to raw pixels.  This faithfully replicates
    the blocking and blurring artifacts that the model must learn to overcome.

    Parameters
    ----------
    frame_np : np.ndarray, shape (H, W, 3), dtype float32, range [0, 1]
    crf : int
        Constant Rate Factor (0 = lossless, 51 = worst).  Default is 23.

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype float32, range [0, 1]
    """
    # Compress frame using `Constant Rate Factor`
    h, w = frame_np.shape[:2]
    raw = (frame_np * 255).clip(0, 255).astype(np.uint8).tobytes()

    encode_proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-i", "pipe:0",
         "-vcodec", "libx264", "-crf", str(crf), "-preset", "ultrafast",
         "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    encoded, _ = encode_proc.communicate(raw)

    decode_proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    decoded_raw, _ = decode_proc.communicate(encoded)
    decoded = np.frombuffer(decoded_raw[:h*w*3], dtype=np.uint8).reshape(h, w, 3)
    return decoded.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Hint generation
# ---------------------------------------------------------------------------

def downsample_hint(frame_np, factor=4):
    """
    Bicubic-downsample a frame to generate the low-resolution hint.

    Parameters
    ----------
    frame_np : np.ndarray, shape (H, W, 3), dtype float32, range [0, 1]
    factor : int
        Downscale factor.  Default is 4 (1/16 pixel count of the original).

    Returns
    -------
    np.ndarray, shape (H//factor, W//factor, 3), dtype float32, range [0, 1]
    """
    img = Image.fromarray((frame_np * 255).astype(np.uint8))
    h, w = frame_np.shape[:2]
    try:
        bicubic = Image.Resampling.BICUBIC
    except AttributeError:
        bicubic = Image.BICUBIC
    img_small = img.resize((w // factor, h // factor), bicubic)
    return np.array(img_small).astype(np.float32) / 255.0


def upsample_hint_bicubic(hint_np, scale=4):
    """
    Bicubic-upsample the hint back to full resolution (baseline method).

    Parameters
    ----------
    hint_np : np.ndarray, shape (H, W, 3), dtype float32, range [0, 1]
    scale : int
        Upscale factor.

    Returns
    -------
    np.ndarray, shape (H*scale, W*scale, 3), dtype float32, range [0, 1]
    """
    t  = torch.from_numpy(hint_np.transpose(2, 0, 1)).unsqueeze(0).float()
    up = F.interpolate(t, scale_factor=scale, mode="bicubic", align_corners=False)
    return up.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def numpy_to_tensor(np_array):
    """Convert an HWC float32 numpy array to a (1, C, H, W) torch tensor."""
    return torch.from_numpy(np_array.transpose(2, 0, 1)).float().unsqueeze(0)


def tensor_to_numpy(tensor):
    """Convert a (1, C, H, W) torch tensor to an HWC float32 numpy array."""
    return tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Vimeo90KDataset(Dataset):
    """
    Loads compressed triplets with optional augmentation:
      - Random 256×256 crop
      - Horizontal flip (50%)
      - Temporal reverse (swap f0 ↔ f2) (50%)
    """
    def __init__(self, image_paths, crop_size=256, use_augmentation=True):
        self.image_paths     = image_paths
        self.crop_size       = crop_size
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        im1_path, im2_path, im3_path = self.image_paths[idx]

        def _load_compressed(path):
            cached = path.replace(".png", "_h264.png")
            if os.path.exists(cached):
                return np.array(Image.open(cached)).astype(np.float32) / 255.0
            frame = np.array(Image.open(path)).astype(np.float32) / 255.0
            return compress_frame_h264(frame, crf=23)

        f0  = _load_compressed(im1_path)
        fgt = _load_compressed(im2_path)
        f2  = _load_compressed(im3_path)
        hint = downsample_hint(fgt, factor=4)

        if self.use_augmentation and self.crop_size:
            h, w = f0.shape[:2]
            if h > self.crop_size and w > self.crop_size:
                y = random.randint(0, h - self.crop_size)
                x = random.randint(0, w - self.crop_size)
                s = self.crop_size
                f0   = f0[y:y+s,   x:x+s]
                fgt  = fgt[y:y+s,  x:x+s]
                f2   = f2[y:y+s,   x:x+s]
                hy, hx, hs = y//4, x//4, s//4
                hint = hint[hy:hy+hs, hx:hx+hs]

        if self.use_augmentation and random.random() < 0.5:
            f0, fgt, f2, hint = [a[:, ::-1].copy() for a in [f0, fgt, f2, hint]]

        if self.use_augmentation and random.random() < 0.5:
            f0, f2 = f2.copy(), f0.copy()

        return {
            "frame0": numpy_to_tensor(f0).squeeze(0),
            "frame2": numpy_to_tensor(f2).squeeze(0),
            "hint":   numpy_to_tensor(hint).squeeze(0),
            "gt":     numpy_to_tensor(fgt).squeeze(0),
        }
