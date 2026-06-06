"""
hgvfi — Hint-Guided Video Frame Interpolation
=============================================

A PyTorch implementation of HGVFI: a video frame interpolation model
that leverages a low-resolution compressed hint frame alongside two reference
frames to reconstruct the intermediate frame with high fidelity.

Submodules
----------
hgvfi.models     — neural network architectures
hgvfi.data       — dataset loading and preprocessing
hgvfi.metrics    — PSNR / SSIM evaluation
hgvfi.train      — training and validation loops
hgvfi.evaluate   — benchmarking and qualitative comparison
hgvfi.visualize  — plotting utilities
"""
