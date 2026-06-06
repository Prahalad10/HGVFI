"""
models.py
---------
Neural network architectures for Hint-Guided Video Frame Interpolation (HGVFI).

Components
----------
- CharbonnierLoss         : Robust training loss
- ResidualAttentionBlock  : Conv block with Squeeze-and-Excitation attention
- HintBranch              : Upsamples the compressed hint to three feature scales
- CrossFrameAttention     : Bidirectional transformer attention at 1/8 scale
- FlowEstimator           : Lightweight CNN for bidirectional flow prediction
- WarpLayer               : Bilinear grid-sample warping
- EMAVFIBackbone          : Full encoder-decoder with hint injection and flow warping
- RefineNet               : Residual refinement network
- HintGuidedVFI           : Top-level model combining all of the above
- EMAVFINoHint            : Ablation backbone without hint conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Robust loss: sqrt(diff² + ε²). Better than L2 for outliers/compression artifacts."""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResidualAttentionBlock(nn.Module):
    """RAB: Conv-BN-ReLU-Conv-BN with Squeeze-and-Excitation channel attention."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.se_avg = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Linear(channels, channels // 4, bias=False)
        self.se_fc2 = nn.Linear(channels // 4, channels, bias=False)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        se  = self.se_avg(out).view(out.size(0), -1)
        se  = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(se))))
        out = out * se.view(se.size(0), se.size(1), 1, 1)
        return F.relu(out + residual)


class HintBranch(nn.Module):
    """
    Upsample hint from 1/4 res to 3 multi-scale feature maps via PixelShuffle + RABs.
    Input : (B, 3, H/4, W/4)
    Output: s1 (B,64,H/4,W/4), s2 (B,64,H/2,W/2), s3 (B,64,H,W)
    """
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Conv2d(3, 64, 3, padding=1)
        self.rab1      = ResidualAttentionBlock(64)
        self.ps1       = nn.PixelShuffle(2)
        self.reconv1   = nn.Conv2d(16, 64, 3, padding=1)
        self.rab2      = ResidualAttentionBlock(64)
        self.ps2       = nn.PixelShuffle(2)
        self.reconv2   = nn.Conv2d(16, 64, 3, padding=1)
        self.rab3      = ResidualAttentionBlock(64)

    def forward(self, hint):
        f  = F.relu(self.init_conv(hint))
        s1 = self.rab1(f)
        f  = F.relu(self.reconv1(self.ps1(s1)))
        s2 = self.rab2(f)
        f  = F.relu(self.reconv2(self.ps2(s2)))
        s3 = self.rab3(f)
        return s1, s2, s3


class CrossFrameAttention(nn.Module):
    """Bidirectional cross-frame attention at 1/8 scale for motion reasoning."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn1  = nn.Linear(dim, 4 * dim)
        self.ffn2  = nn.Linear(4 * dim, dim)

    def forward(self, f0, f2):
        B, C, H, W = f0.shape
        f0_seq = f0.permute(0, 2, 3, 1).reshape(B, H*W, C)
        f2_seq = f2.permute(0, 2, 3, 1).reshape(B, H*W, C)
        attn_out, _ = self.attn(self.norm(f0_seq), self.norm(f2_seq), self.norm(f2_seq))
        attn_out = attn_out + f0_seq
        ffn_out  = self.ffn2(F.relu(self.ffn1(attn_out))) + attn_out
        return ffn_out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class FlowEstimator(nn.Module):
    """Lightweight CNN: predicts bidirectional flow at 1/8 scale."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1    = nn.Conv2d(in_channels, 128, 3, padding=1)
        self.conv2    = nn.Conv2d(128, 64, 3, padding=1)
        self.flow_out = nn.Conv2d(64, 4, 3, padding=1)

    def forward(self, x):
        return self.flow_out(F.relu(self.conv2(F.relu(self.conv1(x)))))


class WarpLayer(nn.Module):
    """Bilinear grid-sample warping."""
    def forward(self, feat, flow):
        B, C, H, W = feat.shape
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device),
            torch.linspace(-1, 1, W, device=feat.device),
            indexing="ij",
        )
        grid   = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        flow_n = flow.permute(0, 2, 3, 1) / torch.tensor([W/2, H/2], device=feat.device)
        return F.grid_sample(feat, grid + flow_n, mode="bilinear", align_corners=True)


# ---------------------------------------------------------------------------
# Backbone and refinement
# ---------------------------------------------------------------------------

class EMAVFIBackbone(nn.Module):
    """
    Encoder-decoder with:
      - 4-level stride-2 encoder, 3-ch input — frames encoded independently (fix 1)
      - U-Net-style hint injection at encoder AND decoder (per paper)
      - 8 CrossFrameAttention blocks at 1/8 scale with proper Q=f0, K/V=f2 (fix 2)
      - Flow estimator whose output is now used to warp frames toward t=0.5 (fix 3)
      - Coarse head registered as nn.Module parameter, not created in forward (fix 4)
      - 3-level stride-2 transposed-conv decoder
    """
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3,   32,  3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(32,  64,  3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64,  128, 3, stride=2, padding=1)
        self.enc4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.hint_adapter_enc1 = nn.Conv2d(32  + 64, 32,  1)
        self.hint_adapter_enc2 = nn.Conv2d(64  + 64, 64,  1)
        self.hint_adapter_enc3 = nn.Conv2d(128 + 64, 128, 1)
        self.hint_adapter_dec3 = nn.Conv2d(128 + 64, 128, 1)
        self.hint_adapter_dec2 = nn.Conv2d(64  + 64, 64,  1)
        self.hint_adapter_dec1 = nn.Conv2d(32  + 64, 32,  1)

        self.cfa_blocks     = nn.ModuleList([CrossFrameAttention(256) for _ in range(8)])
        self.flow_estimator = FlowEstimator(256)
        self.warp_layer     = WarpLayer()

        self.dec3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1)

        self.coarse_head = nn.Conv2d(32, 3, 3, padding=1)

    def _encode_frame(self, frame, s1, s2, s3):
        """Encode one frame through the shared encoder with hint injection."""
        e1 = F.relu(self.enc1(frame))
        if s3 is not None:
            e1 = self.hint_adapter_enc1(torch.cat([e1, s3], dim=1))
        e2 = F.relu(self.enc2(e1))
        if s2 is not None:
            e2 = self.hint_adapter_enc2(torch.cat([e2, s2], dim=1))
        e3 = F.relu(self.enc3(e2))
        if s1 is not None:
            e3 = self.hint_adapter_enc3(torch.cat([e3, s1], dim=1))
        e4 = F.relu(self.enc4(e3))
        return e4

    def forward(self, frame0, frame2, hint_features=None):
        s1, s2, s3 = hint_features if hint_features is not None else (None, None, None)

        e4_0 = self._encode_frame(frame0, s1, s2, s3)  # (B, 256, H/8, W/8)
        e4_2 = self._encode_frame(frame2, s1, s2, s3)

        e4_att = e4_0
        for blk in self.cfa_blocks:
            e4_att = blk(e4_att, e4_2)

        flow_coarse = self.flow_estimator(e4_att)
        H_full, W_full = frame0.shape[2], frame0.shape[3]
        H_c,    W_c    = flow_coarse.shape[2], flow_coarse.shape[3]
        scale_h, scale_w = H_full / H_c, W_full / W_c
        flow_up = F.interpolate(flow_coarse, size=(H_full, W_full),
                                mode="bilinear", align_corners=False)
        flow_up = flow_up * torch.tensor(
            [scale_w, scale_h, scale_w, scale_h], device=flow_up.device
        ).view(1, 4, 1, 1)
        warped0 = self.warp_layer(frame0, flow_up[:, :2] * 0.5)
        warped2 = self.warp_layer(frame2, flow_up[:, 2:] * 0.5)
        warped_blend = (warped0 + warped2) * 0.5

        d3 = F.relu(self.dec3(e4_att))
        if hint_features is not None:
            d3 = self.hint_adapter_dec3(torch.cat([d3, s1], dim=1))
        d2 = F.relu(self.dec2(d3))
        if hint_features is not None:
            d2 = self.hint_adapter_dec2(torch.cat([d2, s2], dim=1))
        d1 = F.relu(self.dec1(d2))
        if hint_features is not None:
            d1 = self.hint_adapter_dec1(torch.cat([d1, s3], dim=1))

        coarse_decoder = torch.sigmoid(self.coarse_head(d1))
        coarse = torch.clamp(0.5 * coarse_decoder + 0.5 * warped_blend, 0.0, 1.0)
        return coarse, [d1]


class RefineNet(nn.Module):
    """
    Residual refinement: coarse_pred + hint_context (s3) + decoder_feat → output.
    Input channels: 3 + 64 + 32 = 99
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3 + 64 + 32, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.out   = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, coarse, hint_ctx, dec_feat):
        x = F.relu(self.conv1(torch.cat([coarse, hint_ctx, dec_feat], dim=1)))
        x = F.relu(self.conv2(x))
        return torch.sigmoid(coarse + self.out(x))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class HintGuidedVFI(nn.Module):
    """
    Full model:
      1. HintBranch   — upsample hint to 3 scales
      2. EMAVFIBackbone — encode frames + inject hint + cross-frame attention + decode
      3. RefineNet    — residual refinement with hint context
    """
    def __init__(self):
        super().__init__()
        self.hint_branch = HintBranch()
        self.backbone    = EMAVFIBackbone()
        self.refine      = RefineNet()

    def forward(self, frame0, frame2, hint):
        s1, s2, s3         = self.hint_branch(hint)
        coarse, dec_feats  = self.backbone(frame0, frame2, (s1, s2, s3))
        return self.refine(coarse, s3, dec_feats[0])


# ---------------------------------------------------------------------------
# Ablation variant
# ---------------------------------------------------------------------------

class EMAVFINoHint(nn.Module):
    """Ablation backbone — same architecture as EMAVFIBackbone but without hint conditioning."""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3,   32,  3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(32,  64,  3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64,  128, 3, stride=2, padding=1)
        self.enc4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.cfa_blocks     = nn.ModuleList([CrossFrameAttention(256) for _ in range(8)])
        self.flow_estimator = FlowEstimator(256)
        self.warp_layer     = WarpLayer()

        self.dec3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(64,  32,  4, stride=2, padding=1)

        self.coarse_head  = nn.Conv2d(32, 3, 3, padding=1)
        self.refine_conv1 = nn.Conv2d(3 + 32, 64, 3, padding=1)
        self.refine_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.refine_out   = nn.Conv2d(64, 3,  3, padding=1)

    def _encode_frame(self, frame):
        e1 = F.relu(self.enc1(frame))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        e4 = F.relu(self.enc4(e3))
        return e4

    def forward(self, frame0, frame2):
        e4_0 = self._encode_frame(frame0)
        e4_2 = self._encode_frame(frame2)

        e4_att = e4_0
        for blk in self.cfa_blocks:
            e4_att = blk(e4_att, e4_2)

        flow_coarse = self.flow_estimator(e4_att)
        H_full, W_full = frame0.shape[2], frame0.shape[3]
        flow_up = F.interpolate(flow_coarse, size=(H_full, W_full),
                                mode="bilinear", align_corners=False)
        scale_h = H_full / flow_coarse.shape[2]
        scale_w = W_full / flow_coarse.shape[3]
        flow_up = flow_up * torch.tensor(
            [scale_w, scale_h, scale_w, scale_h], device=flow_up.device
        ).view(1, 4, 1, 1)
        warped0      = self.warp_layer(frame0, flow_up[:, :2] * 0.5)
        warped2      = self.warp_layer(frame2, flow_up[:, 2:] * 0.5)
        warped_blend = (warped0 + warped2) * 0.5

        d3 = F.relu(self.dec3(e4_att))
        d2 = F.relu(self.dec2(d3))
        d1 = F.relu(self.dec1(d2))

        coarse_decoder = torch.sigmoid(self.coarse_head(d1))
        coarse = torch.clamp(0.5 * coarse_decoder + 0.5 * warped_blend, 0.0, 1.0)

        x = F.relu(self.refine_conv1(torch.cat([coarse, d1], dim=1)))
        x = F.relu(self.refine_conv2(x))
        return torch.sigmoid(coarse + self.refine_out(x))
