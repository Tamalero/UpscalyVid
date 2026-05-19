#!/usr/bin/env python3
"""UpscalyVid — AI video upscaler using Real-ESRGAN (self-contained, no basicsr/opencv required)"""

import argparse
import datetime
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEMP_PREFIX = "upscalyvid_"
_VIDEO_EXTS  = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv"}


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _cleanup_stale_temp() -> None:
    """Remove leftover temp dirs from crashed/killed previous runs (older than 2 h)."""
    cutoff = time.time() - 7200
    for d in Path(tempfile.gettempdir()).glob(f"{_TEMP_PREFIX}*"):
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass


class _TeeLogger:
    """Duplicates every print() / stdout write to a log file as well as the terminal."""

    def __init__(self, log_path: Path) -> None:
        self._real  = sys.stdout
        self._fh    = open(log_path, "w", encoding="utf-8", buffering=1)
        sys.stdout  = self

    def write(self, s: str) -> None:
        self._real.write(s)
        self._fh.write(s)

    def flush(self) -> None:
        self._real.flush()
        self._fh.flush()

    def fileno(self) -> int:          # needed by subprocess when verbose=True
        return self._real.fileno()

    def isatty(self) -> bool:
        return self._real.isatty()

    def close(self) -> None:
        sys.stdout = self._real
        try:
            self._fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent / "models"

MODELS = {
    # ── RRDBNet family (high quality, heavier) ────────────────────────────────
    "realesrgan-x4plus": {
        "arch": "rrdbnet", "scale": 4, "num_block": 23,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "description": "General-purpose 4x — best quality for photos/video",
        "tags": ["general", "photo", "video"],
    },
    "realesrgan-x4plus-anime": {
        "arch": "rrdbnet", "scale": 4, "num_block": 6,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "description": "Optimised for anime / illustration / cartoon",
        "tags": ["anime", "cartoon", "illustration"],
    },
    "realesrgan-x2plus": {
        "arch": "rrdbnet", "scale": 2, "num_block": 23,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "filename": "RealESRGAN_x2plus.pth",
        "description": "2x upscale — HD→2K or 2K→4K",
        "tags": ["general", "2x"],
    },
    "realesrnet-x4plus": {
        "arch": "rrdbnet", "scale": 4, "num_block": 23,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
        "filename": "RealESRNet_x4plus.pth",
        "description": "4x restoration — softer, less aggressive sharpening",
        "tags": ["general", "restoration", "soft"],
    },
    # ── SRVGGNetCompact family (faster, lighter, great for video) ─────────────
    "realesr-general-x4v3": {
        "arch": "srvgg", "scale": 4, "num_feat": 64, "num_conv": 32,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "filename": "realesr-general-x4v3.pth",
        "description": "Compact 4x — fast, real-world footage, recommended for video",
        "tags": ["general", "video", "fast"],
    },
    "realesr-general-wdn-x4v3": {
        "arch": "srvgg", "scale": 4, "num_feat": 64, "num_conv": 32,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        "filename": "realesr-general-wdn-x4v3.pth",
        "description": "Compact 4x + denoising — best for compressed or noisy footage",
        "tags": ["general", "video", "fast", "denoise"],
    },
    # ── Specialised / community models ────────────────────────────────────────
    # Fine organic textures: fur, feathers, scales, hair, fabric weave.
    # Trained with emphasis on high-frequency detail preservation.
    "4x-UltraSharp": {
        "arch": "rrdbnet", "scale": 4, "num_block": 23,
        "url": "https://huggingface.co/Kim2091/4x-UltraSharp/resolve/main/4x-UltraSharp.pth",
        "filename": "4x-UltraSharp.pth",
        "description": "Community 4x — exceptional fine-texture detail (fur, feathers, hair, fabric)",
        "tags": ["fur", "feathers", "hair", "texture", "detail", "community"],
    },
    # Old / degraded footage: VHS, film grain, heavy MPEG artefacts.
    # Strong denoising + sharpening pass, trades some texture for cleanliness.
    "realesr-general-wdn-x4v3-denoise": {
        "arch": "srvgg", "scale": 4, "num_feat": 64, "num_conv": 32,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        "filename": "realesr-general-wdn-x4v3.pth",
        "description": "Same wdn weights, listed here as a reminder for heavily degraded footage",
        "tags": ["degraded", "vhs", "film", "artefacts", "denoise"],
    },
}

# ---------------------------------------------------------------------------
# Custom model discovery
# ---------------------------------------------------------------------------

def _remap_esrgan_keys(weights: dict) -> dict:
    """Convert old ESRGAN model.* keys to Real-ESRGAN RRDBNet body.* keys."""
    # Determine the conv_body index (highest model.1.sub.N index = num_block)
    sub_indices = set()
    for k in weights:
        m = re.match(r"model\.1\.sub\.(\d+)\.", k)
        if m:
            sub_indices.add(int(m.group(1)))
    conv_body_idx = max(sub_indices) if sub_indices else 23

    fixed = {
        "model.0":                        "conv_first",
        f"model.1.sub.{conv_body_idx}":   "conv_body",
        "model.3":                        "conv_up1",
        "model.6":                        "conv_up2",
        "model.8":                        "conv_hr",
        "model.10":                       "conv_last",
    }

    new_weights = {}
    for old_key, val in weights.items():
        mapped = False
        for old_prefix, new_prefix in fixed.items():
            if old_key == old_prefix or old_key.startswith(old_prefix + "."):
                suffix = old_key[len(old_prefix):]  # includes leading "." or empty
                new_weights[new_prefix + suffix] = val
                mapped = True
                break
        if mapped:
            continue
        # model.1.sub.N.RDB{k}.conv{j}.0.weight  →  body.N.rdb{k}.conv{j}.weight
        m = re.match(r"model\.1\.sub\.(\d+)\.(RDB\d+)\.conv(\d+)\.0\.(weight|bias)$", old_key)
        if m:
            n, rdb, j, param = m.group(1), m.group(2).lower(), m.group(3), m.group(4)
            new_weights[f"body.{n}.{rdb}.conv{j}.{param}"] = val
        else:
            new_weights[old_key] = val
    return new_weights


def _safe_load_checkpoint(path: str) -> dict:
    """Load a .pth checkpoint, falling back to weights_only=False for legacy pickle formats."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        pass
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        hint = ""
        try:
            with open(path, "rb") as _f:
                first = _f.read(16)
            if first.startswith(b"version https://"):
                hint = (
                    "\n  The cached file is a Git LFS pointer, not the actual weights."
                    "\n  Delete it and re-run to trigger a proper download:"
                    f"\n    rm '{path}'"
                )
            elif first[:4] == b"v1.0":
                hint = (
                    "\n  The file uses an old Torch format not readable by modern PyTorch."
                    "\n  Delete it and re-run to download the current version:"
                    f"\n    rm '{path}'"
                )
        except OSError:
            pass
        raise RuntimeError(f"Cannot load checkpoint '{Path(path).name}': {e}{hint}") from None


def _detect_hat_params(weights: dict) -> dict:
    """Extract HAT hyperparameters from a HAT state dict."""
    embed_dim = weights["conv_first.weight"].shape[0]

    layer_indices = set()
    for k in weights:
        m = re.match(r"layers\.(\d+)\.", k)
        if m:
            layer_indices.add(int(m.group(1)))
    num_layers = len(layer_indices) if layer_indices else 6

    block_indices = set()
    for k in weights:
        m = re.match(r"layers\.0\.blocks\.(\d+)\.", k)
        if m:
            block_indices.add(int(m.group(1)))
    num_blocks_per_layer = len(block_indices) if block_indices else 6

    rpi_key = "layers.0.blocks.0.attn.attns.0.relative_position_index"
    window_size = int(math.isqrt(weights[rpi_key].shape[0])) if rpi_key in weights else 16

    temp_key = "layers.0.blocks.1.attn.temperature"
    num_heads = weights[temp_key].shape[0] if temp_key in weights else 6
    head_dim = embed_dim // num_heads

    attn_indices = set()
    for k in weights:
        m = re.match(r"layers\.0\.blocks\.0\.attn\.attns\.(\d+)\.", k)
        if m:
            attn_indices.add(int(m.group(1)))
    num_groups = len(attn_indices) if attn_indices else 2
    heads_per_attn = num_heads // num_groups

    ffn_key = "layers.0.blocks.0.ffn.fc1.weight"
    ffn_ratio = weights[ffn_key].shape[0] // embed_dim if ffn_key in weights else 2

    ch_key = "layers.0.blocks.0.attn.channel_interaction.1.weight"
    ch_sq = weights[ch_key].shape[0] if ch_key in weights else 22

    sp_key = "layers.0.blocks.0.attn.spatial_interaction.0.weight"
    sp_sq = weights[sp_key].shape[0] if sp_key in weights else 11

    mid_key = "conv_before_upsample.0.weight"
    upsample_mid_ch = weights[mid_key].shape[0] if mid_key in weights else 64

    upsample_conv_keys = [k for k in weights if re.match(r"upsample\.\d+\.weight", k)]
    n_stages = len(upsample_conv_keys)
    scale = 2 ** n_stages if n_stages > 0 else 4

    return {
        "arch": "hat",
        "scale": scale,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "window_size": window_size,
        "shift_size": window_size // 2,
        "num_layers": num_layers,
        "num_blocks_per_layer": num_blocks_per_layer,
        "ffn_ratio": ffn_ratio,
        "ch_sq": ch_sq,
        "sp_sq": sp_sq,
        "heads_per_attn": heads_per_attn,
        "upsample_mid_ch": upsample_mid_ch,
    }


def _detect_arch_from_weights(weights: dict) -> dict:
    """Detect arch, num_block/num_feat/num_conv, and scale from state dict."""
    keys = set(weights.keys())
    has_body = any(k.startswith("body.") for k in keys)
    has_rdb  = any("rdb" in k for k in keys if k.startswith("body."))

    if has_body and not has_rdb:
        # SRVGGNetCompact — flat body.N.weight keys
        body_conv = sorted(
            [k for k in keys if k.startswith("body.") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[1]),
        )
        scale = 4
        if body_conv:
            out_ch = weights[body_conv[-1]].shape[0]
            for s in (2, 4, 8):
                if out_ch == 3 * s * s:
                    scale = s
                    break
        return {"arch": "srvgg", "num_feat": 64, "num_conv": 32, "scale": scale}

    if has_rdb:
        # RRDBNet — count RRDB blocks, detect scale from upsample conv keys
        body_indices = set()
        for k in keys:
            m = re.match(r"body\.(\d+)\.", k)
            if m:
                body_indices.add(int(m.group(1)))
        num_block = len(body_indices) if body_indices else 23
        scale = 8 if "conv_up3.weight" in keys else (4 if "conv_up2.weight" in keys else 2)
        return {"arch": "rrdbnet", "num_block": num_block, "scale": scale}

    # Old ESRGAN format: keys like model.1.sub.N.RDBN.convM.0.weight
    if any(k.startswith("model.") for k in keys):
        return {"arch": "unsupported", "scale": 4,
                "note": "old ESRGAN format (pre-Real-ESRGAN key naming) — not compatible"}

    # HAT: identified by before_RG keys + layers.*.blocks.*.attn.attns.*
    has_before_rg = any(k.startswith("before_RG") for k in keys)
    has_hat_attns = any(re.match(r"layers\.\d+\.blocks\.\d+\.attn\.attns\.", k) for k in keys)
    if has_before_rg and has_hat_attns:
        return _detect_hat_params(weights)

    # Unknown — identify from key patterns to give a useful error
    if any("attn" in k for k in keys):
        note = "transformer-based (SwinIR / SPAN / other)"
    elif any(k.startswith("layers.") for k in keys):
        note = "layer-based transformer (unknown variant)"
    else:
        note = "unrecognised architecture"
    return {"arch": "unsupported", "scale": 4, "note": note}


def _load_custom_models() -> None:
    """Scan models/ for .pth files not in MODELS and register them as Custom-<stem>."""
    if not MODELS_DIR.exists():
        return
    known_filenames = {info["filename"] for info in MODELS.values()}
    for path in sorted(MODELS_DIR.glob("*.pth")):
        if path.name in known_filenames:
            continue
        key = f"Custom-{path.stem}"
        if key in MODELS:
            continue
        name_lower = path.stem.lower()
        scale_hint = 8 if any(x in name_lower for x in ("x8", "8x")) else \
                     2 if any(x in name_lower for x in ("x2", "2x")) else 4
        MODELS[key] = {
            "arch": "auto",
            "scale": scale_hint,
            "filename": path.name,
            "url": "",
            "description": f"Custom model — {path.name}",
            "tags": ["custom"],
        }


_load_custom_models()


# ---------------------------------------------------------------------------
# RRDBNet architecture (self-contained, matches xinntao/Real-ESRGAN weights)
# ---------------------------------------------------------------------------

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale >= 4:
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.scale >= 4:
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        if self.scale == 8:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ---------------------------------------------------------------------------
# SRVGGNetCompact architecture (realesr-general-x4v3 family)
# ---------------------------------------------------------------------------

class SRVGGNetCompact(nn.Module):
    """Lightweight SRVGG used by the realesr-general-x4v3 / wdn models."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 4,
        act_type: str = "prelu",
    ):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._act(act_type, num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._act(act_type, num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _act(act_type: str, num_feat: int) -> nn.Module:
        if act_type == "relu":
            return nn.ReLU(inplace=True)
        if act_type == "prelu":
            return nn.PReLU(num_parameters=num_feat)
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


# ---------------------------------------------------------------------------
# HAT architecture (Hybrid Attention Transformer — 4x-UltraSharpV2 family)
# ---------------------------------------------------------------------------

def _win_part(x, ws):
    """Partition [B,H,W,C] into windows: returns [-1, ws*ws, C]."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)


def _win_unpart(w, ws, H, W):
    """Reverse window partition: [nW*B, ws*ws, C] -> [B,H,W,C]."""
    nW = (H // ws) * (W // ws)
    B = w.shape[0] // nW
    C = w.shape[-1]
    x = w.view(B, H // ws, W // ws, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)


def _shift_mask(H, W, ws, ss, device):
    """Compute shifted-window attention mask. Returns [nW, ws^2, ws^2]."""
    img_mask = torch.zeros(1, H, W, 1, device=device)
    cnt = 0
    for h in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
        for w in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mw = _win_part(img_mask, ws).squeeze(-1)        # [nW, ws^2]
    mask = mw.unsqueeze(1) - mw.unsqueeze(2)        # [nW, ws^2, ws^2]
    return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)


class CPBMLP(nn.Module):
    """
    Continuous Position Bias MLP.
    Takes unique relative-position coordinates (passed in from the parent
    WinAttnHead which owns rpe_biases and relative_position_index buffers),
    projects them through a small residual MLP, then indexes by
    relative_position_index to get per-pair biases [L*L, heads_per_attn].

    State-dict layout:
      pos_proj.weight/bias
      pos1.0.weight/bias  (LayerNorm)
      pos1.2.weight/bias  (Linear)
      pos2.0, pos2.2, pos3.0, pos3.2  (same pattern)
    """

    def __init__(self, heads: int = 3):
        super().__init__()
        self.pos_proj = nn.Linear(2, 5, bias=True)
        # Each posN: Sequential index 0 = LayerNorm(5), index 2 = Linear
        # (index 1 is GELU — no parameters)
        self.pos1 = nn.Sequential(nn.LayerNorm(5), nn.GELU(), nn.Linear(5, 5))
        self.pos2 = nn.Sequential(nn.LayerNorm(5), nn.GELU(), nn.Linear(5, 5))
        self.pos3 = nn.Sequential(nn.LayerNorm(5), nn.GELU(), nn.Linear(5, heads))

    def forward(self, rpe_biases, relative_position_index):
        """
        rpe_biases: [N_unique, 2]  (owned by parent WinAttnHead)
        relative_position_index: [L, L]
        Returns: [L*L, heads]
        """
        x = self.pos_proj(rpe_biases)                  # [N, 5]
        x = x + self.pos1(x)                           # residual
        x = x + self.pos2(x)                           # residual
        x = self.pos3(x)                                # [N, heads]
        idx = relative_position_index.view(-1)          # [L*L]
        return x[idx]                                  # [L*L, heads]


class WinAttnHead(nn.Module):
    """
    One head-group in WinAttn.
    Owns rpe_biases and relative_position_index buffers (as in the checkpoint),
    and pos (CPBMLP) as a sub-module.

    State-dict keys at this level:
      rpe_biases [945, 2]
      relative_position_index [256, 256]
      pos.pos_proj.*, pos.pos1.*, pos.pos2.*, pos.pos3.*
    """

    def __init__(self, heads_per_attn: int = 3, head_dim: int = 30):
        super().__init__()
        self.heads = heads_per_attn
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.pos = CPBMLP(heads=heads_per_attn)
        # rpe_biases and relative_position_index live here (not in pos)
        self.register_buffer("rpe_biases", torch.zeros(945, 2))
        self.register_buffer("relative_position_index", torch.zeros(256, 256, dtype=torch.long))

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: [nW*B, ws^2, heads*head_dim] (already split out for this group)
        mask: [nW, ws^2, ws^2] or None
        Returns: [nW*B, ws^2, heads*head_dim]
        """
        nWB, L, C = q.shape
        heads = self.heads
        hd = self.head_dim
        q = q.view(nWB, L, heads, hd).permute(0, 2, 1, 3)  # [nWB, heads, L, hd]
        k = k.view(nWB, L, heads, hd).permute(0, 2, 1, 3)
        v = v.view(nWB, L, heads, hd).permute(0, 2, 1, 3)

        # RPE bias: [L*L, heads] → [1, heads, L, L]
        bias = self.pos(self.rpe_biases, self.relative_position_index)  # [L*L, heads]
        bias = bias.view(L, L, heads).permute(2, 0, 1).unsqueeze(0)    # [1, heads, L, L]

        if mask is not None:
            nW = mask.shape[0]
            B_ = nWB // nW
            # mask [nW,L,L] + bias [1,heads,L,L] → [nW,heads,L,L]
            attn_mask = (bias + mask.unsqueeze(1)).to(q.dtype)
            if B_ > 1:
                attn_mask = attn_mask.expand(B_, nW, heads, L, L).reshape(nWB, heads, L, L)
        else:
            attn_mask = bias.to(q.dtype)  # [1, heads, L, L] — broadcast over nWB

        # SDPA: uses Memory-Efficient Attention on CUDA (sm70+) or Flash Attention (sm80+
        # when mask is None). Avoids materialising the full [nWB, heads, L, L] score tensor.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)
        return out.permute(0, 2, 1, 3).contiguous().view(nWB, L, heads * hd)


class SimpleGate(nn.Module):
    """Gate: split input to [x1,x2], apply DWConv+LN on x2, return x1*x2."""

    def __init__(self, dim: int = 180):
        super().__init__()
        # DWConv on the gating half
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H: int, W: int):
        """x: [B, L, 2*dim]. Returns [B, L, dim]."""
        x1, x2 = x.chunk(2, dim=-1)          # each [B, L, dim]
        B, L, C = x2.shape
        # Apply DWConv spatially
        x2_2d = x2.transpose(1, 2).view(B, C, H, W)
        x2_2d = self.conv(x2_2d)
        x2 = x2_2d.view(B, C, L).transpose(1, 2)    # [B, L, C]
        x2 = self.norm(x2)
        return x1 * x2


class GatedFFN(nn.Module):
    def __init__(self, dim: int = 180, ffn_ratio: int = 2):
        super().__init__()
        hidden = dim * ffn_ratio           # 360
        self.fc1 = nn.Linear(dim, hidden)
        self.sg = SimpleGate(dim=dim)
        self.fc2 = nn.Linear(dim, dim)    # input dim (after gate halves to dim)

    def forward(self, x, H: int, W: int):
        """x: [B, L, dim]"""
        x = self.fc1(x)          # [B, L, 2*dim]
        x = self.sg(x, H, W)    # [B, L, dim]
        x = self.fc2(x)          # [B, L, dim]
        return x


def _make_channel_interaction(in_ch: int = 180, sq_ch: int = 22):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),           # 0
        nn.Conv2d(in_ch, sq_ch, 1),        # 1
        nn.BatchNorm2d(sq_ch),             # 2
        nn.ReLU(inplace=True),             # 3
        nn.Conv2d(sq_ch, in_ch, 1),        # 4
    )


def _make_spatial_interaction(in_ch: int = 180, sq_ch: int = 11):
    return nn.Sequential(
        nn.Conv2d(in_ch, sq_ch, 1),        # 0
        nn.BatchNorm2d(sq_ch),             # 1
        nn.ReLU(inplace=True),             # 2
        nn.Conv2d(sq_ch, 1, 1),            # 3
    )


def _make_dwconv(dim: int = 180):
    return nn.Sequential(
        nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True),  # 0
        nn.BatchNorm2d(dim),                                    # 1
    )


class WinAttn(nn.Module):
    def __init__(
        self,
        dim: int = 180,
        num_heads: int = 6,
        head_dim: int = 30,
        window_size: int = 16,
        shift_size: int = 0,
        ch_sq: int = 22,
        sp_sq: int = 11,
        heads_per_attn: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.heads_per_attn = heads_per_attn

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Two WinAttnHead groups (3 heads each = 6 total)
        num_groups = num_heads // heads_per_attn   # 2
        self.attns = nn.ModuleList([
            WinAttnHead(heads_per_attn, head_dim) for _ in range(num_groups)
        ])

        self.channel_interaction = _make_channel_interaction(dim, ch_sq)
        self.spatial_interaction = _make_spatial_interaction(dim, sp_sq)
        self.dwconv = _make_dwconv(dim)

    def forward(self, x, H: int, W: int):
        """x: [B, L, C]  L=H*W. Returns [B, L, C]."""
        B, L, C = x.shape
        ws = self.window_size
        ss = self.shift_size

        # QKV projection
        qkv = self.qkv(x)                              # [B, L, 3C]
        q, k, v = qkv.chunk(3, dim=-1)                 # each [B, L, C]

        # DW conv on V (before shift/window)
        v_2d = v.transpose(1, 2).view(B, C, H, W)      # [B,C,H,W]
        v_dw = self.dwconv[0](v_2d)                    # DWConv
        v_dw = self.dwconv[1](v_dw)                    # BN
        # v_dw: [B, C, H, W]

        # Cyclic shift
        if ss > 0:
            q_2d = q.transpose(1, 2).view(B, C, H, W)
            k_2d = k.transpose(1, 2).view(B, C, H, W)
            q_2d = torch.roll(q_2d, shifts=(-ss, -ss), dims=(2, 3))
            k_2d = torch.roll(k_2d, shifts=(-ss, -ss), dims=(2, 3))
            v_2d_s = torch.roll(v_2d, shifts=(-ss, -ss), dims=(2, 3))
            v_dw_s = torch.roll(v_dw, shifts=(-ss, -ss), dims=(2, 3))
            # Reformat to sequence
            q = q_2d.view(B, C, L).transpose(1, 2)
            k = k_2d.view(B, C, L).transpose(1, 2)
            v_seq = v_2d_s.view(B, C, L).transpose(1, 2)
        else:
            v_seq = v_2d.view(B, C, L).transpose(1, 2)  # [B, L, C]
            v_dw_s = v_dw

        # Reshape Q,K,V to [B,H,W,C] channel-last, then partition into windows
        q_cl = q.view(B, H, W, C)
        k_cl = k.view(B, H, W, C)
        v_cl = v_seq.view(B, H, W, C)

        q_win = _win_part(q_cl, ws)   # [nWB, ws^2, C]
        k_win = _win_part(k_cl, ws)
        v_win = _win_part(v_cl, ws)

        # Attention mask
        if ss > 0:
            mask = _shift_mask(H, W, ws, ss, x.device)  # [nW, ws^2, ws^2]
        else:
            mask = None

        # Split into head groups and run attention
        hd = self.head_dim
        hpa = self.heads_per_attn
        group_dim = hpa * hd   # 90
        attn_out_parts = []
        for i, attn_head in enumerate(self.attns):
            q_g = q_win[..., i*group_dim:(i+1)*group_dim]
            k_g = k_win[..., i*group_dim:(i+1)*group_dim]
            v_g = v_win[..., i*group_dim:(i+1)*group_dim]
            out_g = attn_head(q_g, k_g, v_g, mask=mask)   # [nWB, ws^2, group_dim]
            attn_out_parts.append(out_g)

        attn_win = torch.cat(attn_out_parts, dim=-1)       # [nWB, ws^2, C]

        # Unpartition windows
        attn_cl = _win_unpart(attn_win, ws, H, W)          # [B, H, W, C]

        # Reverse cyclic shift
        if ss > 0:
            attn_2d = attn_cl.permute(0, 3, 1, 2)          # [B, C, H, W]
            attn_2d = torch.roll(attn_2d, shifts=(ss, ss), dims=(2, 3))
            v_dw_s = torch.roll(v_dw_s, shifts=(ss, ss), dims=(2, 3))
        else:
            attn_2d = attn_cl.permute(0, 3, 1, 2)          # [B, C, H, W]

        # Channel interaction (on attn output)
        ch_gate = torch.sigmoid(self.channel_interaction(attn_2d))  # [B, C, 1, 1]

        # Spatial interaction (on DW-conv output)
        sp_gate = torch.sigmoid(self.spatial_interaction(v_dw_s))   # [B, 1, H, W]

        # Combine: (window_attn + dw*sp_gate) * ch_gate
        combined = (attn_2d + v_dw_s * sp_gate) * ch_gate           # [B, C, H, W]

        # Back to sequence
        out = combined.view(B, C, L).transpose(1, 2)                 # [B, L, C]
        out = self.proj(out)
        return out


class ChanAttn(nn.Module):
    def __init__(
        self,
        dim: int = 180,
        num_heads: int = 6,
        head_dim: int = 30,
        ch_sq: int = 22,
        sp_sq: int = 11,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.channel_interaction = _make_channel_interaction(dim, ch_sq)
        self.spatial_interaction = _make_spatial_interaction(dim, sp_sq)
        self.dwconv = _make_dwconv(dim)

    def forward(self, x, H: int, W: int):
        """x: [B, L, C]. Returns [B, L, C]."""
        B, L, C = x.shape
        nh = self.num_heads
        hd = self.head_dim

        # QKV
        qkv = self.qkv(x)                              # [B, L, 3C]
        q, k, v = qkv.chunk(3, dim=-1)                 # each [B, L, C]

        # DW conv on v
        v_2d = v.transpose(1, 2).view(B, C, H, W)
        v_dw = self.dwconv[0](v_2d)
        v_dw = self.dwconv[1](v_dw)

        # Reshape to [B, nh, hd, L] for channel attention
        q = q.view(B, L, nh, hd).permute(0, 2, 3, 1)  # [B, nh, hd, L]
        k = k.view(B, L, nh, hd).permute(0, 2, 3, 1)
        v = v.view(B, L, nh, hd).permute(0, 2, 3, 1)

        # Normalize Q,K
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Channel attention: [B, nh, hd, hd]
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1).to(q.dtype)

        # Apply to V: [B, nh, hd, L]
        out = attn @ v

        # Back to [B, L, C]
        out = out.permute(0, 3, 1, 2).contiguous().view(B, L, C)
        out_2d = out.transpose(1, 2).view(B, C, H, W)

        # Channel interaction
        ch_gate = torch.sigmoid(self.channel_interaction(out_2d))   # [B, C, 1, 1]

        # Spatial interaction (on DW output)
        sp_gate = torch.sigmoid(self.spatial_interaction(v_dw))      # [B, 1, H, W]

        # Combine
        combined = (out_2d + v_dw * sp_gate) * ch_gate               # [B, C, H, W]
        combined = combined.view(B, C, L).transpose(1, 2)            # [B, L, C]
        combined = self.proj(combined)
        return combined


class WinAttnBlock(nn.Module):
    def __init__(self, dim=180, num_heads=6, head_dim=30,
                 window_size=16, shift_size=0,
                 ch_sq=22, sp_sq=11, heads_per_attn=3, ffn_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = WinAttn(
            dim=dim, num_heads=num_heads, head_dim=head_dim,
            window_size=window_size, shift_size=shift_size,
            ch_sq=ch_sq, sp_sq=sp_sq, heads_per_attn=heads_per_attn,
        )
        self.ffn = GatedFFN(dim=dim, ffn_ratio=ffn_ratio)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class ChanAttnBlock(nn.Module):
    def __init__(self, dim=180, num_heads=6, head_dim=30,
                 ch_sq=22, sp_sq=11, ffn_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = ChanAttn(
            dim=dim, num_heads=num_heads, head_dim=head_dim,
            ch_sq=ch_sq, sp_sq=sp_sq,
        )
        self.ffn = GatedFFN(dim=dim, ffn_ratio=ffn_ratio)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class ResidualGroup(nn.Module):
    """
    One residual group ("layer") with num_blocks alternating WinAttn/ChanAttn blocks
    followed by a residual Conv2d.

    State-dict keys at this level:
      blocks.N.*   (the transformer blocks)
      conv.weight / conv.bias   [embed_dim, embed_dim, 3, 3]
    """

    def __init__(self, dim=180, num_heads=6, head_dim=30,
                 window_size=16, shift_size=8,
                 num_blocks=6, ch_sq=22, sp_sq=11,
                 heads_per_attn=3, ffn_ratio=2,
                 layer_idx=0):
        super().__init__()
        blocks = []
        even_idx = 0  # Tracks position among even blocks (0,2,4)
        for b in range(num_blocks):
            if b % 2 == 0:
                # WinAttnBlock — even block
                is_shifted = (layer_idx + even_idx) % 2 == 1
                ss = shift_size if is_shifted else 0
                blocks.append(WinAttnBlock(
                    dim=dim, num_heads=num_heads, head_dim=head_dim,
                    window_size=window_size, shift_size=ss,
                    ch_sq=ch_sq, sp_sq=sp_sq,
                    heads_per_attn=heads_per_attn, ffn_ratio=ffn_ratio,
                ))
                even_idx += 1
            else:
                # ChanAttnBlock — odd block
                blocks.append(ChanAttnBlock(
                    dim=dim, num_heads=num_heads, head_dim=head_dim,
                    ch_sq=ch_sq, sp_sq=sp_sq, ffn_ratio=ffn_ratio,
                ))
        self.blocks = nn.ModuleList(blocks)
        # Residual conv at the end of the group (applied in spatial domain)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, H, W):
        """x: [B, L, C]"""
        B = x.shape[0]
        C = x.shape[-1]
        x_skip = x
        for block in self.blocks:
            x = block(x, H, W)
        # Apply residual conv in spatial domain, then add skip
        x_2d = x.transpose(1, 2).view(B, C, H, W)
        x_2d = self.conv(x_2d)
        x = x_2d.view(B, C, H * W).transpose(1, 2) + x_skip
        return x


class HAT(nn.Module):
    """
    Hybrid Attention Transformer for image super-resolution.
    Matches 4x-UltraSharpV2.pth state dict.
    """

    def __init__(
        self,
        in_ch: int = 3,
        embed_dim: int = 180,
        num_heads: int = 6,
        head_dim: int = 30,
        window_size: int = 16,
        shift_size: int = 8,
        num_layers: int = 6,
        num_blocks_per_layer: int = 6,
        ffn_ratio: int = 2,
        ch_sq: int = 22,
        sp_sq: int = 11,
        heads_per_attn: int = 3,
        upsample_mid_ch: int = 64,
        scale: int = 4,
    ):
        super().__init__()
        self.window_size = window_size

        # Initial feature extraction
        self.conv_first = nn.Conv2d(in_ch, embed_dim, 3, 1, 1)

        # before_RG: Sequential(Identity @ 0, LayerNorm @ 1)
        self.before_RG = nn.Sequential(
            nn.Identity(),
            nn.LayerNorm(embed_dim),
        )

        # Residual groups (= "layers")
        self.layers = nn.ModuleList([
            ResidualGroup(
                dim=embed_dim, num_heads=num_heads, head_dim=head_dim,
                window_size=window_size, shift_size=shift_size,
                num_blocks=num_blocks_per_layer,
                ch_sq=ch_sq, sp_sq=sp_sq,
                heads_per_attn=heads_per_attn, ffn_ratio=ffn_ratio,
                layer_idx=i,
            )
            for i in range(num_layers)
        ])

        # Top-level LayerNorm (applied after all layers)
        self.norm = nn.LayerNorm(embed_dim)

        # Reconstruction head
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, upsample_mid_ch, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )
        # Two PixelShuffle 2x stages = 4x total
        # Each: Conv2d(mid, mid*4) + PixelShuffle(2)  → mid channels out
        self.upsample = nn.Sequential(
            nn.Conv2d(upsample_mid_ch, upsample_mid_ch * 4, 3, 1, 1),  # 0
            nn.PixelShuffle(2),                                          # 1
            nn.Conv2d(upsample_mid_ch, upsample_mid_ch * 4, 3, 1, 1),  # 2
            nn.PixelShuffle(2),                                          # 3
        )
        self.conv_last = nn.Conv2d(upsample_mid_ch, in_ch, 3, 1, 1)

    def forward(self, x):
        """x: [B, 3, H, W] — H,W must be multiples of window_size (16)."""
        B, C, H, W = x.shape

        # Feature extraction
        x = self.conv_first(x)      # [B, embed_dim, H, W]
        x_skip = x

        # Flatten to sequence [B, L, C]
        x = x.flatten(2).transpose(1, 2)   # [B, H*W, embed_dim]

        # before_RG: Identity then LayerNorm
        x = self.before_RG(x)

        # Residual groups
        for layer in self.layers:
            x = layer(x, H, W)

        # Top-level norm
        x = self.norm(x)

        # Back to spatial [B, embed_dim, H, W]
        x = x.transpose(1, 2).view(B, -1, H, W)

        # Long skip connection
        x = self.conv_after_body(x) + x_skip

        # Upsample
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)

        return x


# ---------------------------------------------------------------------------
# Inference engine with optional tiling
# ---------------------------------------------------------------------------

def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


class RealESRGANer:
    def __init__(
        self,
        model: nn.Module,
        model_path: str,
        scale: int,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        half: bool = True,
        device: torch.device = None,
        verbose: bool = False,
        strict: bool = True,
        window_size: int = 1,
        compile_model: bool = True,
    ):
        self.scale = scale
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.half = half
        self.verbose = verbose
        self.window_size = window_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vram_per_pixel = 0  # bytes/pixel estimate for auto-tile; set by build_upsampler

        if verbose:
            print(f"  Loading checkpoint: {model_path}")
        ckpt = _safe_load_checkpoint(model_path)
        weights = ckpt.get("params_ema") or ckpt.get("params") or ckpt
        if isinstance(weights, dict) and next(iter(weights), "").startswith("model."):
            weights = _remap_esrgan_keys(weights)
        model.load_state_dict(weights, strict=strict)
        model.eval()
        if half:
            model = model.half()
        self.model = model.to(self.device)
        if verbose:
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  Parameters: {params:.1f}M  |  dtype: {'FP16' if half else 'FP32'}")
        if compile_model and self.device.type == "cuda" and hasattr(torch, "compile"):
            try:
                if _triton_available():
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    vprint("  Compile  : inductor  (first frame slow — JIT compiling)", verbose)
                else:
                    # cudagraphs records the CUDA kernel sequence on frame 1 and replays it.
                    # No Triton needed; good speedup for fixed-shape video inference.
                    self.model = torch.compile(self.model, backend="cudagraphs")
                    vprint("  Compile  : cudagraphs  (Triton not found)", verbose)
            except Exception as e:
                vprint(f"  Compile  : skipped ({e})", verbose)

    def _to_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img_bgr.astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return t.half() if self.half else t

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        out = t.squeeze(0).float()
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
        out = out.clamp(0, 1).permute(1, 2, 0)
        return (out.cpu().numpy() * 255.0).round().astype(np.uint8)

    @torch.no_grad()
    def _infer(self, img_bgr: np.ndarray) -> np.ndarray:
        inp = self._to_tensor(img_bgr)
        pad_h = pad_w = 0
        if self.window_size > 1:
            h, w = inp.shape[2], inp.shape[3]
            pad_h = (self.window_size - h % self.window_size) % self.window_size
            pad_w = (self.window_size - w % self.window_size) % self.window_size
            if pad_h > 0 or pad_w > 0:
                inp = F.pad(inp, (0, pad_w, 0, pad_h), mode="reflect")
        out = self.model(inp)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :out.shape[2] - pad_h * self.scale, :out.shape[3] - pad_w * self.scale]
        return self._to_numpy(out)

    @torch.no_grad()
    def _tiled_infer(self, img_bgr: np.ndarray, tile: int = None) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        tile = tile if tile is not None else self.tile
        pad = self.tile_pad
        scale = self.scale

        out_h = h * scale
        out_w = w * scale
        output = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        tiles_x = math.ceil(w / tile)
        tiles_y = math.ceil(h / tile)

        for iy in range(tiles_y):
            for ix in range(tiles_x):
                x0 = max(ix * tile - pad, 0)
                y0 = max(iy * tile - pad, 0)
                x1 = min((ix + 1) * tile + pad, w)
                y1 = min((iy + 1) * tile + pad, h)

                tile_in = img_bgr[y0:y1, x0:x1]
                tile_out = self._infer(tile_in)

                ox0 = (x0 if ix == 0 else ix * tile) * scale
                oy0 = (y0 if iy == 0 else iy * tile) * scale
                ox1 = min((ix + 1) * tile * scale, out_w)
                oy1 = min((iy + 1) * tile * scale, out_h)

                cx0 = (0 if ix == 0 else pad) * scale
                cy0 = (0 if iy == 0 else pad) * scale
                cx1 = cx0 + (ox1 - ox0)
                cy1 = cy0 + (oy1 - oy0)

                output[oy0:oy1, ox0:ox1] = tile_out[cy0:cy1, cx0:cx1]

        return output

    def _auto_tile(self, h: int, w: int) -> int:
        """Return a safe tile size based on actual free VRAM, or 0 for full-frame."""
        if self.tile > 0:
            return self.tile
        if self.device.type != "cuda":
            return 0

        h_eff = h + 2 * self.pre_pad
        w_eff = w + 2 * self.pre_pad

        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        HEADROOM = 512 * 1024 ** 2  # keep 512 MB for CUDA/driver overhead
        avail_bytes = max(free_bytes - HEADROOM, 0)

        if self.window_size > 1:
            # HAT peak = attn_mask [nW, heads, L, L] (FP32) + feature map activations.
            # The attn_mask formula (3×nW×ws⁴×4) underestimates by ~10× because it ignores
            # Q/K/V projections, DWConv buffers, upsample intermediates, etc.
            # 12× safety factor brings the estimate in line with observed OOM behaviour.
            ws = self.window_size
            nW = math.ceil(h_eff / ws) * math.ceil(w_eff / ws)
            peak_bytes = 3 * nW * ws ** 4 * 4 * 12
            if peak_bytes <= avail_bytes:
                return 0
            for mult in (48, 40, 32, 24, 16, 12, 8):
                tile = ws * mult
                nW_t = math.ceil(tile / ws) ** 2
                if 3 * nW_t * ws ** 4 * 4 * 12 <= avail_bytes // 2:
                    print(
                        f"  [auto-tile] {w_eff}×{h_eff}: ~{peak_bytes/1024**3:.1f} GB needed, "
                        f"{free_bytes/1024**3:.1f} GB free → tile={tile}",
                        flush=True,
                    )
                    return tile
            tile = ws * 8
            print(
                f"  [auto-tile] Low VRAM ({free_bytes/1024**3:.1f} GB free) → minimum tile={tile}",
                flush=True,
            )
            return tile

        elif self._vram_per_pixel > 0:
            # RRDBNet / SRVGGNet: empirical bytes-per-pixel estimate.
            peak_bytes = h_eff * w_eff * self._vram_per_pixel
            if peak_bytes <= avail_bytes:
                return 0
            safe_pixels = max(avail_bytes // 2 // self._vram_per_pixel, 128 * 128)
            tile_raw = int(math.sqrt(safe_pixels)) - 2 * self.tile_pad
            tile = max(tile_raw // 64 * 64, 128)
            print(
                f"  [auto-tile] {w_eff}×{h_eff}: ~{peak_bytes/1024**3:.1f} GB needed, "
                f"{free_bytes/1024**3:.1f} GB free → tile={tile}",
                flush=True,
            )
            return tile

        return 0

    @torch.no_grad()
    def enhance(self, img_rgb: np.ndarray, outscale: float = None) -> np.ndarray:
        """
        img_rgb: numpy H×W×3 uint8 RGB  →  returns numpy H×W×3 uint8 RGB
        """
        outscale = outscale or self.scale
        h, w = img_rgb.shape[:2]

        img_bgr = img_rgb[:, :, ::-1].copy()
        img_bgr_orig = img_bgr  # keep original (pre-pad) for dark-area blend

        if self.pre_pad > 0:
            img_bgr = np.pad(
                img_bgr,
                ((self.pre_pad, self.pre_pad), (self.pre_pad, self.pre_pad), (0, 0)),
                mode="reflect",
            )

        tile = self._auto_tile(h, w)
        output_bgr = self._tiled_infer(img_bgr, tile=tile) if tile > 0 else self._infer(img_bgr)

        if self.pre_pad > 0:
            ps = self.pre_pad * self.scale
            output_bgr = output_bgr[ps:-ps, ps:-ps]

        # HAT activation-divergence fix (dark-area rainbow correction).
        # For near-zero input, learned biases dominate the signal; LayerNorm amplifies them
        # to std=1; each residual group compounds by ~2.5× → ±900 K at layer 6 for pure
        # black → bright rainbow colour on what should be dark fabric/shadow.
        # The divergence is observable up to ~brightness 100; a simple linear blend from 0
        # to some threshold leaves too much wrong model colour in the 60-100 range.
        #
        # Two-stage blend:
        #   brightness ≤ SOLID_THR → 100% bicubic  (hard replace — model definitely wrong)
        #   SOLID_THR < brightness < DARK_THR → linear taper from 100% to 0% bicubic
        #   brightness ≥ DARK_THR → 100% model output  (model correct for bright areas)
        if self.window_size > 1:
            SOLID_THR = 100  # hard cutoff: pure bicubic below this value
            DARK_THR  = 180  # model takes over completely above this value
            brightness = img_bgr_orig.astype(np.float32).max(axis=2)  # [h, w]
            bic_w = np.where(
                brightness <= SOLID_THR,
                1.0,
                np.clip(1.0 - (brightness - SOLID_THR) / (DARK_THR - SOLID_THR), 0.0, 1.0),
            ).astype(np.float32)
            if bic_w.max() > 0:
                bic_rgb = Image.fromarray(img_bgr_orig[:, :, ::-1])  # BGR→RGB for PIL
                bic_rgb = bic_rgb.resize((w * self.scale, h * self.scale), Image.BICUBIC)
                bic_bgr = np.array(bic_rgb)[:, :, ::-1].astype(np.float32)
                # Bilinear-upsample blend weight via F.interpolate (smooth edges, no PIL mode-F)
                bic_w_t = torch.from_numpy(bic_w).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                bic_w_big = (
                    F.interpolate(bic_w_t, scale_factor=float(self.scale),
                                  mode='bilinear', align_corners=False)
                    .squeeze().numpy()[:, :, None]  # [H*scale, W*scale, 1]
                )
                output_bgr = np.clip(
                    output_bgr.astype(np.float32) * (1.0 - bic_w_big) + bic_bgr * bic_w_big,
                    0, 255,
                ).astype(np.uint8)

        if abs(outscale - self.scale) > 1e-3:
            out_h = round(h * outscale)
            out_w = round(w * outscale)
            pil_out = Image.fromarray(output_bgr[:, :, ::-1])
            pil_out = pil_out.resize((out_w, out_h), Image.LANCZOS)
            return np.array(pil_out)

        return output_bgr[:, :, ::-1].copy()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def vprint(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, flush=True)


def run(cmd: list, desc: str = None, verbose: bool = False) -> None:
    if desc:
        print(f"  {desc}...", flush=True)
    if verbose:
        print(f"  $ {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\nCommand failed:\n  {' '.join(cmd)}", file=sys.stderr)
        if not verbose and hasattr(result, "stderr"):
            print(result.stderr, file=sys.stderr)
        sys.exit(1)


def probe_video(path: str, verbose: bool = False) -> tuple:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(r.stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = [s for s in data["streams"] if s["codec_type"] == "audio"]
    has_audio = bool(audio)

    num, den = video.get("r_frame_rate", "30/1").split("/")
    fps = float(num) / float(den)

    if verbose:
        print(f"  Codec      : {video.get('codec_name', '?')}  {video.get('profile', '')}")
        print(f"  Pixel fmt  : {video.get('pix_fmt', '?')}")
        print(f"  Duration   : {float(video.get('duration', 0)):.2f}s")
        if audio:
            ac = audio[0]
            print(f"  Audio      : {ac.get('codec_name','?')}  {ac.get('sample_rate','?')} Hz  "
                  f"{ac.get('channel_layout', str(ac.get('channels','?')) + 'ch')}")

    return fps, int(video["width"]), int(video["height"]), has_audio


def download_model(info: dict, verbose: bool = False) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODELS_DIR / info["filename"]

    if not info.get("url"):
        if dst.exists():
            size_mb = dst.stat().st_size / 1024 / 1024
            print(f"  Custom model: {dst.name}  ({size_mb:.0f} MB)")
            return dst
        sys.exit(f"Error: custom model file not found: {dst}\n  Place the .pth file in {MODELS_DIR}")

    if dst.exists():
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  Model cached: {dst.name}  ({size_mb:.0f} MB)")
        return dst

    url = info["url"]
    print(f"  Downloading {info['filename']} …")
    if verbose:
        print(f"  URL: {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "UpscalyVid/1.0", "Accept": "*/*"})
    tmp = dst.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = min(100, downloaded * 100 // total)
                        done = pct // 5
                        bar = "█" * done + "░" * (20 - done)
                        mb = downloaded / 1024 / 1024
                        total_mb = total / 1024 / 1024
                        print(f"  [{bar}] {pct:3d}%  {mb:.1f}/{total_mb:.1f} MB", end="\r", flush=True)
        tmp.rename(dst)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        print(f"\n  Download failed: {e}", file=sys.stderr)
        print(f"  Try finding the model manually at https://openmodeldb.info", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Saved: {dst}  ({dst.stat().st_size/1024/1024:.0f} MB)")
    return dst


def build_upsampler(
    model_key: str,
    model_path: Path,
    tile: int,
    tile_pad: int,
    half: bool,
    device: torch.device,
    verbose: bool = False,
    compile_model: bool = True,
) -> RealESRGANer:
    info = MODELS[model_key]

    if info.get("arch") == "auto":
        try:
            ckpt = _safe_load_checkpoint(str(model_path))
            weights = ckpt.get("params_ema") or ckpt.get("params") or ckpt
            if not isinstance(weights, dict):
                raise ValueError("no recognisable weight dict in checkpoint")
            detected = _detect_arch_from_weights(weights)
        except Exception as e:
            sys.exit(f"Error: cannot probe custom model {model_path.name}: {e}")

        if detected["arch"] == "unsupported":
            note = detected.get("note", "unknown architecture")
            sys.exit(
                f"Error: {model_path.name} uses an unsupported architecture ({note}).\n"
                f"  UpscalyVid supports RRDBNet, SRVGGNet, and HAT-based models.\n"
                f"  For SwinIR, SPAN, and other architectures use chaiNNer or ComfyUI."
            )

        info = {**info, **detected}
        if verbose:
            print(f"  Detected arch  : {detected['arch'].upper()}")
            print(f"  Detected scale : {detected['scale']}x")
            print(f"  Detected blocks: {detected.get('num_block', detected.get('num_conv', '?'))}")

    if verbose:
        print(f"  Architecture : {info['arch'].upper()}")
        print(f"  Scale        : {info['scale']}x")

    if info.get("arch") == "srvgg":
        net = SRVGGNetCompact(
            num_feat=info["num_feat"],
            num_conv=info["num_conv"],
            upscale=info["scale"],
        )
    elif info.get("arch") == "hat":
        net = HAT(
            embed_dim=info["embed_dim"],
            num_heads=info["num_heads"],
            head_dim=info["head_dim"],
            window_size=info["window_size"],
            shift_size=info["shift_size"],
            num_layers=info["num_layers"],
            num_blocks_per_layer=info["num_blocks_per_layer"],
            ffn_ratio=info["ffn_ratio"],
            ch_sq=info["ch_sq"],
            sp_sq=info["sp_sq"],
            heads_per_attn=info["heads_per_attn"],
            upsample_mid_ch=info["upsample_mid_ch"],
            scale=info["scale"],
        )
    else:
        net = RRDBNet(scale=info["scale"], num_block=info["num_block"])

    is_hat = info.get("arch") == "hat"
    # HAT overflows in fp16 after 5+ residual layers of accumulation — force fp32
    hat_half = False
    if is_hat and half:
        vprint("  Note: HAT runs in FP32 (fp16 overflows in deep transformer layers)", verbose)
        hat_half = False
    use_half = hat_half if is_hat else half
    # HAT tiles vary in H/W per tile (window-size padding differs per tile).
    # cudagraphs records a graph from the first tile shape and replays it on all others,
    # producing wrong shift masks and rainbow boundary artifacts. Disable compile for HAT.
    upsampler = RealESRGANer(
        model=net,
        model_path=str(model_path),
        scale=info["scale"],
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=10,
        half=use_half,
        device=device,
        verbose=verbose,
        strict=not is_hat,
        window_size=info["window_size"] if is_hat else 1,
        compile_model=compile_model and not is_hat,
    )
    # Empirical peak-VRAM estimates for auto-tile (bytes per input pixel).
    # RRDBNet-23 FP16 ≈ 3.2 GB at 720p (921 600 px) → ~3 500 B/px.
    # SRVGGNet FP16 is roughly 3× lighter.
    arch = info.get("arch")
    if arch == "rrdbnet":
        upsampler._vram_per_pixel = 3500 if use_half else 7000
    elif arch == "srvgg":
        upsampler._vram_per_pixel = 1200 if use_half else 2400
    # HAT uses the window-size formula in _auto_tile; _vram_per_pixel stays 0.
    return upsampler


def suggest_tile(
    model_key: str,
    half: bool,
    device: torch.device,
    ref_h: int = 1080,
    ref_w: int = 1920,
) -> int:
    """
    Estimate a safe tile size from current *free* VRAM for a reference 1080p frame.
    Returns 0 if full-frame inference fits comfortably (no tiling needed).
    Uses the same formulas as RealESRGANer._auto_tile so the suggestion is accurate.
    """
    if device.type != "cuda":
        return 0
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return 0

    HEADROOM = 512 * 1024 ** 2
    avail = max(free_bytes - HEADROOM, 0)
    pre_pad = 10
    h_eff = ref_h + 2 * pre_pad
    w_eff = ref_w + 2 * pre_pad

    info = MODELS.get(model_key, {})
    arch = info.get("arch", "rrdbnet")

    if arch == "hat":
        ws = info.get("window_size", 16)
        nW = math.ceil(h_eff / ws) * math.ceil(w_eff / ws)
        if 3 * nW * ws ** 4 * 4 * 12 <= avail:
            return 0
        for mult in (48, 40, 32, 24, 16, 12, 8):
            tile = ws * mult
            nW_t = math.ceil(tile / ws) ** 2
            if 3 * nW_t * ws ** 4 * 4 * 12 <= avail // 2:
                return tile
        return ws * 8

    vpp = (1200 if arch == "srvgg" else 3500) if half else (2400 if arch == "srvgg" else 7000)
    if h_eff * w_eff * vpp <= avail:
        return 0
    safe_px  = max(avail // 2 // vpp, 128 * 128)
    tile_raw = int(math.sqrt(safe_px)) - 2 * pre_pad
    return max(tile_raw // 64 * 64, 128)


def upscale_frames(
    frames_dir: Path,
    out_dir: Path,
    upsampler: RealESRGANer,
    outscale: float,
    ext: str,
    verbose: bool = False,
    sample_dir: "Path | None" = None,
) -> None:
    frames = sorted(frames_dir.glob(f"*.{ext}"))
    total = len(frames)
    t_start = time.perf_counter()

    def _read(fp):
        return np.array(Image.open(fp).convert("RGB"))

    def _write(arr, path):
        Image.fromarray(arr).save(path)

    # Overlap disk I/O with GPU compute:
    #   read(N+1) and write(N-1) happen in background threads while the GPU runs frame N.
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending_read  = executor.submit(_read, frames[0]) if total > 0 else None
        pending_write = None

        for i, fp in enumerate(frames):
            t_frame = time.perf_counter()

            img = pending_read.result()
            if i + 1 < total:
                pending_read = executor.submit(_read, frames[i + 1])

            result = upsampler.enhance(img, outscale=outscale)

            if pending_write is not None:
                pending_write.result()
            pending_write = executor.submit(_write, result, out_dir / fp.name)

            elapsed = time.perf_counter() - t_start
            frame_time = time.perf_counter() - t_frame
            speed = (i + 1) / elapsed
            eta = (total - i - 1) / speed if speed > 0 else 0
            pct = (i + 1) / total * 100
            done = int(pct) // 5
            bar = "█" * done + "░" * (20 - done)

            if verbose:
                vram_str = ""
                if torch.cuda.is_available():
                    used = torch.cuda.memory_allocated() / 1024**3
                    peak = torch.cuda.max_memory_allocated() / 1024**3
                    vram_str = f"  VRAM {used:.1f}/{peak:.1f}GB"
                print(
                    f"  [{bar}] {pct:5.1f}%  {i+1}/{total}  "
                    f"{frame_time:.2f}s/frame  ETA {eta:.0f}s{vram_str}",
                    end="\r", flush=True,
                )
                if i == 0:
                    _sd = sample_dir or out_dir.parent
                    sample_path = _sd / "sample_frame.png"
                    source_path = _sd / "source_frame.png"
                    Image.fromarray(result).save(sample_path)
                    Image.fromarray(img).save(source_path)
                    print(f"\n  [verbose] Sample frame : {sample_path}", flush=True)
                    print(f"  [verbose] Source frame : {source_path}", flush=True)
            else:
                print(f"  [{bar}] {pct:5.1f}%  {i+1}/{total} frames", end="\r", flush=True)

        if pending_write is not None:
            pending_write.result()

    total_time = time.perf_counter() - t_start
    print(f"\n  {total} frames in {total_time:.1f}s  ({total/total_time:.2f} fps avg)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="upscalyvid",
        description="UpscalyVid — AI video upscaler using Real-ESRGAN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Models:",
            *[f"  {k:35s} {v['description']}" for k, v in MODELS.items()],
            "",
            "Examples:",
            "  upscalyvid.py video.mp4 output.mp4",
            "  upscalyvid.py video.mp4 output.mp4 -m realesrgan-x4plus-anime",
            "  upscalyvid.py video.mp4 output.mp4 -m 4x-UltraSharp -v",
            "  upscalyvid.py video.mp4 output.mp4 -m realesrgan-x2plus --codec hevc_nvenc",
            "  upscalyvid.py /path/to/folder/                     (batch — outputs to folder/upscaled/)",
            "  upscalyvid.py /path/to/folder/ /path/to/out/       (batch — custom output dir)",
        ]),
    )
    p.add_argument("input",  nargs="?", default=None,
                   help="Input video file or directory (batch mode)")
    p.add_argument("output", nargs="?", default=None,
                   help="Output file (single) or directory (batch — default: {input}/upscaled/)")
    p.add_argument(
        "-m", "--model",
        default="realesrgan-x4plus",
        choices=list(MODELS.keys()),
        metavar="MODEL",
        help="Model name (default: realesrgan-x4plus)",
    )
    p.add_argument("--outscale", type=float, default=None,
                   help="Final output scale (default: model native scale)")
    p.add_argument("--codec", default="h264",
                   choices=["h264", "h265", "h264_nvenc", "hevc_nvenc"],
                   help="Output video codec (default: h264)")
    p.add_argument("-q", "--quality", type=int, default=18,
                   help="CRF for h264/h265 or CQ for NVENC — 0=lossless, 18=near-lossless (default: 18)")
    p.add_argument("--tile", type=int, default=0,
                   help="Tile size px for VRAM-limited GPUs (0=auto, default: 0)")
    p.add_argument("--tile-pad", type=int, default=10,
                   help="Tile overlap padding (default: 10)")
    p.add_argument("--frame-format", default="png", choices=["png", "jpg"],
                   help="Intermediate frame format (default: png)")
    p.add_argument("--work-dir", default=None,
                   help="Custom temp directory for frames — lets the GUI watch for preview frames")
    p.add_argument("--cpu", action="store_true", help="Force CPU mode (very slow)")
    p.add_argument("--fp32", action="store_true", help="Use FP32 instead of FP16")
    p.add_argument("--keep-frames", action="store_true", help="Retain temp frame directories")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose output: ffmpeg logs, per-frame timing, VRAM usage")
    p.add_argument("--no-compile", action="store_true",
                   help="Disable torch.compile (faster startup, slower per-frame)")
    p.add_argument("--list-models", action="store_true", help="List available models and exit")
    return p.parse_args()


CODEC_ARGS = {
    "h264":       ["-c:v", "libx264",    "-preset", "slow"],
    "h265":       ["-c:v", "libx265",    "-preset", "slow"],
    "h264_nvenc": ["-c:v", "h264_nvenc", "-preset", "p7", "-rc", "vbr"],
    "hevc_nvenc": ["-c:v", "hevc_nvenc", "-preset", "p7", "-rc", "vbr"],
}

QUALITY_ARG = {
    "h264":       "-crf",
    "h265":       "-crf",
    "h264_nvenc": "-cq",
    "hevc_nvenc": "-cq",
}


def main() -> None:
    _cleanup_stale_temp()
    args = parse_args()
    v = args.verbose

    if args.list_models:
        print("\nAvailable models:\n")
        for name, info in MODELS.items():
            cached = (MODELS_DIR / info["filename"]).exists()
            tag = "[cached]" if cached else "[not downloaded]"
            tags = ", ".join(info.get("tags", []))
            print(f"  {name}  {tag}")
            print(f"    {info['scale']}x  |  {info['description']}")
            print(f"    Tags: {tags}\n")
        return

    if not args.input:
        sys.exit("Error: input path required.\nRun with --help for usage.")

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        sys.exit(f"Error: input not found: {in_path}")

    # Batch mode when input is a directory
    if in_path.is_dir():
        video_files = sorted(p for p in in_path.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
        if not video_files:
            sys.exit(f"Error: no video files found in {in_path}\n"
                     f"Supported: {', '.join(sorted(_VIDEO_EXTS))}")
        out_dir = Path(args.output).resolve() if args.output else (in_path / "upscaled")
        out_dir.mkdir(parents=True, exist_ok=True)
        file_pairs = [(vf, out_dir / f"{vf.stem}_upscaled.mp4") for vf in video_files]
        log_parent = in_path
    else:
        if not args.output:
            sys.exit("Error: output path required for single-file mode.\nRun with --help for usage.")
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        file_pairs = [(in_path, out_path)]
        log_parent = in_path.parent

    model_info = MODELS[args.model]
    outscale = args.outscale or model_info["scale"]
    ext = args.frame_format

    # Open log file — captures all stdout from this point on (including cancellation)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_parent / f"upscalyvid_{stamp}.log"
    tee = _TeeLogger(log_path)

    upsampler = None
    try:
        # Header
        print(f"\n{'='*62}")
        print(f"  UpscalyVid — AI Video Upscaler")
        print(f"{'='*62}")
        if in_path.is_dir():
            print(f"  Input   : {in_path}/  ({len(file_pairs)} videos)")
            print(f"  Output  : {out_dir}/")
        else:
            print(f"  Input   : {in_path.name}")
            print(f"  Output  : {file_pairs[0][1].name}")
        print(f"  Model   : {args.model}  ({model_info['scale']}x native)")
        print(f"  Compile : {'no' if args.no_compile else 'yes (first frame slow)'}")
        print(f"  Verbose : {'yes' if v else 'no'}")
        print(f"  Log     : {log_path}")

        # Device
        if args.cpu or not torch.cuda.is_available():
            device = torch.device("cpu")
            half = False
        else:
            device = torch.device("cuda")
            half = not args.fp32
            # TF32 gives ~2× free speedup on FP32 matmul (sm80+) with negligible precision loss.
            torch.set_float32_matmul_precision("high")

        print(f"\n[Device]")
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / 1024**3
            print(f"  GPU  : {props.name}  ({vram:.1f} GB VRAM)")
            suggested = suggest_tile(args.model, half, device)
            if suggested == 0:
                print(f"  Tile : full frame fits in free VRAM (no tiling needed)")
            else:
                print(f"  Tile : suggested {suggested}px based on free VRAM")
            if v:
                sm = props.major * 10 + props.minor
                sdpa_backend = (
                    "Flash Attention (sm80+)" if sm >= 80
                    else "Memory-Efficient Attention (sm70+)" if sm >= 70
                    else "Math fallback"
                )
                print(f"  CUDA : {torch.version.cuda}")
                print(f"  SM   : {props.major}.{props.minor}  ({sdpa_backend})")
        else:
            print(f"  CPU  (FP32)")

        # Model — downloaded and built once for all files
        print(f"\n[Model]")
        model_path = download_model(model_info, verbose=v)
        upsampler = build_upsampler(args.model, model_path, args.tile, args.tile_pad, half, device,
                                    verbose=v, compile_model=not args.no_compile)
        print(f"  Mode  : {'FP16' if upsampler.half else 'FP32'}")
        print(f"  Ready.")

        # Per-file loop
        for file_idx, (input_path, output_path) in enumerate(file_pairs):
            is_batch = len(file_pairs) > 1
            if is_batch:
                print(f"\n{'─'*62}")
                print(f"  File {file_idx + 1}/{len(file_pairs)}: {input_path.name}")
                print(f"{'─'*62}")

            # Probe
            print(f"\n[Probe]")
            fps, w, h, has_audio = probe_video(str(input_path), verbose=v)
            out_w = round(w * outscale)
            out_h = round(h * outscale)
            print(f"  Resolution : {w}×{h}  →  {out_w}×{out_h}  ({outscale}x)")
            print(f"  FPS        : {fps:.4f}")
            print(f"  Audio      : {'yes' if has_audio else 'no'}")

            # Work dir: honour --work-dir only for single-file mode (GUI preview)
            if args.work_dir and not is_batch:
                work_dir = Path(args.work_dir)
                work_dir.mkdir(parents=True, exist_ok=True)
                own_work_dir = False
            else:
                work_dir = Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX))
                own_work_dir = True

            frames_dir   = work_dir / "frames"
            upscaled_dir = work_dir / "upscaled"
            frames_dir.mkdir(exist_ok=True)
            upscaled_dir.mkdir(exist_ok=True)

            # Announce upscaled dir so the GUI can watch it for preview frames
            print(f"  UPSCALED_DIR={upscaled_dir}", flush=True)

            ffmpeg_loglevel = [] if v else ["-loglevel", "error"]

            try:
                # 1 — extract frames
                print(f"\n[1/3] Extracting frames …")
                q_args = (["-q:v", "2"] if ext == "jpg" else [])
                run(
                    ["ffmpeg", *ffmpeg_loglevel, "-i", str(input_path), *q_args,
                     str(frames_dir / f"frame_%08d.{ext}"), "-y"],
                    desc="ffmpeg frame extraction",
                    verbose=v,
                )
                frame_count = len(list(frames_dir.glob(f"*.{ext}")))
                print(f"  {frame_count} frames extracted")

                # 2 — upscale
                sample_dir = input_path.parent if v else None
                print(f"\n[2/3] Upscaling {frame_count} frames …")
                try:
                    upscale_frames(frames_dir, upscaled_dir, upsampler, outscale, ext,
                                   verbose=v, sample_dir=sample_dir)
                except KeyboardInterrupt:
                    print("\n\nCancelled by user.")
                    raise

                print(f"  Done.")

                # 3 — assemble
                print(f"\n[3/3] Assembling video …")
                codec_args = CODEC_ARGS[args.codec]
                q_flag = QUALITY_ARG[args.codec]

                encode_cmd = [
                    "ffmpeg", *ffmpeg_loglevel,
                    "-framerate", f"{fps}",
                    "-i", str(upscaled_dir / f"frame_%08d.{ext}"),
                    *codec_args, q_flag, str(args.quality),
                    "-pix_fmt", "yuv420p",
                ]

                if has_audio:
                    tmp_vid = work_dir / "video_only.mp4"
                    run([*encode_cmd, str(tmp_vid), "-y"], "encoding upscaled frames", verbose=v)
                    run(
                        ["ffmpeg", *ffmpeg_loglevel,
                         "-i", str(tmp_vid),
                         "-i", str(input_path),
                         "-map", "0:v:0", "-map", "1:a?",
                         "-c:v", "copy", "-c:a", "copy", "-shortest",
                         str(output_path), "-y"],
                        "muxing audio",
                        verbose=v,
                    )
                else:
                    run([*encode_cmd, str(output_path), "-y"], "encoding upscaled frames", verbose=v)

                size_mb = output_path.stat().st_size / 1024 / 1024
                print(f"\n  Done!  →  {output_path}")
                print(f"  Output size : {size_mb:.1f} MB")

            finally:
                if args.keep_frames:
                    print(f"Frames kept at: {work_dir}")
                elif own_work_dir:
                    shutil.rmtree(work_dir, ignore_errors=True)

        # Summary
        print(f"\n{'='*62}")
        if len(file_pairs) > 1:
            print(f"  Batch complete — {len(file_pairs)} files processed.")
        else:
            print(f"  Complete.")
        print(f"{'='*62}\n")

    except KeyboardInterrupt:
        sys.exit(130)

    finally:
        if upsampler is not None:
            try:
                upsampler.model = None
                del upsampler
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tee.close()


if __name__ == "__main__":
    main()
