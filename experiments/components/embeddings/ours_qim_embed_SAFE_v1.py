#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ours_qim_embed_SAFE_v1.py — Ours-QIM hybrid masking + cover-conditioned dither/cost-adaptive embedding for BOSSBase.

This script adapts the structure of the uploaded S-UNIWARD embedding script for the
Ours-QIM variant discussed in the manuscript. It simulates the C3 stego-object
embedding stage offline; it does not require a physical channel.

Pipeline per cover image and payload bpp:
  1) L = round(payload_bpp * H * W)
  2) m <- random payload bits of length L
  3) gamma1, gamma2 <- keyed/generated cover-stream bits, expanded/truncated to L
  4) rho_dit <- Dither(k_stego, psi(o)) as a cover-conditioned bitstream of length L
  5) b_qim = m XOR gamma1 XOR gamma2 XOR rho_dit
  6) Modulate the base adaptive cost map using rho_i^ours = rho_i^base * g(psi_i(o), rho_dit_i)
  7) Embed b_qim with pySTC or the adaptive-cost proxy using the modulated costs.

Important note on STC:
  The uploaded S-UNIWARD script is a simulator rather than a true STC implementation.
  This script therefore implements an STC-compatible, adaptive-cost embedding proxy
  and logs stc_mode='cost_adaptive_lsb_matching_proxy'. If you have a compiled STC
  library/binary, this script can be extended to call it at the function
  embed_cost_adaptive_lsb_matching(). Logs include embedding_efficiency and
  coding_loss_proxy to support publication tables/plots.

Outputs:
  out-root/
    ours_qim_0.100/*.pgm
    ours_qim_0.200/*.pgm
    ours_qim_0.400/*.pgm
    logs/embed_log_ours_qim.csv
    logs/embed_log_ours_qim.jsonl
    logs/embed_summary_ours_qim.csv

Author: generated for Hybrid Multichannel Steganography evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import math
import os
import subprocess
import tempfile
import resource
import socket
import sys
import time
import uuid
import getpass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import pywt
except Exception as exc:
    pywt = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import psutil
except Exception:
    psutil = None

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:
    _ssim = None

try:
    import torch
except Exception:
    torch = None

try:
    import pystc
except Exception:
    pystc = None

# ------------------------------- Basic utilities -------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def read_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    return np.array(im, dtype=np.uint8)


def write_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def mse_psnr_ssim(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    diff = x.astype(np.float64) - y.astype(np.float64)
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse <= 0 else float(10.0 * math.log10((255.0 * 255.0) / mse))
    if _ssim is not None:
        try:
            ssim = float(_ssim(x, y, data_range=255))
        except Exception:
            ssim = float("nan")
    else:
        ssim = float("nan")
    return mse, psnr, ssim


def process_rss_mb() -> float:
    if psutil is not None:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        except Exception:
            pass
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def peak_rss_mb() -> float:
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return process_rss_mb()


def system_ram_info() -> Dict[str, float]:
    if psutil is None:
        return {"ram_total_mb": float("nan"), "ram_available_mb": float("nan"), "ram_used_mb": float("nan")}
    try:
        vm = psutil.virtual_memory()
        return {
            "ram_total_mb": vm.total / (1024 ** 2),
            "ram_available_mb": vm.available / (1024 ** 2),
            "ram_used_mb": vm.used / (1024 ** 2),
        }
    except Exception:
        return {"ram_total_mb": float("nan"), "ram_available_mb": float("nan"), "ram_used_mb": float("nan")}


def cpu_times() -> Dict[str, float]:
    if psutil is None:
        return {"cpu_user_time_sec": float("nan"), "cpu_system_time_sec": float("nan")}
    try:
        t = psutil.Process(os.getpid()).cpu_times()
        return {"cpu_user_time_sec": float(t.user), "cpu_system_time_sec": float(t.system)}
    except Exception:
        return {"cpu_user_time_sec": float("nan"), "cpu_system_time_sec": float("nan")}


def gpu_info() -> Dict[str, object]:
    out = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_mem_allocated_mb": 0.0,
        "gpu_mem_reserved_mb": 0.0,
    }
    if torch is None:
        return out
    try:
        if torch.cuda.is_available():
            out["gpu_available"] = True
            idx = torch.cuda.current_device()
            out["gpu_name"] = torch.cuda.get_device_name(idx)
            out["gpu_mem_allocated_mb"] = float(torch.cuda.memory_allocated(idx) / (1024 ** 2))
            out["gpu_mem_reserved_mb"] = float(torch.cuda.memory_reserved(idx) / (1024 ** 2))
    except Exception:
        pass
    return out


def append_csv(path: Path, header: List[str], row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def append_jsonl(path: Path, obj: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=convert, separators=(",", ":")) + "\n")


# ------------------------------- Bit generation -------------------------------

def bytes_to_bits(data: bytes) -> np.ndarray:
    if len(data) == 0:
        return np.zeros(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> Tuple[bytes, int]:
    """Pack bits into bytes and return (bytes, number_of_zero_padding_bits)."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    pad = (-int(bits.size)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes(), int(pad)


def hmac_bytes(key: bytes, msg: bytes, nbytes: int) -> bytes:
    """Expand HMAC-SHA256(key,msg||counter) to nbytes."""
    out = bytearray()
    ctr = 0
    while len(out) < nbytes:
        out.extend(hmac.new(key, msg + ctr.to_bytes(8, "big"), hashlib.sha256).digest())
        ctr += 1
    return bytes(out[:nbytes])


def hmac_bits(key: bytes, msg: bytes, nbits: int) -> np.ndarray:
    nbytes = (nbits + 7) // 8
    bits = bytes_to_bits(hmac_bytes(key, msg, nbytes))
    return bits[:nbits]


def load_cover_stream_bits(path: Path) -> np.ndarray:
    """Read a text/bytes file and convert to raw UTF-8/byte bits."""
    data = path.read_bytes()
    return bytes_to_bits(data)


def expand_stream_bits(base_bits: np.ndarray, nbits: int, key: bytes, label: str) -> np.ndarray:
    """
    Deterministically expand a cover stream to nbits.
    If the file has enough bits, repeat+truncate with a keyed rotation so different
    images do not always use the same prefix. If not, append HMAC expansion.
    """
    if nbits <= 0:
        return np.zeros(0, dtype=np.uint8)
    if base_bits.size == 0:
        return hmac_bits(key, f"empty-gamma|{label}".encode(), nbits)

    digest = hmac.new(key, f"rotate|{label}".encode(), hashlib.sha256).digest()
    offset = int.from_bytes(digest[:8], "big") % int(base_bits.size)
    rotated = np.concatenate([base_bits[offset:], base_bits[:offset]])
    if rotated.size >= nbits:
        return rotated[:nbits].astype(np.uint8)

    reps = int(math.ceil(nbits / rotated.size))
    tiled = np.tile(rotated, reps)[:nbits]
    # Mix with HMAC expansion to avoid simple periodicity when repeated.
    pad = hmac_bits(key, f"gamma-expand|{label}".encode(), nbits)
    return np.bitwise_xor(tiled.astype(np.uint8), pad.astype(np.uint8))


# ------------------------------- Cost maps -------------------------------

def _resize_to(arr: np.ndarray, shape_hw: Tuple[int, int]) -> np.ndarray:
    Ht, Wt = shape_hw
    H, W = arr.shape
    out = np.zeros((Ht, Wt), dtype=arr.dtype)
    hs, ws = min(H, Ht), min(W, Wt)
    h0, w0 = (H - hs) // 2, (W - ws) // 2
    ho, wo = (Ht - hs) // 2, (Wt - ws) // 2
    out[ho:ho + hs, wo:wo + ws] = arr[h0:h0 + hs, w0:w0 + ws]
    return out


def suniward_like_cost_map(img_u8: np.ndarray, wavelet: str = "db8", levels: int = 2,
                           eps: float = 1e-3, mode: str = "periodization") -> np.ndarray:
    """
    S-UNIWARD-like spatial cost map adapted from uploaded script.
    If PyWavelets is unavailable, fall back to a local-variance adaptive cost.
    """
    if pywt is None:
        var = local_variance_map(img_u8)
        rho = 1.0 / (eps + var)
        rho = np.nan_to_num(rho, nan=np.nanmedian(rho), posinf=np.nanmax(rho[np.isfinite(rho)]), neginf=1.0)
        lo, hi = np.percentile(rho, [0.1, 99.9])
        return np.clip(rho, lo, hi).astype(np.float32)
    img = img_u8.astype(np.float32)
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=levels, mode=mode)
    details = coeffs[1:]
    H, W = img.shape
    acc = np.zeros_like(img, dtype=np.float32)

    for lev, (cH, cV, cD) in enumerate(details):
        a = np.zeros_like(cH)
        for band in range(3):
            bh = cH if band == 0 else np.zeros_like(cH)
            bv = cV if band == 1 else np.zeros_like(cV)
            bd = cD if band == 2 else np.zeros_like(cD)
            tmp = pywt.idwt2((a, (bh, bv, bd)), wavelet=wavelet, mode=mode)
            remaining = len(details) - lev - 1
            for _ in range(remaining):
                z = np.zeros_like(tmp)
                tmp = pywt.idwt2((np.zeros_like(tmp), (z, z, z)), wavelet=wavelet, mode=mode)
            if tmp.shape != (H, W):
                tmp = _resize_to(tmp, (H, W))
            acc += np.abs(tmp.astype(np.float32))

    rho = 1.0 / (eps + acc)
    # Avoid pathological extremes; keep finite positive costs.
    rho = np.nan_to_num(rho, nan=np.nanmedian(rho), posinf=np.nanmax(rho[np.isfinite(rho)]), neginf=1.0)
    lo, hi = np.percentile(rho, [0.1, 99.9])
    rho = np.clip(rho, lo, hi).astype(np.float32)
    return rho


def local_variance_map(img_u8: np.ndarray) -> np.ndarray:
    """3x3 local variance without scipy; sufficient for logging high-variance hit rate."""
    x = img_u8.astype(np.float32)
    pads = np.pad(x, ((1, 1), (1, 1)), mode="reflect")
    vals = []
    for dy in range(3):
        for dx in range(3):
            vals.append(pads[dy:dy + x.shape[0], dx:dx + x.shape[1]])
    stack = np.stack(vals, axis=0)
    return np.var(stack, axis=0).astype(np.float32)


# ------------------------------- Embedding -------------------------------

def keyed_noise(shape: Tuple[int, int], key: bytes, label: str) -> np.ndarray:
    n = shape[0] * shape[1]
    raw = hmac_bytes(key, f"noise|{label}".encode(), n * 8)
    vals = np.frombuffer(raw, dtype=">u8").astype(np.float64)
    vals = vals / float(2 ** 64 - 1)
    return vals.reshape(shape)


def robust_normalize_feature_map(x: np.ndarray, p_low: float = 5.0, p_high: float = 95.0) -> np.ndarray:
    """Robustly normalise a public cover feature map psi(o) to [0,1]."""
    y = x.astype(np.float64)
    lo, hi = np.percentile(y, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(y)), float(np.max(y))
    if hi <= lo:
        return np.zeros_like(y, dtype=np.float32)
    y = (y - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def cover_feature_digest(psi_norm: np.ndarray) -> bytes:
    """Stable digest of quantised public cover features for cover-conditioned dither."""
    q = np.clip(np.rint(psi_norm.astype(np.float64) * 255.0), 0, 255).astype(np.uint8)
    return hashlib.sha256(np.ascontiguousarray(q).tobytes()).digest()


def dither_uniform_map(shape: Tuple[int, int], key: bytes, label: str, psi_digest: bytes) -> np.ndarray:
    """Keyed cover-conditioned dither map U(0,1), conditioned on psi(o)."""
    n = shape[0] * shape[1]
    raw = hmac_bytes(key, b"rho-dit-map|" + label.encode() + b"|" + psi_digest, n * 8)
    vals = np.frombuffer(raw, dtype=">u8").astype(np.float64) / float(2 ** 64 - 1)
    return vals.reshape(shape).astype(np.float32)


def dither_bits(key: bytes, label: str, psi_digest: bytes, nbits: int) -> np.ndarray:
    """Keyed cover-conditioned dither bitstream rho_dit for b_qim."""
    return hmac_bits(key, b"rho-dit-bits|" + label.encode() + b"|" + psi_digest, nbits)


def qim_modulated_cost_map(
    rho_base: np.ndarray,
    psi_norm: np.ndarray,
    rho_dit_u: np.ndarray,
    texture_discount: float,
    smooth_penalty: float,
    dither_strength: float,
    min_factor: float,
    max_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implement rho_i^ours = rho_i^base * g(psi_i(o), rho_dit_i).

    The default g reduces costs in textured/high-variance regions and increases
    costs in smooth regions, while the keyed cover-conditioned dither introduces
    a small deterministic perturbation to avoid identical cost ordering.
    """
    psi = np.clip(psi_norm.astype(np.float64), 0.0, 1.0)
    dit = np.clip(rho_dit_u.astype(np.float64), 0.0, 1.0)
    g_texture = 1.0 + float(smooth_penalty) * (1.0 - psi) - float(texture_discount) * psi
    g_dither = 1.0 + float(dither_strength) * (dit - 0.5)
    g = np.clip(g_texture * g_dither, float(min_factor), float(max_factor))
    rho = rho_base.astype(np.float64) * g
    rho = np.nan_to_num(rho, nan=np.nanmedian(rho), posinf=np.nanmax(rho[np.isfinite(rho)]), neginf=1.0)
    lo, hi = np.percentile(rho, [0.1, 99.9])
    rho = np.clip(rho, lo, hi).astype(np.float32)
    return rho, g.astype(np.float32)

def sha256_bits(bits: np.ndarray) -> str:
    """SHA-256 digest of a bit array, packed into bytes for stable logging."""
    b, _ = bits_to_bytes(np.asarray(bits, dtype=np.uint8).reshape(-1))
    return hashlib.sha256(b).hexdigest()


def safe_mean(x) -> float:
    x = np.asarray(x)
    return float(np.mean(x)) if x.size else float("nan")


def safe_median(x) -> float:
    x = np.asarray(x)
    return float(np.median(x)) if x.size else float("nan")


def safe_sum(x) -> float:
    x = np.asarray(x)
    return float(np.sum(x)) if x.size else float("nan")


def compute_qim_base_stats(
    rho_base: np.ndarray,
    rho_modulated: np.ndarray,
    qim_g: np.ndarray,
    rho_dit_bits: np.ndarray,
    args,
) -> Dict[str, object]:
    """
    Compute QIM-specific statistics that are independent of the final modified mask.
    These fields explain how cover-conditioned dither reshaped the adaptive cost map.
    """
    eps = 1e-12
    ratio = rho_modulated.astype(np.float64) / np.maximum(rho_base.astype(np.float64), eps)
    ratio = ratio[np.isfinite(ratio)]

    return {
        "qim_enabled": True,
        "qim_psi_type": "robust_normalized_local_variance",
        "qim_dither_source": "HMAC-SHA256(k_stego, psi(o))",
        "qim_dither_sha256": sha256_bits(rho_dit_bits),
        "qim_dither_length": int(rho_dit_bits.size),
        "qim_dither_ones_ratio": float(np.mean(rho_dit_bits)) if rho_dit_bits.size else float("nan"),

        "qim_base_cost_mean": float(np.mean(rho_base)),
        "qim_base_cost_median": float(np.median(rho_base)),
        "qim_base_cost_sum": float(np.sum(rho_base)),

        "qim_modulated_cost_mean": float(np.mean(rho_modulated)),
        "qim_modulated_cost_median": float(np.median(rho_modulated)),
        "qim_modulated_cost_sum": float(np.sum(rho_modulated)),

        "qim_cost_ratio_mean": float(np.mean(ratio)) if ratio.size else float("nan"),
        "qim_cost_ratio_median": float(np.median(ratio)) if ratio.size else float("nan"),
        "qim_cost_ratio_min": float(np.min(ratio)) if ratio.size else float("nan"),
        "qim_cost_ratio_max": float(np.max(ratio)) if ratio.size else float("nan"),

        "qim_texture_alpha": float(args.qim_texture_discount),
        "qim_dither_alpha": float(args.qim_dither_strength),

        "qim_g_mean": float(np.mean(qim_g)),
        "qim_g_median": float(np.median(qim_g)),
        "qim_g_min": float(np.min(qim_g)),
        "qim_g_max": float(np.max(qim_g)),

        # Filled after embedding, once modified_mask is known.
        "qim_modified_dither_one_pct": "",
        "qim_modified_high_texture_pct": "",
    }


def add_qim_modified_stats(
    qim_stats: Dict[str, object],
    modified_mask: np.ndarray,
    rho_dit_u: np.ndarray,
    psi_norm: np.ndarray,
    high_texture_threshold: float = 0.75,
) -> Dict[str, object]:
    """
    Add QIM statistics that depend on the final modified pixels.
    """
    num_modified = int(np.count_nonzero(modified_mask))
    if num_modified <= 0:
        qim_stats["qim_modified_dither_one_pct"] = 0.0
        qim_stats["qim_modified_high_texture_pct"] = 0.0
        return qim_stats

    dither_one_map = rho_dit_u >= 0.5
    high_texture_map = psi_norm >= high_texture_threshold

    qim_stats["qim_modified_dither_one_pct"] = float(
        np.mean(dither_one_map[modified_mask]) * 100.0
    )
    qim_stats["qim_modified_high_texture_pct"] = float(
        np.mean(high_texture_map[modified_mask]) * 100.0
    )
    return qim_stats


def empty_qim_stats() -> Dict[str, object]:
    """
    Default QIM fields for failure rows where QIM setup did not complete.
    """
    return {
        "qim_enabled": True,
        "qim_psi_type": "robust_normalized_local_variance",
        "qim_dither_source": "HMAC-SHA256(k_stego, psi(o))",
        "qim_dither_sha256": "",
        "qim_dither_length": "",
        "qim_dither_ones_ratio": "",
        "qim_base_cost_mean": "",
        "qim_base_cost_median": "",
        "qim_base_cost_sum": "",
        "qim_modulated_cost_mean": "",
        "qim_modulated_cost_median": "",
        "qim_modulated_cost_sum": "",
        "qim_cost_ratio_mean": "",
        "qim_cost_ratio_median": "",
        "qim_cost_ratio_min": "",
        "qim_cost_ratio_max": "",
        "qim_texture_alpha": "",
        "qim_dither_alpha": "",
        "qim_g_mean": "",
        "qim_g_median": "",
        "qim_g_min": "",
        "qim_g_max": "",
        "qim_modified_dither_one_pct": "",
        "qim_modified_high_texture_pct": "",
    }


def embed_cost_adaptive_lsb_matching(
    cover: np.ndarray,
    payload_bits: np.ndarray,
    rho: np.ndarray,
    key: bytes,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Adaptive-cost LSB matching proxy for STC-based embedding.
    Selects L low-cost pixels under a keyed cost tie-breaker, writes payload bits
    to LSBs, and uses ±1 matching with edge guards.

    Returns stego, selected_mask, modified_mask, stats.
    """
    H, W = cover.shape
    N = H * W
    L = int(payload_bits.size)
    if L > N:
        raise ValueError(f"payload length L={L} exceeds image capacity N={N}; reduce bpp or add coding layer")

    rho_flat = rho.reshape(-1).astype(np.float64)
    # Keyed tie-breaker to avoid always selecting the identical set for equal costs.
    noise = keyed_noise((H, W), key, label).reshape(-1)
    score = rho_flat + 1e-9 * noise
    selected_idx = np.argpartition(score, L - 1)[:L] if L > 0 else np.array([], dtype=np.int64)
    # Stable order for extraction.
    selected_idx = selected_idx[np.argsort(score[selected_idx])]

    stego = cover.copy().reshape(-1)
    cover_flat = cover.reshape(-1)
    current_lsb = (stego[selected_idx] & 1).astype(np.uint8)
    need_change = current_lsb != payload_bits
    change_idx = selected_idx[need_change]

    # ±1 LSB matching with deterministic keyed sign.
    sign_bits = hmac_bits(key, f"sign|{label}".encode(), max(1, len(change_idx)))[:len(change_idx)]
    for j, idx in enumerate(change_idx):
        val = int(stego[idx])
        if val <= 0:
            stego[idx] = 1
        elif val >= 255:
            stego[idx] = 254
        else:
            stego[idx] = val + (1 if sign_bits[j] == 1 else -1)

    stego2 = stego.reshape(H, W).astype(np.uint8)
    selected_mask = np.zeros(N, dtype=bool)
    selected_mask[selected_idx] = True
    selected_mask = selected_mask.reshape(H, W)
    modified_mask = (cover != stego2)

    flips = int(np.count_nonzero(modified_mask))
    change_rate = flips / float(N)
    payload_bpp_actual = L / float(N)
    embedding_eff = (L / flips) if flips > 0 else float("inf")

    actual_distortion = float(np.sum(rho[modified_mask])) if flips > 0 else 0.0
    if flips > 0:
        lower = float(np.sum(np.partition(rho_flat, flips - 1)[:flips]))
        coding_loss = (actual_distortion / lower - 1.0) if lower > 0 else float("nan")
    else:
        lower = 0.0
        coding_loss = 0.0

    stats = {
        "payload_bits": L,
        "capacity_pixels": N,
        "num_selected_pixels": int(selected_mask.sum()),
        "num_modified_pixels": flips,
        "change_rate": change_rate,
        "actual_payload_bpp": payload_bpp_actual,
        "embedding_efficiency": embedding_eff,
        "actual_distortion": actual_distortion,
        "oracle_min_distortion_same_flips": lower,
        "coding_loss": coding_loss,
        "stc_used": False,
        "stc_mode": "cost_adaptive_lsb_matching_proxy",
    }
    return stego2, selected_mask, modified_mask, stats


def extract_lsb(stego: np.ndarray, selected_mask: np.ndarray, rho: np.ndarray, key: bytes, label: str, L: int) -> np.ndarray:
    """Extraction mirrors sorted selected positions; selected_mask is logged in memory only."""
    # Recompute selected order from rho+key to avoid saving mask.
    H, W = stego.shape
    rho_flat = rho.reshape(-1).astype(np.float64)
    noise = keyed_noise((H, W), key, label).reshape(-1)
    score = rho_flat + 1e-9 * noise
    idx = np.argpartition(score, L - 1)[:L] if L > 0 else np.array([], dtype=np.int64)
    idx = idx[np.argsort(score[idx])]
    return (stego.reshape(-1)[idx] & 1).astype(np.uint8)


def pystc_seed_from_key(key: bytes, label: str) -> int:
    """Derive a deterministic 31-bit seed accepted by pySTC."""
    digest = hmac.new(key, f"pystc-seed|{label}".encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def run_pystc_direct(
    cover: np.ndarray,
    payload_bits: np.ndarray,
    rho: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Run pySTC directly in the current Python process. This is fast but unsafe if the C++ layer aborts."""
    if pystc is None:
        raise RuntimeError("pySTC is not available; install with: pip install pystcstego")
    msg_bytes, padded = bits_to_bytes(payload_bits)
    t0 = time.perf_counter()
    stego = pystc.hide(msg_bytes, cover.astype(np.uint8), rho.astype(np.float64), rho.astype(np.float64), int(seed), mx=255, mn=0)
    t1 = time.perf_counter()
    recovered = pystc.unhide(stego.astype(np.uint8), int(seed))
    t2 = time.perf_counter()
    extracted_bits = bytes_to_bits(bytes(recovered))[:payload_bits.size]
    if extracted_bits.size < payload_bits.size:
        extracted_bits = np.pad(extracted_bits, (0, payload_bits.size - extracted_bits.size), constant_values=0)
    stats = {
        "stc_used": True,
        "stc_mode": "pystc_direct",
        "stc_backend_actual": "pystc",
        "stc_seed": int(seed),
        "stc_message_bytes": len(msg_bytes),
        "stc_padded_bits": int(padded),
        "stc_safe_subprocess": False,
        "worker_returncode": 0,
        "worker_stderr": "",
        "embed_time_sec_backend": float(t1 - t0),
        "extract_time_sec_backend": float(t2 - t1),
    }
    return stego.astype(np.uint8), extracted_bits.astype(np.uint8), stats


def run_pystc_subprocess(
    cover: np.ndarray,
    payload_bits: np.ndarray,
    rho: np.ndarray,
    seed: int,
    timeout_sec: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Run pySTC in a child Python process so C++ aborts do not kill the main experiment."""
    msg_bytes, padded = bits_to_bytes(payload_bits)
    with tempfile.TemporaryDirectory(prefix="ours_qim_pystc_") as td:
        td_path = Path(td)
        inp = td_path / "worker_in.npz"
        outp = td_path / "worker_out.npz"
        np.savez_compressed(
            inp,
            cover=cover.astype(np.uint8),
            rho=rho.astype(np.float64),
            msg_bytes=np.frombuffer(msg_bytes, dtype=np.uint8),
            seed=np.array([int(seed)], dtype=np.int64),
            L=np.array([int(payload_bits.size)], dtype=np.int64),
        )
        cmd = [sys.executable, str(Path(__file__).resolve()), "--_pystc-worker", "--worker-in", str(inp), "--worker-out", str(outp)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec)
        stderr_tail = (proc.stderr or "")[-2000:]
        if proc.returncode != 0 or not outp.exists():
            raise RuntimeError(f"pySTC worker failed rc={proc.returncode}: {stderr_tail}")
        data = np.load(outp, allow_pickle=False)
        stego = data["stego"].astype(np.uint8)
        extracted_bits = data["extracted_bits"].astype(np.uint8)[:payload_bits.size]
        if extracted_bits.size < payload_bits.size:
            extracted_bits = np.pad(extracted_bits, (0, payload_bits.size - extracted_bits.size), constant_values=0)
        stats = {
            "stc_used": True,
            "stc_mode": "pystc_safe_subprocess",
            "stc_backend_actual": "pystc",
            "stc_seed": int(seed),
            "stc_message_bytes": len(msg_bytes),
            "stc_padded_bits": int(padded),
            "stc_safe_subprocess": True,
            "worker_returncode": int(proc.returncode),
            "worker_stderr": stderr_tail,
            "embed_time_sec_backend": float(data["embed_time_sec"][0]),
            "extract_time_sec_backend": float(data["extract_time_sec"][0]),
        }
        return stego, extracted_bits, stats


def pystc_worker_main(worker_in: str, worker_out: str) -> int:
    """Hidden worker entry point used by --stc-safe-subprocess."""
    import pystc as _pystc
    data = np.load(worker_in, allow_pickle=False)
    cover = data["cover"].astype(np.uint8)
    rho = data["rho"].astype(np.float64)
    msg = data["msg_bytes"].astype(np.uint8).tobytes()
    seed = int(data["seed"][0])
    L = int(data["L"][0])
    t0 = time.perf_counter()
    stego = _pystc.hide(msg, cover, rho, rho, seed, mx=255, mn=0)
    t1 = time.perf_counter()
    recovered = _pystc.unhide(stego.astype(np.uint8), seed)
    t2 = time.perf_counter()
    extracted_bits = bytes_to_bits(bytes(recovered))[:L]
    np.savez_compressed(
        worker_out,
        stego=stego.astype(np.uint8),
        extracted_bits=extracted_bits.astype(np.uint8),
        embed_time_sec=np.array([float(t1 - t0)], dtype=np.float64),
        extract_time_sec=np.array([float(t2 - t1)], dtype=np.float64),
    )
    return 0


def embed_with_backend(
    cover: np.ndarray,
    payload_bits: np.ndarray,
    rho: np.ndarray,
    key: bytes,
    label: str,
    args,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Dispatch embedding to pySTC or the adaptive-cost proxy."""
    requested = args.stc_backend
    seed = pystc_seed_from_key(key, label)
    pystc_available = pystc is not None

    def proxy(mode_suffix: str = "cost_adaptive_lsb_matching_proxy"):
        stego, selected_mask, modified_mask, estats = embed_cost_adaptive_lsb_matching(cover, payload_bits, rho, key, label)
        extracted = extract_lsb(stego, selected_mask, rho, key, label, int(payload_bits.size))
        estats.update({
            "stc_backend_requested": requested,
            "stc_backend_actual": "proxy",
            "pystc_available": bool(pystc_available),
            "stc_seed": int(seed),
            "stc_message_bytes": int((payload_bits.size + 7) // 8),
            "stc_padded_bits": int((-payload_bits.size) % 8),
            "stc_safe_subprocess": bool(args.stc_safe_subprocess),
            "worker_returncode": "",
            "worker_stderr": "",
            "stc_mode": mode_suffix,
        })
        return stego, extracted, selected_mask, modified_mask, estats

    if requested == "proxy":
        return proxy()

    if requested in {"pystc", "auto"}:
        if not pystc_available:
            if requested == "auto":
                return proxy("proxy_pystc_unavailable")
            raise RuntimeError("--stc-backend pystc requested, but pySTC is unavailable. Install with: pip install pystcstego")
        try:
            if args.stc_safe_subprocess:
                stego, extracted, stc_stats = run_pystc_subprocess(cover, payload_bits, rho, seed, args.stc_timeout_sec)
            else:
                stego, extracted, stc_stats = run_pystc_direct(cover, payload_bits, rho, seed)
        except Exception as exc:
            if requested == "auto":
                stego, extracted, selected_mask, modified_mask, estats = proxy("proxy_after_pystc_failure")
                estats["error"] = f"pySTC failed; used proxy fallback: {type(exc).__name__}: {exc}"
                return stego, extracted, selected_mask, modified_mask, estats
            raise
        modified_mask = (cover != stego)
        selected_mask = modified_mask.copy()
        flips = int(np.count_nonzero(modified_mask))
        N = int(cover.size)
        L = int(payload_bits.size)
        actual_distortion = float(np.sum(rho[modified_mask])) if flips else 0.0
        rho_flat = rho.reshape(-1).astype(np.float64)
        if flips > 0:
            lower = float(np.sum(np.partition(rho_flat, flips - 1)[:flips]))
            coding_loss = (actual_distortion / lower - 1.0) if lower > 0 else float("nan")
        else:
            lower = 0.0
            coding_loss = 0.0
        estats = {
            "payload_bits": L,
            "capacity_pixels": N,
            "num_selected_pixels": L,
            "num_modified_pixels": flips,
            "change_rate": flips / float(N),
            "actual_payload_bpp": L / float(N),
            "embedding_efficiency": (L / flips) if flips > 0 else float("inf"),
            "actual_distortion": actual_distortion,
            "oracle_min_distortion_same_flips": lower,
            "coding_loss": coding_loss,
            "stc_backend_requested": requested,
            "pystc_available": bool(pystc_available),
            **stc_stats,
        }
        return stego, extracted, selected_mask, modified_mask, estats

    raise ValueError(f"Unknown stc backend: {requested}")


# ------------------------------- Dataset discovery -------------------------------

def discover_images(root: Path, max_images: Optional[int] = None) -> List[Path]:
    exts = {".pgm", ".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
    paths = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    paths.sort(key=lambda p: str(p))
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]
    return paths


def iter_progress(items: List[Tuple[Path, float]], progress: str):
    if tqdm is not None and progress in {"auto", "bar"}:
        yield from tqdm(items, desc="Ours-QIM embedding", unit="job")
    else:
        total = len(items)
        step = max(1, total // 20)
        for i, item in enumerate(items, 1):
            if progress != "none" and (i == 1 or i % step == 0 or i == total):
                print(f"[Ours-QIM embedding] {100*i/total:5.1f}% ({i}/{total})", flush=True)
            yield item


# ------------------------------- Logging schema -------------------------------

CSV_HEADER = [
    "run_id", "timestamp", "image_id", "method", "payload_bpp", "payload_length_L",
    "cover_path", "stego_path", "cover_sha256", "stego_sha256",
    "height", "width", "num_pixels", "num_selected_pixels", "num_modified_pixels", "change_rate",
    "actual_payload_bpp", "embedding_efficiency", "coding_loss", "stc_used", "stc_mode",
    "stc_backend_requested", "stc_backend_actual", "pystc_available", "stc_seed",
    "stc_message_bytes", "stc_padded_bits", "stc_safe_subprocess", "worker_returncode", "worker_stderr",
    "mse", "psnr_db", "ssim",
    "mean_cost", "median_cost", "sum_cost", "mean_cost_modified", "modified_in_high_variance_pct",
    "embed_time_sec", "extract_time_sec", "success_extract", "ber",
    "cpu_user_time_sec", "cpu_system_time_sec", "process_rss_mb", "peak_rss_mb",
    "ram_total_mb", "ram_used_mb", "ram_available_mb",
    "gpu_available", "gpu_name", "gpu_mem_allocated_mb", "gpu_mem_reserved_mb",
    "gamma1_path", "gamma2_path", "gamma1_sha256", "gamma2_sha256",

    # QIM-specific audit fields
    "qim_enabled",
    "qim_psi_type",
    "qim_dither_source",
    "qim_dither_sha256",
    "qim_dither_length",
    "qim_dither_ones_ratio",
    "qim_base_cost_mean",
    "qim_base_cost_median",
    "qim_base_cost_sum",
    "qim_modulated_cost_mean",
    "qim_modulated_cost_median",
    "qim_modulated_cost_sum",
    "qim_cost_ratio_mean",
    "qim_cost_ratio_median",
    "qim_cost_ratio_min",
    "qim_cost_ratio_max",
    "qim_texture_alpha",
    "qim_dither_alpha",
    "qim_g_mean",
    "qim_g_median",
    "qim_g_min",
    "qim_g_max",
    "qim_modified_dither_one_pct",
    "qim_modified_high_texture_pct",

    "seed", "algo_params", "ok", "error"
]

SUMMARY_HEADER = [
    "run_id", "method", "payload_bpp", "n", "fail_rate", "mean_embed_time_sec", "mean_extract_time_sec",
    "mean_ber", "success_rate", "mean_change_rate", "mean_embedding_efficiency", "mean_coding_loss",
    "mean_mse", "median_psnr_db", "mean_ssim", "out_dir"
]


# ------------------------------- Main processing -------------------------------

def process_one(
    cover_path: Path,
    payload_bpp: float,
    args,
    run_id: str,
    gamma1_bits_base: np.ndarray,
    gamma2_bits_base: np.ndarray,
    master_key: bytes,
    logs_dir: Path,
) -> Dict[str, object]:
    row: Dict[str, object] = {}
    image_id = cover_path.stem
    method = "Ours-QIM"
    out_dir = Path(args.out_root) / f"ours_qim_{payload_bpp:.3f}"
    stego_path = out_dir / cover_path.name

    t_embed0 = time.perf_counter()
    cpu0 = cpu_times()
    err = ""
    ok = 1

    try:
        cover = read_gray(cover_path)
        H, W = cover.shape
        N = int(H * W)
        L = int(round(payload_bpp * N))
        if L <= 0:
            raise ValueError("payload length is zero; increase payload_bpp")

        # Per-image/payload keys/labels.
        label = f"{image_id}|{payload_bpp:.3f}|{args.seed}" 
        payload_key = hmac.new(master_key, f"payload|{label}".encode(), hashlib.sha256).digest()
        gamma_key = hmac.new(master_key, f"gamma|{label}".encode(), hashlib.sha256).digest()
        kstego_key = hmac.new(master_key, f"k-stego|{label}".encode(), hashlib.sha256).digest()

        m_bits = hmac_bits(payload_key, b"m", L) if args.deterministic_payload else np.random.default_rng(args.seed).integers(0, 2, L, dtype=np.uint8)
        g1_bits = expand_stream_bits(gamma1_bits_base, L, gamma_key, "gamma1")
        g2_bits = expand_stream_bits(gamma2_bits_base, L, gamma_key, "gamma2")

        rho_base = suniward_like_cost_map(cover, args.wavelet, args.levels, args.eps, args.wavelet_mode)
        var_map = local_variance_map(cover)
        psi_norm = robust_normalize_feature_map(var_map)
        psi_digest = cover_feature_digest(psi_norm)
        rho_dit_bits = dither_bits(kstego_key, label, psi_digest, L)
        rho_dit_u = dither_uniform_map(cover.shape, kstego_key, label, psi_digest)
        b_bits = np.bitwise_xor(np.bitwise_xor(m_bits, g1_bits), np.bitwise_xor(g2_bits, rho_dit_bits)).astype(np.uint8)

        rho, qim_g = qim_modulated_cost_map(
            rho_base, psi_norm, rho_dit_u,
            texture_discount=args.qim_texture_discount,
            smooth_penalty=args.qim_smooth_penalty,
            dither_strength=args.qim_dither_strength,
            min_factor=args.qim_min_factor,
            max_factor=args.qim_max_factor,
        )

        # QIM-specific log fields before embedding.
        qim_stats = compute_qim_base_stats(
            rho_base=rho_base,
            rho_modulated=rho,
            qim_g=qim_g,
            rho_dit_bits=rho_dit_bits,
            args=args,
        )

        high_var_threshold = float(np.percentile(var_map, args.high_variance_percentile))
        high_var_mask = var_map >= high_var_threshold

        embed_key = hmac.new(master_key, f"embed-qim|{label}".encode(), hashlib.sha256).digest()
        t_backend0 = time.perf_counter()
        stego, extracted, selected_mask, modified_mask, estats = embed_with_backend(
            cover, b_bits, rho, embed_key, label, args
        )
        t_backend1 = time.perf_counter()
        write_gray(stego_path, stego)
        t_embed1 = time.perf_counter()

        # Backend routines perform extraction as part of verification. For pySTC safe
        # subprocesses, embed/extract timing comes from the worker; otherwise use the
        # measured total backend time.
        extract_time_backend = float(estats.get("extract_time_sec_backend", 0.0) or 0.0)
        if extract_time_backend <= 0.0:
            extract_time_backend = max(0.0, (t_backend1 - t_backend0) - float(estats.get("embed_time_sec_backend", 0.0) or 0.0))
        t_extract0 = time.perf_counter()
        t_extract1 = t_extract0 + extract_time_backend
        ber = float(np.mean(extracted != b_bits)) if L > 0 and extracted.size == b_bits.size else 1.0
        success_extract = int(ber == 0.0)

        mse, psnr, ssim_val = mse_psnr_ssim(cover, stego)
        cost_mod = rho[modified_mask]
        mean_cost_modified = float(np.mean(cost_mod)) if cost_mod.size else 0.0
        mod_high = float(np.mean(high_var_mask[modified_mask]) * 100.0) if np.count_nonzero(modified_mask) else 0.0
        # QIM-specific log fields after embedding.
        qim_stats = add_qim_modified_stats(
            qim_stats=qim_stats,
            modified_mask=modified_mask,
            rho_dit_u=rho_dit_u,
            psi_norm=psi_norm,
            high_texture_threshold=0.75,
        )

        cpu1 = cpu_times()
        ram = system_ram_info()
        gpu = gpu_info()

        row.update({
            "run_id": run_id,
            "timestamp": now_iso(),
            "image_id": image_id,
            "method": method,
            "payload_bpp": payload_bpp,
            "payload_length_L": L,
            "cover_path": str(cover_path),
            "stego_path": str(stego_path),
            "cover_sha256": sha256_file(cover_path),
            "stego_sha256": sha256_file(stego_path),
            "height": H,
            "width": W,
            "num_pixels": N,
            "num_selected_pixels": estats["num_selected_pixels"],
            "num_modified_pixels": estats["num_modified_pixels"],
            "change_rate": estats["change_rate"],
            "actual_payload_bpp": estats["actual_payload_bpp"],
            "embedding_efficiency": estats["embedding_efficiency"],
            "coding_loss": estats["coding_loss"],
            "stc_used": estats["stc_used"],
            "stc_mode": estats["stc_mode"],
            "stc_backend_requested": estats.get("stc_backend_requested", args.stc_backend),
            "stc_backend_actual": estats.get("stc_backend_actual", "proxy"),
            "pystc_available": estats.get("pystc_available", pystc is not None),
            "stc_seed": estats.get("stc_seed", ""),
            "stc_message_bytes": estats.get("stc_message_bytes", ""),
            "stc_padded_bits": estats.get("stc_padded_bits", ""),
            "stc_safe_subprocess": estats.get("stc_safe_subprocess", False),
            "worker_returncode": estats.get("worker_returncode", ""),
            "worker_stderr": estats.get("worker_stderr", ""),
            "mse": mse,
            "psnr_db": psnr,
            "ssim": ssim_val,
            "mean_cost": float(np.mean(rho)),
            "median_cost": float(np.median(rho)),
            "sum_cost": float(np.sum(rho)),
            "mean_cost_modified": mean_cost_modified,
            "modified_in_high_variance_pct": mod_high,
            "embed_time_sec": t_embed1 - t_embed0,
            "extract_time_sec": t_extract1 - t_extract0,
            "success_extract": success_extract,
            "ber": ber,
            "cpu_user_time_sec": cpu1["cpu_user_time_sec"],
            "cpu_system_time_sec": cpu1["cpu_system_time_sec"],
            "process_rss_mb": process_rss_mb(),
            "peak_rss_mb": peak_rss_mb(),
            **ram,
            **gpu,
            "gamma1_path": str(args.gamma1),
            "gamma2_path": str(args.gamma2),
            "gamma1_sha256": sha256_file(Path(args.gamma1)),
            "gamma2_sha256": sha256_file(Path(args.gamma2)),
            **qim_stats,
            "seed": args.seed,
            "algo_params": json.dumps({
                "mask": "m XOR gamma1 XOR gamma2 XOR rho_dit",
                "rho_dit": "Dither(k_stego, psi(o)) using HMAC-SHA256 over robust local-variance features",
                "cost_map": "rho_ours = rho_base * g(psi_i(o), rho_dit_i)",
                "base_cost_map": "suniward_like_wavelet",
                "psi": "robust_normalized_local_variance",
                "wavelet": args.wavelet,
                "levels": args.levels,
                "eps": args.eps,
                "wavelet_mode": args.wavelet_mode,
                "qim_texture_discount": args.qim_texture_discount,
                "qim_smooth_penalty": args.qim_smooth_penalty,
                "qim_dither_strength": args.qim_dither_strength,
                "qim_min_factor": args.qim_min_factor,
                "qim_max_factor": args.qim_max_factor,
                "high_variance_percentile": args.high_variance_percentile,
                "gamma_expansion": "file_bits_plus_hmac_expand",
                "payload_source": "hmac_deterministic" if args.deterministic_payload else "numpy_rng",
            }, separators=(",", ":")),
            "ok": ok,
            "error": err,
        })

    except Exception as exc:
        ok = 0
        err = f"{type(exc).__name__}: {exc}"
        cpu1 = cpu_times()
        ram = system_ram_info()
        gpu = gpu_info()
        # Preserve all metadata that was available before the backend failed.
        H_fail = locals().get("H", "")
        W_fail = locals().get("W", "")
        N_fail = locals().get("N", "")
        L_fail = locals().get("L", "")
        rho_fail = locals().get("rho", None)
        qim_stats_fail = locals().get("qim_stats", empty_qim_stats())
        embed_key_fail = locals().get("embed_key", b"")
        label_fail = locals().get("label", f"{image_id}|{payload_bpp:.3f}|{getattr(args, 'seed', '')}")
        try:
            stc_seed_fail = pystc_seed_from_key(embed_key_fail, label_fail) if embed_key_fail else ""
        except Exception:
            stc_seed_fail = ""
        mean_cost_fail = float(np.mean(rho_fail)) if isinstance(rho_fail, np.ndarray) else ""
        median_cost_fail = float(np.median(rho_fail)) if isinstance(rho_fail, np.ndarray) else ""
        sum_cost_fail = float(np.sum(rho_fail)) if isinstance(rho_fail, np.ndarray) else ""
        actual_payload_bpp_fail = (float(L_fail) / float(N_fail)) if isinstance(L_fail, int) and isinstance(N_fail, int) and N_fail else ""
        # Pull worker return information from the exception text when available.
        worker_rc_fail = ""
        worker_stderr_fail = ""
        err_text = str(exc)
        if "pySTC worker failed rc=" in err_text:
            try:
                worker_rc_fail = err_text.split("pySTC worker failed rc=", 1)[1].split(":", 1)[0].strip()
                worker_stderr_fail = err_text.split(":", 1)[1].strip()[-2000:]
            except Exception:
                worker_stderr_fail = err_text[-2000:]
        row.update({
            "run_id": run_id,
            "timestamp": now_iso(),
            "image_id": image_id,
            "method": method,
            "payload_bpp": payload_bpp,
            "payload_length_L": L_fail,
            "cover_path": str(cover_path),
            "stego_path": "",
            "cover_sha256": sha256_file(cover_path) if cover_path.exists() else "",
            "stego_sha256": "",
            "height": H_fail, "width": W_fail, "num_pixels": N_fail, "num_selected_pixels": "",
            "num_modified_pixels": "", "change_rate": "", "actual_payload_bpp": actual_payload_bpp_fail,
            "embedding_efficiency": "", "coding_loss": "", "stc_used": False,
            "stc_mode": "failed",
            "stc_backend_requested": getattr(args, "stc_backend", "proxy"),
            "stc_backend_actual": "failed",
            "pystc_available": pystc is not None,
            "stc_seed": stc_seed_fail,
            "stc_message_bytes": int((L_fail + 7)//8) if isinstance(L_fail, int) else "",
            "stc_padded_bits": int((-L_fail) % 8) if isinstance(L_fail, int) else "",
            "stc_safe_subprocess": getattr(args, "stc_safe_subprocess", False),
            "worker_returncode": worker_rc_fail,
            "worker_stderr": worker_stderr_fail,
            "mse": "", "psnr_db": "", "ssim": "",
            "mean_cost": mean_cost_fail, "median_cost": median_cost_fail, "sum_cost": sum_cost_fail, "mean_cost_modified": "",
            "modified_in_high_variance_pct": "",
            "embed_time_sec": time.perf_counter() - t_embed0,
            "extract_time_sec": "", "success_extract": 0, "ber": "",
            "cpu_user_time_sec": cpu1["cpu_user_time_sec"],
            "cpu_system_time_sec": cpu1["cpu_system_time_sec"],
            "process_rss_mb": process_rss_mb(),
            "peak_rss_mb": peak_rss_mb(),
            **ram, **gpu,
            "gamma1_path": str(args.gamma1),
            "gamma2_path": str(args.gamma2),
            "gamma1_sha256": sha256_file(Path(args.gamma1)) if Path(args.gamma1).exists() else "",
            "gamma2_sha256": sha256_file(Path(args.gamma2)) if Path(args.gamma2).exists() else "",
            **qim_stats_fail,
            "seed": args.seed,
            "algo_params": "{}",
            "ok": ok,
            "error": err,
        })

    append_csv(logs_dir / "embed_log_ours_qim.csv", CSV_HEADER, row)
    append_jsonl(logs_dir / "embed_log_ours_qim.jsonl", row)
    return row


def summarize(rows: List[Dict[str, object]], out_root: Path, run_id: str) -> None:
    logs = out_root / "logs"
    by_payload: Dict[float, List[Dict[str, object]]] = {}
    for r in rows:
        try:
            p = float(r["payload_bpp"])
        except Exception:
            continue
        by_payload.setdefault(p, []).append(r)

    def fvals(rs, key):
        vals = []
        for x in rs:
            try:
                v = float(x[key])
                if math.isfinite(v): vals.append(v)
            except Exception:
                pass
        return np.array(vals, dtype=float)

    for payload, rs in sorted(by_payload.items()):
        n = len(rs)
        ok = np.array([int(r.get("ok", 0)) for r in rs], dtype=int)
        row = {
            "run_id": run_id,
            "method": "Ours-QIM",
            "payload_bpp": payload,
            "n": n,
            "fail_rate": float(np.mean(ok == 0)) if n else float("nan"),
            "mean_embed_time_sec": float(np.mean(fvals(rs, "embed_time_sec"))) if fvals(rs, "embed_time_sec").size else float("nan"),
            "mean_extract_time_sec": float(np.mean(fvals(rs, "extract_time_sec"))) if fvals(rs, "extract_time_sec").size else float("nan"),
            "mean_ber": float(np.mean(fvals(rs, "ber"))) if fvals(rs, "ber").size else float("nan"),
            "success_rate": float(np.mean(fvals(rs, "success_extract"))) if fvals(rs, "success_extract").size else float("nan"),
            "mean_change_rate": float(np.mean(fvals(rs, "change_rate"))) if fvals(rs, "change_rate").size else float("nan"),
            "mean_embedding_efficiency": float(np.mean(fvals(rs, "embedding_efficiency"))) if fvals(rs, "embedding_efficiency").size else float("nan"),
            "mean_coding_loss": float(np.mean(fvals(rs, "coding_loss"))) if fvals(rs, "coding_loss").size else float("nan"),
            "mean_mse": float(np.mean(fvals(rs, "mse"))) if fvals(rs, "mse").size else float("nan"),
            "median_psnr_db": float(np.median(fvals(rs, "psnr_db"))) if fvals(rs, "psnr_db").size else float("nan"),
            "mean_ssim": float(np.mean(fvals(rs, "ssim"))) if fvals(rs, "ssim").size else float("nan"),
            "out_dir": str(out_root / f"ours_qim_{payload:.3f}"),
        }
        append_csv(logs / "embed_summary_ours_qim.csv", SUMMARY_HEADER, row)


# ------------------------------- CLI -------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Ours-QIM hybrid masking + cover-conditioned dither/cost-adaptive embedding for BOSSBase.")
    p.add_argument("--boss-root", default="/home/omego/Downloads/BOSSbase", help="BOSSBase root containing .pgm images")
    p.add_argument("--out-root", default=None, help="Output root for stego images and logs")
    p.add_argument("--gamma1", default="/home/omego/Documents/Hybrid_Stego_Eva/cover_parameter/hybrid-stego/gamma1_pride_selected_seed44.txt")
    p.add_argument("--gamma2", default="/home/omego/Documents/Hybrid_Stego_Eva/cover_parameter/hybrid-stego/gamma2_pride_selected_seed44.txt")
    p.add_argument("--payloads", default="0.1,0.2,0.4", help="Comma-separated payloads in bpp")
    p.add_argument("--max-images", type=int, default=0, help="Process only first N images for testing; 0 means all")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--master-secret", default="change-this-demo-secret", help="Used only to derive deterministic evaluation bits")
    p.add_argument("--deterministic-payload", action="store_true", help="Use HMAC-derived payload bits instead of numpy RNG")
    p.add_argument("--wavelet", default="db8")
    p.add_argument("--levels", type=int, default=2)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--wavelet-mode", default="periodization")
    p.add_argument("--high-variance-percentile", type=float, default=75.0)
    p.add_argument("--qim-texture-discount", type=float, default=0.30,
                   help="QIM cost modulation: discount applied to high-texture/high-variance regions.")
    p.add_argument("--qim-smooth-penalty", type=float, default=0.30,
                   help="QIM cost modulation: penalty applied to smooth/low-variance regions.")
    p.add_argument("--qim-dither-strength", type=float, default=0.10,
                   help="Strength of the keyed cover-conditioned dither perturbation in the cost factor.")
    p.add_argument("--qim-min-factor", type=float, default=0.50,
                   help="Minimum multiplicative cost factor for rho_ours.")
    p.add_argument("--qim-max-factor", type=float, default=1.75,
                   help="Maximum multiplicative cost factor for rho_ours.")
    p.add_argument("--run-id", default=None)
    p.add_argument("--progress", choices=["auto", "bar", "none"], default="auto")
    p.add_argument("--stc-backend", choices=["auto", "pystc", "proxy"], default="proxy",
                   help="Embedding backend: proxy, pystc, or auto fallback to proxy if pySTC fails.")
    p.add_argument("--stc-safe-subprocess", action="store_true",
                   help="Run pySTC in a subprocess per image/payload so C++ aborts are logged instead of killing the full run.")
    p.add_argument("--stc-timeout-sec", type=float, default=120.0,
                   help="Timeout for each pySTC subprocess job when --stc-safe-subprocess is enabled.")
    # Hidden worker arguments used internally by --stc-safe-subprocess.
    p.add_argument("--_pystc-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-in", default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-out", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    return args


def main():
    args = parse_args()
    if getattr(args, "_pystc_worker", False):
        if not args.worker_in or not args.worker_out:
            raise ValueError("Worker mode requires --worker-in and --worker-out")
        raise SystemExit(pystc_worker_main(args.worker_in, args.worker_out))

    if not getattr(args, "out_root", None):
        raise SystemExit("error: --out-root is required unless running internal pySTC worker mode")
    boss_root = Path(args.boss_root)
    out_root = Path(args.out_root)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    gamma1_path = Path(args.gamma1)
    gamma2_path = Path(args.gamma2)
    if not boss_root.exists():
        raise FileNotFoundError(f"BOSSBase root not found: {boss_root}")
    if not gamma1_path.exists():
        raise FileNotFoundError(f"gamma1 file not found: {gamma1_path}")
    if not gamma2_path.exists():
        raise FileNotFoundError(f"gamma2 file not found: {gamma2_path}")

    run_id = args.run_id or f"ours-qim-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    payloads = [float(x.strip()) for x in args.payloads.split(",") if x.strip()]
    paths = discover_images(boss_root, max_images=args.max_images if args.max_images > 0 else None)
    if not paths:
        raise RuntimeError(f"No image files found under {boss_root}")

    gamma1_bits = load_cover_stream_bits(gamma1_path)
    gamma2_bits = load_cover_stream_bits(gamma2_path)
    master_key = args.master_secret.encode("utf-8")

    jobs = [(p, payload) for payload in payloads for p in paths]
    rows = []
    print(f"[INFO] run_id={run_id}")
    print(f"[INFO] images={len(paths)}, payloads={payloads}, jobs={len(jobs)}")
    print(f"[INFO] gamma1_bits={gamma1_bits.size}, gamma2_bits={gamma2_bits.size}")
    print(f"[INFO] stc_backend={args.stc_backend}, safe_subprocess={args.stc_safe_subprocess}, pystc_available={pystc is not None}")

    for cover_path, payload in iter_progress(jobs, args.progress):
        row = process_one(cover_path, payload, args, run_id, gamma1_bits, gamma2_bits, master_key, logs_dir)
        rows.append(row)

    summarize(rows, out_root, run_id)
    print(f"[OK] Completed Ours-QIM run_id={run_id}")
    print(f"Logs:\n  - {logs_dir/'embed_log_ours_qim.csv'}\n  - {logs_dir/'embed_log_ours_qim.jsonl'}\n  - {logs_dir/'embed_summary_ours_qim.csv'}")


if __name__ == "__main__":
    main()
