#!/usr/bin/env python3
"""Generate cover, unshaped-stego, and admissibly shaped-stego timing traces.

The generator is a controlled multichannel scheduler experiment aligned with
Algorithm 3 of the manuscript. It does not embed images and does not transmit
network traffic. It records event-level timestamps for C1, C2, and C3.

The cover and shaped worlds use independent draws from the same bounded-delay
traffic model. The unshaped world exposes a nearly periodic, tightly coupled
three-channel schedule. Shaping is admissible because each data-frame delay is
bounded by rho and cross-channel skew is bounded by Delta.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CHANNELS = ("C1", "C2", "C3")


@dataclass(frozen=True)
class Config:
    epochs: int
    seeds: tuple[int, ...]
    epoch_period_ms: float
    rho_ms: float
    skew_delta_ms: float
    timeout_ms: float
    window_w: int
    heartbeat_h: int
    cover_phi: float
    cover_noise_ms: float
    channel_jitter_ms: float
    unshaped_jitter_ms: float
    network_delay_ms: float
    network_jitter_ms: float
    c1_bytes: int
    c2_bytes: int
    c3_bytes: int
    heartbeat_bytes: int
    include_heartbeats: bool


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one seed is required.")
    return seeds


def bounded_ar_delay(
    rng: np.random.Generator,
    n: int,
    rho_ms: float,
    phi: float,
    innovation_ms: float,
) -> np.ndarray:
    """Stationary-looking bounded delay process in [0, rho_ms]."""
    if n <= 0:
        return np.empty(0, dtype=float)
    center = rho_ms / 2.0
    values = np.empty(n, dtype=float)
    values[0] = rng.uniform(0.25 * rho_ms, 0.75 * rho_ms)
    for idx in range(1, n):
        proposal = center + phi * (values[idx - 1] - center)
        proposal += rng.normal(0.0, innovation_ms)
        # Reflection avoids an artificial pile-up caused by hard clipping.
        while proposal < 0.0 or proposal > rho_ms:
            if proposal < 0.0:
                proposal = -proposal
            if proposal > rho_ms:
                proposal = 2.0 * rho_ms - proposal
        values[idx] = proposal
    return values


def channel_offsets(
    rng: np.random.Generator,
    n: int,
    sigma_ms: float,
    skew_delta_ms: float,
) -> np.ndarray:
    """Create per-epoch C1/C2/C3 offsets with max skew <= Delta."""
    raw = rng.normal(0.0, sigma_ms, size=(n, 3))
    raw -= raw.mean(axis=1, keepdims=True)
    spans = np.ptp(raw, axis=1)
    limit = max(skew_delta_ms * 0.95, 1e-9)
    scale = np.ones(n)
    mask = spans > limit
    scale[mask] = limit / spans[mask]
    return raw * scale[:, None]


def build_condition(
    condition: str,
    seed: int,
    cfg: Config,
) -> pd.DataFrame:
    # Use condition-specific child seeds so all worlds are independent while
    # remaining reproducible.
    salt = {"cover": 1009, "unshaped": 2027, "shaped": 3041}[condition]
    rng = np.random.default_rng(seed * 100_003 + salt)

    epoch_ids = np.arange(1, cfg.epochs + 1, dtype=int)
    ready = (epoch_ids - 1) * cfg.epoch_period_ms

    if condition in {"cover", "shaped"}:
        common_delay = bounded_ar_delay(
            rng,
            cfg.epochs,
            cfg.rho_ms,
            cfg.cover_phi,
            cfg.cover_noise_ms,
        )
        offsets = channel_offsets(
            rng,
            cfg.epochs,
            cfg.channel_jitter_ms,
            cfg.skew_delta_ms,
        )
    else:
        common_delay = np.full(cfg.epochs, min(5.0, cfg.rho_ms * 0.05))
        common_delay += rng.normal(0.0, cfg.unshaped_jitter_ms, cfg.epochs)
        common_delay = np.clip(common_delay, 0.0, cfg.rho_ms)
        # Near-simultaneous regular sends create the unshaped footprint.
        fixed = np.array([0.0, 1.5, 3.0])
        offsets = np.tile(fixed, (cfg.epochs, 1))
        offsets += rng.normal(0.0, cfg.unshaped_jitter_ms, size=(cfg.epochs, 3))

    send = ready[:, None] + common_delay[:, None] + offsets
    # No frame is sent before it becomes ready.
    send = np.maximum(send, ready[:, None])

    # Enforce the per-frame shaping budget after channel offsets.
    shaping_delay = send - ready[:, None]
    clipped = shaping_delay > cfg.rho_ms
    send = np.minimum(send, ready[:, None] + cfg.rho_ms)
    shaping_delay = send - ready[:, None]

    # Enforce cross-channel skew after budget clipping.
    row_min = send.min(axis=1, keepdims=True)
    relative = send - row_min
    spans = relative.max(axis=1)
    too_wide = spans > cfg.skew_delta_ms
    if np.any(too_wide):
        relative[too_wide] *= (
            cfg.skew_delta_ms / spans[too_wide]
        )[:, None]
        send = row_min + relative
        shaping_delay = send - ready[:, None]

    network_delay = cfg.network_delay_ms + rng.normal(
        0.0, cfg.network_jitter_ms, size=(cfg.epochs, 3)
    )
    network_delay = np.maximum(network_delay, 0.01)
    arrival = send + network_delay

    epoch_skew = np.ptp(send, axis=1)
    epoch_complete = arrival.max(axis=1)
    epoch_latency = epoch_complete - ready

    sizes = {"C1": cfg.c1_bytes, "C2": cfg.c2_bytes, "C3": cfg.c3_bytes}
    rows: list[dict[str, object]] = []
    for channel_idx, channel in enumerate(CHANNELS):
        for idx, epoch_id in enumerate(epoch_ids):
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "epoch_id": int(epoch_id),
                    "channel_id": channel,
                    "frame_type": "data",
                    "ready_time_ms": float(ready[idx]),
                    "send_time_ms": float(send[idx, channel_idx]),
                    "arrival_time_ms": float(arrival[idx, channel_idx]),
                    "shaping_delay_ms": float(shaping_delay[idx, channel_idx]),
                    "network_delay_ms": float(network_delay[idx, channel_idx]),
                    "size_bytes": int(sizes[channel]),
                    "budget_clipped": bool(clipped[idx, channel_idx]),
                    "epoch_skew_ms": float(epoch_skew[idx]),
                    "epoch_complete_time_ms": float(epoch_complete[idx]),
                    "epoch_latency_ms": float(epoch_latency[idx]),
                }
            )

    if cfg.include_heartbeats and cfg.heartbeat_h > 0:
        for idx, epoch_id in enumerate(epoch_ids):
            if epoch_id % cfg.heartbeat_h != 0:
                continue
            # The optional heartbeat is public protocol metadata and appears in
            # every world, preventing a trivial frame-count distinguisher.
            for channel_idx, channel in enumerate(CHANNELS):
                hb_send = send[idx, channel_idx] + 0.5
                hb_net = max(
                    0.01,
                    cfg.network_delay_ms
                    + float(rng.normal(0.0, cfg.network_jitter_ms)),
                )
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "epoch_id": int(epoch_id),
                        "channel_id": channel,
                        "frame_type": "heartbeat",
                        "ready_time_ms": float(ready[idx]),
                        "send_time_ms": float(hb_send),
                        "arrival_time_ms": float(hb_send + hb_net),
                        "shaping_delay_ms": float(hb_send - ready[idx]),
                        "network_delay_ms": float(hb_net),
                        "size_bytes": int(cfg.heartbeat_bytes),
                        "budget_clipped": False,
                        "epoch_skew_ms": float(epoch_skew[idx]),
                        "epoch_complete_time_ms": float(epoch_complete[idx]),
                        "epoch_latency_ms": float(epoch_latency[idx]),
                    }
                )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("42,43,44"))
    parser.add_argument("--epoch-period-ms", type=float, default=1000.0)
    parser.add_argument("--rho-ms", type=float, default=100.0)
    parser.add_argument("--skew-delta-ms", type=float, default=100.0)
    parser.add_argument("--timeout-ms", type=float, default=200.0)
    parser.add_argument("--window-w", type=int, default=5)
    parser.add_argument("--heartbeat-h", type=int, default=10)
    parser.add_argument("--cover-phi", type=float, default=0.80)
    parser.add_argument("--cover-noise-ms", type=float, default=18.0)
    parser.add_argument("--channel-jitter-ms", type=float, default=5.0)
    parser.add_argument("--unshaped-jitter-ms", type=float, default=0.40)
    parser.add_argument("--network-delay-ms", type=float, default=20.0)
    parser.add_argument("--network-jitter-ms", type=float, default=5.0)
    parser.add_argument("--c1-bytes", type=int, default=127)
    parser.add_argument("--c2-bytes", type=int, default=127)
    parser.add_argument("--c3-bytes", type=int, default=262_253)
    parser.add_argument("--heartbeat-bytes", type=int, default=40)
    parser.add_argument("--include-heartbeats", action="store_true")
    args = parser.parse_args()

    if args.epochs < 100:
        raise SystemExit("--epochs must be at least 100.")
    if not 0.0 <= args.cover_phi < 1.0:
        raise SystemExit("--cover-phi must be in [0,1).")
    if args.rho_ms <= 0 or args.skew_delta_ms <= 0:
        raise SystemExit("rho and Delta must be positive.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        epochs=args.epochs,
        seeds=args.seeds,
        epoch_period_ms=args.epoch_period_ms,
        rho_ms=args.rho_ms,
        skew_delta_ms=args.skew_delta_ms,
        timeout_ms=args.timeout_ms,
        window_w=args.window_w,
        heartbeat_h=args.heartbeat_h,
        cover_phi=args.cover_phi,
        cover_noise_ms=args.cover_noise_ms,
        channel_jitter_ms=args.channel_jitter_ms,
        unshaped_jitter_ms=args.unshaped_jitter_ms,
        network_delay_ms=args.network_delay_ms,
        network_jitter_ms=args.network_jitter_ms,
        c1_bytes=args.c1_bytes,
        c2_bytes=args.c2_bytes,
        c3_bytes=args.c3_bytes,
        heartbeat_bytes=args.heartbeat_bytes,
        include_heartbeats=args.include_heartbeats,
    )

    frames = []
    for seed in cfg.seeds:
        for condition in ("cover", "unshaped", "shaped"):
            print(f"[RUN] seed={seed} condition={condition}")
            frames.append(build_condition(condition, seed, cfg))

    traces = pd.concat(frames, ignore_index=True)
    traces.sort_values(
        ["condition", "seed", "channel_id", "send_time_ms", "frame_type"],
        inplace=True,
    )
    trace_path = out_dir / "scheduling_event_traces.csv"
    traces.to_csv(trace_path, index=False)

    metadata = {
        "experiment": "scheduling-footprint-controlled-simulator",
        "worlds": ["cover", "unshaped", "shaped"],
        "notes": [
            "No image embedding was performed.",
            "KS/MMD should be interpreted as empirical footprint proxies, not direct TV estimates.",
            "Cover and shaped worlds use independent draws from the same bounded-delay model.",
        ],
        "config": asdict(cfg),
        "rows": int(len(traces)),
    }
    (out_dir / "scheduling_trace_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[OK] Event traces: {trace_path}")
    print(f"[OK] Rows: {len(traces):,}")
    print("[OK] No embedding, extraction, or network transmission was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
