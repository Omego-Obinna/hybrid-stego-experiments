#!/usr/bin/env python3
"""Separate sender and receiver processes using the existing native pySTC backend."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import sys
import time
import resource

FUSION_WEIGHT = 0.75
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
import pystc
import epoch_protocol as ep


def options():
    import legacy_embedder as old
    saved = sys.argv
    try:
        sys.argv = ["legacy_embedder.py", "--base-cost", "fused",
                    "--fused-suniward-weight", str(FUSION_WEIGHT), "--qim-texture-discount", "0.08",
                    "--qim-smooth-penalty", "0.08", "--cf-use-slic",
                    "--cluster-penalty", "1.02", "--cost-smooth-mix", "0.05",
                    "--cf-complexity-discount", "0.03", "--cf-smooth-penalty", "0.03"]
        return old, old.parse_args()
    finally:
        sys.argv = saved


def costs(cover, master, e):
    old, args = options()
    plus, minus, energy, base_stats = old.compute_base_costs(cover, args)
    psi, maps, cf_stats = old.compute_complex_feature_map(cover, energy, args)
    digest = old.cover_feature_digest(psi)
    label = hashlib.sha256(ep.context(e)).hexdigest()
    cost_key = ep.key(master, e.sid, b"cost-modulation")
    dither = old.dither_uniform_map(cover.shape, cost_key, label, digest)
    _, factor = old.qim_modulated_cost_map(
        np.minimum(plus, minus), psi, dither,
        texture_discount=args.qim_texture_discount, smooth_penalty=args.qim_smooth_penalty,
        dither_strength=args.qim_dither_strength, min_factor=args.qim_min_factor,
        max_factor=args.qim_max_factor, normalize_factor=True)
    plus, minus, _ = old.qim_directional_costs(
        cover=cover, rho_p1=plus * factor, rho_m1=minus * factor,
        psi_norm=psi, rho_dit_u=dither, mode="none", strength=args.qim_direction_strength,
        dither_mix=args.qim_direction_dither_mix, residual_scale=args.qim_direction_residual_scale,
        wet_cost=args.wet_cost)
    plus, minus, _ = old.apply_spreading_clustering_costs(
        plus, minus, psi, maps["region_mask"], cover, args)
    return plus.astype(np.float64), minus.astype(np.float64), dict(
        fused_weight=FUSION_WEIGHT, slic_segments=256, slic_compactness=10.0,
        cost_smooth_mix=0.05, cluster_penalty=1.02, cf_complexity_discount=0.03,
        cf_smooth_penalty=0.03, direction_mode="none",
        feature_digest_used_only_for_sender_costs=digest.hex(),
        pystc_cost_argument_order="minus,plus")


def decode_image(data, seed):
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PPM" or image.mode != "L":
            raise ep.ProtocolError("Expected lossless 8-bit grayscale PGM image")
        if not 1 <= image.width * image.height <= 1024 * 1024:
            raise ep.ProtocolError("Unexpected image dimensions")
        pixels = np.asarray(image, dtype=np.uint8).copy()
    return bytes(pystc.unhide(pixels, seed))


def send(args):
    master = Path(args.master_file).read_bytes()
    sid = bytes.fromhex(args.session)
    with Image.open(args.cover) as image:
        if image.mode != "L":
            raise ep.ProtocolError("Use the native grayscale cover")
        cover = np.asarray(image, dtype=np.uint8).copy()
    bits = int(round(args.payload * cover.size))
    nonces = [ep.mac(ep.key(master, sid, b"nonce"), label, ep.uint(args.epoch))[:16]
              for label in [b"C1", b"C2", b"C3"]]
    e = ep.Epoch(sid, args.epoch, ep.header(master, sid, args.epoch), nonces[0],
                 Path(args.gamma1).read_bytes(), nonces[1], Path(args.gamma2).read_bytes(),
                 nonces[2], bits)
    message = ep.truncate(Path(args.message).read_bytes(), bits)
    masked = ep.mask_message(master, e, message)
    start = time.perf_counter()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    plus, minus, profile = costs(cover, master, e)
    seed = ep.embedding_seed(master, e)
    # The native API explicitly expects minus costs followed by plus costs.
    stego = np.asarray(pystc.hide(masked, cover, minus, plus, seed, mx=255, mn=0),
                       dtype=np.uint8)
    embedded_time = time.perf_counter() - start
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    embedding_cpu_sec = ((usage_end.ru_utime + usage_end.ru_stime)
                         - (usage_start.ru_utime + usage_start.ru_stime))
    stream = io.BytesIO()
    Image.fromarray(stego).save(stream, format="PPM")
    data = stream.getvalue()
    check_start = time.perf_counter()
    extracted = decode_image(data, seed)
    if extracted != masked:
        raise ep.ProtocolError("Sender preflight failed: STC did not recover the masked bytes")
    e = ep.authenticate(master, replace(e, stego=data))
    ep.write_frames(e, args.out)
    metrics = dict(status="ok", protocol=ep.VERSION.decode(), application_payload_bits=bits,
                   byte_padding_bits=len(masked) * 8 - bits, stc_length_header_bits=32,
                   pixel_count=int(cover.size), target_bpp=args.payload,
                   effective_stc_bpp=(len(masked) * 8 + 32) / cover.size,
                   message_sha256=hashlib.sha256(message).hexdigest(),
                   masked_sha256=hashlib.sha256(masked).hexdigest(),
                   transcript_sha256=hashlib.sha256(ep.transcript(e)).hexdigest(),
                   stego_sha256=hashlib.sha256(data).hexdigest(),
                   embed_time_sec=embedded_time, embedding_cpu_sec=embedding_cpu_sec,
                   preflight_decode_time_sec=time.perf_counter() - check_start,
                   source_cover_sha256=hashlib.sha256(Path(args.cover).read_bytes()).hexdigest(),
                   backend="pystc", profile=profile)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2))


def receive(args):
    master = Path(args.master_file).read_bytes()
    e = ep.read_frames(args.frames)
    state = ep.ReceiverState.read(args.state)
    start = time.perf_counter()
    message = ep.recover(master, e, bytes.fromhex(args.session), state, decode_image)
    # Commit the persistent freshness state before releasing the recovered payload.
    state.save(args.state)
    Path(args.out).write_bytes(message)
    print(json.dumps(dict(status="accepted", payload_bits=e.payload_bits,
                          message_sha256=hashlib.sha256(message).hexdigest(),
                          receiver_time_sec=time.perf_counter() - start,
                          receiver_inputs=["three_received_frames", "master_secret",
                                           "expected_session_id", "persistent_replay_state"],
                          original_cover_used=False, sender_mask_used=False)))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("send")
    for flag in ["cover", "gamma1", "gamma2", "message", "master-file", "session", "out", "metrics"]:
        s.add_argument("--" + flag, required=True)
    s.add_argument("--epoch", type=int, required=True)
    s.add_argument("--payload", type=float, required=True)
    s.add_argument("--fusion-weight", type=float, required=True)
    r = sub.add_parser("receive")
    for flag in ["frames", "master-file", "session", "state", "out"]:
        r.add_argument("--" + flag, required=True)
    args = p.parse_args()
    try:
        global FUSION_WEIGHT
        if args.mode == "send":
            if not 0.0 <= args.fusion_weight <= 1.0:
                raise ep.ProtocolError("Fusion weight must lie in [0,1]")
            FUSION_WEIGHT = args.fusion_weight
        (send if args.mode == "send" else receive)(args)
    except (ep.ProtocolError, ValueError, OSError) as exc:
        print(json.dumps(dict(status="rejected_or_failed", reason=str(exc))))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
