#!/usr/bin/env python3
"""Timed simulation adapter; cryptography/image extraction remain frozen pilot code.

This process receives frame bytes and virtual arrival times only. It never reads
original covers, messages, attack labels, or the evaluator's expected outcomes.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter, OrderedDict
import hashlib
import hmac
import itertools
import json
import math
import os
from pathlib import Path
import sys


def unique(items):
    result = {}
    for k, v in items:
        if k in result:
            raise ValueError("duplicate JSON field")
        result[k] = v
    return result


class WindowReceiver:
    def __init__(self, ep, decode, master, sid, config, work):
        self.ep, self.decode, self.master, self.sid = ep, decode, master, sid
        self.cfg, self.work = config, Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.assembly = self.work / "assembly"
        self.assembly.mkdir(exist_ok=True)
        self.state_path = self.work / "protected_state.json"
        self.state = ep.ReceiverState.read(self.state_path)
        self.buffers = {}
        self.failed = {}
        self.first = {}
        self.a = 0
        self.now = 0
        self.count = Counter()
        self.maxima = Counter()

    def ident(self, e):
        return self.sid.hex() + ":" + str(e)

    def usage(self):
        candidates = [c for channels in self.buffers.values() for queue in channels.values()
                      for c in queue]
        return len(candidates), sum(c["size"] for c in candidates)

    def measure(self):
        n, size = self.usage()
        self.maxima["buffered_candidates"] = max(n, self.maxima["buffered_candidates"])
        self.maxima["buffered_serialized_bytes"] = max(size, self.maxima["buffered_serialized_bytes"])
        self.maxima["buffered_epochs"] = max(len(self.buffers), self.maxima["buffered_epochs"])
        self.maxima["failed_triple_cache"] = max(sum(map(len, self.failed.values())),
                                                 self.maxima["failed_triple_cache"])
        assert n <= self.cfg["window"] * 3 * self.cfg["candidates_per_channel"]
        assert size <= self.cfg["buffer_bytes"]
        assert len(self.buffers) <= self.cfg["window"]
        assert all(len(v) <= self.cfg["failed_cache_per_epoch"] for v in self.failed.values())

    def clear(self, e):
        self.buffers.pop(e, None)
        self.failed.pop(e, None)
        self.first.pop(e, None)

    def advance(self, t):
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < self.now:
            raise ValueError("virtual arrivals must be finite and nondecreasing")
        self.now = t
        # Explicit agreed clock/skip policy. Receipt of headers never advances it.
        new_a = max(0, math.floor((t - self.cfg["timeout_ms"]) /
                                  self.cfg["epoch_interval_ms"]) + 1)
        for e in range(self.a, min(new_a, self.cfg["epoch_count"])):
            if self.ident(e) not in self.state.accepted:
                self.count["expired_missing_epochs"] += 1
            self.clear(e)
        self.a = max(self.a, new_a)
        for e in list(self.buffers):
            if t - self.first[e] > self.cfg["timeout_ms"]:
                self.count["candidate_lifetime_expiries"] += 1
                self.clear(e)

    def reject(self, reason):
        self.count[reason] += 1
        return dict(public="rejected", reason=reason, accepted=[])

    def parse_one(self, raw):
        if not isinstance(raw, str) or len(raw.encode()) > 2 * self.ep.MAX_FRAME_BYTES:
            raise ValueError("oversized or non-text frame")
        r = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(r, dict):
            raise ValueError("frame is not an object")
        channel = r.get("channel")
        allowed = {"version", "session", "epoch", "header", "channel", "nonce", "content"}
        if channel == "C3":
            allowed |= {"codec", "payload_bits", "tag"}
        elif channel not in ("C1", "C2"):
            raise ValueError("invalid channel")
        if set(r) != allowed or r["version"] != self.ep.VERSION.decode():
            raise ValueError("invalid fields/version")
        if any(not isinstance(r[x], str) for x in ("session", "header", "nonce", "content")):
            raise ValueError("invalid field type")
        sid, hdr, nonce = (bytes.fromhex(r[x]) for x in ("session", "header", "nonce"))
        self.ep.uint(r["epoch"])
        if len(sid) != 16 or len(hdr) != 16 or len(nonce) != 16:
            raise ValueError("invalid identifier length")
        content = base64.b64decode(r["content"], validate=True)
        if not content or len(content) > self.ep.MAX_FRAME_BYTES:
            raise ValueError("invalid content length")
        if channel == "C3":
            self.ep.uint(r["payload_bits"])
            if (r["codec"] != self.ep.CODEC.decode() or not 1 <= r["payload_bits"] <=
                    8 * self.ep.MAX_FRAME_BYTES or len(bytes.fromhex(r["tag"])) != 32):
                raise ValueError("invalid codec, payload length or tag")
        return r, sid, hdr, nonce

    def frame(self, raw, t):
        self.advance(t)
        self.count["received_frames"] += 1
        try:
            r, sid, hdr, nonce = self.parse_one(raw)
        except (ValueError, TypeError, KeyError):
            return self.reject("front_end_parse_rejects")
        if sid != self.sid:
            return self.reject("session_rejects")
        e, channel = r["epoch"], r["channel"]
        if not hmac.compare_digest(hdr, self.ep.header(self.master, sid, e)):
            return self.reject("header_rejects")
        if self.ident(e) in self.state.accepted or nonce.hex() in self.state.nonces:
            return self.reject("freshness_rejects")
        if not self.a <= e < self.a + self.cfg["window"]:
            return self.reject("window_rejects")
        digest = hashlib.sha256(raw.encode()).hexdigest()
        channels = self.buffers.get(e, {})
        queue = channels.get(channel, [])
        if any(c["sha256"] == digest for c in queue):
            return self.reject("duplicate_buffer_rejects")
        size = len(raw.encode())
        # Per-channel FIFO eviction, followed by a global byte-budget check.
        if len(queue) >= self.cfg["candidates_per_channel"]:
            queue.pop(0)
            self.count["candidate_evictions"] += 1
        if self.usage()[1] + size > self.cfg["buffer_bytes"]:
            return self.reject("byte_budget_rejects")
        self.buffers.setdefault(e, {})[channel] = queue
        queue.append(dict(raw=raw, time=t, sha256=digest, size=size))
        self.first.setdefault(e, t)
        self.measure()
        if not all(self.buffers[e].get(c) for c in ("C1", "C2", "C3")):
            return dict(public="buffered", reason="incomplete", accepted=[])
        failures = 0
        for triple in itertools.product(*(self.buffers[e][c] for c in ("C1", "C2", "C3"))):
            if max(c["time"] for c in triple) - min(c["time"] for c in triple) > self.cfg["skew_ms"]:
                self.count["skew_ineligible_triples"] += 1
                continue
            fp = tuple(c["sha256"] for c in triple)
            cache = self.failed.setdefault(e, OrderedDict())
            if fp in cache:
                continue
            for i, candidate in enumerate(triple, 1):
                (self.assembly / f"C{i}.json").write_text(candidate["raw"])
            before = (self.state.accepted.copy(), self.state.nonces.copy())
            decodes_before = self.count["decode_calls"]
            try:
                assembled = self.ep.read_frames(self.assembly)
                self.count["mac_verification_attempts"] += 1
                def native(data, seed):
                    self.count["decode_calls"] += 1
                    return self.decode(data, seed)
                message = self.ep.recover(self.master, assembled, self.sid, self.state, native)
            except self.ep.ProtocolError as exc:
                self.count["authentication_rejects" if "Complete-epoch" in str(exc)
                           else "assembly_or_decode_rejects"] += 1
                if "Complete-epoch" in str(exc):
                    assert decodes_before == self.count["decode_calls"]
                assert before == (self.state.accepted, self.state.nonces)
                cache[fp] = True
                if len(cache) > self.cfg["failed_cache_per_epoch"]:
                    cache.popitem(last=False)
                failures += 1
                self.measure()
                continue
            # Commit and flush state before emitting plaintext to the caller.
            self.state.save(self.state_path)
            with self.state_path.open("rb") as persisted:
                os.fsync(persisted.fileno())
            directory_fd = os.open(self.work, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            saved = self.ep.ReceiverState.read(self.state_path)
            assert saved == self.state
            self.count["accepted_epochs"] += 1
            self.clear(e)
            self.measure()
            return dict(public="accepted", accepted=[dict(epoch=e, time_ms=t,
                plaintext_b64=base64.b64encode(message).decode(),
                transcript_sha256=hashlib.sha256(self.ep.transcript(assembled)).hexdigest())])
        return dict(public="rejected" if failures else "buffered",
                    reason="candidate_verification" if failures else "no_new_eligible_triple", accepted=[])

    def restart(self, t):
        self.advance(t)
        old = self.state
        self.state = self.ep.ReceiverState.read(self.state_path)
        assert old == self.state
        self.buffers.clear()
        self.failed.clear()
        self.first.clear()
        self.count["state_reloads"] += 1
        return dict(public="restarted", accepted=[])

    def snapshot(self):
        self.measure()
        return dict(counts=dict(self.count), maxima=dict(self.maxima), window_start=self.a,
                    protected_accepted=len(self.state.accepted), protected_nonces=len(self.state.nonces),
                    protected_state_bytes=self.state_path.stat().st_size if self.state_path.exists() else 0)


def main():
    p = argparse.ArgumentParser()
    for flag in ("source", "master-file", "session", "config", "work"):
        p.add_argument("--" + flag, required=True)
    args = p.parse_args()
    sys.path.insert(0, args.source)
    import epoch_protocol as ep
    from image_worker import decode_image
    receiver = WindowReceiver(ep, decode_image, Path(args.master_file).read_bytes(),
                              bytes.fromhex(args.session), json.loads(Path(args.config).read_text()), args.work)
    for line in sys.stdin:
        request = json.loads(line)
        op = request["op"]
        if op == "frame":
            result = receiver.frame(request["raw"], request["time_ms"])
        elif op == "restart":
            result = receiver.restart(request["time_ms"])
        elif op == "finish":
            receiver.advance(request["time_ms"])
            result = dict(public="finished", accepted=[], **receiver.snapshot())
        else:
            raise ValueError("unknown operation")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
