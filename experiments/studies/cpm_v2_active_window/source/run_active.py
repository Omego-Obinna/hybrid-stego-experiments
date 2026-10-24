#!/usr/bin/env python3
"""Private evaluator and seeded event generator, separate from receiver process."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import heapq
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

SCENARIOS = ("honest", "bounded_reordering", "burst_delay", "burst_suppression",
             "persistent_C2_suppression", "accepted_replay", "tamper_and_splice",
             "bounded_injection_flood", "adaptive_injection", "malformed_frames",
             "restart_then_replay")
PROFILES = (
    dict(name="W3_T200", window=3, timeout_ms=200),
    dict(name="W5_T400", window=5, timeout_ms=400),
)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)

def mutate(raw, salt, field="content"):
    record = json.loads(raw)
    if field == "payload_bits":
        record[field] += 8
    elif field == "tag":
        record[field] = sha((record[field] + str(salt)).encode())
    else:
        data = bytearray(base64.b64decode(record[field]))
        data[(salt * 7919) % len(data)] ^= (salt % 255) + 1
        record[field] = base64.b64encode(data).decode()
    return json.dumps(record, sort_keys=True)

def make_trace(scenario, seed, fixtures):
    rng = random.Random(seed)
    events, serial = [], 0
    def add(t, raw=None, origin="authorised", epoch=None, channel=None, op="frame"):
        nonlocal serial
        serial += 1
        heapq.heappush(events, (t, serial, dict(op=op, raw=raw, origin=origin,
                                              epoch=epoch, channel=channel)))
    for e, fixture in enumerate(fixtures):
        target = 8 <= e <= 11
        reordered_epoch_delay = rng.choice((20, 145)) if scenario == "bounded_reordering" else None
        for c in (1, 2, 3):
            raw = fixture["frames"][c-1]
            delay = 20 + rng.randrange(16)
            if scenario == "bounded_reordering":
                # Shared epoch shift permits e+1 to complete before e while
                # keeping each triple's cross-channel skew below the limit.
                delay = reordered_epoch_delay + rng.randrange(16)
            if scenario == "burst_delay" and target and c == 2:
                delay += 350
            if scenario in ("burst_suppression", "persistent_C2_suppression") and c == 2:
                if target or scenario == "persistent_C2_suppression":
                    continue
            if scenario in ("tamper_and_splice", "adaptive_injection") and target:
                delay += 50
                bad = mutate(raw, e + c)
                if c == 1:
                    donor = json.loads(fixtures[e-1]["frames"][0])
                    rebound = json.loads(raw)
                    # Rebind recorded public content to the target metadata;
                    # no sender key/MAC is available to the attack generator.
                    rebound["content"] = donor["content"]
                    bad = json.dumps(rebound, sort_keys=True)
                if c == 3:
                    field = {8:"content", 9:"tag", 10:"payload_bits", 11:"content"}[e]
                    bad = mutate(raw, e + c, field)
                add(e*100 + 3 + c, bad, "unauthorised_mutation_or_splice", e, c)
            if scenario == "bounded_injection_flood" and target:
                for j in range(16):
                    add(e*100 + 2 + j, mutate(raw, 100*e+10*c+j,
                        "tag" if c == 3 else "content"), "bounded_injection", e, c)
            add(e*100 + delay, raw, epoch=e, channel=c)
            if scenario == "accepted_replay":
                add(e*100 + 70, raw, "recorded_duplicate", e, c)
        if scenario == "malformed_frames" and target:
            raw = fixture["frames"][0]
            bad = json.loads(raw)
            if e == 8:
                bad_raw = raw[:-1] + ', "epoch": 8}'  # duplicate key
            elif e == 9:
                bad["session"] = "00"*16
                bad_raw = json.dumps(bad)
            elif e == 10:
                bad["header"] = "00"*16
                bad_raw = json.dumps(bad)
            else:
                bad["content"] = "!invalid-base64!"
                bad_raw = json.dumps(bad)
            add(e*100 + 2, bad_raw, "malformed_or_wrong_context", e, 1)
    if scenario == "restart_then_replay":
        add(1580, origin="process_state_reload", op="restart")
        for e in range(8, 12):
            for c in (1, 2, 3):
                add(1581 + (e-8)*3+c, fixtures[e]["frames"][c-1], "post_restart_replay", e, c)
    return events, serial

def load_fixtures(pilot, count):
    result = []
    rows = [json.loads(line) for line in (pilot/"pilot_results.jsonl").read_text().splitlines()]
    for e in range(count):
        folder = pilot/"epochs"/f"epoch_{e:03d}"
        result.append(dict(epoch=e, cover_id=rows[e]["cover_id"], payload_bpp=rows[e]["payload_bpp"],
            frames=[(folder/"frames"/f"C{c}.json").read_text() for c in (1,2,3)],
            message=(folder/"message.bin").read_bytes(),
            transcript_sha256=json.loads((folder/"sender_metrics.json").read_text())["transcript_sha256"]))
    return result

def one_trace(root, pilot, config, fixtures, scenario, seed, source):
    trace_id = f'{config["name"]}__{scenario}__seed{seed}'
    work = root/"traces"/trace_id
    work.mkdir(parents=True, exist_ok=False)
    save(work/"config.json", config)
    pilot_cfg = json.loads((pilot/"run_config.json").read_text())
    command = [sys.executable, str(source/"window_receiver.py"), "--source", str(pilot/"source"),
        "--master-file", pilot_cfg["master_file"], "--session", pilot_cfg["session_hex"],
        "--config", str(work/"config.json"), "--work", str(work/"receiver")]
    start = time.perf_counter()
    accepted, seen, transcripts_bad, plaintext_bad, repeated = [], set(), 0, 0, 0
    arrival_origins = Counter()
    events, serial = make_trace(scenario, seed, fixtures)
    adapted = set()
    with (work/"receiver_stderr.log").open("w") as err, (work/"events.jsonl").open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=err, text=True, bufsize=1)
        def exchange(request):
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError(f"Receiver stopped; inspect {work/'receiver_stderr.log'}")
            return json.loads(line)
        try:
            while events:
                t, order, event = heapq.heappop(events)
                request = dict(op=event["op"], time_ms=t)
                if event["op"] == "frame":
                    request["raw"] = event["raw"]
                response = exchange(request)
                arrival_origins[event["origin"]] += 1
                summaries = []
                for item in response["accepted"]:
                    e = item["epoch"]
                    repeated += e in seen
                    seen.add(e)
                    authorised = 0 <= e < len(fixtures) and item["transcript_sha256"] == fixtures[e]["transcript_sha256"]
                    exact = 0 <= e < len(fixtures) and base64.b64decode(item["plaintext_b64"]) == fixtures[e]["message"]
                    transcripts_bad += not authorised
                    plaintext_bad += not exact
                    record = dict(epoch=e, arrival_time_ms=t, latency_ms=t-e*100,
                        plaintext_sha256=sha(base64.b64decode(item["plaintext_b64"])),
                        authorised_transcript=authorised, exact_plaintext=exact,
                        transcript_sha256=item["transcript_sha256"])
                    accepted.append(record)
                    summaries.append(record)
                log.write(json.dumps(dict(time_ms=t, order=order, op=event["op"],
                    origin=event["origin"], source_epoch=event["epoch"], channel=event["channel"],
                    frame_sha256=sha(event["raw"].encode()) if event["raw"] is not None else None,
                    public=response["public"], reason=response.get("reason"), accepted=summaries)) + "\n")
                # Attacker sees the packet response only, never protected state/ground truth.
                if (scenario == "adaptive_injection" and event["epoch"] is not None and
                    8 <= event["epoch"] <= 11 and response["public"] == "rejected" and
                    response.get("reason") == "candidate_verification" and event["epoch"] not in adapted):
                    e = event["epoch"]
                    adapted.add(e)
                    for j in range(2):
                        serial += 1
                        heapq.heappush(events, (t+1+j, serial, dict(op="frame",
                            raw=mutate(fixtures[e]["frames"][0], seed*100+j),
                            origin="public_feedback_adaptive_injection", epoch=e, channel=1)))
            final = exchange(dict(op="finish", time_ms=(len(fixtures)-1)*100+config["timeout_ms"]+1))
            process.stdin.close()
            rc = process.wait(timeout=20)
            if rc:
                raise RuntimeError(f"receiver exit {rc}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    result = dict(trace_id=trace_id, scenario=scenario, seed=seed, profile=config["name"],
        offered_epochs=len(fixtures), accepted_epochs=len(accepted), erased_epochs=len(fixtures)-len(seen),
        unauthorised_transcript_accepts=transcripts_bad, wrong_plaintext_accepts=plaintext_bad,
        repeated_epoch_accepts=repeated, counts=final["counts"], maxima=final["maxima"],
        protected_accepted=final["protected_accepted"], protected_nonces=final["protected_nonces"],
        protected_state_bytes=final["protected_state_bytes"], arrival_origins=dict(arrival_origins),
        adaptive_responses=len(adapted), wall_sec=time.perf_counter()-start,
        all_fragments_recovered=len(seen)==len(fixtures), accepted=accepted)
    result["out_of_order_epoch_acceptances"] = sum(
        item["epoch"] < max(x["epoch"] for x in accepted[:i])
        for i,item in enumerate(accepted) if i)
    if scenario in ("burst_delay", "burst_suppression"):
        tail = [x for x in accepted if x["epoch"] >= 12]
        result["clean_tail_accepted"] = len(tail)
        result["first_clean_tail_recovery_after_epoch12_start_ms"] = tail[0]["arrival_time_ms"]-1200 if tail else None
    save(work/"result.json", result)
    # No acceptance assertion is based on an attack label inside the receiver.
    assert not (transcripts_bad or plaintext_bad or repeated), "integrity/correctness failure: retained for review"
    expected = (0 if scenario == "persistent_C2_suppression" else len(fixtures)-4
                if scenario in ("burst_delay", "burst_suppression") else len(fixtures))
    assert len(accepted) == expected, f"unexpected delivery in {trace_id}: {len(accepted)} vs {expected}"
    assert final["protected_nonces"] == 3*len(accepted)
    assert final["counts"].get("decode_calls", 0) == len(accepted)
    if scenario in ("accepted_replay", "restart_then_replay"):
        assert final["counts"].get("freshness_rejects", 0) > 0
    if scenario in ("tamper_and_splice", "bounded_injection_flood", "adaptive_injection"):
        assert final["counts"].get("authentication_rejects", 0) > 0
    if scenario == "bounded_injection_flood":
        assert final["counts"].get("candidate_evictions", 0) > 0
    if scenario == "adaptive_injection":
        assert len(adapted) == 4
    if scenario == "bounded_reordering":
        assert result["out_of_order_epoch_acceptances"] > 0
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--status", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[11,23,37,53,71])
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--profiles", nargs="+", default=[x["name"] for x in PROFILES])
    args = p.parse_args()
    root, source = args.root.resolve(), Path(__file__).resolve().parent
    if (root/"run_config.json").exists():
        raise ValueError("Run already exists; no automatic retry/overwrite")
    root.mkdir(parents=True, exist_ok=True)
    fixtures = load_fixtures(args.pilot, 30)
    cfgs = [dict(x, epoch_count=30, epoch_interval_ms=100, skew_ms=100,
                 candidates_per_channel=2, buffer_bytes=8*1024*1024, failed_cache_per_epoch=128)
            for x in PROFILES if x["name"] in args.profiles]
    assert cfgs
    run_cfg = dict(created_at=now(), phase="timed-window active attacks", pilot=str(args.pilot),
        source_sha256={name:sha((source/name).read_bytes()) for name in ("run_active.py", "window_receiver.py")},
        frozen_protocol_sha256=sha((args.pilot/"source/epoch_protocol.py").read_bytes()),
        seeds=args.seeds, scenarios=args.scenarios, profiles=cfgs, fixture_epoch_count=30,
        unique_cover_count=len({x["cover_id"] for x in fixtures}),
        expected_trace_count=len(cfgs)*len(args.scenarios)*len(args.seeds),
        trial_scope="Repeated finite real-image fixtures; correlated seeded simulation traces, not independent cryptographic security trials",
        virtual_clock="Known shared epoch origin; a(t)=max(0,floor((t-timeout)/100)+1); deadline equality expires; no heartbeat/ECC",
        receiver_inputs=["received frame bytes", "virtual arrival time", "shared configuration", "fixture session and secret", "protected replay state"],
        cpu_threads=1, automatic_retry=False)
    save(root/"run_config.json", run_cfg)
    save(root/"fixture_manifest.json", [dict(epoch=x["epoch"],cover_id=x["cover_id"],
        payload_bpp=x["payload_bpp"], frame_sha256=[sha(f.encode()) for f in x["frames"]],
        message_sha256=sha(x["message"]), transcript_sha256=x["transcript_sha256"]) for x in fixtures])
    started = time.perf_counter()
    results = []
    def status(state, stage, error=None):
        elapsed = time.perf_counter()-started
        save(args.status, dict(state=state, stage=stage, completed=len(results),
            total=run_cfg["expected_trace_count"], elapsed_sec=elapsed,
            eta_sec=elapsed/len(results)*(run_cfg["expected_trace_count"]-len(results)) if results else None,
            accepted=sum(x["accepted_epochs"] for x in results),
            unauthorised_accepts=sum(x["unauthorised_transcript_accepts"] for x in results),
            wrong_plaintexts=sum(x["wrong_plaintext_accepts"] for x in results),
            repeats=sum(x["repeated_epoch_accepts"] for x in results),
            verification_attempts=sum(x["counts"].get("mac_verification_attempts",0) for x in results),
            authentication_rejects=sum(x["counts"].get("authentication_rejects",0) for x in results),
            recent=[{k:x[k] for k in ("trace_id","scenario","profile","accepted_epochs","erased_epochs",
                                      "unauthorised_transcript_accepts","wall_sec")} for x in results[-8:]],
            updated_at=now(), last_error=error))
    status("running", "Checking honest controls and attack traces")
    try:
        with (root/"trace_results.jsonl").open("x") as out:
            for cfg in cfgs:
                for scenario in args.scenarios:
                    for seed in args.seeds:
                        status("running", f'{cfg["name"]} · {scenario} · seed {seed}')
                        result = one_trace(root,args.pilot,cfg,fixtures,scenario,seed,source)
                        out.write(json.dumps(result)+"\n")
                        out.flush()
                        results.append(result)
                        status("running", f'Finished {result["trace_id"]}')
                        print(json.dumps({k:result[k] for k in ("trace_id","accepted_epochs","erased_epochs","wall_sec")}), flush=True)
        save(root/"completion.json", dict(completed_at=now(), traces=len(results),
            elapsed_sec=time.perf_counter()-started, all_checks_passed=True,
            accepted_epoch_events=sum(x["accepted_epochs"] for x in results),
            unauthorised_accepts=0, wrong_plaintexts=0, repeated_accepts=0))
        status("completed", "Timed-window attack batch finished; all scenario checks passed")
    except Exception as exc:
        status("failed", "Stopped on an unexpected outcome; evidence preserved", str(exc))
        (root/"failure.txt").write_text(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
