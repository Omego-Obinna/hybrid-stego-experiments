#!/usr/bin/env python3
"""Independent-process real-image recovery; all attempted covers remain in the log."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import epoch_protocol as ep

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CPM_RESULTS_DIR", str(ROOT / "results")))
COVERS = Path(os.environ.get("BOSSBASE_ROOT", "data/BOSSbase_1.01"))
TEXT = Path(os.environ.get("TEXT_STREAM_ROOT", str(ROOT / "fixtures")))


def main():
    OUT.mkdir(exist_ok=True)
    master = hashlib.sha256(b"reviewer-one-validation-fixture-only").digest()
    keyfile = OUT / "fixture_master.bin"
    keyfile.write_bytes(master)
    sid = hashlib.sha256(b"reviewer-one-validation-session-v2").digest()[:16]
    state = OUT / "receiver_state.json"
    if state.exists() or (OUT / "receiver_validation.json").exists():
        raise SystemExit("Validation results already exist; preserve or move them before a new run")
    texts = []
    for name in ["gamma1_pride_selected_seed44.txt", "gamma2_pride_selected_seed44.txt"]:
        texts.append([line for line in (TEXT / name).read_bytes().splitlines() if line.strip()])
    outcomes = []
    for index in range(1, 13):
        for payload in [0.10, 0.20, 0.40]:
            epoch = len(outcomes)
            folder = OUT / f"epoch_{epoch:03d}"
            folder.mkdir()
            bits = int(round(payload * 512 * 512))
            msg = ep.truncate(ep.expand(master, b"fixture-message", ep.uint(epoch),
                                       (bits + 7) // 8), bits)
            (folder / "sender_message.bin").write_bytes(msg)
            for j in [0, 1]:
                (folder / f"gamma{j+1}.txt").write_bytes(texts[j][index - 1])
            prefix = [sys.executable, str(ROOT / "image_worker.py")]
            cmd = prefix + ["send", "--cover", str(COVERS / f"{index}.pgm"),
                            "--gamma1", str(folder / "gamma1.txt"),
                            "--gamma2", str(folder / "gamma2.txt"),
                            "--message", str(folder / "sender_message.bin"),
                            "--master-file", str(keyfile), "--session", sid.hex(),
                            "--epoch", str(epoch), "--payload", str(payload),
                            "--out", str(folder / "received_frames"),
                            "--metrics", str(folder / "sender_metrics.json")]
            sent = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            row = dict(cover_id=index, payload=payload, epoch=epoch,
                       sender_returncode=sent.returncode, exact_recovery=False)
            if sent.returncode == 0:
                received = subprocess.run(prefix + ["receive", "--frames", str(folder / "received_frames"),
                    "--master-file", str(keyfile), "--session", sid.hex(), "--state", str(state),
                    "--out", str(folder / "recovered_message.bin")],
                    capture_output=True, text=True, timeout=120)
                row["receiver_returncode"] = received.returncode
                row["receiver_report"] = received.stdout.strip()
                if received.returncode == 0:
                    actual = (folder / "recovered_message.bin").read_bytes()
                    row["plaintext_error_bits"] = sum((a ^ b).bit_count() for a, b in zip(msg, actual))
                    row["plaintext_ber"] = row["plaintext_error_bits"] / bits
                    row["exact_recovery"] = actual == msg
                else:
                    row["error"] = received.stderr[-1000:]
            else:
                row["error"] = (sent.stdout + sent.stderr)[-1800:]
            outcomes.append(row)
            (OUT / "receiver_validation_progress.json").write_text(json.dumps(outcomes, indent=2))
            print(f"epoch={epoch} cover={index} payload={payload} exact={row['exact_recovery']}", flush=True)
    result = dict(protocol=ep.VERSION.decode(), attempted=len(outcomes),
                  successful=sum(r["exact_recovery"] for r in outcomes),
                  sender_failures=sum(r["sender_returncode"] != 0 for r in outcomes),
                  receiver_failures=sum(r.get("receiver_returncode", 0) != 0 for r in outcomes),
                  payloads=[0.10, 0.20, 0.40], covers=list(range(1, 13)),
                  receiver_original_cover_available_as_input=False,
                  scope="Bounded integration validation, not a steganalysis or population-security experiment",
                  outcomes=outcomes)
    (OUT / "receiver_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "outcomes"}, indent=2))
    return 0 if result["successful"] == result["attempted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
