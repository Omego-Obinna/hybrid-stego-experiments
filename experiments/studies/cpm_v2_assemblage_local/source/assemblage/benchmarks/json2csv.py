#!/usr/bin/env python3
import json
import sys
import csv
from pathlib import Path
from typing import List, Dict, Any, Iterable


def _flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    sections = ("offline", "send", "receive")
    if len({k for k in rec if k in sections}) == 0:
        # Skip flattening if doesn't have any of the sections
        return rec

    flat: Dict[str, Any] = {}

    # Keep the simple top‑level fields
    for key in ("model", "index"):
        if key in rec:
            flat[key] = rec[key]

    # The three sub‑objects for each run
    for sec in sections:
        sub = rec.get(sec, {})
        if isinstance(sub, dict):
            for k, v in sub.items():
                flat[f"{sec}_{k}"] = v
        else:
            # Defensive: keep the whole thing if it's not a dict
            flat[sec] = sub

    return flat


def read_json(source: Iterable[str]) -> List[Dict[str, Any]]:
    try:
        return json.load(source)  # type: ignore[arg-type]
    except json.JSONDecodeError:
        txt = "".join(source)
        return json.loads(txt)


def write_csv(rows: List[Dict[str, Any]], out_fp) -> None:
    if not rows:
        return  # nothing to write

    fieldnames = rows[0].keys()

    writer = csv.DictWriter(out_fp, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Pick the input source
    if not argv:
        source = sys.stdin
    else:
        # First argument is the path to the JSON file
        json_path = Path(argv[0])
        if not json_path.is_file():
            sys.stderr.write(f"❌  File not found: {json_path}\n")
            return 1
        source = json_path.open("r", encoding="utf‑8")

    # Load JSON, flatten each record
    raw_data = read_json(source)
    flat_rows = [_flatten_record(rec) for rec in raw_data]

    # Write CSV to stdout
    write_csv(flat_rows, sys.stdout)

    # Clean‑up if we opened a file ourselves
    if hasattr(source, "close"):
        source.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
