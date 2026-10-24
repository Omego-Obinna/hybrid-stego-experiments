# keyed_select_gamma.py
import argparse
import hashlib
import hmac
import random
import re
from pathlib import Path


def hmac_seed(key: str, label: bytes) -> int:
    d = hmac.new(key.encode("utf-8"), label, hashlib.sha256).digest()
    return int.from_bytes(d[:8], "big") % (2**31 - 1)


def clean_line(x: str) -> str:
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x


def word_count(x: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", x))


def load_lines(path: str, min_words=6, max_words=60):
    root = Path(path)

    if root.is_dir():
        files = sorted(root.glob("*.txt"))
    else:
        files = [root]

    lines = []
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            line = clean_line(line)
            if min_words <= word_count(line) <= max_words:
                lines.append(line)

    # Remove duplicates while preserving order.
    seen = set()
    unique = []
    for line in lines:
        key = line.lower()
        if key not in seen:
            unique.append(line)
            seen.add(key)

    return unique


def write_keyed_sample(lines, key, label, n, out_path):
    if len(lines) < n:
        raise ValueError(f"Need at least {n} usable lines, but found {len(lines)}.")

    seed = hmac_seed(key, label.encode("utf-8"))
    rng = random.Random(seed)

    indices = list(range(len(lines)))
    rng.shuffle(indices)

    selected = [lines[i] for i in indices[:n]]
    Path(out_path).write_text("\n".join(selected) + "\n", encoding="utf-8")

    print(f"Wrote {len(selected)} lines to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--n_sent", type=int, default=1000)
    ap.add_argument("--out1", default="gamma1_selected.txt")
    ap.add_argument("--out2", default="gamma2_selected.txt")
    args = ap.parse_args()

    lines = load_lines(args.corpus)
    print(f"Loaded {len(lines)} usable unique benign lines.")

    write_keyed_sample(lines, args.key, "gamma1-selected", args.n_sent, args.out1)
    write_keyed_sample(lines, args.key, "gamma2-selected", args.n_sent, args.out2)


if __name__ == "__main__":
    main()
