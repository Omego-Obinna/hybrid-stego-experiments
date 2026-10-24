# sample_lines.py
import argparse, random
from pathlib import Path

def read_lines(path):
    return [x.strip() for x in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lines = read_lines(args.infile)
    random.Random(args.seed).shuffle(lines)
    lines = lines[:args.n]

    Path(args.outfile).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} lines to {args.outfile}")

if __name__ == "__main__":
    main()
