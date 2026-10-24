#!/usr/bin/env python3
# synth_streams.py  —  compact, natural Markov sentences with deterministic dual streams
import argparse, hashlib, hmac, random, glob, re
from pathlib import Path
import markovify, nltk, unidecode

# Ensure NLTK's punkt is present (quiet if already installed)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# -------------------------- helpers --------------------------

def derive_seed(key: str, label: bytes) -> int:
    """Deterministic 32-bit seed from key+label (HMAC-SHA256)."""
    d = hmac.new(key.encode("utf-8"), label, hashlib.sha256).digest()
    return int.from_bytes(d[:4], "big")

def read_corpus(corpus_dir: Path) -> str:
    """Read all .txt files (recursively) and normalize to ASCII-compatible text."""
    files = glob.glob(str(corpus_dir / "**" / "*.txt"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No .txt files found in {corpus_dir}")
    chunks = []
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()
        # light normalization only (to keep the original style you liked)
        txt = unidecode.unidecode(txt)
        txt = re.sub(r"[ \t]+\n", "\n", txt)         # trim trailing spaces
        txt = re.sub(r"\n{3,}", "\n\n", txt)         # collapse tall gaps
        chunks.append(txt.strip())
    return "\n\n".join(chunks)

def build_model(text: str, state_size: int = 2):
    # Markov model over sentences; well_formed=True favors natural boundaries
    return markovify.Text(text, state_size=state_size, well_formed=True)

_punct_end = re.compile(r"[.!?…]$")

def _tidy_sentence(s: str) -> str:
    """Small cleanups for flow without changing the generator's character."""
    s = s.strip()
    # Collapse excessive inner spaces
    s = re.sub(r"\s+", " ", s)
    # Capitalize first letter if reasonable
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    # Ensure end punctuation
    if s and not _punct_end.search(s):
        s += "."
    return s

def _sig(sent: str) -> set:
    """A tiny signature for near-duplicate detection (word 3-grams, lowercased)."""
    toks = [w.lower() for w in re.findall(r"[A-Za-z0-9']+", sent)]
    return {tuple(toks[i:i+3]) for i in range(len(toks)-2)}

def generate_stream(model, n_sent, min_len, max_len, rng_seed,
                    tries_per_sent=200, allow_near_dupes=False, dupe_threshold=0.85):
    """
    Produce n_sent sentences in [min_len, max_len] words, deterministic via rng_seed.
    Near-duplicate filtering keeps flow natural without obvious repetition.
    """
    random.seed(rng_seed)
    out, sigs = [], []
    for _ in range(n_sent):
        chosen = None
        for _ in range(tries_per_sent):
            s = model.make_sentence(tries=100)
            if not s:
                continue
            wc = len(s.split())
            if wc < min_len or wc > max_len:
                continue
            s = _tidy_sentence(s)

            if not allow_near_dupes and sigs:
                sig = _sig(s)
                # Jaccard against recent few (limit to last 200 to keep O(1) memory)
                recent = sigs[-200:]
                sim_ok = True
                for old in recent:
                    inter = len(sig & old)
                    union = len(sig | old) or 1
                    j = inter / union
                    if j >= dupe_threshold:
                        sim_ok = False
                        break
                if not sim_ok:
                    continue
                sigs.append(sig)
            chosen = s
            break
        if not chosen:
            # Fallback: character-limited sentence, then tidy
            s = model.make_short_sentence(max_len * 12) or "..."
            chosen = _tidy_sentence(s)
            if not allow_near_dupes:
                sigs.append(_sig(chosen))
        out.append(chosen)
    return out

# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser(description="Deterministic dual-stream sentence synthesizer")
    ap.add_argument("--corpus", required=True, help="Directory with .txt corpus files (recurses)")
    ap.add_argument("--key", required=True, help="Shared session key (string)")
    ap.add_argument("--n_sent", type=int, default=60)
    ap.add_argument("--min_len", type=int, default=6)
    ap.add_argument("--max_len", type=int, default=18)
    ap.add_argument("--out1", required=True)
    ap.add_argument("--out2", required=True)
    ap.add_argument("--state_size", type=int, default=2, help="Markov state size (1–3 recommended)")
    ap.add_argument("--tries_per_sent", type=int, default=200)
    ap.add_argument("--allow_near_dupes", action="store_true", help="Permit near-duplicate sentences")
    ap.add_argument("--dupe_threshold", type=float, default=0.85,
                    help="Jaccard threshold (3-grams) to treat as near-duplicate")
    args = ap.parse_args()

    text = read_corpus(Path(args.corpus))
    model = build_model(text, state_size=args.state_size)

    seed1 = derive_seed(args.key, b"gamma1")
    seed2 = derive_seed(args.key, b"gamma2")

    g1 = generate_stream(model, args.n_sent, args.min_len, args.max_len, seed1,
                         tries_per_sent=args.tries_per_sent,
                         allow_near_dupes=args.allow_near_dupes,
                         dupe_threshold=args.dupe_threshold)
    g2 = generate_stream(model, args.n_sent, args.min_len, args.max_len, seed2,
                         tries_per_sent=args.tries_per_sent,
                         allow_near_dupes=args.allow_near_dupes,
                         dupe_threshold=args.dupe_threshold)

    Path(args.out1).write_text("\n".join(g1) + "\n", encoding="utf-8")
    Path(args.out2).write_text("\n".join(g2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()

