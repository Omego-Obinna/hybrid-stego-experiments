#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Stego: Two independent Markov sentence streams (γ1, γ2)
- Corpus: local .txt files (Project Gutenberg + Wikisource)
- PRG: HKDF(key, label) -> HMAC-SHA256 counter-mode stream (deterministic)
- Model: 2nd-order word-level Markov; sentence-safe generation

Usage:
  python synth_streams.py --corpus /path/to/corpus_root \
                          --key "correct horse battery staple" \
                          --n_sent 50 --min_len 5 --max_len 20 \
                          --out1 gamma1.txt --out2 gamma2.txt
"""
import argparse, glob, hmac, hashlib, os, re, unicodedata
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Iterator

# ---------------------- CSPRNG: HKDF + HMAC-CTR ----------------------

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, hashlib.sha256).digest()

def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 style HKDF-Expand (SHA-256)."""
    okm = b""
    t = b""
    counter = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        okm += t
        counter += 1
    return okm[:length]

class DeterministicPRG:
    """HMAC-SHA256( key , counter || label ) stream -> bytes/ints/floats."""
    def __init__(self, master_key: bytes, label: str):
        # Derive per-stream subkey via HKDF (salt+info separation)
        salt = b"hybstego-hkdf-salt-v1"
        prk = hkdf_extract(salt, master_key)
        self.key = hkdf_expand(prk, info=b"stream-"+label.encode(), length=32)
        self.counter = 0

    def _block(self) -> bytes:
        self.counter += 1
        msg = self.counter.to_bytes(16, "big")  # 128-bit counter
        return hmac.new(self.key, msg, hashlib.sha256).digest()

    def randbytes(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += self._block()
        return out[:n]

    def randint(self, a: int, b: int) -> int:
        # inclusive [a,b]
        span = b - a + 1
        # rejection sampling using 64-bit chunks
        while True:
            x = int.from_bytes(self.randbytes(8), "big")
            r = x % span
            # Optional: reduce bias by rejecting if x is in the tail
            return a + r

    def rand(self) -> float:
        # 53-bit mantissa uniform in [0,1)
        x = int.from_bytes(self.randbytes(8), "big") >> 11
        return x / float(1 << 53)

# ---------------------- Text normalization & corpus ----------------------

_SENT_END = re.compile(r"[.!?]+[\"\')\]]*\s+")
_TOKENIZE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?|[.,!?;:()\"'\[\]—–-]")

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    # Remove Gutenberg headers/footers heuristically
    s = re.sub(r"\*\*\* START OF.*?\*\*\*", " ", s, flags=re.I|re.S)
    s = re.sub(r"\*\*\* END OF.*?\*\*\*", " ", s, flags=re.I|re.S)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def sentences_from_text(s: str) -> List[str]:
    # split by sentence-ish boundaries, keep punctuation with prev token
    parts = _SENT_END.split(s)
    return [p.strip() for p in parts if p and len(p.split()) >= 3]

def load_corpus_texts(root: str) -> List[str]:
    files = []
    for ext in ("*.txt", "**/*.txt"):
        files.extend(glob.glob(os.path.join(root, ext), recursive=True))
    texts = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(normalize_text(f.read()))
        except Exception:
            pass
    return texts

# ---------------------- Markov(2) model ----------------------

START = "<s>"
END = "</s>"

class Markov2:
    def __init__(self):
        self.nexts: Dict[Tuple[str,str], Counter] = defaultdict(Counter)
        self.starts: Counter = Counter()

    def fit(self, sentences: List[str]) -> None:
        for sent in sentences:
            toks = _TOKENIZE.findall(sent)
            if not toks: 
                continue
            # record starts
            if len(toks) >= 2:
                self.starts[(toks[0].lower(), toks[1].lower())] += 1
            # add BOS/EOS padding
            seq = [START, START] + [t.lower() for t in toks] + [END]
            for i in range(len(seq)-2):
                key = (seq[i], seq[i+1])
                self.nexts[key][seq[i+2]] += 1

    def _weighted_choice(self, prg: DeterministicPRG, counter: Counter) -> str:
        total = sum(counter.values())
        r = prg.randint(1, total)
        cum = 0
        for tok, w in counter.items():
            cum += w
            if r <= cum:
                return tok
        # fallback (shouldn’t happen)
        return next(iter(counter.keys()))

    def generate_sentence(self, prg: DeterministicPRG, min_len=5, max_len=20) -> str:
        # choose starting bigram by frequency
        start_pair = self._weighted_choice(prg, self.starts)
        w1, w2 = start_pair
        out = [w1, w2]
        while len(out) < max_len:
            nxt = self._weighted_choice(prg, self.nexts.get((w1, w2), Counter({END:1})))
            if nxt == END and len(out) >= min_len:
                break
            if nxt == END:
                # force continue until min_len
                nxt = self._weighted_choice(prg, (self.nexts.get((w1, w2)) or Counter({".":1})))
            out.append(nxt)
            w1, w2 = w2, nxt

        # Light post-process: capitalize first, join with spaces, fix spacing before punctuation
        s = " ".join(out)
        s = re.sub(r"\s+([.,!?;:)\]])", r"\1", s)
        s = re.sub(r"([(\[])\s+", r"\1", s)
        if s and s[0].isalpha():
            s = s[0].upper() + s[1:]
        if not re.search(r"[.!?]$", s):
            s += "."
        return s

    def stream(self, prg: DeterministicPRG, n_sent=50, min_len=5, max_len=20) -> Iterator[str]:
        for _ in range(n_sent):
            yield self.generate_sentence(prg, min_len=min_len, max_len=max_len)

# ---------------------- Glue: two streams from one key ----------------------

def derive_master_key(user_key_str: str) -> bytes:
    # Convert a passphrase or hex to bytes in a stable way
    return hashlib.sha256(user_key_str.encode("utf-8")).digest()

def build_model(corpus_root: str) -> Markov2:
    texts = load_corpus_texts(corpus_root)
    sents: List[str] = []
    for t in texts:
        sents.extend(sentences_from_text(t))
    if len(sents) < 100:
        raise RuntimeError("Corpus too small: add more .txt files from Gutenberg/Wikisource.")
    m = Markov2()
    m.fit(sents)
    return m

def generate_two_streams(corpus_root: str, user_key: str,
                         n_sent: int, min_len: int, max_len: int) -> Tuple[List[str], List[str]]:
    model = build_model(corpus_root)
    master = derive_master_key(user_key)
    prg1 = DeterministicPRG(master, label="gamma1")
    prg2 = DeterministicPRG(master, label="gamma2")
    gamma1 = list(model.stream(prg1, n_sent=n_sent, min_len=min_len, max_len=max_len))
    gamma2 = list(model.stream(prg2, n_sent=n_sent, min_len=min_len, max_len=max_len))
    return gamma1, gamma2

# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Directory with .txt files (Gutenberg + Wikisource)")
    ap.add_argument("--key", required=True, help="User key / passphrase (shared)")
    ap.add_argument("--n_sent", type=int, default=50)
    ap.add_argument("--min_len", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=20)
    ap.add_argument("--out1", default="gamma1.txt")
    ap.add_argument("--out2", default="gamma2.txt")
    args = ap.parse_args()

    g1, g2 = generate_two_streams(args.corpus, args.key, args.n_sent, args.min_len, args.max_len)

    with open(args.out1, "w", encoding="utf-8") as f:
        f.write("\n".join(g1) + "\n")
    with open(args.out2, "w", encoding="utf-8") as f:
        f.write("\n".join(g2) + "\n")

    print(f"Wrote {len(g1)} sentences to {args.out1} and {len(g2)} to {args.out2}")

if __name__ == "__main__":
    main()
