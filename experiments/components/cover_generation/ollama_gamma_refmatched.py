#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reference-matched gamma stream generator for cover-message experiments.

Goal:
    Generate gamma1.txt and gamma2.txt so that their length, style,
    and surface distribution are closer to a benign reference corpus.

Example:
    python3 ollama_gamma_refmatched.py \
        --model phi3:mini-academic \
        --key "shared-secret-key" \
        --benign benign.txt \
        --n_sent 1000 \
        --out1 gamma1.txt \
        --out2 gamma2.txt
"""

import argparse
import hashlib
import hmac
import json
import random
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"


# -----------------------------
# Seeding and text utilities
# -----------------------------

def hmac_seed(key: str, label: bytes) -> int:
    """
    Deterministically derive a generation seed from a secret key and label.
    We reduce to a 31-bit positive integer for compatibility with local
    generation backends that expect ordinary integer seeds.
    """
    d = hmac.new(key.encode("utf-8"), label, hashlib.sha256).digest()
    return int.from_bytes(d[:8], "big") % (2**31 - 1)


def clean_line(x: str) -> str:
    x = x.strip()
    x = re.sub(r"^[\"'`]+|[\"'`]+$", "", x)
    x = re.sub(r"^\s*[-*\d.)]+\s*", "", x)
    x = x.replace("\n", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def words(x: str):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", x)


def word_count(x: str) -> int:
    return len(words(x))


def is_one_sentence(x: str) -> bool:
    # Accept one ordinary sentence ending with . ! or ?
    endings = re.findall(r"[.!?]", x)
    return 1 <= len(endings) <= 2


def too_similar(candidate: str, references, threshold: float = 0.82) -> bool:
    """
    Avoid copying benign examples too closely.
    """
    c = candidate.lower()
    for r in references:
        ratio = SequenceMatcher(None, c, r.lower()).ratio()
        if ratio >= threshold:
            return True
    return False


def has_bad_surface_form(x: str) -> bool:
    """
    Remove outputs that often reveal prompting or model artefacts.
    """
    bad_patterns = [
        r"\bHere is\b",
        r"\bHere are\b",
        r"\bAs an AI\b",
        r"\bI cannot\b",
        r"\bThis sentence\b",
        r"\bIn conclusion\b",
        r"\bFurthermore\b",
        r"\bMoreover\b",
        r"\bAdditionally\b",
        r"\bIt is important to note\b",
        r"[:;]",
        r"[{}[\]<>]",
    ]
    return any(re.search(p, x, flags=re.IGNORECASE) for p in bad_patterns)


def load_benign(path: str):
    lines = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = clean_line(line)
        if 5 <= word_count(line) <= 60 and is_one_sentence(line):
            lines.append(line)
    if len(lines) < 50:
        raise ValueError("The benign corpus is too small after cleaning. Use at least 300-1000 benign sentences if possible.")
    return lines


# -----------------------------
# Prompt construction
# -----------------------------

PROMPT_TEMPLATES = [
    """
You are writing ordinary benign English text from the same general domain as the examples.
Study the style, vocabulary level, sentence length, and rhythm of the examples.
Write exactly ONE new sentence.
Do not copy any phrase from the examples.
Do not mention that you are generating text.
Do not use quotation marks, lists, headings, semicolons, or colons.
Target length: about {target_len} words.

Examples:
{examples}

New sentence:
""",
    """
Write one plain English sentence that could naturally belong with the examples below.
Keep it mundane, specific, and human-like.
Avoid rare words, formal transitions, lists, quotation marks, and exaggerated claims.
The sentence should be about {target_len} words.

Examples:
{examples}

Sentence:
""",
    """
Generate exactly one natural sentence in the same register as these benign samples.
Match their simplicity, topic range, and sentence rhythm without reusing their wording.
Use ordinary punctuation only.
Length target: {target_len} words.

Samples:
{examples}

Output one sentence only:
""",
]


def build_prompt(rng: random.Random, benign_lines, target_len: int, k_examples: int = 4) -> str:
    examples = rng.sample(benign_lines, k=min(k_examples, len(benign_lines)))
    examples_block = "\n".join(f"- {e}" for e in examples)
    template = rng.choice(PROMPT_TEMPLATES)
    return template.format(target_len=target_len, examples=examples_block)


def sample_target_length(rng: random.Random, benign_lines) -> int:
    """
    Sample length from the empirical benign distribution rather than imposing
    a fixed 14-22 word range.
    """
    return word_count(rng.choice(benign_lines))


# -----------------------------
# Ollama generation
# -----------------------------

def call_ollama(model: str, prompt: str, seed: int, options: dict, timeout: int = 180) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            **options,
            "seed": seed,
        }
    }
    r = requests.post(OLLAMA_URL, data=json.dumps(body), timeout=timeout)
    r.raise_for_status()
    return clean_line(r.json().get("response", ""))


def valid_candidate(x: str, benign_lines, target_len: int, seen: set, tolerance: int = 4) -> bool:
    if not x:
        return False

    wc = word_count(x)

    if wc < max(5, target_len - tolerance):
        return False

    if wc > target_len + tolerance:
        return False

    if not is_one_sentence(x):
        return False

    if has_bad_surface_form(x):
        return False

    lx = x.lower()
    if lx in seen:
        return False

    if too_similar(x, benign_lines, threshold=0.82):
        return False

    # Avoid excessive repetition inside one sentence.
    toks = [w.lower() for w in words(x)]
    if len(toks) > 0 and len(set(toks)) / len(toks) < 0.68:
        return False

    return True


def generate_stream(
    model: str,
    benign_lines,
    n: int,
    base_seed: int,
    out_path: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    max_attempts_per_item: int = 12,
):
    rng = random.Random(base_seed)
    out = []
    seen = set()

    options = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "num_predict": 80,
    }

    attempts_total = 0

    while len(out) < n:
        target_len = sample_target_length(rng, benign_lines)

        accepted = None
        for attempt in range(max_attempts_per_item):
            attempts_total += 1

            prompt = build_prompt(rng, benign_lines, target_len)
            seed = (base_seed + attempts_total * 7919 + attempt) % (2**31 - 1)

            try:
                candidate = call_ollama(model, prompt, seed, options)
            except requests.RequestException as e:
                print(f"[WARN] Ollama request failed: {e}")
                time.sleep(1.0)
                continue

            candidate = clean_line(candidate)

            if valid_candidate(candidate, benign_lines, target_len, seen):
                accepted = candidate
                break

        if accepted is None:
            # Fallback: widen tolerance slightly after repeated failures.
            prompt = build_prompt(rng, benign_lines, target_len)
            seed = (base_seed + attempts_total * 104729) % (2**31 - 1)
            candidate = clean_line(call_ollama(model, prompt, seed, options))
            if candidate and candidate.lower() not in seen and not has_bad_surface_form(candidate):
                accepted = candidate

        if accepted:
            out.append(accepted)
            seen.add(accepted.lower())

        if len(out) % 50 == 0 and len(out) > 0:
            print(f"[INFO] Generated {len(out)}/{n} for {out_path}")

    Path(out_path).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote {len(out)} sentences to {out_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", default="phi3:mini-academic")
    ap.add_argument("--key", required=True)
    ap.add_argument("--benign", required=True)

    ap.add_argument("--n_sent", type=int, default=1000)
    ap.add_argument("--out1", default="gamma1.txt")
    ap.add_argument("--out2", default="gamma2.txt")

    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top_p", type=float, default=0.92)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--repeat_penalty", type=float, default=1.04)

    args = ap.parse_args()

    benign_lines = load_benign(args.benign)

    seed1 = hmac_seed(args.key, b"gamma1-refmatched")
    seed2 = hmac_seed(args.key, b"gamma2-refmatched")

    generate_stream(
        model=args.model,
        benign_lines=benign_lines,
        n=args.n_sent,
        base_seed=seed1,
        out_path=args.out1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repeat_penalty=args.repeat_penalty,
    )

    generate_stream(
        model=args.model,
        benign_lines=benign_lines,
        n=args.n_sent,
        base_seed=seed2,
        out_path=args.out2,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repeat_penalty=args.repeat_penalty,
    )


if __name__ == "__main__":
    main()
