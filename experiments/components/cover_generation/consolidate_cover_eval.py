#!/usr/bin/env python3
import argparse
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
import spacy


CORPUS_PREFIX = {
    "sherlock": "sherlock",
    "alice": "alice",
    "pride_prejudice": "pride",
}

EXPECTED = [
    ("sherlock", 42), ("sherlock", 43), ("sherlock", 44),
    ("alice", 42), ("alice", 43), ("alice", 44),
    ("pride_prejudice", 42), ("pride_prejudice", 43), ("pride_prejudice", 44),
]


def read_lines(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return [x.strip() for x in path.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]


def words(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def char_ngrams(text, n):
    text = re.sub(r"\s+", " ", text.lower())
    return [text[i:i+n] for i in range(max(0, len(text) - n + 1))]


def word_ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(max(0, len(tokens) - n + 1))]


def js_divergence_from_counts(c1, c2):
    vocab = sorted(set(c1) | set(c2))
    if not vocab:
        return np.nan

    p = np.array([c1.get(v, 0) for v in vocab], dtype=float)
    q = np.array([c2.get(v, 0) for v in vocab], dtype=float)

    if p.sum() == 0 or q.sum() == 0:
        return np.nan

    p /= p.sum()
    q /= q.sum()

    # scipy.spatial.distance.jensenshannon returns JS distance,
    # i.e., sqrt(JS divergence), so square it for divergence.
    d = jensenshannon(p, q, base=2.0)
    return float(d * d)


def compute_js_metrics(p_lines, q_lines):
    p_text = " ".join(p_lines)
    q_text = " ".join(q_lines)

    p_words = words(p_text)
    q_words = words(q_text)

    out = {}

    for n in (1, 2, 3):
        out[f"js_char_{n}"] = js_divergence_from_counts(
            Counter(char_ngrams(p_text, n)),
            Counter(char_ngrams(q_text, n)),
        )
        out[f"js_word_{n}"] = js_divergence_from_counts(
            Counter(word_ngrams(p_words, n)),
            Counter(word_ngrams(q_words, n)),
        )

    return out


def pos_counts(lines, nlp, exclude_punct=False):
    counts = Counter()
    for doc in nlp.pipe(lines, batch_size=64):
        for tok in doc:
            if tok.is_space:
                continue
            if exclude_punct and tok.pos_ == "PUNCT":
                continue
            counts[tok.pos_] += 1
    return counts


def compute_pos_metrics(p_lines, q_lines, nlp, min_pos_count=5, exclude_punct=False):
    c1 = pos_counts(p_lines, nlp, exclude_punct=exclude_punct)
    c2 = pos_counts(q_lines, nlp, exclude_punct=exclude_punct)

    all_tags = sorted(set(c1) | set(c2))
    kept = []
    other1 = 0
    other2 = 0

    for tag in all_tags:
        total = c1.get(tag, 0) + c2.get(tag, 0)
        if total < min_pos_count:
            other1 += c1.get(tag, 0)
            other2 += c2.get(tag, 0)
        else:
            kept.append(tag)

    row1 = [c1.get(tag, 0) for tag in kept]
    row2 = [c2.get(tag, 0) for tag in kept]

    if other1 + other2 > 0:
        row1.append(other1)
        row2.append(other2)

    table = np.array([row1, row2], dtype=float)

    if table.shape[1] < 2 or table.sum() == 0:
        return {
            "pos_chi2": np.nan,
            "pos_pvalue": np.nan,
            "pos_dof": np.nan,
            "pos_cramers_v": np.nan,
        }

    chi2, pvalue, dof, expected = chi2_contingency(table, correction=False)

    n = table.sum()
    r, k = table.shape
    denom = n * min(r - 1, k - 1)
    cramers_v = math.sqrt(chi2 / denom) if denom > 0 else np.nan

    return {
        "pos_chi2": float(chi2),
        "pos_pvalue": float(pvalue),
        "pos_dof": int(dof),
        "pos_cramers_v": float(cramers_v),
    }


def build_paths(corpus, seed, stream):
    prefix = CORPUS_PREFIX[corpus]

    test_path = f"corpus/{prefix}_test_seed{seed}_1000.txt"

    if stream == "human_reference":
        p_path = f"corpus/{prefix}_reference_seed{seed}_1000.txt"
        comparison_type = "human_vs_test"
    else:
        p_path = f"{stream}_{prefix}_selected_seed{seed}.txt"
        comparison_type = "selected_vs_test"

    return p_path, test_path, comparison_type


def expected_rows():
    rows = []
    for corpus, seed in EXPECTED:
        for stream in ("gamma1", "gamma2", "human_reference"):
            _, _, comparison_type = build_paths(corpus, seed, stream)
            rows.append({
                "corpus": corpus,
                "seed": seed,
                "stream": stream,
                "comparison_type": comparison_type,
            })
    return pd.DataFrame(rows)


def ensure_expected_rows(df):
    exp = expected_rows()

    key_cols = ["corpus", "seed", "stream", "comparison_type"]
    merged = exp.merge(df, on=key_cols, how="left", suffixes=("", "_old"))

    # Preserve run_id if present; otherwise infer from corpus/seed ordering.
    if "run_id" not in merged.columns:
        run_map = {
            ("sherlock", 42): 1, ("sherlock", 43): 2, ("sherlock", 44): 3,
            ("alice", 42): 4, ("alice", 43): 5, ("alice", 44): 6,
            ("pride_prejudice", 42): 7, ("pride_prejudice", 43): 8, ("pride_prejudice", 44): 9,
        }
        merged["run_id"] = merged.apply(lambda r: run_map[(r["corpus"], int(r["seed"]))], axis=1)

    return merged


def flatten_columns(df):
    df.columns = [
        "_".join([str(x) for x in col if str(x) != ""]).strip("_")
        if isinstance(col, tuple) else col
        for col in df.columns
    ]
    return df


def make_summaries(df, out_dir):
    metric_cols = [
        "mauve",
        "roberta_accuracy",
        "roberta_auc",
        "js_char_1", "js_char_2", "js_char_3",
        "js_word_1", "js_word_2", "js_word_3",
        "pos_chi2", "pos_pvalue", "pos_cramers_v",
    ]

    available = [c for c in metric_cols if c in df.columns]

    selected = df[df["comparison_type"] == "selected_vs_test"].copy()
    human = df[df["comparison_type"] == "human_vs_test"].copy()

    by_corpus_stream = selected.groupby(["corpus", "stream"])[available].agg(["count", "mean", "std"])
    by_corpus_stream = flatten_columns(by_corpus_stream.reset_index())
    by_corpus_stream.to_csv(out_dir / "cover_eval_mean_std_by_corpus_stream.csv", index=False)

    by_corpus = selected.groupby(["corpus"])[available].agg(["count", "mean", "std"])
    by_corpus = flatten_columns(by_corpus.reset_index())
    by_corpus.to_csv(out_dir / "cover_eval_mean_std_by_corpus.csv", index=False)

    overall = selected[available].agg(["count", "mean", "std"]).T.reset_index()
    overall.columns = ["metric", "count", "mean", "std"]
    overall.to_csv(out_dir / "cover_eval_mean_std_overall_selected.csv", index=False)

    human_summary = human.groupby(["corpus"])[available].agg(["count", "mean", "std"])
    human_summary = flatten_columns(human_summary.reset_index())
    human_summary.to_csv(out_dir / "cover_eval_human_baseline_mean_std_by_corpus.csv", index=False)

    # Compact LaTeX-ready table.
    pub_cols = [
        "mauve", "roberta_accuracy", "roberta_auc",
        "js_word_1", "js_word_2", "js_word_3",
        "pos_cramers_v",
    ]
    pub_cols = [c for c in pub_cols if c in selected.columns]

    pub = selected.groupby(["corpus"])[pub_cols].agg(["mean", "std"])
    pub = flatten_columns(pub.reset_index())
    pub.to_csv(out_dir / "cover_eval_publication_table.csv", index=False)

    try:
        pub.to_latex(out_dir / "cover_eval_publication_table.tex", index=False, float_format="%.4f")
    except Exception as e:
        print(f"[WARN] Could not write LaTeX table: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", default="results/cover_eval_summary.csv")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--spacy_model", default="en_core_web_sm")
    ap.add_argument("--min_pos_count", type=int, default=5)
    ap.add_argument("--exclude_punct", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary_csv)
    if summary_path.exists():
        df = pd.read_csv(summary_path)
    else:
        df = pd.DataFrame(columns=[
            "run_id", "corpus", "seed", "stream", "comparison_type",
            "mauve", "roberta_accuracy", "roberta_auc", "notes"
        ])

    df = ensure_expected_rows(df)

    print("[INFO] Loading spaCy model...")
    nlp = spacy.load(args.spacy_model, disable=["ner", "parser", "lemmatizer", "textcat"])

    detailed_rows = []

    for _, row in df.iterrows():
        corpus = row["corpus"]
        seed = int(row["seed"])
        stream = row["stream"]

        p_path, q_path, comparison_type = build_paths(corpus, seed, stream)

        print(f"[INFO] Processing {corpus} seed={seed} stream={stream}")
        print(f"       p={p_path}")
        print(f"       q={q_path}")

        new_row = row.to_dict()
        new_row["p_file"] = p_path
        new_row["q_file"] = q_path
        new_row["comparison_type"] = comparison_type

        try:
            p_lines = read_lines(p_path)
            q_lines = read_lines(q_path)

            js = compute_js_metrics(p_lines, q_lines)
            pos = compute_pos_metrics(
                p_lines, q_lines, nlp,
                min_pos_count=args.min_pos_count,
                exclude_punct=args.exclude_punct
            )

            new_row.update(js)
            new_row.update(pos)
            new_row["p_n_lines"] = len(p_lines)
            new_row["q_n_lines"] = len(q_lines)
            new_row["file_status"] = "ok"

        except Exception as e:
            print(f"[WARN] Failed for {corpus} seed={seed} stream={stream}: {e}")
            new_row["file_status"] = f"failed: {e}"

        detailed_rows.append(new_row)

    detailed = pd.DataFrame(detailed_rows)

    detailed_path = out_dir / "cover_eval_detailed_with_js_pos.csv"
    detailed.to_csv(detailed_path, index=False)

    missing_numeric = detailed[
        detailed[["mauve", "roberta_accuracy", "roberta_auc"]].isna().any(axis=1)
        & (detailed["comparison_type"] == "selected_vs_test")
    ]
    missing_numeric.to_csv(out_dir / "cover_eval_missing_numeric_metrics.csv", index=False)

    make_summaries(detailed, out_dir)

    print("\n[DONE]")
    print(f"Wrote: {detailed_path}")
    print(f"Wrote: {out_dir / 'cover_eval_mean_std_by_corpus_stream.csv'}")
    print(f"Wrote: {out_dir / 'cover_eval_mean_std_by_corpus.csv'}")
    print(f"Wrote: {out_dir / 'cover_eval_mean_std_overall_selected.csv'}")
    print(f"Wrote: {out_dir / 'cover_eval_publication_table.csv'}")
    print(f"Wrote: {out_dir / 'cover_eval_publication_table.tex'}")
    print(f"Wrote: {out_dir / 'cover_eval_missing_numeric_metrics.csv'}")


if __name__ == "__main__":
    main()
