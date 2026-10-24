#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CORPUS_LABELS = {
    "sherlock": "Sherlock",
    "alice": "Alice",
    "pride_prejudice": "Pride & Prejudice",
}

STREAM_LABELS = {
    "gamma1": r"$\gamma_1$",
    "gamma2": r"$\gamma_2$",
    "human_reference": "Human baseline",
}

COLORS = {
    "gamma1": "#2F6BFF",
    "gamma2": "#F28E2B",
    "human_reference": "#4D4D4D",
    "baseline": "#6C757D",
    "grid": "#D8DEE9",
    "text": "#222222",
}

METRIC_LABELS = {
    "mauve": "MAUVE ↑",
    "roberta_accuracy": "RoBERTa Accuracy ↓",
    "roberta_auc": "RoBERTa AUC ↓",
    "js_word_1": "Word-1 JS Divergence ↓",
    "js_word_2": "Word-2 JS Divergence ↓",
    "js_word_3": "Word-3 JS Divergence ↓",
    "js_char_1": "Char-1 JS Divergence ↓",
    "js_char_2": "Char-2 JS Divergence ↓",
    "js_char_3": "Char-3 JS Divergence ↓",
    "pos_cramers_v": "POS Cramér's V ↓",
}


def setup_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 400,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_fig(fig, out_dir, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def finish_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def mean_std(df, group_cols, metric):
    out = df.groupby(group_cols)[metric].agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def plot_mauve_by_corpus(df, out_dir):
    selected = df[df["comparison_type"] == "selected_vs_test"].copy()
    human = df[df["comparison_type"] == "human_vs_test"].copy()

    corpora = ["sherlock", "alice", "pride_prejudice"]
    streams = ["gamma1", "gamma2"]

    sel = mean_std(selected, ["corpus", "stream"], "mauve")
    hum = mean_std(human, ["corpus"], "mauve")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    x = np.arange(len(corpora))
    width = 0.22

    for i, stream in enumerate(streams):
        vals = []
        errs = []
        for corpus in corpora:
            row = sel[(sel["corpus"] == corpus) & (sel["stream"] == stream)]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["std"].iloc[0]) if not row.empty else 0.0)

        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            yerr=errs,
            capsize=4,
            label=STREAM_LABELS[stream],
            color=COLORS[stream],
            alpha=0.92,
            edgecolor="#222222",
            linewidth=0.6,
        )

    baseline_vals = []
    for corpus in corpora:
        row = hum[hum["corpus"] == corpus]
        baseline_vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)

    ax.plot(
        x,
        baseline_vals,
        marker="D",
        linewidth=2.0,
        color=COLORS["baseline"],
        label="Human-vs-human baseline",
    )

    ax.set_title("Distributional Similarity of Keyed Cover Streams")
    ax.set_ylabel("MAUVE score")
    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABELS[c] for c in corpora])
    ax.set_ylim(0.90, 1.01)
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    finish_axes(ax)
    save_fig(fig, out_dir, "fig01_mauve_by_corpus")


def plot_roberta_by_corpus(df, out_dir):
    selected = df[df["comparison_type"] == "selected_vs_test"].copy()
    corpora = ["sherlock", "alice", "pride_prejudice"]
    streams = ["gamma1", "gamma2"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharex=True)

    for ax, metric, title in [
        (axes[0], "roberta_accuracy", "Detector Accuracy"),
        (axes[1], "roberta_auc", "Detector ROC-AUC"),
    ]:
        summary = mean_std(selected, ["corpus", "stream"], metric)
        x = np.arange(len(corpora))
        width = 0.22

        for i, stream in enumerate(streams):
            vals = []
            errs = []
            for corpus in corpora:
                row = summary[(summary["corpus"] == corpus) & (summary["stream"] == stream)]
                vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
                errs.append(float(row["std"].iloc[0]) if not row.empty else 0.0)

            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width,
                yerr=errs,
                capsize=4,
                color=COLORS[stream],
                alpha=0.92,
                edgecolor="#222222",
                linewidth=0.6,
                label=STREAM_LABELS[stream],
            )

        ax.axhline(0.5, color="#111111", linestyle="--", linewidth=1.2, label="Chance level" if metric == "roberta_accuracy" else None)
        ax.set_title(title)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_xticks(x)
        ax.set_xticklabels([CORPUS_LABELS[c] for c in corpora], rotation=15, ha="right")
        ax.set_ylim(0.45, 0.70)
        finish_axes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Classifier-Based Separability of Keyed Cover Streams", y=1.03)
    save_fig(fig, out_dir, "fig02_roberta_by_corpus")


def plot_mauve_gap(df, out_dir):
    selected = df[df["comparison_type"] == "selected_vs_test"].copy()
    human = df[df["comparison_type"] == "human_vs_test"].copy()

    human_mean = human.groupby("corpus")["mauve"].mean().to_dict()
    selected["mauve_gap_to_human"] = selected.apply(
        lambda r: human_mean.get(r["corpus"], np.nan) - r["mauve"],
        axis=1
    )

    summary = mean_std(selected, ["corpus", "stream"], "mauve_gap_to_human")
    corpora = ["sherlock", "alice", "pride_prejudice"]
    streams = ["gamma1", "gamma2"]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(corpora))
    width = 0.22

    for i, stream in enumerate(streams):
        vals = []
        errs = []
        for corpus in corpora:
            row = summary[(summary["corpus"] == corpus) & (summary["stream"] == stream)]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["std"].iloc[0]) if not row.empty else 0.0)

        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            yerr=errs,
            capsize=4,
            color=COLORS[stream],
            alpha=0.92,
            edgecolor="#222222",
            linewidth=0.6,
            label=STREAM_LABELS[stream],
        )

    ax.axhline(0, color="#111111", linewidth=1.0)
    ax.set_title("MAUVE Gap from Human-vs-Human Baseline")
    ax.set_ylabel("Human baseline MAUVE − selected stream MAUVE")
    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABELS[c] for c in corpora])
    ax.legend(frameon=False, ncol=2)
    finish_axes(ax)
    save_fig(fig, out_dir, "fig03_mauve_gap_to_human_baseline")


def plot_js_word_divergence(df, out_dir):
    selected = df[df["comparison_type"] == "selected_vs_test"].copy()
    corpora = ["sherlock", "alice", "pride_prejudice"]
    metrics = ["js_word_1", "js_word_2", "js_word_3"]

    summary = selected.groupby("corpus")[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(corpora))
    width = 0.20

    palette = ["#4C78A8", "#F58518", "#54A24B"]

    for i, metric in enumerate(metrics):
        vals = []
        errs = []
        for corpus in corpora:
            row = summary[summary["corpus"] == corpus]
            vals.append(float(row[f"{metric}_mean"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row[f"{metric}_std"].iloc[0]) if not row.empty else 0.0)

        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            yerr=errs,
            capsize=4,
            color=palette[i],
            alpha=0.92,
            edgecolor="#222222",
            linewidth=0.6,
            label=METRIC_LABELS[metric],
        )

    ax.set_title("Word-Level Jensen-Shannon Divergence")
    ax.set_ylabel("JS divergence")
    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABELS[c] for c in corpora])
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.30))
    finish_axes(ax)
    save_fig(fig, out_dir, "fig04_word_js_divergence")


def plot_pos_effect(df, out_dir):
    selected = df[df["comparison_type"] == "selected_vs_test"].copy()
    corpora = ["sherlock", "alice", "pride_prejudice"]
    streams = ["gamma1", "gamma2"]

    summary = mean_std(selected, ["corpus", "stream"], "pos_cramers_v")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(corpora))
    width = 0.22

    for i, stream in enumerate(streams):
        vals = []
        errs = []
        for corpus in corpora:
            row = summary[(summary["corpus"] == corpus) & (summary["stream"] == stream)]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else np.nan)
            errs.append(float(row["std"].iloc[0]) if not row.empty else 0.0)

        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            yerr=errs,
            capsize=4,
            color=COLORS[stream],
            alpha=0.92,
            edgecolor="#222222",
            linewidth=0.6,
            label=STREAM_LABELS[stream],
        )

    ax.set_title("Syntactic Distribution Difference")
    ax.set_ylabel("POS Cramér's V")
    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABELS[c] for c in corpora])
    ax.legend(frameon=False, ncol=2)
    finish_axes(ax)
    save_fig(fig, out_dir, "fig05_pos_cramers_v")


def plot_metric_heatmap(df, out_dir):
    selected = df[df["comparison_type"] == "selected_vs_test"].copy()

    cols = ["mauve", "roberta_accuracy", "roberta_auc", "js_word_1", "js_word_2", "js_word_3", "pos_cramers_v"]
    cols = [c for c in cols if c in selected.columns]

    plot_df = selected.copy()
    plot_df["row_label"] = plot_df.apply(
        lambda r: f"{CORPUS_LABELS.get(r['corpus'], r['corpus'])}\nseed {int(r['seed'])}, {STREAM_LABELS.get(r['stream'], r['stream'])}",
        axis=1
    )

    matrix = plot_df[cols].astype(float).to_numpy()

    # Min-max normalise each metric for visual comparison only.
    norm = matrix.copy()
    for j in range(norm.shape[1]):
        col = norm[:, j]
        finite = np.isfinite(col)
        if finite.sum() > 0:
            mn, mx = np.nanmin(col), np.nanmax(col)
            if mx > mn:
                norm[:, j] = (col - mn) / (mx - mn)
            else:
                norm[:, j] = 0.5

    fig, ax = plt.subplots(figsize=(9.6, 8.4))
    im = ax.imshow(norm, aspect="auto", cmap="viridis")

    ax.set_title("Normalised Metric Heatmap Across Runs")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([METRIC_LABELS.get(c, c) for c in cols], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["row_label"].tolist())

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            txt = "NA" if not np.isfinite(value) else f"{value:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 0.55 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("Normalised value for visual comparison")

    save_fig(fig, out_dir, "fig06_metric_heatmap")


def plot_dashboard(df, out_dir):
    selected = df[df["comparison_type"] == "selected_vs_test"].copy()
    human = df[df["comparison_type"] == "human_vs_test"].copy()

    corpora = ["sherlock", "alice", "pride_prejudice"]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    axes = axes.ravel()

    # Panel A: MAUVE by corpus.
    ax = axes[0]
    mauve_summary = selected.groupby("corpus")["mauve"].agg(["mean", "std"]).reindex(corpora)
    human_summary = human.groupby("corpus")["mauve"].mean().reindex(corpora)
    x = np.arange(len(corpora))
    ax.bar(x, mauve_summary["mean"], yerr=mauve_summary["std"], capsize=4, color="#2F6BFF", alpha=0.9, edgecolor="#222222", linewidth=0.6)
    ax.plot(x, human_summary, marker="D", color="#4D4D4D", linewidth=2, label="Human baseline")
    ax.set_title("(a) MAUVE")
    ax.set_ylabel("Score")
    ax.set_ylim(0.90, 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABELS[c] for c in corpora], rotation=15, ha="right")
    ax.legend(frameon=False)
    finish_axes(ax)

    # Panel B: AUC by corpus.
    ax = axes[1]
    auc_summary = selected.groupby("corpus")["roberta_auc"].agg(["mean", "std"]).reindex(corpora)
    ax.bar(x, auc_summary["mean"], yerr=auc_summary["std"], capsize=4, color="#F28E2B", alpha=0.9, edgecolor="#222222", linewidth=0.6)
    ax.axhline(0.5, color="#111111", linestyle="--", linewidth=1.2)
    ax.set_title("(b) RoBERTa AUC")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.45, 0.70)
    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABELS[c] for c in corpora], rotation=15, ha="right")
    finish_axes(ax)

    # Panel C: Word JS.
    ax = axes[2]
    js_summary = selected.groupby("corpus")[["js_word_1", "js_word_2", "js_word_3"]].mean().reindex(corpora)
    width = 0.22
    for i, metric in enumerate(["js_word_1", "js_word_2", "js_word_3"]):
        ax.bar(x + (i - 1) * width, js_summary[metric], width, label=metric.replace("js_word_", "word-"), alpha=0.9, edgecolor="#222222", linewidth=0.6)
    ax.set_title("(c) Word JS divergence")
    ax.set_ylabel("JS divergence")
    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABELS[c] for c in corpora], rotation=15, ha="right")
    ax.legend(frameon=False, ncol=3)
    finish_axes(ax)

    # Panel D: POS Cramer's V.
    ax = axes[3]
    pos_summary = selected.groupby("corpus")["pos_cramers_v"].agg(["mean", "std"]).reindex(corpora)
    ax.bar(x, pos_summary["mean"], yerr=pos_summary["std"], capsize=4, color="#54A24B", alpha=0.9, edgecolor="#222222", linewidth=0.6)
    ax.set_title("(d) POS Cramér's V")
    ax.set_ylabel("Effect size")
    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABELS[c] for c in corpora], rotation=15, ha="right")
    finish_axes(ax)

    fig.suptitle("Keyed Corpus-Conditioned Cover Selection: Summary Dashboard", y=1.02, fontsize=14)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig07_summary_dashboard")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/cover_eval_detailed_with_js_pos.csv")
    ap.add_argument("--out_dir", default="results/plots")
    args = ap.parse_args()

    setup_matplotlib()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = {"corpus", "seed", "stream", "comparison_type", "mauve"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    plot_mauve_by_corpus(df, out_dir)
    plot_roberta_by_corpus(df, out_dir)
    plot_mauve_gap(df, out_dir)

    if {"js_word_1", "js_word_2", "js_word_3"}.issubset(df.columns):
        plot_js_word_divergence(df, out_dir)

    if "pos_cramers_v" in df.columns:
        plot_pos_effect(df, out_dir)

    plot_metric_heatmap(df, out_dir)

    if {"js_word_1", "js_word_2", "js_word_3", "pos_cramers_v"}.issubset(df.columns):
        plot_dashboard(df, out_dir)

    print(f"[DONE] Plots written to: {out_dir}")
    for p in sorted(out_dir.glob("*")):
        print(p)


if __name__ == "__main__":
    main()
