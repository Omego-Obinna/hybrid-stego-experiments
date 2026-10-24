#!/usr/bin/env python3
"""
Aggregate and plot QIM-style multichannel robustness/re-synchronisation results.

This script reads the run_summary.csv produced by multichannel_qim_resync_sim.py
and generates:
  - full-grid CSV summaries for robustness and latency
  - publication-ready LaTeX tables using booktabs
  - cleaner representative-condition plots using 95% confidence intervals
  - one combined three-panel figure and three individual figures

Design choice:
  Tables use the full parameter grid and report mean +/- standard deviation.
  Plots use one representative configuration and report mean +/- 95% CI.
  This avoids visually misleading large SD bars caused by averaging over
  heterogeneous windows, heartbeat values, skew values and timeout settings.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Use a non-interactive backend so the script works on servers/SSH.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


REQUIRED_COLUMNS = [
    "loss_rate", "jitter_ms", "skew_ms", "window_W", "heartbeat_h", "ecc_enabled",
    "acceptance_rate", "recovery_rate", "resync_success_rate",
    "mean_e2e_latency_ms", "p95_e2e_latency_ms", "mean_resync_delay_ms",
    "throughput_epochs_per_sec", "integrity_failure_rate", "false_accepts",
]


@dataclass(frozen=True)
class PlotConfig:
    window_W: int
    heartbeat_h: int
    skew_ms: float
    timeout_factor: Optional[float]
    ci_level: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate publication-ready tables and cleaner representative-condition "
            "plots from QIM multichannel re-synchronisation results."
        )
    )
    p.add_argument("--summary-csv", required=True, help="Path to run_summary.csv from multichannel_qim_resync_sim.py")
    p.add_argument("--attack-summary-csv", default=None, help="Optional replay/tamper run_summary.csv")
    p.add_argument("--out-dir", required=True, help="Output directory for CSV, LaTeX and figures")
    p.add_argument("--representative-window", type=int, default=5, help="Window W used for plots")
    p.add_argument("--representative-heartbeat", type=int, default=10, help="Heartbeat h used for plots")
    p.add_argument("--representative-skew-ms", type=float, default=100.0, help="Skew value used for plots")
    p.add_argument("--representative-timeout-factor", type=float, default=2.0, help="Timeout factor used for plots if available")
    p.add_argument("--ci-level", type=float, default=0.95, help="Confidence level for plot error bars")
    p.add_argument("--dpi", type=int, default=600, help="PNG output DPI")
    p.add_argument("--format", choices=["pdf", "png", "both"], default="both", help="Figure output format")
    p.add_argument("--make-individual", action="store_true", help="Also write individual plots in addition to the three-panel figure")
    p.add_argument("--no-latex", action="store_true", help="Do not write LaTeX tables")
    p.add_argument("--strict", action="store_true", help="Fail if representative filter returns no rows")
    return p.parse_args()


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing) +
            f"\nAvailable columns: {', '.join(df.columns)}"
        )
    return normalise_df(df)


def normalise_bool(v) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


def normalise_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ecc_enabled"] = df["ecc_enabled"].map(normalise_bool)
    numeric_cols = [
        "loss_rate", "jitter_ms", "skew_ms", "window_W", "heartbeat_h",
        "acceptance_rate", "recovery_rate", "resync_success_rate",
        "mean_e2e_latency_ms", "median_e2e_latency_ms", "p95_e2e_latency_ms",
        "mean_resync_delay_ms", "median_resync_delay_ms", "p95_resync_delay_ms",
        "throughput_epochs_per_sec", "throughput_bits_per_sec",
        "integrity_failure_rate", "false_accepts", "timeout_tau_ms",
        "timeout_factor",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["window_W"] = df["window_W"].astype("Int64")
    df["heartbeat_h"] = df["heartbeat_h"].astype("Int64")

    # Derive timeout_factor if possible and not already present.
    if "timeout_factor" not in df.columns and {"timeout_tau_ms", "skew_ms"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["timeout_factor"] = df["timeout_tau_ms"] / df["skew_ms"]

    return df


def is_fraction_series(s: pd.Series) -> bool:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty:
        return True
    return float(vals.quantile(0.99)) <= 1.5


def to_percent_array(s: Sequence[float]) -> np.ndarray:
    arr = np.asarray(s, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanpercentile(finite, 99) <= 1.5:
        return arr * 100.0
    return arr


def metric_to_percent(df: pd.DataFrame, col: str) -> pd.Series:
    if is_fraction_series(df[col]):
        return df[col] * 100.0
    return df[col]


def ci_half_width(x: Iterable[float], level: float = 0.95) -> float:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n <= 1:
        return 0.0
    sem = float(np.std(arr, ddof=1) / math.sqrt(n))
    if stats is not None:
        alpha = 1.0 - level
        return float(stats.t.ppf(1.0 - alpha / 2.0, n - 1) * sem)
    # Normal approximation fallback.
    z = 1.96 if abs(level - 0.95) < 1e-9 else 1.96
    return float(z * sem)


def summarise_mean_sd(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    out = df.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std", "count"]).reset_index()
    # Flatten columns.
    flat = []
    for c in out.columns:
        if isinstance(c, tuple):
            flat.append("_".join([str(x) for x in c if x != ""]).strip("_"))
        else:
            flat.append(str(c))
    out.columns = flat
    return out


def summarise_for_plot(df: pd.DataFrame, group_cols: List[str], metric: str, level: float = 0.95) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = pd.to_numeric(g[metric], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        rows.append({
            **dict(zip(group_cols, keys)),
            "n": int(vals.size),
            "mean": float(np.mean(vals)) if vals.size else np.nan,
            "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "ci": ci_half_width(vals, level=level) if vals.size else np.nan,
        })
    return pd.DataFrame(rows)


def pct_fmt_mean_sd(mean: float, sd: float, decimals: int = 2) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(sd):
        sd = 0.0
    return f"${mean:.{decimals}f} \\pm {sd:.{decimals}f}$"


def num_fmt(n) -> str:
    try:
        return f"{int(round(float(n))):,}"
    except Exception:
        return "--"


def ecc_label(v: bool) -> str:
    return "On" if bool(v) else "Off"


def write_robustness_table(df: pd.DataFrame, out_dir: Path, write_latex: bool = True) -> pd.DataFrame:
    tmp = df.copy()
    for c in ["acceptance_rate", "recovery_rate", "resync_success_rate", "integrity_failure_rate"]:
        tmp[c + "_pct"] = metric_to_percent(tmp, c)
    tmp["loss_pct"] = tmp["loss_rate"] * 100.0 if is_fraction_series(tmp["loss_rate"]) else tmp["loss_rate"]

    # False accepts are totals, not mean+sd.
    summary = summarise_mean_sd(
        tmp,
        ["loss_pct", "ecc_enabled"],
        ["acceptance_rate_pct", "recovery_rate_pct", "resync_success_rate_pct", "integrity_failure_rate_pct"],
    )
    false_counts = tmp.groupby(["loss_pct", "ecc_enabled"], dropna=False)["false_accepts"].sum().reset_index(name="false_accepts_total")
    summary = summary.merge(false_counts, on=["loss_pct", "ecc_enabled"], how="left")
    summary = summary.sort_values(["loss_pct", "ecc_enabled"])
    summary.to_csv(out_dir / "robustness_by_loss_ecc.csv", index=False)

    if write_latex:
        lines = []
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{QIM-style multichannel robustness under channel loss. Results are reported as mean~$\pm$~standard deviation over simulator runs, except false accepts, which are reported as total counts.}")
        lines.append(r"\label{tab:qim_resync_robustness}")
        lines.append(r"\setlength{\tabcolsep}{5pt}")
        lines.append(r"\renewcommand{\arraystretch}{1.15}")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{@{}llrccccc@{}}")
        lines.append(r"\toprule")
        lines.append(r"Loss $r$ & ECC & Runs & Acceptance (\%) & Recovery (\%) & Re-sync Success (\%) & Integrity Fail. (\%) & False Accepts \\")
        lines.append(r"\midrule")
        prev_loss = None
        for _, row in summary.iterrows():
            loss = row["loss_pct"]
            if prev_loss is not None and loss != prev_loss:
                lines.append(r"\addlinespace[2pt]")
            prev_loss = loss
            runs = row.get("acceptance_rate_pct_count", np.nan)
            lines.append(
                f"${loss:.0f}\\%$ & {ecc_label(row['ecc_enabled'])} & {num_fmt(runs)} & "
                f"{pct_fmt_mean_sd(row['acceptance_rate_pct_mean'], row['acceptance_rate_pct_std'])} & "
                f"{pct_fmt_mean_sd(row['recovery_rate_pct_mean'], row['recovery_rate_pct_std'])} & "
                f"{pct_fmt_mean_sd(row['resync_success_rate_pct_mean'], row['resync_success_rate_pct_std'])} & "
                f"{pct_fmt_mean_sd(row['integrity_failure_rate_pct_mean'], row['integrity_failure_rate_pct_std'], decimals=4)} & "
                f"{num_fmt(row['false_accepts_total'])} \\\\" 
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}%")
        lines.append(r"}")
        lines.append(r"\vspace{2pt}")
        lines.append(r"\begin{minipage}{0.98\textwidth}")
        lines.append(r"\footnotesize")
        lines.append(r"\emph{Note:} ECC denotes the optional erasure-recovery mechanism used by the receiver. Acceptance measures the proportion of epochs accepted after freshness, window, and MAC checks; recovery measures epochs reconstructed after partial channel loss; and re-sync success measures successful restoration of receiver alignment after incomplete or delayed epochs.")
        lines.append(r"\end{minipage}")
        lines.append(r"\end{table*}")
        (out_dir / "table_resync_robustness.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_latency_table(df: pd.DataFrame, out_dir: Path, write_latex: bool = True) -> pd.DataFrame:
    tmp = df.copy()
    summary = summarise_mean_sd(
        tmp,
        ["jitter_ms", "ecc_enabled"],
        ["mean_e2e_latency_ms", "p95_e2e_latency_ms", "mean_resync_delay_ms", "throughput_epochs_per_sec"],
    ).sort_values(["jitter_ms", "ecc_enabled"])
    summary.to_csv(out_dir / "latency_by_jitter_ecc.csv", index=False)

    if write_latex:
        lines = []
        lines.append(r"\begin{table*}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{QIM-style multichannel robustness under channel jitter. Results are reported as mean~$\pm$~standard deviation over simulator runs.}")
        lines.append(r"\label{tab:qim_resync_latency}")
        lines.append(r"\setlength{\tabcolsep}{5pt}")
        lines.append(r"\renewcommand{\arraystretch}{1.15}")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{@{}llrcccc@{}}")
        lines.append(r"\toprule")
        lines.append(r"Jitter (ms) & ECC & Runs & Mean Latency (ms) & P95 Latency (ms) & Mean Re-sync Delay (ms) & Throughput (epochs/s) \\")
        lines.append(r"\midrule")
        prev_j = None
        for _, row in summary.iterrows():
            j = row["jitter_ms"]
            if prev_j is not None and j != prev_j:
                lines.append(r"\addlinespace[2pt]")
            prev_j = j
            runs = row.get("mean_e2e_latency_ms_count", np.nan)
            lines.append(
                f"${j:.0f}$ & {ecc_label(row['ecc_enabled'])} & {num_fmt(runs)} & "
                f"{pct_fmt_mean_sd(row['mean_e2e_latency_ms_mean'], row['mean_e2e_latency_ms_std'])} & "
                f"{pct_fmt_mean_sd(row['p95_e2e_latency_ms_mean'], row['p95_e2e_latency_ms_std'])} & "
                f"{pct_fmt_mean_sd(row['mean_resync_delay_ms_mean'], row['mean_resync_delay_ms_std'])} & "
                f"{pct_fmt_mean_sd(row['throughput_epochs_per_sec_mean'], row['throughput_epochs_per_sec_std'])} \\\\" 
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}%")
        lines.append(r"}")
        lines.append(r"\vspace{2pt}")
        lines.append(r"\begin{minipage}{0.98\textwidth}")
        lines.append(r"\footnotesize")
        lines.append(r"\emph{Note:} ECC denotes the optional erasure-recovery mechanism used by the receiver. Mean latency and P95 latency measure end-to-end epoch acceptance delay, while mean re-synchronisation delay measures the time required to restore receiver alignment after incomplete, delayed, or reordered channel observations.")
        lines.append(r"\end{minipage}")
        lines.append(r"\end{table*}")
        (out_dir / "table_resync_latency.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def representative_filter(df: pd.DataFrame, cfg: PlotConfig, strict: bool = False) -> pd.DataFrame:
    plot_df = df.copy()
    conditions = []
    if "window_W" in plot_df.columns:
        conditions.append(plot_df["window_W"].astype(float) == float(cfg.window_W))
    if "heartbeat_h" in plot_df.columns:
        conditions.append(plot_df["heartbeat_h"].astype(float) == float(cfg.heartbeat_h))
    if "skew_ms" in plot_df.columns:
        conditions.append(np.isclose(plot_df["skew_ms"].astype(float), float(cfg.skew_ms)))
    if cfg.timeout_factor is not None and "timeout_factor" in plot_df.columns:
        conditions.append(np.isclose(plot_df["timeout_factor"].astype(float), float(cfg.timeout_factor), rtol=0, atol=1e-6))
    if conditions:
        mask = np.logical_and.reduce(conditions)
        filtered = plot_df.loc[mask].copy()
    else:
        filtered = plot_df

    if filtered.empty:
        msg = (
            "Representative filter returned no rows. Falling back to full grid for plots. "
            f"Requested W={cfg.window_W}, h={cfg.heartbeat_h}, skew={cfg.skew_ms}, "
            f"timeout_factor={cfg.timeout_factor}."
        )
        if strict:
            raise ValueError(msg)
        print("[WARNING] " + msg, file=sys.stderr)
        return plot_df
    return filtered


def setup_axis(ax, xlabel: str, ylabel: str, ylim: Optional[Tuple[float, float]] = None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.30)


def plot_series_errorbar(ax, data: pd.DataFrame, x_col: str, y_col: str, x_scale: float, y_scale: float, label_prefix: str = "ECC") -> None:
    markers = {False: "o", True: "s"}
    for ecc in [False, True]:
        sub = data[data["ecc_enabled"] == ecc].sort_values(x_col)
        if sub.empty:
            continue
        x = sub[x_col].to_numpy(dtype=float) * x_scale
        y = sub["mean"].to_numpy(dtype=float) * y_scale
        yerr = sub["ci"].to_numpy(dtype=float) * y_scale
        ax.errorbar(
            x, y, yerr=yerr,
            marker=markers[ecc], linewidth=2.0,
            capsize=4, capthick=1.2,
            label=f"{label_prefix} {'on' if ecc else 'off'}",
        )


def save_fig(fig, out_dir: Path, stem: str, fmt: str, dpi: int) -> None:
    if fmt in {"pdf", "both"}:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    if fmt in {"png", "both"}:
        fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")


def make_plots(df: pd.DataFrame, out_dir: Path, cfg: PlotConfig, fmt: str, dpi: int, make_individual: bool, strict: bool) -> pd.DataFrame:
    plot_df = representative_filter(df, cfg, strict=strict)
    plot_df = plot_df.copy()
    plot_df["acceptance_rate_pct"] = metric_to_percent(plot_df, "acceptance_rate")
    plot_df["resync_success_rate_pct"] = metric_to_percent(plot_df, "resync_success_rate")
    plot_df["loss_pct"] = plot_df["loss_rate"] * 100.0 if is_fraction_series(plot_df["loss_rate"]) else plot_df["loss_rate"]

    acc = summarise_for_plot(plot_df, ["loss_pct", "ecc_enabled"], "acceptance_rate_pct", cfg.ci_level)
    resync = summarise_for_plot(plot_df, ["loss_pct", "ecc_enabled"], "resync_success_rate_pct", cfg.ci_level)
    lat = summarise_for_plot(plot_df, ["jitter_ms", "ecc_enabled"], "p95_e2e_latency_ms", cfg.ci_level)

    acc.to_csv(out_dir / "plot_acceptance_vs_loss_representative.csv", index=False)
    resync.to_csv(out_dir / "plot_resync_success_vs_loss_representative.csv", index=False)
    lat.to_csv(out_dir / "plot_latency_vs_jitter_representative.csv", index=False)

    # Combined three-panel figure.
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.8))

    plot_series_errorbar(axes[0], acc, "loss_pct", "acceptance_rate_pct", x_scale=1.0, y_scale=1.0)
    axes[0].set_title("(a) Acceptance")
    setup_axis(axes[0], "Loss rate (%)", "Acceptance rate (%)", ylim=(0, 100))
    axes[0].set_xticks(sorted(acc["loss_pct"].dropna().unique()))

    plot_series_errorbar(axes[1], resync, "loss_pct", "resync_success_rate_pct", x_scale=1.0, y_scale=1.0)
    axes[1].set_title("(b) Re-synchronisation")
    setup_axis(axes[1], "Loss rate (%)", "Re-synchronisation success rate (%)", ylim=(0, 100))
    axes[1].set_xticks(sorted(resync["loss_pct"].dropna().unique()))

    plot_series_errorbar(axes[2], lat, "jitter_ms", "p95_e2e_latency_ms", x_scale=1.0, y_scale=1.0)
    axes[2].set_title("(c) P95 latency")
    setup_axis(axes[2], "Jitter (ms)", "P95 end-to-end latency (ms)", ylim=None)
    axes[2].set_xticks(sorted(lat["jitter_ms"].dropna().unique()))

    # One shared legend.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_fig(fig, out_dir, "figure_qim_resync_combined_representative", fmt, dpi)
    plt.close(fig)

    if make_individual:
        specs = [
            (acc, "loss_pct", "Acceptance rate (%)", "Loss rate (%)", "Acceptance rate (%)", (0, 100), "figure_acceptance_vs_loss_representative"),
            (resync, "loss_pct", "Re-synchronisation success rate (%)", "Loss rate (%)", "Re-synchronisation success rate (%)", (0, 100), "figure_resync_success_vs_loss_representative"),
            (lat, "jitter_ms", "P95 end-to-end latency (ms)", "Jitter (ms)", "P95 end-to-end latency (ms)", None, "figure_latency_vs_jitter_representative"),
        ]
        for data, x_col, title, xlabel, ylabel, ylim, stem in specs:
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            plot_series_errorbar(ax, data, x_col, title, x_scale=1.0, y_scale=1.0)
            setup_axis(ax, xlabel, ylabel, ylim=ylim)
            ax.set_xticks(sorted(data[x_col].dropna().unique()))
            ax.legend(frameon=False)
            fig.tight_layout()
            save_fig(fig, out_dir, stem, fmt, dpi)
            plt.close(fig)

    manifest_plot = pd.DataFrame({
        "filter": ["representative"],
        "rows_used_for_plots": [len(plot_df)],
        "window_W": [cfg.window_W],
        "heartbeat_h": [cfg.heartbeat_h],
        "skew_ms": [cfg.skew_ms],
        "timeout_factor": [cfg.timeout_factor],
        "ci_level": [cfg.ci_level],
    })
    manifest_plot.to_csv(out_dir / "representative_plot_filter.csv", index=False)
    return manifest_plot


def write_attack_table(attack_csv: Optional[str], out_dir: Path, write_latex: bool = True) -> Optional[pd.DataFrame]:
    if not attack_csv:
        return None
    attack_df = read_summary(Path(attack_csv))
    # Use available active-interference columns where present.
    group_cols = ["loss_rate", "ecc_enabled"]
    if "replay_rate" in attack_df.columns:
        group_cols.append("replay_rate")
    if "tamper_rate" in attack_df.columns:
        group_cols.append("tamper_rate")

    for c in ["integrity_failure_rate"]:
        attack_df[c + "_pct"] = metric_to_percent(attack_df, c)

    value_cols = ["mac_failures", "replay_rejections", "stale_epoch_rejections", "false_accepts", "integrity_failure_rate_pct"]
    value_cols = [c for c in value_cols if c in attack_df.columns]
    if not value_cols:
        return None
    summary = summarise_mean_sd(attack_df, group_cols, value_cols)
    summary.to_csv(out_dir / "active_interference_summary.csv", index=False)

    if write_latex:
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\caption{Active replay/tamper robustness of the QIM-style multichannel simulator. Results are reported over simulator runs.}")
        lines.append(r"\label{tab:qim_resync_active_interference}")
        lines.append(r"\setlength{\tabcolsep}{5pt}")
        lines.append(r"\renewcommand{\arraystretch}{1.15}")
        lines.append(r"\resizebox{\linewidth}{!}{%")
        lines.append(r"\begin{tabular}{@{}llrrrr@{}}")
        lines.append(r"\toprule")
        lines.append(r"Loss $r$ & ECC & MAC Failures & Replay Reject. & Stale Reject. & False Accepts \\")
        lines.append(r"\midrule")
        for _, row in summary.iterrows():
            loss = row["loss_rate"] * 100.0 if row["loss_rate"] <= 1.5 else row["loss_rate"]
            lines.append(
                f"${loss:.0f}\\%$ & {ecc_label(row['ecc_enabled'])} & "
                f"{num_fmt(row.get('mac_failures_mean', 0))} & "
                f"{num_fmt(row.get('replay_rejections_mean', 0))} & "
                f"{num_fmt(row.get('stale_epoch_rejections_mean', 0))} & "
                f"{num_fmt(row.get('false_accepts_mean', 0))} \\\\" 
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}%")
        lines.append(r"}")
        lines.append(r"\end{table}")
        (out_dir / "table_active_interference.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def write_manifest(args: argparse.Namespace, out_dir: Path, df: pd.DataFrame, plot_manifest: pd.DataFrame) -> None:
    manifest = {
        "script": "aggregate_qim_resync_results_v2.py",
        "summary_csv": str(args.summary_csv),
        "attack_summary_csv": str(args.attack_summary_csv) if args.attack_summary_csv else None,
        "out_dir": str(out_dir),
        "input_rows": int(len(df)),
        "representative_filter": plot_manifest.to_dict(orient="records"),
        "notes": [
            "Tables use the full grid with mean +/- standard deviation.",
            "Plots use a representative configuration with mean +/- confidence interval.",
            "Rate axes are clipped to 0--100% for interpretability.",
        ],
    }
    (out_dir / "aggregation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    df = read_summary(Path(args.summary_csv))
    write_latex = not args.no_latex

    print(f"[INFO] Loaded {len(df):,} rows from {args.summary_csv}")
    print("[INFO] Writing full-grid tables/CSVs...")
    write_robustness_table(df, out_dir, write_latex=write_latex)
    write_latency_table(df, out_dir, write_latex=write_latex)

    print("[INFO] Writing representative-condition plots...")
    cfg = PlotConfig(
        window_W=args.representative_window,
        heartbeat_h=args.representative_heartbeat,
        skew_ms=args.representative_skew_ms,
        timeout_factor=args.representative_timeout_factor,
        ci_level=args.ci_level,
    )
    plot_manifest = make_plots(
        df, out_dir, cfg,
        fmt=args.format,
        dpi=args.dpi,
        make_individual=args.make_individual,
        strict=args.strict,
    )

    if args.attack_summary_csv:
        print("[INFO] Writing active-interference summary...")
        write_attack_table(args.attack_summary_csv, out_dir, write_latex=write_latex)

    write_manifest(args, out_dir, df, plot_manifest)

    print(f"[OK] Results written to: {out_dir}")
    print("[OK] Key outputs:")
    print(f"  - {out_dir / 'table_resync_robustness.tex'}")
    print(f"  - {out_dir / 'table_resync_latency.tex'}")
    print(f"  - {out_dir / 'figure_qim_resync_combined_representative.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
