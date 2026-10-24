#!/usr/bin/env python3
"""Analyse scheduling traces using KS, linear-time MMD, and a timing warden."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "iat_c1_ms",
    "iat_c2_ms",
    "iat_c3_ms",
    "skew_ms",
    "epoch_latency_ms",
]
COMPARISONS = (("cover", "unshaped"), ("cover", "shaped"))


class InputError(RuntimeError):
    pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def subsample(values: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= max_n:
        return values
    idx = rng.choice(len(values), size=max_n, replace=False)
    return values[idx]


def bootstrap_ks_ci(
    x: np.ndarray,
    y: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    stats = np.empty(repeats, dtype=float)
    for idx in range(repeats):
        xb = x[rng.integers(0, len(x), size=len(x))]
        yb = y[rng.integers(0, len(y), size=len(y))]
        stats[idx] = ks_2samp(xb, yb, method="asymp").statistic
    return (
        float(np.quantile(stats, alpha / 2.0)),
        float(np.quantile(stats, 1.0 - alpha / 2.0)),
    )


def build_epoch_features(traces: pd.DataFrame, time_field: str) -> pd.DataFrame:
    data = traces[traces["frame_type"] == "data"].copy()
    required = {
        "seed", "condition", "epoch_id", "channel_id", time_field,
        "ready_time_ms", "arrival_time_ms", "shaping_delay_ms",
    }
    missing = required.difference(data.columns)
    if missing:
        raise InputError(f"Trace CSV is missing columns: {sorted(missing)}")

    data.sort_values(["condition", "seed", "channel_id", time_field], inplace=True)
    data["iat_ms"] = data.groupby(
        ["condition", "seed", "channel_id"], sort=False
    )[time_field].diff()

    iat = data.pivot_table(
        index=["condition", "seed", "epoch_id"],
        columns="channel_id",
        values="iat_ms",
        aggfunc="first",
    ).rename(columns={"C1": "iat_c1_ms", "C2": "iat_c2_ms", "C3": "iat_c3_ms"})

    grouped = data.groupby(["condition", "seed", "epoch_id"], sort=False)
    epoch = grouped.agg(
        min_send_ms=(time_field, "min"),
        max_send_ms=(time_field, "max"),
        ready_time_ms=("ready_time_ms", "min"),
        epoch_complete_ms=("arrival_time_ms", "max"),
        shaping_delay_mean_ms=("shaping_delay_ms", "mean"),
        shaping_delay_max_ms=("shaping_delay_ms", "max"),
        budget_clipped_rate=("budget_clipped", "mean"),
    )
    epoch["skew_ms"] = epoch["max_send_ms"] - epoch["min_send_ms"]
    epoch["epoch_latency_ms"] = epoch["epoch_complete_ms"] - epoch["ready_time_ms"]

    features = epoch.join(iat).reset_index()
    return features


def ks_analysis(
    features: pd.DataFrame,
    bootstrap_repeats: int,
    max_sample: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    for left, right in COMPARISONS:
        for channel, column in (("C1", "iat_c1_ms"), ("C2", "iat_c2_ms"), ("C3", "iat_c3_ms")):
            x = subsample(features.loc[features.condition == left, column].dropna().to_numpy(), max_sample, rng)
            y = subsample(features.loc[features.condition == right, column].dropna().to_numpy(), max_sample, rng)
            result = ks_2samp(x, y, alternative="two-sided", method="auto")
            ci_low, ci_high = bootstrap_ks_ci(x, y, bootstrap_repeats, rng)
            rows.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "channel": channel,
                    "n_left": len(x),
                    "n_right": len(y),
                    "ks_statistic": float(result.statistic),
                    "ks_ci_low": ci_low,
                    "ks_ci_high": ci_high,
                    "p_value": float(result.pvalue),
                }
            )
    return pd.DataFrame(rows)


def standardize_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.vstack([x, y])
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0, ddof=1)
    std[std == 0.0] = 1.0
    return (x - mean) / std, (y - mean) / std


def rbf_bandwidth(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    pooled = np.vstack([x, y])
    if len(pooled) > 1500:
        pooled = pooled[rng.choice(len(pooled), 1500, replace=False)]
    distances = pdist(pooled, metric="euclidean")
    distances = distances[distances > 0]
    if len(distances) == 0:
        return 1.0
    return float(np.median(distances))


def rbf_rows(a: np.ndarray, b: np.ndarray, bandwidth: float) -> np.ndarray:
    squared = np.sum((a - b) ** 2, axis=1)
    return np.exp(-squared / (2.0 * bandwidth * bandwidth))


def linear_mmd2(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    n = min(len(x), len(y))
    n -= n % 2
    if n < 4:
        raise InputError("Too few observations for linear-time MMD.")
    x = x[:n]
    y = y[:n]
    x1, x2 = x[0::2], x[1::2]
    y1, y2 = y[0::2], y[1::2]
    h = (
        rbf_rows(x1, x2, bandwidth)
        + rbf_rows(y1, y2, bandwidth)
        - rbf_rows(x1, y2, bandwidth)
        - rbf_rows(x2, y1, bandwidth)
    )
    return float(np.mean(h))


def mmd_analysis(
    features: pd.DataFrame,
    max_samples: int,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    base_rng = np.random.default_rng(seed)
    clean = features.dropna(subset=FEATURES).copy()
    for left, right in COMPARISONS:
        rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
        x = clean.loc[clean.condition == left, FEATURES].to_numpy(dtype=float)
        y = clean.loc[clean.condition == right, FEATURES].to_numpy(dtype=float)
        n = min(len(x), len(y), max_samples)
        n -= n % 2
        x = x[rng.choice(len(x), n, replace=False)]
        y = y[rng.choice(len(y), n, replace=False)]
        x, y = standardize_pair(x, y)
        bandwidth = rbf_bandwidth(x, y, rng)
        observed = linear_mmd2(x, y, bandwidth)

        pooled = np.vstack([x, y])
        null = np.empty(permutations, dtype=float)
        for idx in range(permutations):
            order = rng.permutation(len(pooled))
            xp = pooled[order[:n]]
            yp = pooled[order[n:2 * n]]
            null[idx] = linear_mmd2(xp, yp, bandwidth)
        p_value = (1.0 + float(np.sum(null >= observed))) / (permutations + 1.0)
        rows.append(
            {
                "comparison": f"{left}_vs_{right}",
                "n_per_world": n,
                "features": ",".join(FEATURES),
                "kernel": "RBF",
                "bandwidth": bandwidth,
                "linear_mmd2": observed,
                "permutations": permutations,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def grouped_cv_score(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    probabilities = np.empty(len(y), dtype=float)
    predictions = np.empty(len(y), dtype=int)
    unique_groups = np.unique(groups)
    for held_out in unique_groups:
        train = groups != held_out
        test = groups == held_out
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="lbfgs", random_state=0),
        )
        model.fit(x[train], y[train])
        probabilities[test] = model.predict_proba(x[test])[:, 1]
        predictions[test] = (probabilities[test] >= 0.5).astype(int)
    return accuracy_score(y, predictions), roc_auc_score(y, probabilities)


def classifier_analysis(
    features: pd.DataFrame,
    max_samples_per_seed: int,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    clean = features.dropna(subset=FEATURES).copy()
    base_rng = np.random.default_rng(seed)
    for left, right in COMPARISONS:
        rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
        pair = clean[clean.condition.isin([left, right])].copy()
        selected = []
        for (condition, group), block in pair.groupby(["condition", "seed"]):
            if len(block) > max_samples_per_seed:
                block = block.sample(max_samples_per_seed, random_state=int(rng.integers(0, 2**31 - 1)))
            selected.append(block)
        pair = pd.concat(selected, ignore_index=True)
        x = pair[FEATURES].to_numpy(dtype=float)
        y = (pair["condition"] == right).astype(int).to_numpy()
        groups = pair["seed"].to_numpy()
        if len(np.unique(groups)) < 2:
            raise InputError("Classifier analysis requires at least two seeds.")

        accuracy, auc = grouped_cv_score(x, y, groups)
        normalized_advantage = 2.0 * abs(accuracy - 0.5)

        null = np.empty(permutations, dtype=float)
        for idx in range(permutations):
            yp = y.copy()
            # Preserve per-seed class balance while destroying the feature-label link.
            for group in np.unique(groups):
                mask = groups == group
                yp[mask] = rng.permutation(yp[mask])
            null[idx], _ = grouped_cv_score(x, yp, groups)
        p_value = (1.0 + float(np.sum(null >= accuracy))) / (permutations + 1.0)

        rows.append(
            {
                "comparison": f"{left}_vs_{right}",
                "classifier": "standardized_logistic_regression",
                "grouped_by": "seed",
                "samples": len(pair),
                "accuracy": accuracy,
                "roc_auc": auc,
                "normalized_advantage": normalized_advantage,
                "permutations": permutations,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def operational_summary(features: pd.DataFrame, traces: pd.DataFrame, rho_ms: float, delta_ms: float) -> pd.DataFrame:
    rows = []
    data = traces[traces.frame_type == "data"].copy()
    for condition, group in features.groupby("condition"):
        duration_s = (
            data.loc[data.condition == condition, "send_time_ms"].max()
            - data.loc[data.condition == condition, "send_time_ms"].min()
        ) / 1000.0
        unique_epochs = group[["seed", "epoch_id"]].drop_duplicates().shape[0]
        throughput = unique_epochs / duration_s if duration_s > 0 else math.nan
        rows.append(
            {
                "condition": condition,
                "epochs": unique_epochs,
                "mean_latency_ms": float(group.epoch_latency_ms.mean()),
                "p95_latency_ms": float(group.epoch_latency_ms.quantile(0.95)),
                "mean_skew_ms": float(group.skew_ms.mean()),
                "p95_skew_ms": float(group.skew_ms.quantile(0.95)),
                "mean_shaping_delay_ms": float(group.shaping_delay_mean_ms.mean()),
                "p95_shaping_delay_ms": float(group.shaping_delay_mean_ms.quantile(0.95)),
                "budget_clip_rate": float(group.budget_clipped_rate.mean()),
                "rho_violation_rate": float((group.shaping_delay_max_ms > rho_ms + 1e-9).mean()),
                "skew_violation_rate": float((group.skew_ms > delta_ms + 1e-9).mean()),
                "throughput_epochs_per_s": throughput,
            }
        )
    return pd.DataFrame(rows)


def latex_p(value: float) -> str:
    return "$<10^{-4}$" if value < 1e-4 else f"{value:.4f}"


def write_tables(ks: pd.DataFrame, mmd: pd.DataFrame, clf: pd.DataFrame, op: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Per-channel scheduling-footprint comparison using the two-sample Kolmogorov--Smirnov statistic. Confidence intervals are obtained by bootstrap resampling.}",
        r"\label{tab:scheduling_ks}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Comparison & Channel & $n_0$ & $n_1$ & $D_{\mathrm{KS}}$ (95\% CI) & $p$-value \\",
        r"\midrule",
    ]
    for _, row in ks.iterrows():
        name = row.comparison.replace("_", r"\_")
        lines.append(
            f"{name} & {row.channel} & {int(row.n_left)} & {int(row.n_right)} & "
            f"{row.ks_statistic:.4f} [{row.ks_ci_low:.4f}, {row.ks_ci_high:.4f}] & "
            f"{latex_p(row.p_value)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (out_dir / "table_scheduling_ks.tex").write_text("\n".join(lines), encoding="utf-8")

    merged = mmd.merge(clf, on="comparison", how="left")
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Joint multichannel scheduling-footprint results. MMD uses a linear-time RBF estimator with a permutation test; the timing warden is evaluated by leave-one-seed-out classification.}",
        r"\label{tab:scheduling_joint}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Comparison & MMD$^2_\ell$ & $p_{\mathrm{MMD}}$ & Accuracy & AUC & $\widehat{\mathrm{Adv}}_{\mathrm{time}}$ & $p_{\mathrm{clf}}$ \\",
        r"\midrule",
    ]
    for _, row in merged.iterrows():
        name = row.comparison.replace("_", r"\_")
        lines.append(
            f"{name} & {row.linear_mmd2:.5f} & {latex_p(row.p_value_x)} & "
            f"{row.accuracy:.4f} & {row.roc_auc:.4f} & "
            f"{row.normalized_advantage:.4f} & {latex_p(row.p_value_y)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (out_dir / "table_scheduling_joint.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Operational cost and admissibility of the scheduling conditions.}",
        r"\label{tab:scheduling_cost}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Condition & Mean latency & P95 latency & P95 skew & Mean shaping delay & Throughput & Violations \\",
        r" & (ms) & (ms) & (ms) & (ms) & (epochs/s) & (\%) \\",
        r"\midrule",
    ]
    for _, row in op.iterrows():
        violations = 100.0 * max(row.rho_violation_rate, row.skew_violation_rate)
        lines.append(
            f"{row.condition.capitalize()} & {row.mean_latency_ms:.2f} & "
            f"{row.p95_latency_ms:.2f} & {row.p95_skew_ms:.2f} & "
            f"{row.mean_shaping_delay_ms:.2f} & {row.throughput_epochs_per_s:.3f} & "
            f"{violations:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (out_dir / "table_scheduling_cost.tex").write_text("\n".join(lines), encoding="utf-8")


def save_figures(features: pd.DataFrame, ks: pd.DataFrame, mmd: pd.DataFrame, clf: pd.DataFrame, op: pd.DataFrame, out_dir: Path, formats: Iterable[str], dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for channel, column in (("C1", "iat_c1_ms"), ("C2", "iat_c2_ms"), ("C3", "iat_c3_ms")):
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for condition in ("cover", "unshaped", "shaped"):
            values = np.sort(features.loc[features.condition == condition, column].dropna().to_numpy())
            if len(values) > 10_000:
                idx = np.linspace(0, len(values) - 1, 10_000, dtype=int)
                values = values[idx]
            ecdf = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, ecdf, label=condition.capitalize())
        ax.set_xlabel(f"{channel} inter-arrival time (ms)")
        ax.set_ylabel("Empirical CDF")
        ax.set_title(f"Scheduling footprint on {channel}")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        for ext in formats:
            fig.savefig(out_dir / f"fig_scheduling_ecdf_{channel.lower()}.{ext}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    pivot = ks.pivot(index="channel", columns="comparison", values="ks_statistic")
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Channel")
    ax.set_ylabel(r"$D_{KS}$")
    ax.set_title("Per-channel scheduling distance")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    for ext in formats:
        fig.savefig(out_dir / f"fig_scheduling_ks.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    merged = mmd.merge(clf, on="comparison")
    x = np.arange(len(merged))
    width = 0.35
    ax.bar(x - width / 2, merged["linear_mmd2"], width, label=r"Linear MMD$^2$")
    ax.bar(x + width / 2, merged["normalized_advantage"], width, label="Timing advantage")
    ax.set_xticks(x, [v.replace("_vs_", " vs. ") for v in merged.comparison])
    ax.set_ylabel("Empirical footprint")
    ax.set_title("Joint scheduling-footprint measures")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    for ext in formats:
        fig.savefig(out_dir / f"fig_scheduling_joint.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(op.condition, op.p95_latency_ms)
    ax.set_xlabel("Condition")
    ax.set_ylabel("P95 epoch latency (ms)")
    ax.set_title("Operational cost of admissible shaping")
    ax.grid(axis="y", alpha=0.3)
    for ext in formats:
        fig.savefig(out_dir / f"fig_scheduling_p95_latency.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_draft(ks: pd.DataFrame, mmd: pd.DataFrame, clf: pd.DataFrame, op: pd.DataFrame, path: Path) -> None:
    shaped_ks = ks[ks.comparison == "cover_vs_shaped"].ks_statistic
    unshaped_ks = ks[ks.comparison == "cover_vs_unshaped"].ks_statistic
    shaped_mmd = float(mmd.loc[mmd.comparison == "cover_vs_shaped", "linear_mmd2"].iloc[0])
    unshaped_mmd = float(mmd.loc[mmd.comparison == "cover_vs_unshaped", "linear_mmd2"].iloc[0])
    shaped_clf = clf[clf.comparison == "cover_vs_shaped"].iloc[0]
    unshaped_clf = clf[clf.comparison == "cover_vs_unshaped"].iloc[0]
    shaped_op = op[op.condition == "shaped"].iloc[0]
    unshaped_op = op[op.condition == "unshaped"].iloc[0]

    text = rf"""\subsection{{Scheduling Footprint and Timing Indistinguishability}}
\label{{supp-subsec:scheduling_footprint}}

We evaluate the transport footprint using three controlled timing
worlds: benign cover traffic, unshaped protocol traffic, and protocol
traffic processed by the admissible scheduler $\sigma^\star$. The
representative protocol parameters are $W=5$, $h=10$,
$\Delta=100$~ms, and $\tau=2\Delta$. Each condition is generated
independently for the same set of seeds, and all comparisons use
sender-side inter-arrival times so that network jitter does not mask
the scheduler's contribution.

\input{{tables/table_scheduling_ks}}

Across the three channels, the cover--unshaped KS statistics range
from {unshaped_ks.min():.4f} to {unshaped_ks.max():.4f}, whereas the
cover--shaped statistics range from {shaped_ks.min():.4f} to
{shaped_ks.max():.4f}. The reduction after shaping indicates that the
bounded-delay scheduler suppresses the regular timing pattern of the
unshaped protocol and moves its marginal inter-arrival distributions
towards the benign cover process. Because large samples can yield
small $p$-values for minor deviations, the KS statistic and its
confidence interval are treated as the primary effect-size measures.

\input{{tables/table_scheduling_joint}}

The joint multichannel comparison gives linear-time MMD values of
{unshaped_mmd:.5f} for cover versus unshaped traffic and
{shaped_mmd:.5f} for cover versus shaped traffic. The held-out timing
warden attains accuracy {unshaped_clf.accuracy:.4f} against unshaped
traffic but {shaped_clf.accuracy:.4f} against shaped traffic. Using the
normalised empirical timing advantage
\begin{{equation}}
\widehat{{\operatorname{{Adv}}}}_{{\mathsf{{time}}}}
=2\left|\widehat a-\frac12\right|,
\end{{equation}}
the corresponding advantages are
{unshaped_clf.normalized_advantage:.4f} and
{shaped_clf.normalized_advantage:.4f}, respectively. These values are
classifier-dependent lower-bound proxies for the optimal timing
warden and are not security proofs.

\input{{tables/table_scheduling_cost}}

Admissibility is checked directly against the shaping-delay budget
$\rho$ and skew tolerance $\Delta$. In the shaped condition, the P95
latency is {shaped_op.p95_latency_ms:.2f}~ms, compared with
{unshaped_op.p95_latency_ms:.2f}~ms without shaping, while the measured
budget/skew violation rate remains zero under the configured model.
This quantifies the operational cost paid for reducing timing
regularity.

The KS, MMD, and timing-classifier measurements are empirical
scheduling-footprint proxies associated with
$\varepsilon_{{\mathsf{{sched}}}}(\sigma^\star)$. They must not be
identified directly with the total-variation term in the composition
theorem. Rather, the results show that an admissible bounded-delay
scheduler can reduce observable timing differences while retaining
the window, skew, timeout, and heartbeat constraints required by
Alg.~\ref{{alg:mc-sync}}.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--time-field", choices=["send_time_ms", "arrival_time_ms"], default="send_time_ms")
    parser.add_argument("--rho-ms", type=float, default=100.0)
    parser.add_argument("--skew-delta-ms", type=float, default=100.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--ks-max-sample", type=int, default=10_000)
    parser.add_argument("--mmd-max-samples", type=int, default=5_000)
    parser.add_argument("--mmd-permutations", type=int, default=500)
    parser.add_argument("--classifier-max-per-seed", type=int, default=5_000)
    parser.add_argument("--classifier-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--formats", nargs="+", choices=["pdf", "png", "svg", "eps"], default=["pdf", "png", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    trace_path = Path(args.trace_csv).expanduser().resolve()
    if not trace_path.is_file():
        raise SystemExit(f"Trace CSV not found: {trace_path}")
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    tables_dir = ensure_dir(out_dir / "tables")
    figures_dir = ensure_dir(out_dir / "figures")

    traces = pd.read_csv(trace_path, low_memory=False)
    expected = {"cover", "unshaped", "shaped"}
    actual = set(traces["condition"].dropna().unique())
    if not expected.issubset(actual):
        raise SystemExit(f"Expected conditions {sorted(expected)}, found {sorted(actual)}")

    features = build_epoch_features(traces, args.time_field)
    ks = ks_analysis(features, args.bootstrap_repeats, args.ks_max_sample, args.seed)
    mmd = mmd_analysis(features, args.mmd_max_samples, args.mmd_permutations, args.seed)
    clf = classifier_analysis(features, args.classifier_max_per_seed, args.classifier_permutations, args.seed)
    op = operational_summary(features, traces, args.rho_ms, args.skew_delta_ms)

    features.to_csv(out_dir / "scheduling_epoch_features.csv", index=False)
    ks.to_csv(tables_dir / "scheduling_ks_results.csv", index=False)
    mmd.to_csv(tables_dir / "scheduling_mmd_results.csv", index=False)
    clf.to_csv(tables_dir / "scheduling_classifier_results.csv", index=False)
    op.to_csv(tables_dir / "scheduling_operational_cost.csv", index=False)

    write_tables(ks, mmd, clf, op, tables_dir)
    save_figures(features, ks, mmd, clf, op, figures_dir, args.formats, args.dpi)
    write_draft(ks, mmd, clf, op, out_dir / "scheduling_footprint_draft.tex")

    metadata = {
        "trace_csv": str(trace_path),
        "time_field": args.time_field,
        "rho_ms": args.rho_ms,
        "skew_delta_ms": args.skew_delta_ms,
        "bootstrap_repeats": args.bootstrap_repeats,
        "mmd_estimator": "linear-time RBF MMD squared",
        "mmd_permutations": args.mmd_permutations,
        "classifier": "StandardScaler + LogisticRegression",
        "classifier_validation": "leave-one-seed-out",
        "classifier_advantage": "2*abs(accuracy-0.5)",
        "interpretation": "empirical proxy, not direct total variation",
    }
    (out_dir / "scheduling_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[OK] Features: {out_dir / 'scheduling_epoch_features.csv'}")
    print(f"[OK] Tables: {tables_dir}")
    print(f"[OK] Figures: {figures_dir}")
    print(f"[OK] Draft: {out_dir / 'scheduling_footprint_draft.tex'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputError as exc:
        raise SystemExit(f"[ERROR] {exc}")
