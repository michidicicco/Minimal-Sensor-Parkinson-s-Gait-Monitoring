r"""
04_analyze_longitudinal_reference.py

Characterizes longitudinal s2-s1 changes in the WearGait-PD reference
gait measures BEFORE any IMU sensor-ablation modeling.

Reads:
WearGait_PD_Longitudinal/
    project_reference_gait_refined/
        longitudinal_reference_changes_core.csv
        longitudinal_reference_changes_strict.csv

Writes:
WearGait_PD_Longitudinal/project_reference_analysis/
    core_longitudinal_stats.csv
    strict_longitudinal_stats.csv
    longitudinal_metric_ranking.csv
    longitudinal_analysis_summary.txt
    figures/
        <metric>_paired_change.png

Important interpretation:
These are longitudinally responsive gait measures, NOT yet proven
Parkinson's disease progression biomarkers. Clinical progression will be
assessed later against visit interval, medication/DBS state, and MDS-UPDRS.

Primary statistical outputs:
- baseline and follow-up mean/SD
- mean and median s2-s1 change
- 95% bootstrap CI for mean change
- Cohen's dz for paired change
- paired t-test
- Wilcoxon signed-rank test
- Benjamini-Hochberg FDR q-values
- proportion of participants changing in the same direction as the group mean

This script does not select a final biomarker solely on p-value.
"""

from pathlib import Path
import math
import re
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    raise SystemExit(
        "scipy is required.\nInstall with:\n    python -m pip install scipy"
    )

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "matplotlib is required.\nInstall with:\n    python -m pip install matplotlib"
    )

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "project_reference_gait_refined"
OUT = ROOT / "project_reference_analysis"
FIG_DIR = OUT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CORE_FILE = IN_DIR / "longitudinal_reference_changes_core.csv"
STRICT_FILE = IN_DIR / "longitudinal_reference_changes_strict.csv"

N_BOOT = 5000
RANDOM_SEED = 20260905

CORE_METRICS = [
    "gait_speed_m_s",
    "cadence_steps_min",
    "step_time_s",
    "stride_time_s",
    "stance_time_s",
    "swing_time_s",
    "double_support_pct",
    "step_length_m",
    "stride_length_m",
]

STRICT_METRICS = [
    "step_time_cv_pct",
    "stride_time_cv_pct",
    "step_length_cv_pct",
    "stride_length_cv_pct",
    "step_time_asym_pct",
    "step_length_asym_pct",
    "stride_time_asym_pct",
    "stride_length_asym_pct",
    "stance_time_asym_pct",
    "swing_time_asym_pct",
]


def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=RANDOM_SEED):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        means[i] = np.mean(sample)

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def bh_fdr(p_values):
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan)

    valid = np.isfinite(p)
    pv = p[valid]

    if len(pv) == 0:
        return q

    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)

    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    valid_indices = np.where(valid)[0]
    q[valid_indices[order]] = adjusted

    return q


def paired_stats(df, metric, family):
    s1_col = f"{metric}_s1"
    s2_col = f"{metric}_s2"
    delta_col = f"delta_{metric}"

    if not all(c in df.columns for c in [s1_col, s2_col, delta_col]):
        return None

    tmp = df[["subject", s1_col, s2_col, delta_col]].copy()

    for c in [s1_col, s2_col, delta_col]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp = tmp.dropna()

    if len(tmp) < 3:
        return None

    s1 = tmp[s1_col].to_numpy(dtype=float)
    s2 = tmp[s2_col].to_numpy(dtype=float)
    delta = tmp[delta_col].to_numpy(dtype=float)

    n = len(delta)
    mean_delta = float(np.mean(delta))
    sd_delta = float(np.std(delta, ddof=1))
    median_delta = float(np.median(delta))
    q1, q3 = np.percentile(delta, [25, 75])

    dz = (
        mean_delta / sd_delta
        if np.isfinite(sd_delta) and sd_delta > 0
        else np.nan
    )

    # Small-sample correction for standardized paired change.
    # Approximate Hedges correction applied to dz.
    correction = 1 - (3 / (4 * n - 5)) if n > 2 else np.nan
    hedges_gz = dz * correction if np.isfinite(dz) else np.nan

    try:
        t_result = stats.ttest_rel(s2, s1, nan_policy="omit")
        t_stat = float(t_result.statistic)
        p_t = float(t_result.pvalue)
    except Exception:
        t_stat = np.nan
        p_t = np.nan

    # Wilcoxon can fail if every difference is exactly zero.
    try:
        if np.allclose(delta, 0):
            w_stat = 0.0
            p_w = 1.0
        else:
            w_result = stats.wilcoxon(
                s2,
                s1,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
            w_stat = float(w_result.statistic)
            p_w = float(w_result.pvalue)
    except Exception:
        w_stat = np.nan
        p_w = np.nan

    ci_low, ci_high = bootstrap_mean_ci(
        delta,
        seed=RANDOM_SEED + sum(ord(ch) for ch in metric),
    )

    # Percent change in the group mean. This is descriptive only.
    baseline_mean = float(np.mean(s1))
    followup_mean = float(np.mean(s2))

    mean_pct_change = (
        100.0 * mean_delta / baseline_mean
        if baseline_mean != 0
        else np.nan
    )

    if mean_delta > 0:
        direction_consistency = float(np.mean(delta > 0))
        group_direction = "increase"
    elif mean_delta < 0:
        direction_consistency = float(np.mean(delta < 0))
        group_direction = "decrease"
    else:
        direction_consistency = float(np.mean(delta == 0))
        group_direction = "no_change"

    return {
        "family": family,
        "metric": metric,
        "n": n,
        "s1_mean": baseline_mean,
        "s1_sd": float(np.std(s1, ddof=1)),
        "s2_mean": followup_mean,
        "s2_sd": float(np.std(s2, ddof=1)),
        "mean_delta": mean_delta,
        "sd_delta": sd_delta,
        "median_delta": median_delta,
        "delta_q1": float(q1),
        "delta_q3": float(q3),
        "mean_pct_change": mean_pct_change,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "cohen_dz": dz,
        "hedges_gz": hedges_gz,
        "paired_t_stat": t_stat,
        "paired_t_p": p_t,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": p_w,
        "group_direction": group_direction,
        "direction_consistency": direction_consistency,
    }


def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def make_paired_plot(df, metric, family):
    s1_col = f"{metric}_s1"
    s2_col = f"{metric}_s2"

    if s1_col not in df.columns or s2_col not in df.columns:
        return

    tmp = df[["subject", s1_col, s2_col]].copy()
    tmp[s1_col] = pd.to_numeric(tmp[s1_col], errors="coerce")
    tmp[s2_col] = pd.to_numeric(tmp[s2_col], errors="coerce")
    tmp = tmp.dropna()

    if tmp.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for _, row in tmp.iterrows():
        ax.plot(
            [1, 2],
            [row[s1_col], row[s2_col]],
            marker="o",
            linewidth=0.8,
            alpha=0.45,
        )

    group_means = [
        tmp[s1_col].mean(),
        tmp[s2_col].mean(),
    ]

    ax.plot(
        [1, 2],
        group_means,
        marker="o",
        linewidth=3,
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Session 1", "Session 2"])
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} ({family})")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / f"{safe_filename(metric)}_paired_change.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def analyze_file(path, metrics, family):
    if not path.exists():
        raise SystemExit(
            f"Missing:\n{path}\n\n"
            "Run 03b_refine_reference_qc.py first."
        )

    df = pd.read_csv(path)

    results = []

    for metric in metrics:
        result = paired_stats(df, metric, family)
        if result is not None:
            results.append(result)
            make_paired_plot(df, metric, family)

    out = pd.DataFrame(results)

    if out.empty:
        return out

    out["paired_t_q_fdr"] = bh_fdr(out["paired_t_p"].to_numpy())
    out["wilcoxon_q_fdr"] = bh_fdr(out["wilcoxon_p"].to_numpy())

    out["abs_hedges_gz"] = out["hedges_gz"].abs()

    out["bootstrap_ci_excludes_zero"] = (
        ((out["bootstrap_ci_low"] > 0) & (out["bootstrap_ci_high"] > 0))
        | ((out["bootstrap_ci_low"] < 0) & (out["bootstrap_ci_high"] < 0))
    )

    # A descriptive responsiveness flag, not a clinical-biomarker claim.
    out["longitudinal_signal_flag"] = (
        (out["abs_hedges_gz"] >= 0.30)
        & (out["direction_consistency"] >= 0.60)
        & out["bootstrap_ci_excludes_zero"]
    )

    return out


def main():
    core = analyze_file(CORE_FILE, CORE_METRICS, "core")
    strict = analyze_file(STRICT_FILE, STRICT_METRICS, "strict")

    core.to_csv(OUT / "core_longitudinal_stats.csv", index=False)
    strict.to_csv(OUT / "strict_longitudinal_stats.csv", index=False)

    combined = pd.concat([core, strict], ignore_index=True)

    ranking = combined.sort_values(
        ["longitudinal_signal_flag", "abs_hedges_gz", "direction_consistency"],
        ascending=[False, False, False],
    )

    ranking.to_csv(
        OUT / "longitudinal_metric_ranking.csv",
        index=False,
    )

    flagged = ranking[ranking["longitudinal_signal_flag"] == True]

    lines = [
        "WearGait-PD Longitudinal Reference Analysis",
        "=" * 50,
        "",
        f"Core metrics analyzed: {len(core)}",
        f"Strict metrics analyzed: {len(strict)}",
        "",
        "Interpretation guardrail:",
        "  These results identify longitudinally responsive gait measures.",
        "  They do NOT establish Parkinson's disease progression biomarkers",
        "  until clinical change, treatment state, and follow-up interval are linked.",
        "",
        "Descriptive longitudinal-signal rule:",
        "  |Hedges gz| >= 0.30",
        "  direction consistency >= 60%",
        "  bootstrap 95% CI for mean change excludes zero",
        "",
        f"Metrics meeting that descriptive rule: {len(flagged)}",
    ]

    for _, row in flagged.iterrows():
        lines.append(
            f"  {row['metric']} [{row['family']}]: "
            f"n={int(row['n'])}, "
            f"mean delta={row['mean_delta']:.6g}, "
            f"Hedges gz={row['hedges_gz']:.3f}, "
            f"direction={row['group_direction']} "
            f"({100*row['direction_consistency']:.1f}% consistent), "
            f"FDR Wilcoxon q={row['wilcoxon_q_fdr']:.4g}"
        )

    lines += [
        "",
        "Top metrics by absolute paired standardized change:",
    ]

    for _, row in ranking.head(10).iterrows():
        lines.append(
            f"  {row['metric']} [{row['family']}]: "
            f"|gz|={row['abs_hedges_gz']:.3f}, "
            f"mean delta={row['mean_delta']:.6g}, "
            f"direction consistency={100*row['direction_consistency']:.1f}%"
        )

    lines += [
        "",
        "NEXT:",
        "  1. Review these distributions and freeze a compact reference target set.",
        "  2. Locate longitudinal visit interval + MDS-UPDRS + medication/DBS metadata.",
        "  3. Extract IMU features only after reference targets are frozen.",
    ]

    summary = "\n".join(lines)

    (OUT / "longitudinal_analysis_summary.txt").write_text(
        summary,
        encoding="utf-8"
    )

    print(summary)
    print(f"\nOutputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
