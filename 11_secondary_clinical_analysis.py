r"""
11_secondary_clinical_analysis.py

Secondary clinical analysis for the WearGait-PD longitudinal study.

Purpose
-------
Use the available BASELINE clinical table to assess:

1. Clinical construct validity:
   Are baseline gait measures associated with baseline PD motor severity?

2. Prognostic association:
   Is baseline severity associated with subsequent longitudinal gait change?

3. Descriptive treatment-state sensitivity:
   Do longitudinal gait changes differ descriptively by baseline medication,
   DBS, or sex groups?

IMPORTANT
---------
There is no linked Version 2 follow-up clinical table, so this script DOES NOT:
- calculate delta MDS-UPDRS
- claim clinical progression
- use clinical variables as primary sensor-model inputs

The primary engineering conclusion remains based on sensor-only longitudinal
gait-change preservation.

Expected inputs
---------------
WearGait_PD_Longitudinal/
    project_clinical_linkage/
        baseline_clinical_linkage.csv
    project_reference_gait/
        session_reference_gait_metrics.csv
    project_reference_gait_refined/
        longitudinal_reference_changes_core.csv

Outputs
-------
WearGait_PD_Longitudinal/project_clinical_secondary/
    baseline_construct_validity.csv
    baseline_severity_vs_longitudinal_change.csv
    categorical_longitudinal_descriptives.csv
    clinical_analysis_dataset.csv
    clinical_secondary_summary.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    raise SystemExit(
        "scipy is required.\nInstall with:\n    python -m pip install scipy"
    )

ROOT = Path(__file__).resolve().parents[1]

CLINICAL_FILE = (
    ROOT / "project_clinical_linkage" / "baseline_clinical_linkage.csv"
)
SESSION_REFERENCE_FILE = (
    ROOT / "project_reference_gait" / "session_reference_gait_metrics.csv"
)
CHANGE_FILE = (
    ROOT
    / "project_reference_gait_refined"
    / "longitudinal_reference_changes_core.csv"
)

OUT = ROOT / "project_clinical_secondary"
OUT.mkdir(parents=True, exist_ok=True)

BASELINE_GAIT_METRICS = [
    "gait_speed_m_s",
    "step_length_m",
    "cadence_steps_min",
    "swing_time_s",
    "double_support_pct",
]

LONGITUDINAL_TARGETS = [
    "step_length_m",
    "cadence_steps_min",
    "swing_time_s",
]

CLINICAL_NUMERIC = {
    "mds_updrs_iii_total_calc": "MDS-UPDRS III",
    "Modified Hoehn & Yahr Score": "Modified Hoehn & Yahr",
    "Age (years)": "Age",
    "Years since PD diagnosis": "Years since PD diagnosis",
}

MIN_CORRELATION_N = 10
MIN_GROUP_N = 5


def clean_numeric(series):
    return pd.to_numeric(
        series.replace(
            {
                "-": np.nan,
                "": np.nan,
                " ": np.nan,
            }
        ),
        errors="coerce",
    )


def bh_fdr(p_values):
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

    valid_idx = np.where(valid)[0]
    q[valid_idx[order]] = adjusted

    return q


def spearman_result(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)

    if n < MIN_CORRELATION_N:
        return n, np.nan, np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return n, np.nan, np.nan

    result = stats.spearmanr(x, y)
    return n, float(result.statistic), float(result.pvalue)


def rank_biserial_from_mannwhitney(u_stat, n1, n2):
    """
    Rank-biserial effect size from Mann-Whitney U.
    Positive means group 1 tends to have larger values.
    """
    if n1 <= 0 or n2 <= 0:
        return np.nan

    return float(2 * u_stat / (n1 * n2) - 1)


def categorical_comparison(df, group_var, group1, group2, target):
    sub = df[
        df[group_var].isin([group1, group2])
    ][[group_var, target]].copy()

    sub[target] = clean_numeric(sub[target])
    sub = sub.dropna()

    x = sub.loc[
        sub[group_var] == group1, target
    ].to_numpy(dtype=float)

    y = sub.loc[
        sub[group_var] == group2, target
    ].to_numpy(dtype=float)

    row = {
        "group_variable": group_var,
        "group_1": group1,
        "group_2": group2,
        "target": target,
        "n_group_1": len(x),
        "n_group_2": len(y),
        "group_1_mean": np.mean(x) if len(x) else np.nan,
        "group_1_median": np.median(x) if len(x) else np.nan,
        "group_2_mean": np.mean(y) if len(y) else np.nan,
        "group_2_median": np.median(y) if len(y) else np.nan,
        "mannwhitney_u": np.nan,
        "mannwhitney_p": np.nan,
        "rank_biserial": np.nan,
        "inferential_test_run": False,
    }

    if len(x) >= MIN_GROUP_N and len(y) >= MIN_GROUP_N:
        result = stats.mannwhitneyu(
            x,
            y,
            alternative="two-sided",
        )

        row["mannwhitney_u"] = float(result.statistic)
        row["mannwhitney_p"] = float(result.pvalue)
        row["rank_biserial"] = rank_biserial_from_mannwhitney(
            result.statistic,
            len(x),
            len(y),
        )
        row["inferential_test_run"] = True

    return row


def main():
    for path in [
        CLINICAL_FILE,
        SESSION_REFERENCE_FILE,
        CHANGE_FILE,
    ]:
        if not path.exists():
            raise SystemExit(
                f"Missing required input:\n{path}"
            )

    clinical = pd.read_csv(CLINICAL_FILE)
    reference = pd.read_csv(SESSION_REFERENCE_FILE)
    changes = pd.read_csv(CHANGE_FILE)

    for frame in [clinical, reference, changes]:
        frame["subject"] = (
            frame["subject"].astype(str).str.upper()
        )

    reference["session"] = (
        reference["session"].astype(str).str.lower()
    )

    # Primary CORE cohort only.
    clinical = clinical[
        clinical["paired_core"] == True
    ].copy()

    baseline = reference[
        reference["session"] == "s1"
    ][
        ["subject"] + BASELINE_GAIT_METRICS
    ].copy()

    # Keep only the three frozen longitudinal targets.
    change_cols = ["subject"] + [
        f"{target}_s1"
        for target in LONGITUDINAL_TARGETS
    ] + [
        f"{target}_s2"
        for target in LONGITUDINAL_TARGETS
    ] + [
        f"delta_{target}"
        for target in LONGITUDINAL_TARGETS
    ]

    change_cols = [
        c for c in change_cols
        if c in changes.columns
    ]

    analysis = (
        clinical
        .merge(
            baseline,
            on="subject",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            changes[change_cols],
            on="subject",
            how="inner",
            validate="one_to_one",
            suffixes=("", "_change"),
        )
    )

    # Clean numeric clinical covariates.
    for col in CLINICAL_NUMERIC:
        if col in analysis.columns:
            analysis[col] = clean_numeric(
                analysis[col]
            )

    analysis.to_csv(
        OUT / "clinical_analysis_dataset.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 1. Baseline construct validity.
    # ---------------------------------------------------------
    construct_rows = []

    for clinical_col, clinical_label in CLINICAL_NUMERIC.items():
        if clinical_col not in analysis.columns:
            continue

        for gait_metric in BASELINE_GAIT_METRICS:
            n, rho, p = spearman_result(
                analysis[clinical_col],
                analysis[gait_metric],
            )

            construct_rows.append(
                {
                    "clinical_variable": clinical_label,
                    "clinical_column": clinical_col,
                    "baseline_gait_metric": gait_metric,
                    "n": n,
                    "spearman_rho": rho,
                    "p_value": p,
                }
            )

    construct = pd.DataFrame(construct_rows)
    construct["q_fdr"] = bh_fdr(
        construct["p_value"].to_numpy()
    )

    construct.to_csv(
        OUT / "baseline_construct_validity.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 2. Baseline severity vs subsequent gait change.
    # ---------------------------------------------------------
    prognosis_rows = []

    for clinical_col, clinical_label in CLINICAL_NUMERIC.items():
        if clinical_col not in analysis.columns:
            continue

        for target in LONGITUDINAL_TARGETS:
            delta_col = f"delta_{target}"

            if delta_col not in analysis.columns:
                continue

            n, rho, p = spearman_result(
                analysis[clinical_col],
                analysis[delta_col],
            )

            prognosis_rows.append(
                {
                    "baseline_clinical_variable": clinical_label,
                    "clinical_column": clinical_col,
                    "longitudinal_gait_target": target,
                    "n": n,
                    "spearman_rho": rho,
                    "p_value": p,
                }
            )

    prognosis = pd.DataFrame(prognosis_rows)
    prognosis["q_fdr"] = bh_fdr(
        prognosis["p_value"].to_numpy()
    )

    prognosis.to_csv(
        OUT / "baseline_severity_vs_longitudinal_change.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 3. Categorical descriptive sensitivity analyses.
    # ---------------------------------------------------------
    categorical_rows = []

    # Medication: compare only clearly classified ON vs OFF.
    if "motor_exam_med_state" in analysis.columns:
        for target in LONGITUDINAL_TARGETS:
            categorical_rows.append(
                categorical_comparison(
                    analysis,
                    "motor_exam_med_state",
                    "ON",
                    "OFF",
                    f"delta_{target}",
                )
            )

    # DBS: only explicit Yes vs No. '-' remains unknown and excluded.
    if "DBS?" in analysis.columns:
        for target in LONGITUDINAL_TARGETS:
            categorical_rows.append(
                categorical_comparison(
                    analysis,
                    "DBS?",
                    "Yes",
                    "No",
                    f"delta_{target}",
                )
            )

    # Sex is included as a descriptive biological covariate.
    if "Sex" in analysis.columns:
        for target in LONGITUDINAL_TARGETS:
            categorical_rows.append(
                categorical_comparison(
                    analysis,
                    "Sex",
                    "Female",
                    "Male",
                    f"delta_{target}",
                )
            )

    categorical = pd.DataFrame(categorical_rows)

    if not categorical.empty:
        categorical["q_fdr"] = bh_fdr(
            categorical["mannwhitney_p"].to_numpy()
        )

    categorical.to_csv(
        OUT / "categorical_longitudinal_descriptives.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------
    lines = [
        "WearGait-PD Secondary Baseline Clinical Analysis",
        "=" * 58,
        f"CORE participants in merged analysis: {len(analysis)}",
        "",
        "Available baseline clinical sample sizes:",
    ]

    for clinical_col, label in CLINICAL_NUMERIC.items():
        if clinical_col in analysis.columns:
            n = int(analysis[clinical_col].notna().sum())
            lines.append(
                f"  {label}: {n}/{len(analysis)}"
            )

    lines += [
        "",
        "Baseline construct-validity correlations with FDR q < 0.05:",
    ]

    sig_construct = construct[
        construct["q_fdr"] < 0.05
    ].sort_values("q_fdr")

    if sig_construct.empty:
        lines.append("  None")
    else:
        for _, row in sig_construct.iterrows():
            lines.append(
                f"  {row['clinical_variable']} vs "
                f"{row['baseline_gait_metric']}: "
                f"n={int(row['n'])}, "
                f"rho={row['spearman_rho']:.3f}, "
                f"q={row['q_fdr']:.4g}"
            )

    lines += [
        "",
        "Baseline clinical variables associated with later gait change",
        "(FDR q < 0.05):",
    ]

    sig_prog = prognosis[
        prognosis["q_fdr"] < 0.05
    ].sort_values("q_fdr")

    if sig_prog.empty:
        lines.append("  None")
    else:
        for _, row in sig_prog.iterrows():
            lines.append(
                f"  {row['baseline_clinical_variable']} vs "
                f"delta {row['longitudinal_gait_target']}: "
                f"n={int(row['n'])}, "
                f"rho={row['spearman_rho']:.3f}, "
                f"q={row['q_fdr']:.4g}"
            )

    lines += [
        "",
        "Categorical sensitivity analyses:",
    ]

    if categorical.empty:
        lines.append("  None")
    else:
        tested = categorical[
            categorical["inferential_test_run"] == True
        ]
        lines.append(
            f"  Comparisons meeting >= {MIN_GROUP_N} per group: "
            f"{len(tested)}/{len(categorical)}"
        )

        significant = tested[
            tested["q_fdr"] < 0.05
        ]

        if significant.empty:
            lines.append(
                "  No categorical comparison survived FDR correction."
            )
        else:
            for _, row in significant.iterrows():
                lines.append(
                    f"  {row['group_variable']} "
                    f"({row['group_1']} vs {row['group_2']}) "
                    f"for {row['target']}: "
                    f"rank-biserial={row['rank_biserial']:.3f}, "
                    f"q={row['q_fdr']:.4g}"
                )

    lines += [
        "",
        "Interpretation guardrail:",
        "  These analyses use BASELINE clinical data only.",
        "  Associations with later gait change are prognostic/exploratory,",
        "  not evidence of delta clinical progression.",
        "",
        "NEXT:",
        "  Use these results to characterize clinical relevance and",
        "  potential confounding, without changing the primary",
        "  sensor-only architecture conclusion.",
    ]

    summary = "\n".join(lines)

    (OUT / "clinical_secondary_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print(f"\nOutputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
