r"""
03b_refine_reference_qc.py

Refines WearGait-PD reference gait QC using metric-specific eligibility.

Why:
The first reference-extraction script used one conservative all-or-nothing
threshold for every metric. That is unnecessarily restrictive because simple
mean gait metrics (e.g., gait speed, step length) need fewer observations than
variability/asymmetry metrics.

This script creates two pre-specified QC tiers:

CORE QC
    Intended for mean spatiotemporal gait measures.
    Requires:
      - >= 2 usable straight-walking bouts
      - >= 5 valid steps
      - >= 3 valid strides

STRICT QC
    Intended for variability/asymmetry measures.
    Requires:
      - >= 2 usable straight-walking bouts
      - >= 6 valid steps
      - >= 4 valid strides

No session is removed globally. Eligibility is tracked by metric family.

Expected:
WearGait_PD_Longitudinal/
    project_reference_gait/
        session_reference_gait_metrics.csv

Outputs:
WearGait_PD_Longitudinal/project_reference_gait_refined/
    reference_metric_eligibility.csv
    paired_reference_eligibility.csv
    longitudinal_reference_changes_core.csv
    longitudinal_reference_changes_strict.csv
    qc_sensitivity_summary.csv
    refined_reference_summary.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "project_reference_gait"
OUT = ROOT / "project_reference_gait_refined"
OUT.mkdir(parents=True, exist_ok=True)

INPUT = IN_DIR / "session_reference_gait_metrics.csv"

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


def finite_required(row, metrics):
    vals = pd.to_numeric(row[metrics], errors="coerce")
    return bool(np.isfinite(vals).all())


def build_changes(session_df, eligible_col, metrics):
    eligible = session_df[session_df[eligible_col] == True].copy()

    s1 = eligible[eligible["session"] == "s1"].set_index("subject")
    s2 = eligible[eligible["session"] == "s2"].set_index("subject")

    subjects = sorted(set(s1.index).intersection(set(s2.index)))
    rows = []

    for subject in subjects:
        row = {"subject": subject}

        for metric in metrics:
            baseline = pd.to_numeric(
                pd.Series([s1.at[subject, metric]]),
                errors="coerce"
            ).iloc[0]
            followup = pd.to_numeric(
                pd.Series([s2.at[subject, metric]]),
                errors="coerce"
            ).iloc[0]

            row[f"{metric}_s1"] = baseline
            row[f"{metric}_s2"] = followup

            row[f"delta_{metric}"] = (
                followup - baseline
                if np.isfinite(baseline) and np.isfinite(followup)
                else np.nan
            )

            row[f"pct_change_{metric}"] = (
                100.0 * (followup - baseline) / baseline
                if np.isfinite(baseline)
                and np.isfinite(followup)
                and baseline != 0
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def paired_n_for_threshold(df, min_bouts, min_steps, min_strides):
    tmp = df.copy()

    tmp["pass_tmp"] = (
        (tmp["n_walk_bouts_usable"] >= min_bouts)
        & (tmp["n_steps"] >= min_steps)
        & (tmp["n_strides"] >= min_strides)
    )

    s1 = set(
        tmp.loc[
            (tmp["session"] == "s1") & tmp["pass_tmp"],
            "subject"
        ]
    )
    s2 = set(
        tmp.loc[
            (tmp["session"] == "s2") & tmp["pass_tmp"],
            "subject"
        ]
    )

    return len(s1.intersection(s2))


def main():
    if not INPUT.exists():
        raise SystemExit(
            f"Missing input:\n{INPUT}\n\n"
            "Run 03_extract_reference_gait_metrics.py first."
        )

    df = pd.read_csv(INPUT)

    required_cols = [
        "subject",
        "session",
        "n_walk_bouts_usable",
        "n_steps",
        "n_strides",
    ] + CORE_METRICS + STRICT_METRICS

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(
            "Missing required columns from session_reference_gait_metrics.csv:\n"
            + "\n".join(missing)
        )

    # ---------------------------------------------------------
    # Metric-family QC
    # ---------------------------------------------------------
    df["core_qc_pass"] = (
        (df["n_walk_bouts_usable"] >= 2)
        & (df["n_steps"] >= 5)
        & (df["n_strides"] >= 3)
        & df.apply(lambda r: finite_required(r, CORE_METRICS), axis=1)
    )

    df["strict_qc_pass"] = (
        (df["n_walk_bouts_usable"] >= 2)
        & (df["n_steps"] >= 6)
        & (df["n_strides"] >= 4)
        & df.apply(lambda r: finite_required(r, STRICT_METRICS), axis=1)
    )

    eligibility_cols = [
        "subject",
        "session",
        "n_walk_bouts_detected",
        "n_walk_bouts_usable",
        "n_complete_footfalls",
        "n_steps",
        "n_strides",
        "core_qc_pass",
        "strict_qc_pass",
        "reference_qc_pass",
        "source_file",
        "extraction_error",
    ]
    eligibility_cols = [c for c in eligibility_cols if c in df.columns]

    df[eligibility_cols].to_csv(
        OUT / "reference_metric_eligibility.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Paired eligibility table
    # ---------------------------------------------------------
    rows = []

    for subject in sorted(df["subject"].unique()):
        sub = df[df["subject"] == subject]

        s1 = sub[sub["session"] == "s1"]
        s2 = sub[sub["session"] == "s2"]

        row = {
            "subject": subject,
            "has_s1": not s1.empty,
            "has_s2": not s2.empty,
            "core_s1": (
                bool(s1.iloc[0]["core_qc_pass"])
                if not s1.empty else False
            ),
            "core_s2": (
                bool(s2.iloc[0]["core_qc_pass"])
                if not s2.empty else False
            ),
            "strict_s1": (
                bool(s1.iloc[0]["strict_qc_pass"])
                if not s1.empty else False
            ),
            "strict_s2": (
                bool(s2.iloc[0]["strict_qc_pass"])
                if not s2.empty else False
            ),
        }

        row["paired_core"] = row["core_s1"] and row["core_s2"]
        row["paired_strict"] = row["strict_s1"] and row["strict_s2"]
        rows.append(row)

    paired = pd.DataFrame(rows)
    paired.to_csv(
        OUT / "paired_reference_eligibility.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Longitudinal changes by metric family
    # ---------------------------------------------------------
    core_changes = build_changes(df, "core_qc_pass", CORE_METRICS)
    strict_changes = build_changes(df, "strict_qc_pass", STRICT_METRICS)

    core_changes.to_csv(
        OUT / "longitudinal_reference_changes_core.csv",
        index=False
    )
    strict_changes.to_csv(
        OUT / "longitudinal_reference_changes_strict.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Sensitivity table: show how paired N changes under
    # reasonable observation-count thresholds.
    # This is descriptive; the primary thresholds above remain fixed.
    # ---------------------------------------------------------
    sensitivity_specs = [
        (1, 4, 3),
        (2, 5, 3),
        (2, 6, 4),
        (2, 8, 5),
        (3, 8, 5),
    ]

    sensitivity_rows = []

    for min_bouts, min_steps, min_strides in sensitivity_specs:
        sensitivity_rows.append(
            {
                "min_usable_bouts": min_bouts,
                "min_steps": min_steps,
                "min_strides": min_strides,
                "paired_n": paired_n_for_threshold(
                    df,
                    min_bouts,
                    min_steps,
                    min_strides,
                ),
            }
        )

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        OUT / "qc_sensitivity_summary.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    n_sessions = len(df)
    n_core_sessions = int(df["core_qc_pass"].sum())
    n_strict_sessions = int(df["strict_qc_pass"].sum())
    n_core_paired = int(paired["paired_core"].sum())
    n_strict_paired = int(paired["paired_strict"].sum())

    excluded_core = paired.loc[~paired["paired_core"], "subject"].tolist()
    excluded_strict = paired.loc[~paired["paired_strict"], "subject"].tolist()

    lines = [
        "WearGait-PD Refined Reference QC Summary",
        "=" * 48,
        f"Total synchronized sessions: {n_sessions}",
        "",
        "CORE metric family:",
        "  QC: >=2 usable bouts, >=5 steps, >=3 strides",
        f"  Passing sessions: {n_core_sessions}/{n_sessions}",
        f"  Paired participants: {n_core_paired}",
        f"  Not paired-core eligible: {', '.join(excluded_core)}",
        "",
        "STRICT variability/asymmetry family:",
        "  QC: >=2 usable bouts, >=6 steps, >=4 strides",
        f"  Passing sessions: {n_strict_sessions}/{n_sessions}",
        f"  Paired participants: {n_strict_paired}",
        f"  Not paired-strict eligible: {', '.join(excluded_strict)}",
        "",
        "Interpretation:",
        "  Core mean gait metrics use the larger paired cohort.",
        "  Variability/asymmetry metrics retain the stricter cohort.",
        "  This avoids globally excluding a participant because one",
        "  metric family lacks enough valid gait cycles.",
        "",
        "QC sensitivity:",
    ]

    for _, r in sensitivity.iterrows():
        lines.append(
            f"  bouts>={int(r['min_usable_bouts'])}, "
            f"steps>={int(r['min_steps'])}, "
            f"strides>={int(r['min_strides'])}: "
            f"paired N={int(r['paired_n'])}"
        )

    lines += [
        "",
        "NEXT:",
        "  Analyze longitudinal effect sizes/distributions separately",
        "  for core and strict metric families before IMU ablation.",
    ]

    summary = "\n".join(lines)
    (OUT / "refined_reference_summary.txt").write_text(
        summary,
        encoding="utf-8"
    )

    print(summary)
    print(f"\nOutputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
