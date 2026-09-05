r"""
03_extract_reference_gait_metrics.py

Extracts reference spatiotemporal gait metrics from WearGait-PD
Version 2 selfpace_mat files using the synchronized pressure walkway.

Reference sources:
- L Foot Contact / R Foot Contact -> temporal gait events
- Walkway_X + WalkwayFoot -> longitudinal footfall position
- GeneralEvent == "Walk" -> straight walking only

The ProtoKinetics Zeno walkway has 0.5-inch (1.27 cm) sensor cells.
Spatial footfall positions are estimated from the median longitudinal
walkway sensor coordinate activated by each foot during a complete stance.

IMPORTANT:
- Boundary-truncated foot contacts are excluded because a foot may already
  be on the walkway when a "Walk" annotation begins or remain in contact
  when the annotation ends.
- Metrics are pooled across usable straight-walking bouts within a session.
- This script creates reference gait measurements only. It does not yet
  derive candidate IMU features or fit machine-learning models.

Expected:
WearGait_PD_Longitudinal/
    scripts/
        03_extract_reference_gait_metrics.py
    project_cohort_v2/
        paired_longitudinal_cohort.csv
        selfpace_mat_session_qc.csv

Outputs:
WearGait_PD_Longitudinal/project_reference_gait/
    session_reference_gait_metrics.csv
    reference_metric_qc.csv
    longitudinal_reference_changes.csv
    reference_summary.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COHORT_DIR = ROOT / "project_cohort_v2"
OUT = ROOT / "project_reference_gait"
OUT.mkdir(parents=True, exist_ok=True)

QC_FILE = COHORT_DIR / "selfpace_mat_session_qc.csv"

# Zeno walkway sensor-cell pitch: 0.5 inch = 1.27 cm.
CELL_SIZE_M = 0.0127

MIN_STANCE_SEC = 0.15
MAX_STANCE_SEC = 2.00
MIN_WALK_BOUT_SEC = 1.0
MIN_USABLE_FOOTFALLS_PER_BOUT = 3

REFERENCE_METRICS = [
    "gait_speed_m_s",
    "cadence_steps_min",
    "step_time_s",
    "stride_time_s",
    "stance_time_s",
    "swing_time_s",
    "double_support_pct",
    "step_length_m",
    "stride_length_m",
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


def parse_time_seconds(series):
    return pd.to_numeric(
        series.astype(str).str.replace(" sec", "", regex=False),
        errors="coerce",
    ).to_numpy()


def contiguous_true_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    starts = np.where(mask & ~np.r_[False, mask[:-1]])[0]
    ends = np.where(mask & ~np.r_[mask[1:], False])[0]
    return list(zip(starts, ends))


def parse_pipe_strings(value):
    if pd.isna(value):
        return []
    return str(value).split("|")


def footfall_longitudinal_centroid(df, start, end, foot):
    """
    Estimate footfall center along the walkway's long axis.

    Walkway_X ranges along the ~16-ft dimension; WalkwayFoot identifies
    each active pressure cell as L/R/Other.
    """
    x_values = []

    for idx in range(start, end + 1):
        xs = parse_pipe_strings(df.at[idx, "Walkway_X"])
        feet = parse_pipe_strings(df.at[idx, "WalkwayFoot"])

        n = min(len(xs), len(feet))
        for x, f in zip(xs[:n], feet[:n]):
            if f != foot:
                continue
            try:
                x_values.append(float(x))
            except (TypeError, ValueError):
                pass

    if not x_values:
        return np.nan

    return float(np.median(x_values)) * CELL_SIZE_M


def safe_mean(values):
    x = pd.Series(values, dtype="float64").dropna()
    return float(x.mean()) if len(x) else np.nan


def safe_cv(values):
    x = pd.Series(values, dtype="float64").dropna()
    if len(x) < 2:
        return np.nan
    mu = x.mean()
    if mu == 0:
        return np.nan
    return float(x.std(ddof=1) / mu * 100.0)


def asymmetry_pct(left, right):
    if not np.isfinite(left) or not np.isfinite(right):
        return np.nan
    denom = (left + right) / 2.0
    if denom == 0:
        return np.nan
    return float(abs(left - right) / denom * 100.0)


def side_mean_df(frame, foot, variable):
    if frame.empty:
        return np.nan
    return safe_mean(frame.loc[frame["foot"] == foot, variable])


def side_mean_pairs(values, foot):
    return safe_mean([value for side, value in values if side == foot])


def extract_reference_metrics(path):
    df = pd.read_csv(path, low_memory=False)

    required = [
        "Time",
        "GeneralEvent",
        "L Foot Contact",
        "R Foot Contact",
        "Walkway_X",
        "WalkwayFoot",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    time = parse_time_seconds(df["Time"])
    finite_time = time[np.isfinite(time)]

    if len(finite_time) < 3:
        raise ValueError("Insufficient valid time data.")

    dt = float(np.nanmedian(np.diff(finite_time)))
    fs = 1.0 / dt if dt > 0 else np.nan

    walk_mask = (
        df["GeneralEvent"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("walk")
        .to_numpy()
    )

    walk_bouts = contiguous_true_runs(walk_mask)

    L_contact = (
        pd.to_numeric(df["L Foot Contact"], errors="coerce")
        .fillna(0)
        .to_numpy()
        > 0.5
    )
    R_contact = (
        pd.to_numeric(df["R Foot Contact"], errors="coerce")
        .fillna(0)
        .to_numpy()
        > 0.5
    )

    steps = []
    strides = []
    stances = []
    swings = []

    double_support_sec = 0.0
    analyzed_gait_sec = 0.0

    usable_bouts = 0
    complete_footfalls = 0

    for bout_number, (bout_start, bout_end) in enumerate(walk_bouts, start=1):
        bout_duration = time[bout_end] - time[bout_start]
        if not np.isfinite(bout_duration) or bout_duration < MIN_WALK_BOUT_SEC:
            continue

        footfalls = []

        for foot, contact_signal in [("L", L_contact), ("R", R_contact)]:
            masked_contact = contact_signal & walk_mask

            for start, end in contiguous_true_runs(masked_contact):
                if start < bout_start or end > bout_end:
                    continue

                # Exclude a contact truncated by the annotation boundary.
                if start == bout_start or end == bout_end:
                    continue

                stance_sec = (end - start + 1) * dt

                if stance_sec < MIN_STANCE_SEC or stance_sec > MAX_STANCE_SEC:
                    continue

                x_m = footfall_longitudinal_centroid(
                    df=df,
                    start=start,
                    end=end,
                    foot=foot,
                )

                footfalls.append(
                    {
                        "foot": foot,
                        "start": start,
                        "end": end,
                        "ic": time[start],
                        "toe_off": time[end] + dt,
                        "stance": stance_sec,
                        "x_m": x_m,
                    }
                )

        footfalls = sorted(footfalls, key=lambda item: item["ic"])

        if len(footfalls) < MIN_USABLE_FOOTFALLS_PER_BOUT:
            continue

        usable_bouts += 1
        complete_footfalls += len(footfalls)

        stances.extend(
            [(footfall["foot"], footfall["stance"]) for footfall in footfalls]
        )

        # Consecutive opposite-foot initial contacts -> step metrics.
        for i in range(1, len(footfalls)):
            previous = footfalls[i - 1]
            current = footfalls[i]

            if previous["foot"] == current["foot"]:
                continue

            step_time = current["ic"] - previous["ic"]
            step_length = (
                abs(current["x_m"] - previous["x_m"])
                if np.isfinite(current["x_m"])
                and np.isfinite(previous["x_m"])
                else np.nan
            )

            if step_time > 0:
                steps.append(
                    {
                        "foot": current["foot"],
                        "time": step_time,
                        "length": step_length,
                    }
                )

        # Same-foot successive IC -> stride metrics and swing time.
        for foot in ["L", "R"]:
            same_foot = [
                footfall for footfall in footfalls if footfall["foot"] == foot
            ]

            for i in range(1, len(same_foot)):
                previous = same_foot[i - 1]
                current = same_foot[i]

                stride_time = current["ic"] - previous["ic"]
                stride_length = (
                    abs(current["x_m"] - previous["x_m"])
                    if np.isfinite(current["x_m"])
                    and np.isfinite(previous["x_m"])
                    else np.nan
                )
                swing_time = current["ic"] - previous["toe_off"]

                if stride_time <= 0:
                    continue

                strides.append(
                    {
                        "foot": foot,
                        "time": stride_time,
                        "length": stride_length,
                        "speed": (
                            stride_length / stride_time
                            if np.isfinite(stride_length)
                            else np.nan
                        ),
                    }
                )

                if swing_time >= 0:
                    swings.append((foot, swing_time))

        # Double-support proportion is calculated only over the interval
        # bracketed by complete retained foot contacts.
        first_idx = footfalls[0]["start"]
        last_idx = footfalls[-1]["end"]

        interval_sec = (last_idx - first_idx + 1) * dt
        ds_sec = np.sum(
            (L_contact & R_contact)[first_idx : last_idx + 1]
        ) * dt

        if interval_sec > 0:
            double_support_sec += ds_sec
            analyzed_gait_sec += interval_sec

    step_df = pd.DataFrame(steps)
    stride_df = pd.DataFrame(strides)

    result = {
        "estimated_hz": fs,
        "n_walk_bouts_detected": len(walk_bouts),
        "n_walk_bouts_usable": usable_bouts,
        "n_complete_footfalls": complete_footfalls,
        "n_steps": len(step_df),
        "n_strides": len(stride_df),
    }

    result["gait_speed_m_s"] = (
        safe_mean(stride_df["speed"]) if not stride_df.empty else np.nan
    )
    result["step_time_s"] = (
        safe_mean(step_df["time"]) if not step_df.empty else np.nan
    )
    result["cadence_steps_min"] = (
        60.0 / result["step_time_s"]
        if np.isfinite(result["step_time_s"])
        and result["step_time_s"] > 0
        else np.nan
    )
    result["stride_time_s"] = (
        safe_mean(stride_df["time"]) if not stride_df.empty else np.nan
    )
    result["stance_time_s"] = safe_mean([v for _, v in stances])
    result["swing_time_s"] = safe_mean([v for _, v in swings])
    result["double_support_pct"] = (
        100.0 * double_support_sec / analyzed_gait_sec
        if analyzed_gait_sec > 0
        else np.nan
    )
    result["step_length_m"] = (
        safe_mean(step_df["length"]) if not step_df.empty else np.nan
    )
    result["stride_length_m"] = (
        safe_mean(stride_df["length"]) if not stride_df.empty else np.nan
    )

    # Within-session variability.
    result["step_time_cv_pct"] = (
        safe_cv(step_df["time"]) if not step_df.empty else np.nan
    )
    result["stride_time_cv_pct"] = (
        safe_cv(stride_df["time"]) if not stride_df.empty else np.nan
    )
    result["step_length_cv_pct"] = (
        safe_cv(step_df["length"]) if not step_df.empty else np.nan
    )
    result["stride_length_cv_pct"] = (
        safe_cv(stride_df["length"]) if not stride_df.empty else np.nan
    )

    # Bilateral means and asymmetry.
    for metric_name, frame, variable in [
        ("step_time", step_df, "time"),
        ("step_length", step_df, "length"),
        ("stride_time", stride_df, "time"),
        ("stride_length", stride_df, "length"),
    ]:
        left = side_mean_df(frame, "L", variable)
        right = side_mean_df(frame, "R", variable)

        result[f"{metric_name}_L"] = left
        result[f"{metric_name}_R"] = right
        result[f"{metric_name}_asym_pct"] = asymmetry_pct(left, right)

    stance_L = side_mean_pairs(stances, "L")
    stance_R = side_mean_pairs(stances, "R")
    result["stance_time_L"] = stance_L
    result["stance_time_R"] = stance_R
    result["stance_time_asym_pct"] = asymmetry_pct(stance_L, stance_R)

    swing_L = side_mean_pairs(swings, "L")
    swing_R = side_mean_pairs(swings, "R")
    result["swing_time_L"] = swing_L
    result["swing_time_R"] = swing_R
    result["swing_time_asym_pct"] = asymmetry_pct(swing_L, swing_R)

    # Conservative QC for reference use.
    result["reference_qc_pass"] = bool(
        usable_bouts >= 2
        and len(step_df) >= 6
        and len(stride_df) >= 4
        and np.isfinite(result["gait_speed_m_s"])
        and np.isfinite(result["step_time_s"])
        and np.isfinite(result["stride_time_s"])
    )

    return result


def main():
    if not QC_FILE.exists():
        raise SystemExit(
            f"Missing:\n{QC_FILE}\n\n"
            "Run 02_build_paired_longitudinal_cohort_v2.py first."
        )

    qc = pd.read_csv(QC_FILE)

    # Only process session files that passed the synchronized-reference QC.
    qc = qc[qc["usable_primary_reference"] == True].copy()

    rows = []

    for i, row in qc.iterrows():
        path = Path(row["selfpace_mat_path"])

        try:
            metrics = extract_reference_metrics(path)
            metrics.update(
                {
                    "subject": row["subject"],
                    "session": row["session"],
                    "source_file": str(path),
                    "extraction_error": "",
                }
            )
        except Exception as exc:
            metrics = {
                "subject": row["subject"],
                "session": row["session"],
                "source_file": str(path),
                "reference_qc_pass": False,
                "extraction_error": str(exc),
            }

        rows.append(metrics)

        if len(rows) % 10 == 0:
            print(f"Processed {len(rows)}/{len(qc)} sessions...")

    sessions = pd.DataFrame(rows)

    # Put identifiers first.
    id_cols = [
        "subject",
        "session",
        "source_file",
        "reference_qc_pass",
        "extraction_error",
    ]
    other_cols = [c for c in sessions.columns if c not in id_cols]
    sessions = sessions[id_cols + other_cols]

    sessions.to_csv(
        OUT / "session_reference_gait_metrics.csv",
        index=False,
    )

    qc_cols = [
        "subject",
        "session",
        "reference_qc_pass",
        "estimated_hz",
        "n_walk_bouts_detected",
        "n_walk_bouts_usable",
        "n_complete_footfalls",
        "n_steps",
        "n_strides",
        "extraction_error",
    ]
    qc_cols = [c for c in qc_cols if c in sessions.columns]

    sessions[qc_cols].to_csv(
        OUT / "reference_metric_qc.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Paired s2 - s1 longitudinal reference changes
    # ---------------------------------------------------------
    passed = sessions[sessions["reference_qc_pass"] == True].copy()

    s1 = passed[passed["session"] == "s1"].set_index("subject")
    s2 = passed[passed["session"] == "s2"].set_index("subject")

    paired_subjects = sorted(set(s1.index).intersection(set(s2.index)))

    change_rows = []

    for subject in paired_subjects:
        row = {"subject": subject}

        for metric in REFERENCE_METRICS:
            if metric not in s1.columns or metric not in s2.columns:
                continue

            baseline = pd.to_numeric(
                pd.Series([s1.at[subject, metric]]), errors="coerce"
            ).iloc[0]
            followup = pd.to_numeric(
                pd.Series([s2.at[subject, metric]]), errors="coerce"
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

        change_rows.append(row)

    changes = pd.DataFrame(change_rows)
    changes.to_csv(
        OUT / "longitudinal_reference_changes.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    n_sessions = len(sessions)
    n_session_pass = int(sessions["reference_qc_pass"].fillna(False).sum())
    n_paired = len(paired_subjects)

    lines = [
        "WearGait-PD Reference Gait Extraction Summary",
        "=" * 50,
        f"Input synchronized sessions: {n_sessions}",
        f"Sessions passing reference metric QC: {n_session_pass}",
        f"Participants with paired QC-passing s1/s2 reference metrics: {n_paired}",
        "",
        "Primary reference metrics:",
    ]

    for metric in REFERENCE_METRICS:
        if metric in sessions.columns:
            valid = int(sessions.loc[sessions["reference_qc_pass"] == True, metric].notna().sum())
            lines.append(f"  {metric}: {valid}/{n_session_pass} valid sessions")

    lines += [
        "",
        "Reference construction:",
        "  Temporal gait events: pressure-walkway left/right foot contact",
        "  Spatial gait measures: pressure-walkway longitudinal footfall position",
        "  Spatial grid conversion: 0.5 in = 1.27 cm per cell",
        "  Straight walking only: GeneralEvent == Walk",
        "  Annotation-boundary partial foot contacts excluded",
        "",
        "NEXT:",
        "  Inspect distributions and longitudinal deltas.",
        "  Freeze the progression-sensitive reference metric set.",
        "  Then derive sensor-specific IMU features for ablation.",
    ]

    summary = "\n".join(lines)
    (OUT / "reference_summary.txt").write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"\nOutputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
