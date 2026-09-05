r"""
02_build_paired_longitudinal_cohort_v2.py

Corrected WearGait-PD Version 2 longitudinal cohort builder.

Key corrections from v1:
1. Recognizes BOTH NLS### and WPD### participant IDs.
2. Uses selfpace_mat as the primary synchronized straight-walking file.
3. Does NOT require every sensor/insole to be present for cohort inclusion.
   Instead, it reports paired availability separately for each sensor.
4. Defines primary-reference eligibility from paired selfpace_mat + usable
   walkway data, preserving participants for sensor-specific ablation analyses.

Expected structure:
WearGait_PD_Longitudinal/
    scripts/
        02_build_paired_longitudinal_cohort_v2.py
    PD Participants/
        00-CSV files/
            nls005s1_selfpace_mat.csv
            wpd001s1_selfpace_mat.csv
            ...

Outputs:
WearGait_PD_Longitudinal/project_cohort_v2/
    session_file_map.csv
    selfpace_mat_session_qc.csv
    paired_longitudinal_cohort.csv
    paired_sensor_availability.csv
    sensor_availability_summary.csv
    cohort_summary.txt
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "project_cohort_v2"
OUT.mkdir(parents=True, exist_ok=True)

# Generic prefix + digits supports both NLS005 and WPD001.
FILE_RE = re.compile(
    r"^(?P<subject>[A-Za-z]+\d+)(?P<session>s\d+)_(?P<task>.+)\.csv$",
    flags=re.IGNORECASE,
)

PRIMARY_TASKS = ("selfpace", "selfpace_mat", "selfpace_matturn")

BODY_SENSOR_PREFIXES = [
    "LowerBack",
    "R_Wrist", "L_Wrist",
    "R_LatShank", "L_LatShank",
    "R_DorsalFoot", "L_DorsalFoot",
    "R_Ankle", "L_Ankle",
    "Xiphoid",
    "Forehead",
]

BODY_REQUIRED_SUFFIXES = [
    "_Acc_X", "_Acc_Y", "_Acc_Z",
    "_Gyr_X", "_Gyr_Y", "_Gyr_Z",
]

INSOLE_SENSOR_PREFIXES = ["Linsole", "Rinsole"]

REFERENCE_COLUMNS = [
    "L Foot Contact", "R Foot Contact",
    "L Foot Pressure", "R Foot Pressure",
    "Walkway_X", "Walkway_Y",
    "WalkwayPressureLevel", "WalkwayFoot",
]


def parse_time_seconds(series):
    return pd.to_numeric(
        series.astype(str).str.replace(" sec", "", regex=False),
        errors="coerce",
    )


def discover_files():
    rows = []
    for p in ROOT.rglob("*.csv"):
        # Avoid recursively reading our generated analysis outputs.
        if (
            OUT in p.parents
            or "project_audit" in p.parts
            or "project_cohort" in p.parts
            or "project_cohort_v2" in p.parts
        ):
            continue

        m = FILE_RE.match(p.name)
        if not m:
            continue

        task = m.group("task").lower()
        if task not in PRIMARY_TASKS:
            continue

        rows.append(
            {
                "subject": m.group("subject").lower(),
                "session": m.group("session").lower(),
                "task": task,
                "path": str(p.resolve()),
                "relative_path": str(p.relative_to(ROOT)),
            }
        )

    return pd.DataFrame(rows)


def sensor_present(columns, prefix):
    needed = [prefix + s for s in BODY_REQUIRED_SUFFIXES]
    return all(c in columns for c in needed)


def insole_present(columns, side_prefix):
    needed = [
        f"{side_prefix}:Acc_X", f"{side_prefix}:Acc_Y", f"{side_prefix}:Acc_Z",
        f"{side_prefix}:Gyr_X", f"{side_prefix}:Gyr_Y", f"{side_prefix}:Gyr_Z",
    ]
    return all(c in columns for c in needed)


def inspect_primary_file(path):
    df = pd.read_csv(path, low_memory=False)
    columns = set(df.columns)

    time = (
        parse_time_seconds(df["Time"])
        if "Time" in df.columns
        else pd.Series(dtype=float)
    )
    dt = time.diff().dropna()
    median_dt = float(dt.median()) if len(dt) else np.nan
    estimated_hz = 1.0 / median_dt if median_dt > 0 else np.nan

    if "GeneralEvent" in df.columns:
        walk_mask = df["GeneralEvent"].astype(str).str.strip().str.lower().eq("walk")
    else:
        walk_mask = pd.Series(False, index=df.index)

    walkway_present = all(c in columns for c in REFERENCE_COLUMNS)

    # Reference coverage is based on the core spatial/pressure fields while walking.
    walkway_core = [
        c for c in ["Walkway_X", "Walkway_Y", "WalkwayPressureLevel", "WalkwayFoot"]
        if c in columns
    ]
    if walk_mask.any() and walkway_core:
        walkway_coverage_walk = float(
            df.loc[walk_mask, walkway_core].notna().mean().mean()
        )
    else:
        walkway_coverage_walk = np.nan

    result = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "duration_sec": (
            float(time.max() - time.min()) if len(time) and time.notna().any() else np.nan
        ),
        "median_dt_sec": median_dt,
        "estimated_hz": estimated_hz,
        "walk_rows": int(walk_mask.sum()),
        "walkway_columns_present": walkway_present,
        "walkway_coverage_walk": walkway_coverage_walk,
    }

    for sensor in BODY_SENSOR_PREFIXES:
        result[f"sensor_{sensor}"] = sensor_present(columns, sensor)

    result["sensor_Linsole"] = insole_present(columns, "Linsole")
    result["sensor_Rinsole"] = insole_present(columns, "Rinsole")

    # Primary reference usability does NOT require all wearable sensors.
    result["usable_primary_reference"] = bool(
        walkway_present
        and walk_mask.sum() > 0
        and np.isfinite(estimated_hz)
        and 95 <= estimated_hz <= 105
        and np.isfinite(walkway_coverage_walk)
        and walkway_coverage_walk > 0.50
    )

    return result


def main():
    files = discover_files()
    if files.empty:
        raise SystemExit(
            f"No matching SelfPace files found under:\n{ROOT}\n"
            "Check the dataset location."
        )

    # Wide file map per participant/session.
    fmap = (
        files.pivot_table(
            index=["subject", "session"],
            columns="task",
            values="path",
            aggfunc="first",
        )
        .reset_index()
    )
    fmap.columns.name = None

    for task in PRIMARY_TASKS:
        if task not in fmap.columns:
            fmap[task] = np.nan

    fmap["has_selfpace"] = fmap["selfpace"].notna()
    fmap["has_selfpace_mat"] = fmap["selfpace_mat"].notna()
    fmap["has_selfpace_matturn"] = fmap["selfpace_matturn"].notna()
    fmap.to_csv(OUT / "session_file_map.csv", index=False)

    # QC the actual PRIMARY analysis file: selfpace_mat.
    qc_rows = []
    for _, row in fmap.iterrows():
        if not row["has_selfpace_mat"]:
            continue

        q = inspect_primary_file(Path(row["selfpace_mat"]))
        q.update(
            {
                "subject": row["subject"],
                "session": row["session"],
                "selfpace_mat_path": row["selfpace_mat"],
                "has_selfpace": row["has_selfpace"],
                "has_selfpace_matturn": row["has_selfpace_matturn"],
            }
        )
        qc_rows.append(q)

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(OUT / "selfpace_mat_session_qc.csv", index=False)

    # Participant-level paired cohort based on PRIMARY REFERENCE availability.
    paired_rows = []
    subjects = sorted(fmap["subject"].unique())

    for subject in subjects:
        qsub = qc[qc["subject"] == subject]
        s1 = qsub[qsub["session"] == "s1"]
        s2 = qsub[qsub["session"] == "s2"]

        has_s1 = not s1.empty
        has_s2 = not s2.empty

        s1_ref = bool(s1.iloc[0]["usable_primary_reference"]) if has_s1 else False
        s2_ref = bool(s2.iloc[0]["usable_primary_reference"]) if has_s2 else False

        paired_rows.append(
            {
                "subject": subject,
                "has_s1_selfpace_mat": has_s1,
                "has_s2_selfpace_mat": has_s2,
                "s1_usable_reference": s1_ref,
                "s2_usable_reference": s2_ref,
                "eligible_primary_reference": bool(
                    has_s1 and has_s2 and s1_ref and s2_ref
                ),
                "s1_selfpace_mat": (
                    s1.iloc[0]["selfpace_mat_path"] if has_s1 else ""
                ),
                "s2_selfpace_mat": (
                    s2.iloc[0]["selfpace_mat_path"] if has_s2 else ""
                ),
            }
        )

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "paired_longitudinal_cohort.csv", index=False)

    # Sensor-specific paired availability.
    all_sensors = BODY_SENSOR_PREFIXES + INSOLE_SENSOR_PREFIXES
    sensor_rows = []

    for subject in subjects:
        qsub = qc[qc["subject"] == subject]
        s1 = qsub[qsub["session"] == "s1"]
        s2 = qsub[qsub["session"] == "s2"]

        primary_ref_ok = bool(
            not s1.empty
            and not s2.empty
            and s1.iloc[0]["usable_primary_reference"]
            and s2.iloc[0]["usable_primary_reference"]
        )

        row = {
            "subject": subject,
            "eligible_primary_reference": primary_ref_ok,
        }

        for sensor in all_sensors:
            col = f"sensor_{sensor}"
            s1_present = bool(s1.iloc[0][col]) if not s1.empty else False
            s2_present = bool(s2.iloc[0][col]) if not s2.empty else False
            row[f"{sensor}_s1"] = s1_present
            row[f"{sensor}_s2"] = s2_present
            row[f"{sensor}_paired"] = bool(
                primary_ref_ok and s1_present and s2_present
            )

        sensor_rows.append(row)

    sensor_avail = pd.DataFrame(sensor_rows)
    sensor_avail.to_csv(OUT / "paired_sensor_availability.csv", index=False)

    summary_rows = []
    for sensor in all_sensors:
        paired_col = f"{sensor}_paired"
        summary_rows.append(
            {
                "sensor": sensor,
                "paired_n_with_reference": int(sensor_avail[paired_col].sum()),
                "paired_percent_of_reference_cohort": (
                    100.0 * sensor_avail[paired_col].sum()
                    / max(1, sensor_avail["eligible_primary_reference"].sum())
                ),
            }
        )

    sensor_summary = pd.DataFrame(summary_rows).sort_values(
        ["paired_n_with_reference", "sensor"],
        ascending=[False, True],
    )
    sensor_summary.to_csv(OUT / "sensor_availability_summary.csv", index=False)

    # Human-readable summary.
    total_subjects = paired["subject"].nunique()
    paired_s1s2 = int(
        (paired["has_s1_selfpace_mat"] & paired["has_s2_selfpace_mat"]).sum()
    )
    reference_n = int(paired["eligible_primary_reference"].sum())

    lines = [
        "WearGait-PD Corrected Longitudinal Cohort Summary",
        "=" * 55,
        f"Dataset root: {ROOT}",
        "",
        f"Subjects detected: {total_subjects}",
        f"Subjects with paired s1/s2 selfpace_mat: {paired_s1s2}",
        f"Primary reference-eligible paired subjects: {reference_n}",
        "",
        "Participant ID families detected:",
    ]

    prefixes = (
        paired["subject"]
        .str.extract(r"^([A-Za-z]+)", expand=False)
        .str.lower()
        .value_counts()
    )
    for prefix, count in prefixes.items():
        lines.append(f"  {prefix}: {count}")

    lines += [
        "",
        "Paired sensor availability within the reference cohort:",
    ]

    for _, r in sensor_summary.iterrows():
        lines.append(
            f"  {r['sensor']}: {int(r['paired_n_with_reference'])}/{reference_n} "
            f"({r['paired_percent_of_reference_cohort']:.1f}%)"
        )

    lines += [
        "",
        "Primary analysis principle:",
        "  Keep the full reference cohort whenever possible.",
        "  Use sensor-specific paired N for each ablation configuration.",
        "  Do not exclude a participant globally because one non-required sensor is missing.",
        "",
        "NEXT:",
        "  Extract reference gait metrics from selfpace_mat.",
        "  Build per-session and s2-s1 longitudinal gait changes.",
    ]

    (OUT / "cohort_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nOutputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
