r"""
02_build_paired_longitudinal_cohort.py

Builds the paired WearGait-PD Version 2 longitudinal SelfPace cohort
from the downloaded dataset.

Expected structure:
WearGait_PD_Longitudinal/
    scripts/
        02_build_paired_longitudinal_cohort.py
    PD Participants/
        00-CSV files/
            nls005s1_selfpace.csv
            nls005s1_selfpace_mat.csv
            nls005s1_selfpace_matturn.csv
            ...

Outputs:
WearGait_PD_Longitudinal/project_cohort/
    session_file_map.csv
    selfpace_session_qc.csv
    paired_longitudinal_cohort.csv
    sensor_schema.txt
    cohort_summary.txt
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "project_cohort"
OUT.mkdir(parents=True, exist_ok=True)

FILE_RE = re.compile(
    r"^(?P<subject>nls\d+)(?P<session>s\d+)_(?P<task>.+)\.csv$",
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

INSOLE_REQUIRED = [
    "Linsole:Acc_X", "Linsole:Acc_Y", "Linsole:Acc_Z",
    "Linsole:Gyr_X", "Linsole:Gyr_Y", "Linsole:Gyr_Z",
    "Rinsole:Acc_X", "Rinsole:Acc_Y", "Rinsole:Acc_Z",
    "Rinsole:Gyr_X", "Rinsole:Gyr_Y", "Rinsole:Gyr_Z",
]

REFERENCE_COLUMNS = [
    "L Foot Contact", "R Foot Contact",
    "L Foot Pressure", "R Foot Pressure",
    "Walkway_X", "Walkway_Y",
    "WalkwayPressureLevel", "WalkwayFoot",
]

PLANTAR_COLUMNS = (
    [f"LPressure{i}" for i in range(1, 17)]
    + [f"RPressure{i}" for i in range(1, 17)]
    + ["LTotalForce", "RTotalForce",
       "LCoP_X", "LCoP_Y", "RCoP_X", "RCoP_Y"]
)


def parse_time_seconds(series):
    """Convert values such as '12.34 sec' to numeric seconds."""
    return pd.to_numeric(
        series.astype(str).str.replace(" sec", "", regex=False),
        errors="coerce"
    )


def discover_files():
    rows = []
    for p in ROOT.rglob("*.csv"):
        # Ignore our own outputs.
        if OUT in p.parents or "project_audit" in p.parts:
            continue

        m = FILE_RE.match(p.name)
        if not m:
            continue

        task = m.group("task").lower()
        if task not in PRIMARY_TASKS:
            continue

        rows.append({
            "subject": m.group("subject").lower(),
            "session": m.group("session").lower(),
            "task": task,
            "path": str(p.resolve()),
            "relative_path": str(p.relative_to(ROOT)),
        })

    return pd.DataFrame(rows)


def inspect_session_file(path):
    df = pd.read_csv(path, low_memory=False)

    time = parse_time_seconds(df["Time"]) if "Time" in df.columns else pd.Series(dtype=float)
    dt = time.diff().dropna()
    median_dt = float(dt.median()) if len(dt) else np.nan
    fs = 1.0 / median_dt if median_dt and median_dt > 0 else np.nan

    events = (
        sorted(df["GeneralEvent"].dropna().astype(str).unique().tolist())
        if "GeneralEvent" in df.columns else []
    )

    body_ok = {}
    for prefix in BODY_SENSOR_PREFIXES:
        needed = [prefix + s for s in BODY_REQUIRED_SUFFIXES]
        body_ok[prefix] = all(c in df.columns for c in needed)

    insole_present = all(c in df.columns for c in INSOLE_REQUIRED)
    walkway_present = all(c in df.columns for c in REFERENCE_COLUMNS)

    walk_mask = (
        df["GeneralEvent"].astype(str).str.lower().eq("walk")
        if "GeneralEvent" in df.columns else pd.Series(False, index=df.index)
    )

    def coverage(cols, mask=None):
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return np.nan
        sub = df.loc[mask, cols] if mask is not None else df[cols]
        if sub.empty:
            return np.nan
        return float(sub.notna().mean().mean())

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "duration_sec": float(time.max() - time.min()) if len(time) else np.nan,
        "median_dt_sec": median_dt,
        "estimated_hz": fs,
        "events": ";".join(events),
        "walk_rows": int(walk_mask.sum()),
        "body_sensors_complete": int(sum(body_ok.values())),
        "body_sensors_expected": len(BODY_SENSOR_PREFIXES),
        "insole_imu_present": insole_present,
        "walkway_columns_present": walkway_present,
        "walkway_pixel_coverage_all": coverage(
            ["Walkway_X", "Walkway_Y", "WalkwayPressureLevel", "WalkwayFoot"]
        ),
        "walkway_pixel_coverage_walk": coverage(
            ["Walkway_X", "Walkway_Y", "WalkwayPressureLevel", "WalkwayFoot"],
            walk_mask
        ),
        "plantar_coverage_all": coverage(PLANTAR_COLUMNS),
        "plantar_coverage_walk": coverage(PLANTAR_COLUMNS, walk_mask),
    }


def main():
    files = discover_files()
    if files.empty:
        raise SystemExit(
            f"No SelfPace CSVs were found under:\n{ROOT}\n"
            "Check that the Version 2 CSV data are inside this project folder."
        )

    # One row per subject/session, with paths pivoted wide.
    fmap = (
        files.pivot_table(
            index=["subject", "session"],
            columns="task",
            values="path",
            aggfunc="first"
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
    fmap["complete_primary_trio"] = (
        fmap["has_selfpace"]
        & fmap["has_selfpace_mat"]
        & fmap["has_selfpace_matturn"]
    )

    fmap.to_csv(OUT / "session_file_map.csv", index=False)

    # QC every SelfPace session file.
    qc_rows = []
    for _, row in fmap.iterrows():
        if not row["has_selfpace"]:
            continue
        result = inspect_session_file(Path(row["selfpace"]))
        result.update({
            "subject": row["subject"],
            "session": row["session"],
            "selfpace_path": row["selfpace"],
            "has_selfpace_mat": row["has_selfpace_mat"],
            "has_selfpace_matturn": row["has_selfpace_matturn"],
            "complete_primary_trio": row["complete_primary_trio"],
        })
        qc_rows.append(result)

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(OUT / "selfpace_session_qc.csv", index=False)

    # Build subject-level paired longitudinal table.
    sessions_by_subject = (
        fmap.groupby("subject")["session"]
        .apply(lambda x: sorted(set(x)))
        .to_dict()
    )

    paired_rows = []
    for subject, sessions in sorted(sessions_by_subject.items()):
        s1 = fmap[(fmap.subject == subject) & (fmap.session == "s1")]
        s2 = fmap[(fmap.subject == subject) & (fmap.session == "s2")]

        has_s1 = not s1.empty
        has_s2 = not s2.empty
        s1_complete = bool(s1.iloc[0]["complete_primary_trio"]) if has_s1 else False
        s2_complete = bool(s2.iloc[0]["complete_primary_trio"]) if has_s2 else False

        paired_rows.append({
            "subject": subject,
            "sessions_detected": ";".join(sessions),
            "has_s1": has_s1,
            "has_s2": has_s2,
            "s1_complete_primary_trio": s1_complete,
            "s2_complete_primary_trio": s2_complete,
            "eligible_primary_longitudinal": has_s1 and has_s2 and s1_complete and s2_complete,
            "s1_selfpace": s1.iloc[0]["selfpace"] if has_s1 else "",
            "s1_selfpace_mat": s1.iloc[0]["selfpace_mat"] if has_s1 else "",
            "s1_selfpace_matturn": s1.iloc[0]["selfpace_matturn"] if has_s1 else "",
            "s2_selfpace": s2.iloc[0]["selfpace"] if has_s2 else "",
            "s2_selfpace_mat": s2.iloc[0]["selfpace_mat"] if has_s2 else "",
            "s2_selfpace_matturn": s2.iloc[0]["selfpace_matturn"] if has_s2 else "",
        })

    paired = pd.DataFrame(paired_rows)
    paired.to_csv(OUT / "paired_longitudinal_cohort.csv", index=False)

    # Sensor schema from the first usable SelfPace file.
    first_path = Path(qc.iloc[0]["selfpace_path"])
    first = pd.read_csv(first_path, nrows=5, low_memory=False)

    schema_lines = [
        "WearGait-PD SelfPace Sensor Schema",
        "=" * 45,
        f"Example file: {first_path.name}",
        f"Total columns: {len(first.columns)}",
        "",
        "Body-mounted IMU locations detected:",
    ]

    for prefix in BODY_SENSOR_PREFIXES:
        cols = [c for c in first.columns if c.startswith(prefix + "_")]
        schema_lines.append(f"  {prefix}: {len(cols)} columns")

    schema_lines += [
        "",
        "Insole IMU channels:",
    ]
    for c in INSOLE_REQUIRED:
        schema_lines.append(f"  {c}: {'YES' if c in first.columns else 'NO'}")

    schema_lines += [
        "",
        f"Plantar pressure/force/CoP columns expected: {len(PLANTAR_COLUMNS)}",
        f"Plantar pressure/force/CoP columns present: "
        f"{sum(c in first.columns for c in PLANTAR_COLUMNS)}",
        "",
        "Walkway/reference columns:",
    ]
    for c in REFERENCE_COLUMNS:
        schema_lines.append(f"  {c}: {'YES' if c in first.columns else 'NO'}")

    (OUT / "sensor_schema.txt").write_text("\n".join(schema_lines), encoding="utf-8")

    # Summary.
    n_subjects = paired["subject"].nunique()
    n_paired = int((paired["has_s1"] & paired["has_s2"]).sum())
    n_eligible = int(paired["eligible_primary_longitudinal"].sum())

    summary = [
        "WearGait-PD Primary Longitudinal Cohort Summary",
        "=" * 52,
        f"Dataset root: {ROOT}",
        "",
        f"Subjects detected in SelfPace family: {n_subjects}",
        f"Subjects with both s1 and s2: {n_paired}",
        f"Subjects eligible with complete SelfPace trio at both sessions: {n_eligible}",
        "",
        "Primary task: selfpace",
        "Reference-focused subsets: selfpace_mat and selfpace_matturn",
        "",
        "QC notes:",
        f"  Expected body-mounted IMU locations: {len(BODY_SENSOR_PREFIXES)}",
        "  Plus bilateral insole IMUs",
        "  Expected sampling rate: approximately 100 Hz",
        "",
        "NEXT:",
        "1. Review incomplete/excluded participants.",
        "2. Confirm clinical metadata/visit interval files.",
        "3. Extract reference gait metrics and IMU gait features.",
    ]

    (OUT / "cohort_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print(f"\nOutputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
