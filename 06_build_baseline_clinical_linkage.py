r"""
06_build_baseline_clinical_linkage.py

Parses the WearGait-PD Version 1 PD Demographic+Clinical table and links
baseline clinical information to the Version 2 longitudinal cohort.

IMPORTANT:
The clinical CSV has a two-row header. The second row contains the actual
column names, so it must be read with header=1.

This script uses baseline clinical data only. It does NOT create
longitudinal clinical change because a Version 2 follow-up clinical table
has not yet been located.

Expected structure:
WearGait_PD_Longitudinal/
    Clinical_Metadata/
        PD - Demographic+Clinical - datasetV1.csv
    project_reference_gait_refined/
        paired_reference_eligibility.csv
    scripts/
        06_build_baseline_clinical_linkage.py

Outputs:
WearGait_PD_Longitudinal/project_clinical_linkage/
    baseline_clinical_linkage.csv
    clinical_completeness.csv
    mds_updrs_part3_item_qc.csv
    clinical_linkage_summary.txt
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "project_clinical_linkage"
OUT.mkdir(parents=True, exist_ok=True)

CLINICAL_DIR = ROOT / "Clinical_Metadata"
ELIGIBILITY_FILE = (
    ROOT / "project_reference_gait_refined" / "paired_reference_eligibility.csv"
)

TARGET_CLINICAL_NAME = "PD - Demographic+Clinical - datasetV1.csv"

PART3_PREFIX = "MDSUPDRS_3-"


def clean_string(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"", "-", "nan", "None"}:
        return np.nan
    return text


def normalize_subject_id(value):
    text = clean_string(value)
    if pd.isna(text):
        return np.nan
    return str(text).upper().replace(" ", "")


def normalize_med_state(value):
    """
    Preserve the original 3b value and create a conservative normalized state.
    Only clearly interpretable ON/OFF labels are collapsed.
    """
    text = clean_string(value)
    if pd.isna(text):
        return np.nan

    low = str(text).strip().lower()

    if low == "on":
        return "ON"
    if low == "off":
        return "OFF"
    if "med off" in low and "dbs on" in low:
        return "MED_OFF_DBS_ON"
    if "wear" in low and "off" in low:
        return "WEARING_OFF"
    if "partial" in low and "kick" in low:
        return "PARTIAL_KICK_IN"

    return "OTHER"


def valid_value(series):
    cleaned = series.map(clean_string)
    return cleaned.notna()


def find_clinical_file():
    exact = CLINICAL_DIR / TARGET_CLINICAL_NAME
    if exact.exists():
        return exact

    matches = list(ROOT.rglob("*Demographic+Clinical*datasetV1*.csv"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        pd_matches = [p for p in matches if "PD" in p.name.upper()]
        if len(pd_matches) == 1:
            return pd_matches[0]

    raise FileNotFoundError(
        "Could not uniquely locate the PD Demographic+Clinical datasetV1 CSV."
    )


def main():
    clinical_path = find_clinical_file()

    if not ELIGIBILITY_FILE.exists():
        raise SystemExit(
            f"Missing:\n{ELIGIBILITY_FILE}\n"
            "Run 03b_refine_reference_qc.py first."
        )

    # Critical: second row contains actual variable names.
    clinical = pd.read_csv(clinical_path, header=1, low_memory=False)

    if "Subject ID" not in clinical.columns:
        raise SystemExit(
            "Clinical table did not parse correctly. "
            "Expected a 'Subject ID' column after reading with header=1."
        )

    clinical["subject"] = clinical["Subject ID"].map(normalize_subject_id)

    eligibility = pd.read_csv(ELIGIBILITY_FILE)
    eligibility["subject"] = eligibility["subject"].astype(str).str.upper()

    # ------------------------------------------------------------
    # Compute MDS-UPDRS Part III total conservatively.
    # A total is produced only when every item is numeric/present.
    # ------------------------------------------------------------
    part3_cols = [
        c for c in clinical.columns if str(c).startswith(PART3_PREFIX)
    ]

    part3_numeric = clinical[part3_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    clinical["mds_updrs_iii_items_expected"] = len(part3_cols)
    clinical["mds_updrs_iii_items_valid"] = part3_numeric.notna().sum(axis=1)
    clinical["mds_updrs_iii_items_missing"] = (
        len(part3_cols) - clinical["mds_updrs_iii_items_valid"]
    )

    clinical["mds_updrs_iii_complete"] = (
        clinical["mds_updrs_iii_items_missing"] == 0
    )

    clinical["mds_updrs_iii_total_calc"] = part3_numeric.sum(
        axis=1,
        min_count=len(part3_cols),
    )

    # Preserve raw medication state and add conservative normalized version.
    clinical["motor_exam_med_state_raw"] = clinical["3b"].map(clean_string)
    clinical["motor_exam_med_state"] = clinical["3b"].map(normalize_med_state)

    # 3c1 is retained as supplied in the source. We do not silently reinterpret
    # its meaning beyond numeric coercion.
    clinical["part3_3c1_numeric"] = pd.to_numeric(
        clinical["3c1"], errors="coerce"
    )

    # ------------------------------------------------------------
    # Select useful baseline covariates.
    # ------------------------------------------------------------
    wanted = [
        "subject",
        "Release version",
        "Assistive Device used during testing?",
        "Time of research session",
        "Height (in)",
        "Age (years)",
        "Weight (kg)",
        "Gender",
        "Sex",
        "Race",
        "Time of last medication dose",
        "PT/OT status",
        "Frequency of PT/OT",
        "Years since PD diagnosis",
        "Current Medications",
        "PD Medication Dose",
        "DBS?",
        "Bilateral vs uilateral",
        "Electrode location(s)",
        "Years since surgery",
        "Days since Part III Clinical Evaluation",
        "Modified Hoehn & Yahr Score",
        "3a",
        "3b",
        "3c",
        "3c1",
        "motor_exam_med_state_raw",
        "motor_exam_med_state",
        "part3_3c1_numeric",
        "mds_updrs_iii_items_expected",
        "mds_updrs_iii_items_valid",
        "mds_updrs_iii_items_missing",
        "mds_updrs_iii_complete",
        "mds_updrs_iii_total_calc",
    ]

    wanted = [c for c in wanted if c in clinical.columns]
    clinical_small = clinical[wanted].copy()

    linked = eligibility.merge(
        clinical_small,
        on="subject",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    linked["baseline_clinical_matched"] = linked["_merge"].eq("both")
    linked = linked.drop(columns="_merge")

    linked.to_csv(
        OUT / "baseline_clinical_linkage.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Clinical completeness within paired analysis cohorts.
    # ------------------------------------------------------------
    fields_to_audit = [
        "Age (years)",
        "Sex",
        "Race",
        "Height (in)",
        "Weight (kg)",
        "Years since PD diagnosis",
        "Modified Hoehn & Yahr Score",
        "DBS?",
        "Time of last medication dose",
        "Time of research session",
        "PD Medication Dose",
        "Current Medications",
        "Days since Part III Clinical Evaluation",
        "motor_exam_med_state_raw",
        "part3_3c1_numeric",
        "mds_updrs_iii_total_calc",
    ]
    fields_to_audit = [c for c in fields_to_audit if c in linked.columns]

    completeness_rows = []

    for cohort_name, mask in [
        ("all_longitudinal_ids", pd.Series(True, index=linked.index)),
        ("paired_core", linked["paired_core"] == True),
        ("paired_strict", linked["paired_strict"] == True),
    ]:
        subset = linked.loc[mask].copy()
        n = len(subset)

        for field in fields_to_audit:
            if pd.api.types.is_numeric_dtype(subset[field]):
                valid = subset[field].notna()
            else:
                valid = valid_value(subset[field])

            count = int(valid.sum())
            completeness_rows.append(
                {
                    "cohort": cohort_name,
                    "field": field,
                    "n_total": n,
                    "n_valid": count,
                    "percent_valid": (
                        100.0 * count / n if n else np.nan
                    ),
                }
            )

    completeness = pd.DataFrame(completeness_rows)
    completeness.to_csv(
        OUT / "clinical_completeness.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Item-level MDS-UPDRS Part III QC for matched participants.
    # ------------------------------------------------------------
    p3_qc = linked[
        [
            "subject",
            "paired_core",
            "paired_strict",
            "mds_updrs_iii_items_expected",
            "mds_updrs_iii_items_valid",
            "mds_updrs_iii_items_missing",
            "mds_updrs_iii_complete",
            "mds_updrs_iii_total_calc",
        ]
    ].copy()

    p3_qc.to_csv(
        OUT / "mds_updrs_part3_item_qc.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------
    n_total = len(linked)
    n_matched = int(linked["baseline_clinical_matched"].sum())

    core = linked[linked["paired_core"] == True]
    strict = linked[linked["paired_strict"] == True]

    core_p3 = int(core["mds_updrs_iii_total_calc"].notna().sum())
    strict_p3 = int(strict["mds_updrs_iii_total_calc"].notna().sum())

    unmatched = linked.loc[
        ~linked["baseline_clinical_matched"], "subject"
    ].tolist()

    incomplete_core_p3 = core.loc[
        core["mds_updrs_iii_total_calc"].isna(),
        ["subject", "mds_updrs_iii_items_missing"],
    ]

    lines = [
        "WearGait-PD Baseline Clinical Linkage Summary",
        "=" * 52,
        f"Clinical source: {clinical_path}",
        f"Clinical table rows: {len(clinical)}",
        f"Clinical table columns: {len(clinical.columns)}",
        "",
        f"Longitudinal IDs in eligibility table: {n_total}",
        f"Matched to baseline clinical table: {n_matched}/{n_total}",
        f"Unmatched IDs: {', '.join(unmatched) if unmatched else 'None'}",
        "",
        f"Paired CORE cohort: {len(core)}",
        f"CORE with complete calculated MDS-UPDRS Part III: {core_p3}/{len(core)}",
        "",
        f"Paired STRICT cohort: {len(strict)}",
        f"STRICT with complete calculated MDS-UPDRS Part III: {strict_p3}/{len(strict)}",
        "",
        f"Part III item columns detected: {len(part3_cols)}",
        "",
        "CORE participants without a complete Part III total:",
    ]

    if incomplete_core_p3.empty:
        lines.append("  None")
    else:
        for _, row in incomplete_core_p3.iterrows():
            lines.append(
                f"  {row['subject']}: "
                f"{int(row['mds_updrs_iii_items_missing'])} item(s) missing/non-numeric"
            )

    lines += [
        "",
        "Use of this file:",
        "  Baseline severity/covariate adjustment: YES",
        "  Baseline association with gait phenotype: YES",
        "  Longitudinal clinical progression (delta MDS-UPDRS): NO",
        "  Reason: no Version 2 follow-up clinical table has been linked yet.",
        "",
        "NEXT:",
        "  Merge baseline covariates with longitudinal gait changes.",
        "  Use baseline MDS-UPDRS/medication/DBS as covariates or secondary validation.",
        "  Keep the primary sensor-ablation endpoint as preservation of longitudinal gait change.",
    ]

    summary = "\n".join(lines)
    (OUT / "clinical_linkage_summary.txt").write_text(
        summary,
        encoding="utf-8"
    )

    print(summary)
    print(f"\nOutputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
