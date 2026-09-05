r"""
05_audit_clinical_metadata.py

Finds and inventories WearGait-PD clinical/demographic metadata files
needed to interpret longitudinal gait change.

It intentionally excludes the large subject-session task CSV files and
searches the rest of the downloaded Version 2 project for spreadsheets
containing fields related to:
- participant/session identifiers
- visit/date/follow-up timing
- MDS-UPDRS
- Hoehn & Yahr
- medication / time since dose / ON-OFF status
- DBS
- demographics

Expected:
WearGait_PD_Longitudinal/
    scripts/
        05_audit_clinical_metadata.py
    ... downloaded Version 2 files ...

Outputs:
WearGait_PD_Longitudinal/project_clinical_audit/
    metadata_file_inventory.csv
    metadata_column_inventory.csv
    metadata_candidate_ranking.csv
    metadata_preview.txt
    clinical_audit_summary.txt
"""

from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "project_clinical_audit"
OUT.mkdir(parents=True, exist_ok=True)

TASK_FILE_RE = re.compile(
    r"^[A-Za-z]+\d+s\d+_.+\.(csv|tsv)$",
    flags=re.IGNORECASE,
)

FILENAME_TERMS = [
    "clinical", "demographic", "updrs", "participant", "subject",
    "metadata", "visit", "session", "medication", "dbs", "hoehn", "yahr"
]

COLUMN_TERMS = [
    "subject", "participant", "id",
    "session", "visit", "date", "time",
    "updrs", "mds",
    "hoehn", "yahr",
    "medication", "med", "dose", "on/off", "on_off", "onoff",
    "dbs",
    "diagnosis", "disease duration", "time since diagnosis",
    "age", "sex", "gender", "height", "weight",
]

HIGH_VALUE_TERMS = [
    "updrs", "mds", "medication", "dose", "dbs",
    "hoehn", "yahr", "visit", "session", "date"
]


def read_table(path):
    attempts = []

    if path.suffix.lower() == ".tsv":
        attempts = [
            {"sep": "\t"},
            {"sep": "\t", "encoding": "utf-8-sig"},
            {"sep": "\t", "encoding": "latin1"},
        ]
    else:
        attempts = [
            {},
            {"encoding": "utf-8-sig"},
            {"encoding": "latin1"},
        ]

    for kwargs in attempts:
        try:
            return pd.read_csv(path, low_memory=False, **kwargs)
        except Exception:
            pass

    return None


def normalized(text):
    return str(text).strip().lower()


def term_hits(texts, terms):
    joined = " | ".join(normalized(x) for x in texts)
    return sorted({term for term in terms if term in joined})


def main():
    candidates = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".csv", ".tsv"}:
            continue

        if (
            OUT in path.parents
            or "project_audit" in path.parts
            or "project_cohort" in path.parts
            or "project_cohort_v2" in path.parts
            or "project_reference_gait" in path.parts
            or "project_reference_gait_refined" in path.parts
            or "project_reference_analysis" in path.parts
            or "project_clinical_audit" in path.parts
        ):
            continue

        # Skip regular sensor task files such as nls005s1_selfpace.csv.
        if TASK_FILE_RE.match(path.name):
            continue

        table = read_table(path)
        if table is None:
            continue

        cols = [str(c) for c in table.columns]

        filename_hits = term_hits([path.name, str(path.parent)], FILENAME_TERMS)
        column_hits = term_hits(cols, COLUMN_TERMS)
        high_value_hits = term_hits(cols, HIGH_VALUE_TERMS)

        score = (
            2 * len(filename_hits)
            + 2 * len(column_hits)
            + 5 * len(high_value_hits)
        )

        candidates.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "filename": path.name,
                "extension": path.suffix.lower(),
                "n_rows": len(table),
                "n_columns": len(cols),
                "filename_hits": ";".join(filename_hits),
                "column_hits": ";".join(column_hits),
                "high_value_hits": ";".join(high_value_hits),
                "candidate_score": score,
                "_columns": cols,
                "_table": table,
            }
        )

    if not candidates:
        raise SystemExit(
            "No non-task CSV/TSV metadata files were found.\n"
            "Confirm that the Version 2 clinical/demographic spreadsheets "
            "were included in the Synapse download."
        )

    # File inventory.
    inventory = pd.DataFrame(
        [
            {k: v for k, v in row.items() if not k.startswith("_")}
            for row in candidates
        ]
    ).sort_values(
        ["candidate_score", "filename"],
        ascending=[False, True],
    )

    inventory.to_csv(
        OUT / "metadata_file_inventory.csv",
        index=False
    )

    # Column inventory.
    column_rows = []

    for row in candidates:
        for i, col in enumerate(row["_columns"]):
            col_norm = normalized(col)
            hits = sorted(
                {term for term in COLUMN_TERMS if term in col_norm}
            )
            column_rows.append(
                {
                    "relative_path": row["relative_path"],
                    "column_index": i,
                    "column_name": col,
                    "keyword_hits": ";".join(hits),
                }
            )

    column_inventory = pd.DataFrame(column_rows)
    column_inventory.to_csv(
        OUT / "metadata_column_inventory.csv",
        index=False
    )

    # Candidate ranking — keep all, but highlight likely clinical files.
    ranking = inventory.copy()
    ranking["likely_clinical_metadata"] = (
        (ranking["candidate_score"] >= 5)
        | (ranking["high_value_hits"].fillna("") != "")
    )

    ranking.to_csv(
        OUT / "metadata_candidate_ranking.csv",
        index=False
    )

    # Human-readable previews of top candidate files.
    preview_lines = [
        "WearGait-PD Clinical Metadata Candidate Preview",
        "=" * 55,
        "",
    ]

    top_paths = ranking.head(10)["relative_path"].tolist()

    for rel in top_paths:
        row = next(r for r in candidates if r["relative_path"] == rel)
        table = row["_table"]

        preview_lines += [
            f"FILE: {rel}",
            f"Rows x columns: {len(table)} x {len(table.columns)}",
            f"Candidate score: {row['candidate_score']}",
            f"Filename hits: {row['filename_hits']}",
            f"High-value column hits: {row['high_value_hits']}",
            "",
            "Columns:",
        ]

        for col in table.columns:
            preview_lines.append(f"  {col}")

        preview_lines += [
            "",
            "First 3 rows (truncated to first 20 columns):",
            table.iloc[:3, :20].to_string(index=False),
            "",
            "-" * 70,
            "",
        ]

    (OUT / "metadata_preview.txt").write_text(
        "\n".join(preview_lines),
        encoding="utf-8",
    )

    # Summary.
    likely = ranking[ranking["likely_clinical_metadata"] == True]

    lines = [
        "WearGait-PD Clinical Metadata Audit Summary",
        "=" * 50,
        f"Project root: {ROOT}",
        "",
        f"Non-task CSV/TSV files inspected: {len(inventory)}",
        f"Likely clinical/metadata candidates: {len(likely)}",
        "",
        "Top candidate files:",
    ]

    for _, row in ranking.head(15).iterrows():
        lines.append(
            f"  {row['relative_path']} | "
            f"score={int(row['candidate_score'])} | "
            f"rows={int(row['n_rows'])}, cols={int(row['n_columns'])} | "
            f"high-value={row['high_value_hits']}"
        )

    lines += [
        "",
        "What we need to locate before calling gait change 'PD progression':",
        "  1. Session/visit date or follow-up interval",
        "  2. MDS-UPDRS (especially Part III / motor score)",
        "  3. Medication status or time since last dose",
        "  4. DBS status/state",
        "  5. Hoehn & Yahr where available",
        "",
        "NEXT:",
        "  Upload clinical_audit_summary.txt and metadata_preview.txt.",
        "  Then build the longitudinal clinical linkage table.",
    ]

    summary = "\n".join(lines)
    (OUT / "clinical_audit_summary.txt").write_text(
        summary,
        encoding="utf-8"
    )

    print(summary)
    print(f"\nOutputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
