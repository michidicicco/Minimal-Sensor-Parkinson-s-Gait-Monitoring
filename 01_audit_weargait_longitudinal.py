r"""
01_audit_weargait_longitudinal.py

Audits the downloaded WearGait-PD Version 2 longitudinal dataset.

Expected structure:
    WearGait_PD_Longitudinal/
        scripts/
            01_audit_weargait_longitudinal.py
        downloaded dataset files/folders
"""

from pathlib import Path
import csv
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "project_audit"

# Typical WearGait-style filenames can contain subject/session/task.
# We deliberately keep parsing permissive and also save raw stems.
SUBJECT_SESSION_RE = re.compile(
    r"(?P<subject>[A-Za-z]+[0-9]+)[_-]?(?P<session>s[0-9]+)?",
    flags=re.IGNORECASE
)

CLINICAL_KEYWORDS = [
    "updrs", "mds", "clinical", "demographic", "metadata", "participant",
    "medication", "meds", "dbs", "diagnosis", "visit", "session", "subject"
]

COLUMN_KEYWORDS = [
    "updrs", "mds", "med", "dbs", "age", "sex", "gender", "diagnosis",
    "visit", "session", "subject", "participant", "date", "time", "hoe",
    "yahr"
]


def human_mb(n_bytes):
    return round(n_bytes / (1024 ** 2), 3)


def read_header(path):
    """Read CSV header with pandas fallback settings."""
    attempts = [
        dict(),
        dict(encoding="utf-8-sig"),
        dict(encoding="latin1"),
    ]
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, nrows=5, low_memory=False, **kwargs)
            return list(df.columns), len(df)
        except Exception:
            pass
    return [], None


def parse_filename(path):
    stem = path.stem

    match = SUBJECT_SESSION_RE.search(stem)
    subject = match.group("subject") if match else None
    session = match.group("session") if match and match.group("session") else None

    # Infer task by removing parsed subject/session from filename and common separators.
    task = stem
    if subject:
        task = re.sub(re.escape(subject), "", task, count=1, flags=re.IGNORECASE)
    if session:
        task = re.sub(re.escape(session), "", task, count=1, flags=re.IGNORECASE)
    task = re.sub(r"^[\s_\-]+|[\s_\-]+$", "", task)

    return subject, session, task if task else None


def main():
    if not ROOT.exists():
        raise SystemExit(
            f"Dataset folder not found:\n{ROOT}\n\n"
            "If you used a different folder name/location, edit ROOT at the top of this script."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    files = [p for p in ROOT.rglob("*") if p.is_file() and OUT not in p.parents]
    csv_files = [p for p in files if p.suffix.lower() == ".csv"]

    # -----------------------------
    # 1. FILE INVENTORY
    # -----------------------------
    file_rows = []
    for p in files:
        file_rows.append({
            "relative_path": str(p.relative_to(ROOT)),
            "filename": p.name,
            "extension": p.suffix.lower(),
            "size_mb": human_mb(p.stat().st_size),
        })

    file_df = pd.DataFrame(file_rows)
    file_df.to_csv(OUT / "file_inventory.csv", index=False)

    # -----------------------------
    # 2. CSV INVENTORY + HEADERS
    # -----------------------------
    csv_rows = []
    column_rows = []

    for i, p in enumerate(csv_files, start=1):
        subject, session, task = parse_filename(p)
        cols, preview_rows = read_header(p)

        lower_name = p.name.lower()
        lower_cols = [str(c).lower() for c in cols]

        name_hits = [k for k in CLINICAL_KEYWORDS if k in lower_name]
        col_hits = sorted({
            k for k in COLUMN_KEYWORDS
            if any(k in c for c in lower_cols)
        })

        csv_rows.append({
            "relative_path": str(p.relative_to(ROOT)),
            "filename": p.name,
            "subject": subject,
            "session": session,
            "task_inferred": task,
            "size_mb": human_mb(p.stat().st_size),
            "n_columns": len(cols),
            "clinical_filename_hits": ";".join(name_hits),
            "clinical_column_hits": ";".join(col_hits),
        })

        for idx, col in enumerate(cols):
            column_rows.append({
                "relative_path": str(p.relative_to(ROOT)),
                "filename": p.name,
                "column_index": idx,
                "column_name": str(col),
            })

        if i % 50 == 0:
            print(f"Inspected {i}/{len(csv_files)} CSV files...")

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(OUT / "csv_inventory.csv", index=False)

    col_df = pd.DataFrame(column_rows)
    col_df.to_csv(OUT / "column_inventory.csv", index=False)

    # -----------------------------
    # 3. SUBJECT / SESSION / TASK SUMMARY
    # -----------------------------
    parsed = csv_df[csv_df["subject"].notna()].copy()

    if not parsed.empty:
        session_summary = (
            parsed.groupby("subject")
            .agg(
                n_csv_files=("filename", "count"),
                n_sessions=("session", lambda x: x.dropna().nunique()),
                sessions=("session", lambda x: ";".join(sorted(set(v for v in x.dropna().astype(str))))),
                n_tasks=("task_inferred", lambda x: x.dropna().nunique()),
            )
            .reset_index()
            .sort_values(["n_sessions", "subject"], ascending=[False, True])
        )
    else:
        session_summary = pd.DataFrame(
            columns=["subject", "n_csv_files", "n_sessions", "sessions", "n_tasks"]
        )

    session_summary.to_csv(OUT / "participant_session_summary.csv", index=False)

    if not parsed.empty:
        pt_task = (
            parsed.groupby(["subject", "session", "task_inferred"], dropna=False)
            .size()
            .reset_index(name="n_files")
        )
    else:
        pt_task = pd.DataFrame(columns=["subject", "session", "task_inferred", "n_files"])

    pt_task.to_csv(OUT / "participant_task_summary.csv", index=False)

    if not parsed.empty:
        task_summary = (
            parsed.groupby("task_inferred", dropna=False)
            .agg(
                n_files=("filename", "count"),
                n_subjects=("subject", "nunique"),
                n_subject_sessions=("session", lambda x: len(x)),
            )
            .reset_index()
            .sort_values("n_files", ascending=False)
        )
    else:
        task_summary = pd.DataFrame(
            columns=["task_inferred", "n_files", "n_subjects", "n_subject_sessions"]
        )

    task_summary.to_csv(OUT / "task_summary.csv", index=False)

    # -----------------------------
    # 4. CANDIDATE CLINICAL FILES
    # -----------------------------
    if not csv_df.empty:
        cand = csv_df[
            (csv_df["clinical_filename_hits"].fillna("") != "") |
            (csv_df["clinical_column_hits"].fillna("") != "")
        ].copy()
    else:
        cand = pd.DataFrame()

    cand.to_csv(OUT / "candidate_clinical_files.csv", index=False)

    # -----------------------------
    # 5. TEXT SUMMARY
    # -----------------------------
    ext_counts = Counter(p.suffix.lower() or "[no extension]" for p in files)
    parsed_subjects = sorted(set(parsed["subject"].dropna().astype(str))) if not parsed.empty else []
    parsed_sessions = sorted(set(parsed["session"].dropna().astype(str))) if not parsed.empty else []

    lines = []
    lines.append("WearGait-PD Version 2 Longitudinal Dataset Audit")
    lines.append("=" * 55)
    lines.append(f"Dataset root: {ROOT}")
    lines.append("")
    lines.append(f"Total files: {len(files)}")
    lines.append(f"CSV files: {len(csv_files)}")
    lines.append("")
    lines.append("File extensions:")
    for ext, count in ext_counts.most_common():
        lines.append(f"  {ext}: {count}")

    lines.append("")
    lines.append(f"Parsed unique participants from filenames: {len(parsed_subjects)}")
    lines.append(f"Parsed session labels: {', '.join(parsed_sessions) if parsed_sessions else 'None detected'}")

    if not session_summary.empty:
        n_2plus = int((session_summary["n_sessions"] >= 2).sum())
        n_3plus = int((session_summary["n_sessions"] >= 3).sum())
        lines.append(f"Participants with >=2 detected sessions: {n_2plus}")
        lines.append(f"Participants with >=3 detected sessions: {n_3plus}")

    lines.append("")
    lines.append("Most common inferred tasks:")
    if not task_summary.empty:
        for _, row in task_summary.head(20).iterrows():
            lines.append(
                f"  {row['task_inferred']}: {int(row['n_files'])} files, "
                f"{int(row['n_subjects'])} subjects"
            )
    else:
        lines.append("  No tasks parsed.")

    lines.append("")
    lines.append(f"Candidate clinical/metadata CSV files: {len(cand)}")
    for name in cand["relative_path"].head(25).tolist() if not cand.empty else []:
        lines.append(f"  {name}")

    lines.append("")
    lines.append("NEXT STEP:")
    lines.append("Review participant_session_summary.csv, task_summary.csv, and")
    lines.append("candidate_clinical_files.csv before defining the primary analysis cohort.")

    summary_path = OUT / "audit_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\nAudit outputs saved to:")
    print(OUT)


if __name__ == "__main__":
    main()
