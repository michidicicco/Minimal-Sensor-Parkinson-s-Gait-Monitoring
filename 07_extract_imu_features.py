r"""
07_extract_imu_features.py

Extracts device-oriented IMU features from WearGait-PD selfpace_mat sessions.

Design principles
-----------------
1. Straight walking only: GeneralEvent == "Walk"
2. Analyze each continuous walking bout independently.
3. Body IMUs use FREE ACCELERATION magnitude (E/N/U) + gyroscope magnitude.
4. Insole IMUs are exploratory because they do not provide free-acceleration
   channels; their raw acceleration axes are centered within each bout.
5. Features are intentionally orientation-robust and reproducible on a
   practical minimal wearable:
      - RMS
      - SD
      - IQR
      - 95th percentile
      - jerk RMS
      - dominant gait-band frequency
      - gait-band spectral entropy
      - periodicity/autocorrelation peak
6. Bout-level features are aggregated to the session MEDIAN.
7. A sensor-session passes QC when >=2 walking bouts are usable.
8. No clinical variables or walkway targets are used here.

Expected structure
------------------
WearGait_PD_Longitudinal/
    project_cohort_v2/
        selfpace_mat_session_qc.csv
    project_reference_gait_refined/
        paired_reference_eligibility.csv
    scripts/
        07_extract_imu_features.py

Outputs
-------
WearGait_PD_Longitudinal/project_imu_features/
    session_imu_features.csv
    sensor_session_qc.csv
    paired_imu_feature_changes.csv
    paired_sensor_feature_availability.csv
    imu_feature_summary.txt
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

try:
    from scipy import signal
except ImportError:
    raise SystemExit(
        "scipy is required.\nInstall with:\n    python -m pip install scipy"
    )

ROOT = Path(__file__).resolve().parents[1]
COHORT_DIR = ROOT / "project_cohort_v2"
REFINED_DIR = ROOT / "project_reference_gait_refined"
OUT = ROOT / "project_imu_features"
OUT.mkdir(parents=True, exist_ok=True)

SESSION_QC_FILE = COHORT_DIR / "selfpace_mat_session_qc.csv"
ELIGIBILITY_FILE = REFINED_DIR / "paired_reference_eligibility.csv"

BODY_SENSORS = [
    "LowerBack",
    "R_Wrist", "L_Wrist",
    "R_LatShank", "L_LatShank",
    "R_DorsalFoot", "L_DorsalFoot",
    "R_Ankle", "L_Ankle",
    "Xiphoid",
    "Forehead",
]

INSOLE_SENSORS = ["Linsole", "Rinsole"]
ALL_SENSORS = BODY_SENSORS + INSOLE_SENSORS

FEATURE_NAMES = [
    "acc_rms",
    "acc_sd",
    "acc_iqr",
    "acc_p95",
    "acc_jerk_rms",
    "acc_dom_freq_hz",
    "acc_spectral_entropy",
    "acc_periodicity",
    "gyr_rms",
    "gyr_sd",
    "gyr_iqr",
    "gyr_p95",
    "gyr_jerk_rms",
    "gyr_dom_freq_hz",
    "gyr_spectral_entropy",
    "gyr_periodicity",
]

MIN_BOUT_SEC = 2.0
MIN_AXIS_COVERAGE = 0.90
MAX_GAP_SEC = 0.20
MIN_USABLE_BOUTS = 2
GAIT_BAND_HZ = (0.5, 4.0)
PERIOD_LAG_SEC = (0.25, 1.50)


def parse_time_seconds(series):
    return pd.to_numeric(
        series.astype(str).str.replace(" sec", "", regex=False),
        errors="coerce",
    ).to_numpy(dtype=float)


def contiguous_true_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    starts = np.where(mask & ~np.r_[False, mask[:-1]])[0]
    ends = np.where(mask & ~np.r_[mask[1:], False])[0]
    return list(zip(starts, ends))


def max_consecutive_nan(values):
    x = np.asarray(values)
    missing = ~np.isfinite(x)

    if not missing.any():
        return 0

    runs = contiguous_true_runs(missing)
    return max((end - start + 1) for start, end in runs)


def required_columns(sensor_name):
    if sensor_name in BODY_SENSORS:
        acc = [
            f"{sensor_name}_FreeAcc_E",
            f"{sensor_name}_FreeAcc_N",
            f"{sensor_name}_FreeAcc_U",
        ]
        gyr = [
            f"{sensor_name}_Gyr_X",
            f"{sensor_name}_Gyr_Y",
            f"{sensor_name}_Gyr_Z",
        ]
    else:
        acc = [
            f"{sensor_name}:Acc_X",
            f"{sensor_name}:Acc_Y",
            f"{sensor_name}:Acc_Z",
        ]
        gyr = [
            f"{sensor_name}:Gyr_X",
            f"{sensor_name}:Gyr_Y",
            f"{sensor_name}:Gyr_Z",
        ]

    return acc, gyr


def interpolate_axes(frame):
    out = frame.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].interpolate(
            method="linear",
            limit_direction="both",
        )
    return out


def magnitude(frame):
    arr = frame.to_numpy(dtype=float)
    return np.sqrt(np.sum(arr ** 2, axis=1))


def center_insole_acc(frame):
    """
    Insole IMUs lack FreeAcc. Remove each bout's median axis value before
    calculating magnitude. This is exploratory and not equivalent to a
    full orientation/gravity compensation algorithm.
    """
    arr = frame.to_numpy(dtype=float)
    arr = arr - np.nanmedian(arr, axis=0, keepdims=True)
    return pd.DataFrame(arr, columns=frame.columns, index=frame.index)


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def iqr(x):
    q25, q75 = np.percentile(x, [25, 75])
    return float(q75 - q25)


def jerk_rms(x, fs):
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or not np.isfinite(fs) or fs <= 0:
        return np.nan
    dx = np.gradient(x, 1.0 / fs)
    return rms(dx)


def gait_band_psd_features(x, fs):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < max(64, int(fs * 2)):
        return np.nan, np.nan

    x = signal.detrend(x, type="constant")

    nperseg = min(len(x), max(64, int(fs * 4)))
    freqs, psd = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        detrend="constant",
        scaling="density",
    )

    band = (
        (freqs >= GAIT_BAND_HZ[0])
        & (freqs <= GAIT_BAND_HZ[1])
    )

    if not band.any():
        return np.nan, np.nan

    f = freqs[band]
    p = psd[band]

    if not np.isfinite(p).any() or np.nansum(p) <= 0:
        return np.nan, np.nan

    dom_freq = float(f[np.nanargmax(p)])

    probs = p / np.nansum(p)
    probs = probs[np.isfinite(probs) & (probs > 0)]

    if len(probs) <= 1:
        entropy = 0.0
    else:
        entropy = float(
            -np.sum(probs * np.log(probs))
            / np.log(len(probs))
        )

    return dom_freq, entropy


def periodicity_feature(x, fs):
    """
    Maximum normalized autocorrelation in a physiologic gait-period lag window.
    Values closer to 1 indicate stronger periodicity.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < max(64, int(fs * 2)):
        return np.nan

    x = signal.detrend(x, type="constant")
    sd = np.std(x)

    if sd <= 0:
        return np.nan

    x = x / sd

    ac = signal.correlate(
        x,
        x,
        mode="full",
        method="fft",
    )
    ac = ac[len(x) - 1:]

    if ac[0] == 0:
        return np.nan

    ac = ac / ac[0]

    lag_min = max(1, int(round(PERIOD_LAG_SEC[0] * fs)))
    lag_max = min(
        len(ac) - 1,
        int(round(PERIOD_LAG_SEC[1] * fs)),
    )

    if lag_max <= lag_min:
        return np.nan

    return float(np.nanmax(ac[lag_min:lag_max + 1]))


def signal_features(x, fs, prefix):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < max(50, int(fs * MIN_BOUT_SEC)):
        return {
            f"{prefix}_rms": np.nan,
            f"{prefix}_sd": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_p95": np.nan,
            f"{prefix}_jerk_rms": np.nan,
            f"{prefix}_dom_freq_hz": np.nan,
            f"{prefix}_spectral_entropy": np.nan,
            f"{prefix}_periodicity": np.nan,
        }

    dom_freq, spec_entropy = gait_band_psd_features(x, fs)

    return {
        f"{prefix}_rms": rms(x),
        f"{prefix}_sd": float(np.std(x, ddof=1)),
        f"{prefix}_iqr": iqr(x),
        f"{prefix}_p95": float(np.percentile(x, 95)),
        f"{prefix}_jerk_rms": jerk_rms(x, fs),
        f"{prefix}_dom_freq_hz": dom_freq,
        f"{prefix}_spectral_entropy": spec_entropy,
        f"{prefix}_periodicity": periodicity_feature(x, fs),
    }


def sensor_bout_features(df, start, end, sensor_name, fs):
    acc_cols, gyr_cols = required_columns(sensor_name)
    needed = acc_cols + gyr_cols

    if not all(c in df.columns for c in needed):
        return None, "missing_columns"

    bout = df.loc[start:end, needed].copy()

    if len(bout) < int(round(MIN_BOUT_SEC * fs)):
        return None, "bout_too_short"

    numeric = bout.apply(pd.to_numeric, errors="coerce")

    for col in numeric.columns:
        coverage = float(numeric[col].notna().mean())

        if coverage < MIN_AXIS_COVERAGE:
            return None, "low_axis_coverage"

        max_gap = max_consecutive_nan(
            numeric[col].to_numpy(dtype=float)
        )

        if max_gap > int(round(MAX_GAP_SEC * fs)):
            return None, "long_missing_gap"

    numeric = interpolate_axes(numeric)

    acc_frame = numeric[acc_cols]
    gyr_frame = numeric[gyr_cols]

    if sensor_name in INSOLE_SENSORS:
        acc_frame = center_insole_acc(acc_frame)

    acc_mag = magnitude(acc_frame)
    gyr_mag = magnitude(gyr_frame)

    features = {}
    features.update(signal_features(acc_mag, fs, "acc"))
    features.update(signal_features(gyr_mag, fs, "gyr"))

    return features, ""


def process_sensor_session(df, walk_bouts, sensor_name, fs):
    bout_features = []
    failure_reasons = []

    for start, end in walk_bouts:
        if (
            not np.isfinite(fs)
            or fs <= 0
            or (end - start + 1) / fs < MIN_BOUT_SEC
        ):
            continue

        feat, reason = sensor_bout_features(
            df,
            start,
            end,
            sensor_name,
            fs,
        )

        if feat is None:
            failure_reasons.append(reason)
        else:
            bout_features.append(feat)

    result = {
        "sensor": sensor_name,
        "n_walk_bouts_detected": len(walk_bouts),
        "n_usable_bouts": len(bout_features),
        "sensor_session_qc_pass": (
            len(bout_features) >= MIN_USABLE_BOUTS
        ),
        "failure_reasons": ";".join(
            sorted(set(r for r in failure_reasons if r))
        ),
    }

    session_features = {}

    if bout_features:
        feat_df = pd.DataFrame(bout_features)

        for feature in FEATURE_NAMES:
            session_features[feature] = (
                float(feat_df[feature].median())
                if feature in feat_df.columns
                and feat_df[feature].notna().any()
                else np.nan
            )
    else:
        for feature in FEATURE_NAMES:
            session_features[feature] = np.nan

    return result, session_features


def main():
    if not SESSION_QC_FILE.exists():
        raise SystemExit(
            f"Missing:\n{SESSION_QC_FILE}\n"
            "Run 02_build_paired_longitudinal_cohort_v2.py first."
        )

    if not ELIGIBILITY_FILE.exists():
        raise SystemExit(
            f"Missing:\n{ELIGIBILITY_FILE}\n"
            "Run 03b_refine_reference_qc.py first."
        )

    session_map = pd.read_csv(SESSION_QC_FILE)
    eligibility = pd.read_csv(ELIGIBILITY_FILE)

    session_map["subject"] = (
        session_map["subject"].astype(str).str.upper()
    )
    eligibility["subject"] = (
        eligibility["subject"].astype(str).str.upper()
    )

    feature_rows = []
    qc_rows = []

    usable_input = session_map[
        session_map["usable_primary_reference"] == True
    ].copy()

    total = len(usable_input)

    for counter, (_, row) in enumerate(
        usable_input.iterrows(),
        start=1,
    ):
        path = Path(row["selfpace_mat_path"])
        subject = row["subject"]
        session_name = str(row["session"]).lower()

        try:
            df = pd.read_csv(path, low_memory=False)

            time = parse_time_seconds(df["Time"])
            finite_t = time[np.isfinite(time)]

            if len(finite_t) < 3:
                raise ValueError("Insufficient valid timestamps.")

            dt = float(np.nanmedian(np.diff(finite_t)))
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

            wide = {
                "subject": subject,
                "session": session_name,
                "source_file": str(path),
                "estimated_hz": fs,
            }

            for sensor_name in ALL_SENSORS:
                qc, feats = process_sensor_session(
                    df,
                    walk_bouts,
                    sensor_name,
                    fs,
                )

                qc.update(
                    {
                        "subject": subject,
                        "session": session_name,
                        "source_file": str(path),
                        "estimated_hz": fs,
                    }
                )
                qc_rows.append(qc)

                for feature_name, value in feats.items():
                    wide[
                        f"{sensor_name}__{feature_name}"
                    ] = value

            feature_rows.append(wide)

        except Exception as exc:
            qc_rows.append(
                {
                    "subject": subject,
                    "session": session_name,
                    "sensor": "SESSION_ERROR",
                    "source_file": str(path),
                    "sensor_session_qc_pass": False,
                    "failure_reasons": str(exc),
                }
            )

        if counter % 10 == 0 or counter == total:
            print(
                f"Processed {counter}/{total} sessions..."
            )

    features = pd.DataFrame(feature_rows)
    sensor_qc = pd.DataFrame(qc_rows)

    features.to_csv(
        OUT / "session_imu_features.csv",
        index=False,
    )

    sensor_qc.to_csv(
        OUT / "sensor_session_qc.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Paired sensor availability within core/strict reference cohorts
    # ---------------------------------------------------------
    availability_rows = []

    for _, erow in eligibility.iterrows():
        subject = erow["subject"]

        qsub = sensor_qc[
            sensor_qc["subject"].eq(subject)
            & sensor_qc["sensor"].isin(ALL_SENSORS)
        ]

        row = {
            "subject": subject,
            "paired_core": bool(erow["paired_core"]),
            "paired_strict": bool(erow["paired_strict"]),
        }

        for sensor_name in ALL_SENSORS:
            ss = qsub[qsub["sensor"] == sensor_name]

            s1_ok = bool(
                (
                    (ss["session"] == "s1")
                    & (ss["sensor_session_qc_pass"] == True)
                ).any()
            )

            s2_ok = bool(
                (
                    (ss["session"] == "s2")
                    & (ss["sensor_session_qc_pass"] == True)
                ).any()
            )

            row[f"{sensor_name}_s1"] = s1_ok
            row[f"{sensor_name}_s2"] = s2_ok
            row[f"{sensor_name}_paired"] = s1_ok and s2_ok

        availability_rows.append(row)

    availability = pd.DataFrame(availability_rows)

    availability.to_csv(
        OUT / "paired_sensor_feature_availability.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Paired s2-s1 feature changes.
    # One wide row per participant.
    # ---------------------------------------------------------
    feature_cols = [
        c for c in features.columns
        if "__" in c
    ]

    f1 = features[
        features["session"] == "s1"
    ].set_index("subject")

    f2 = features[
        features["session"] == "s2"
    ].set_index("subject")

    paired_subjects = sorted(
        set(f1.index).intersection(set(f2.index))
    )

    delta_rows = []

    for subject in paired_subjects:
        row = {"subject": subject}

        for col in feature_cols:
            v1 = pd.to_numeric(
                pd.Series([f1.at[subject, col]]),
                errors="coerce",
            ).iloc[0]

            v2 = pd.to_numeric(
                pd.Series([f2.at[subject, col]]),
                errors="coerce",
            ).iloc[0]

            row[f"{col}__s1"] = v1
            row[f"{col}__s2"] = v2
            row[f"delta__{col}"] = (
                v2 - v1
                if np.isfinite(v1) and np.isfinite(v2)
                else np.nan
            )

        delta_rows.append(row)

    deltas = pd.DataFrame(delta_rows)

    deltas.to_csv(
        OUT / "paired_imu_feature_changes.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    core_ids = set(
        eligibility.loc[
            eligibility["paired_core"] == True,
            "subject",
        ]
    )

    strict_ids = set(
        eligibility.loc[
            eligibility["paired_strict"] == True,
            "subject",
        ]
    )

    lines = [
        "WearGait-PD IMU Feature Extraction Summary",
        "=" * 50,
        f"Input synchronized sessions processed: {len(features)}",
        f"Body IMU locations: {len(BODY_SENSORS)}",
        f"Exploratory insole IMUs: {len(INSOLE_SENSORS)}",
        f"Features per sensor: {len(FEATURE_NAMES)}",
        "",
        "Feature design:",
        "  Body sensors: free-acceleration magnitude + gyro magnitude",
        "  Insole sensors: bout-centered raw acceleration + gyro magnitude",
        "  Straight walking only",
        "  Per-bout extraction, session median aggregation",
        f"  Sensor-session QC requires >= {MIN_USABLE_BOUTS} usable bouts",
        "",
        "Paired feature availability in CORE cohort:",
    ]

    for sensor_name in ALL_SENSORS:
        col = f"{sensor_name}_paired"
        n = int(
            availability.loc[
                availability["subject"].isin(core_ids),
                col,
            ].sum()
        )
        lines.append(
            f"  {sensor_name}: {n}/{len(core_ids)}"
        )

    lines += [
        "",
        "Paired feature availability in STRICT cohort:",
    ]

    for sensor_name in ALL_SENSORS:
        col = f"{sensor_name}_paired"
        n = int(
            availability.loc[
                availability["subject"].isin(strict_ids),
                col,
            ].sum()
        )
        lines.append(
            f"  {sensor_name}: {n}/{len(strict_ids)}"
        )

    lines += [
        "",
        "NEXT:",
        "  Freeze candidate single- and multi-sensor configurations.",
        "  Run participant-grouped cross-validation to estimate",
        "  step length, cadence, and swing time per session.",
        "  Evaluate preservation of observed s2-s1 reference change.",
    ]

    summary = "\n".join(lines)

    (OUT / "imu_feature_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print(f"\nOutputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
