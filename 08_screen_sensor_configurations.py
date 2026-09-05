r"""
08_screen_sensor_configurations.py

Primary fair-comparison screen for WearGait-PD sensor ablation.

Goal
----
Compare a pre-specified set of single- and multi-body-IMU configurations
for their ability to estimate three frozen reference gait targets:

    1. step_length_m
    2. cadence_steps_min
    3. swing_time_s

and, most importantly, preserve each participant's observed longitudinal
change from session 1 to session 2.

Key design safeguards
---------------------
- Uses only participants in the paired CORE reference cohort.
- Uses a COMMON BODY-SENSOR COHORT: participants must have paired usable
  data for every body IMU location included in the screening universe.
  This keeps N identical across configurations for a fair ranking.
- Participant-grouped repeated 5-fold cross-validation keeps s1 and s2
  from the same person in the same fold.
- Feature scaling, imputation, and Elastic Net fitting happen inside the
  training folds.
- Hyperparameters are selected only within each outer-training fold using
  grouped inner CV.
- Clinical variables are NOT inputs to the primary model.
- Insoles are excluded from this primary hardware screen.

Expected inputs
---------------
WearGait_PD_Longitudinal/
    project_imu_features/
        session_imu_features.csv
        paired_sensor_feature_availability.csv
    project_reference_gait/
        session_reference_gait_metrics.csv
    project_reference_gait_refined/
        paired_reference_eligibility.csv

Outputs
-------
WearGait_PD_Longitudinal/project_sensor_screen/
    sensor_configuration_definitions.csv
    common_cohort_subjects.csv
    cv_session_predictions.csv
    cv_longitudinal_predictions.csv
    target_configuration_performance.csv
    configuration_summary.csv
    sensor_screen_ranking.csv
    sensor_screen_summary.txt

Notes
-----
This is a SCREENING analysis, not the final model comparison.
The top few configurations should be re-tested later on their maximum
available cohorts and with secondary model families/sensitivity analyses.
"""

from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    raise SystemExit(
        "scipy is required.\nInstall with:\n    python -m pip install scipy"
    )

try:
    from sklearn.base import clone
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import GroupKFold, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise SystemExit(
        "scikit-learn is required.\nInstall with:\n"
        "    python -m pip install scikit-learn"
    )

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]

IMU_DIR = ROOT / "project_imu_features"
REF_DIR = ROOT / "project_reference_gait"
REFINED_DIR = ROOT / "project_reference_gait_refined"

OUT = ROOT / "project_sensor_screen"
OUT.mkdir(parents=True, exist_ok=True)

IMU_FEATURE_FILE = IMU_DIR / "session_imu_features.csv"
AVAILABILITY_FILE = IMU_DIR / "paired_sensor_feature_availability.csv"
REFERENCE_FILE = REF_DIR / "session_reference_gait_metrics.csv"
ELIGIBILITY_FILE = REFINED_DIR / "paired_reference_eligibility.csv"

TARGETS = [
    "step_length_m",
    "cadence_steps_min",
    "swing_time_s",
]

BODY_SENSORS = [
    "LowerBack",
    "R_Wrist",
    "L_Wrist",
    "R_LatShank",
    "L_LatShank",
    "R_DorsalFoot",
    "L_DorsalFoot",
    "R_Ankle",
    "L_Ankle",
    "Xiphoid",
    "Forehead",
]

# Pre-specified, device-relevant configurations.
# We intentionally do NOT enumerate every possible sensor combination.
CONFIGS = {
    # Single-sensor screen
    "Single_R_Ankle": ["R_Ankle"],
    "Single_L_Ankle": ["L_Ankle"],
    "Single_R_LatShank": ["R_LatShank"],
    "Single_L_LatShank": ["L_LatShank"],
    "Single_R_DorsalFoot": ["R_DorsalFoot"],
    "Single_L_DorsalFoot": ["L_DorsalFoot"],
    "Single_LowerBack": ["LowerBack"],
    "Single_R_Wrist": ["R_Wrist"],
    "Single_L_Wrist": ["L_Wrist"],
    "Single_Xiphoid": ["Xiphoid"],
    "Single_Forehead": ["Forehead"],

    # Symmetric bilateral configurations
    "Bilateral_Ankles": ["R_Ankle", "L_Ankle"],
    "Bilateral_Shanks": ["R_LatShank", "L_LatShank"],
    "Bilateral_DorsalFeet": ["R_DorsalFoot", "L_DorsalFoot"],
    "Bilateral_Wrists": ["R_Wrist", "L_Wrist"],

    # Trunk + distal combinations
    "LowerBack_plus_R_Ankle": ["LowerBack", "R_Ankle"],
    "LowerBack_plus_L_Ankle": ["LowerBack", "L_Ankle"],
    "LowerBack_plus_Bilateral_Ankles": [
        "LowerBack", "R_Ankle", "L_Ankle"
    ],
}

N_REPEATS = 3
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
RANDOM_SEED = 20260905

ELASTIC_GRID = {
    "regressor__model__alpha": [
        0.001, 0.01, 0.1, 0.5, 1.0, 2.0
    ],
    "regressor__model__l1_ratio": [
        0.05, 0.25, 0.50, 0.75, 0.95
    ],
}


def concordance_correlation_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    x = y_true[mask]
    y = y_pred[mask]

    if len(x) < 3:
        return np.nan

    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]

    denom = vx + vy + (mx - my) ** 2

    if denom == 0:
        return np.nan

    return float(2.0 * cov / denom)


def pearson_r(y_true, y_pred):
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(stats.pearsonr(x, y).statistic)


def normalized_rmse(y_true, y_pred):
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    sd = np.std(x, ddof=1)

    if sd == 0:
        return np.nan

    rmse = np.sqrt(mean_squared_error(x, y))
    return float(rmse / sd)


def direction_agreement(y_true_delta, y_pred_delta):
    t = np.asarray(y_true_delta, dtype=float)
    p = np.asarray(y_pred_delta, dtype=float)

    mask = np.isfinite(t) & np.isfinite(p)
    t = t[mask]
    p = p[mask]

    if len(t) == 0:
        return np.nan

    # sign(0) is retained as 0. Exact zeros are rare here.
    return float(np.mean(np.sign(t) == np.sign(p)))


def make_repeated_group_folds(subjects, n_splits, repeat, seed):
    """
    Deterministic randomized assignment of participant IDs to folds.
    Every participant appears exactly once as test in each repeat.
    """
    unique_subjects = np.array(sorted(pd.unique(subjects)))
    rng = np.random.default_rng(seed + repeat)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)

    fold_subjects = [
        set(shuffled[i::n_splits])
        for i in range(n_splits)
    ]

    splits = []

    subject_array = np.asarray(subjects)

    for fold_idx, test_subjects in enumerate(fold_subjects):
        test_mask = np.array(
            [s in test_subjects for s in subject_array],
            dtype=bool,
        )
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]

        splits.append((fold_idx, train_idx, test_idx))

    return splits


def config_feature_columns(all_columns, sensors):
    columns = []

    for sensor in sensors:
        prefix = f"{sensor}__"
        columns.extend(
            [
                c for c in all_columns
                if c.startswith(prefix)
            ]
        )

    return sorted(set(columns))


def build_estimator():
    regressor = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                ElasticNet(
                    max_iter=20000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    # Standardize target inside each training fold as well.
    return TransformedTargetRegressor(
        regressor=regressor,
        transformer=StandardScaler(),
    )


def session_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 3:
        return {
            "session_mae": np.nan,
            "session_rmse": np.nan,
            "session_r2": np.nan,
            "session_pearson_r": np.nan,
            "session_ccc": np.nan,
        }

    return {
        "session_mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "session_rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "session_r2": float(
            r2_score(y_true, y_pred)
        ),
        "session_pearson_r": pearson_r(y_true, y_pred),
        "session_ccc": concordance_correlation_coefficient(
            y_true, y_pred
        ),
    }


def delta_metrics(delta_df):
    y_true = delta_df["true_delta"].to_numpy(dtype=float)
    y_pred = delta_df["pred_delta"].to_numpy(dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 3:
        return {
            "delta_n": len(y_true),
            "delta_mae": np.nan,
            "delta_rmse": np.nan,
            "delta_nrmse": np.nan,
            "delta_r2": np.nan,
            "delta_pearson_r": np.nan,
            "delta_ccc": np.nan,
            "direction_agreement": np.nan,
        }

    return {
        "delta_n": len(y_true),
        "delta_mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "delta_rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "delta_nrmse": normalized_rmse(y_true, y_pred),
        "delta_r2": float(
            r2_score(y_true, y_pred)
        ),
        "delta_pearson_r": pearson_r(y_true, y_pred),
        "delta_ccc": concordance_correlation_coefficient(
            y_true, y_pred
        ),
        "direction_agreement": direction_agreement(
            y_true, y_pred
        ),
    }


def make_delta_table(predictions):
    """
    One row per participant for a single configuration/target/repeat.
    """
    rows = []

    for subject, sub in predictions.groupby("subject"):
        s1 = sub[sub["session"] == "s1"]
        s2 = sub[sub["session"] == "s2"]

        if len(s1) != 1 or len(s2) != 1:
            continue

        true_s1 = float(s1.iloc[0]["y_true"])
        true_s2 = float(s2.iloc[0]["y_true"])
        pred_s1 = float(s1.iloc[0]["y_pred"])
        pred_s2 = float(s2.iloc[0]["y_pred"])

        rows.append(
            {
                "subject": subject,
                "true_s1": true_s1,
                "true_s2": true_s2,
                "pred_s1": pred_s1,
                "pred_s2": pred_s2,
                "true_delta": true_s2 - true_s1,
                "pred_delta": pred_s2 - pred_s1,
            }
        )

    return pd.DataFrame(rows)


def main():
    for required in [
        IMU_FEATURE_FILE,
        AVAILABILITY_FILE,
        REFERENCE_FILE,
        ELIGIBILITY_FILE,
    ]:
        if not required.exists():
            raise SystemExit(
                f"Missing required input:\n{required}"
            )

    imu = pd.read_csv(IMU_FEATURE_FILE)
    availability = pd.read_csv(AVAILABILITY_FILE)
    reference = pd.read_csv(REFERENCE_FILE)
    eligibility = pd.read_csv(ELIGIBILITY_FILE)

    for frame in [imu, availability, reference, eligibility]:
        frame["subject"] = (
            frame["subject"].astype(str).str.upper()
        )

    imu["session"] = imu["session"].astype(str).str.lower()
    reference["session"] = (
        reference["session"].astype(str).str.lower()
    )

    # Core reference cohort only.
    core_subjects = set(
        eligibility.loc[
            eligibility["paired_core"] == True,
            "subject",
        ]
    )

    availability_core = availability[
        availability["subject"].isin(core_subjects)
    ].copy()

    # Common body-sensor cohort:
    # paired usable data from every body sensor.
    common_mask = np.ones(
        len(availability_core),
        dtype=bool,
    )

    for sensor in BODY_SENSORS:
        common_mask &= availability_core[
            f"{sensor}_paired"
        ].astype(bool).to_numpy()

    common_subjects = sorted(
        availability_core.loc[
            common_mask, "subject"
        ].tolist()
    )

    common_df = pd.DataFrame(
        {"subject": common_subjects}
    )
    common_df.to_csv(
        OUT / "common_cohort_subjects.csv",
        index=False,
    )

    # Configuration definitions.
    config_rows = []
    for config_name, sensors in CONFIGS.items():
        config_rows.append(
            {
                "configuration": config_name,
                "n_sensors": len(sensors),
                "sensors": ";".join(sensors),
            }
        )

    pd.DataFrame(config_rows).to_csv(
        OUT / "sensor_configuration_definitions.csv",
        index=False,
    )

    # Session-level merge.
    ref_cols = [
        "subject",
        "session",
        "step_length_m",
        "cadence_steps_min",
        "swing_time_s",
    ]

    analysis = imu.merge(
        reference[ref_cols],
        on=["subject", "session"],
        how="inner",
        validate="one_to_one",
    )

    analysis = analysis[
        analysis["subject"].isin(common_subjects)
    ].copy()

    # Must have both sessions.
    counts = (
        analysis.groupby("subject")["session"]
        .nunique()
    )
    complete_ids = set(
        counts[counts == 2].index
    )
    analysis = analysis[
        analysis["subject"].isin(complete_ids)
    ].copy()

    # Freeze row order.
    analysis = analysis.sort_values(
        ["subject", "session"]
    ).reset_index(drop=True)

    session_prediction_rows = []
    delta_prediction_rows = []
    performance_rows = []

    total_jobs = (
        len(CONFIGS)
        * len(TARGETS)
        * N_REPEATS
    )
    job_counter = 0

    for config_name, sensors in CONFIGS.items():
        feature_cols = config_feature_columns(
            analysis.columns,
            sensors,
        )

        if not feature_cols:
            print(
                f"Skipping {config_name}: no feature columns found."
            )
            continue

        X_all = analysis[feature_cols].copy()
        groups_all = analysis["subject"].to_numpy()

        for target in TARGETS:
            y_all = pd.to_numeric(
                analysis[target],
                errors="coerce",
            )

            valid = y_all.notna()

            X = X_all.loc[valid].reset_index(drop=True)
            y = y_all.loc[valid].reset_index(drop=True)
            meta = analysis.loc[
                valid,
                ["subject", "session"],
            ].reset_index(drop=True)

            groups = meta["subject"].to_numpy()

            # Ensure each retained participant still has both sessions.
            session_counts = (
                meta.groupby("subject")["session"]
                .nunique()
            )
            valid_subjects = set(
                session_counts[session_counts == 2].index
            )

            keep = meta["subject"].isin(valid_subjects)
            X = X.loc[keep].reset_index(drop=True)
            y = y.loc[keep].reset_index(drop=True)
            meta = meta.loc[keep].reset_index(drop=True)
            groups = meta["subject"].to_numpy()

            for repeat in range(N_REPEATS):
                job_counter += 1

                print(
                    f"[{job_counter}/{total_jobs}] "
                    f"{config_name} | {target} | repeat {repeat + 1}"
                )

                outer_splits = make_repeated_group_folds(
                    groups,
                    n_splits=N_OUTER_FOLDS,
                    repeat=repeat,
                    seed=RANDOM_SEED,
                )

                repeat_predictions = []

                for fold_idx, train_idx, test_idx in outer_splits:
                    X_train = X.iloc[train_idx]
                    y_train = y.iloc[train_idx]
                    g_train = groups[train_idx]

                    X_test = X.iloc[test_idx]
                    y_test = y.iloc[test_idx]

                    # Inner grouped CV uses only training participants.
                    inner_cv = GroupKFold(
                        n_splits=N_INNER_FOLDS
                    )

                    estimator = build_estimator()

                    search = GridSearchCV(
                        estimator=estimator,
                        param_grid=ELASTIC_GRID,
                        scoring="neg_mean_absolute_error",
                        cv=inner_cv,
                        n_jobs=-1,
                        refit=True,
                    )

                    search.fit(
                        X_train,
                        y_train,
                        groups=g_train,
                    )

                    y_pred = search.predict(X_test)

                    for local_i, global_i in enumerate(test_idx):
                        record = {
                            "configuration": config_name,
                            "sensors": ";".join(sensors),
                            "n_sensors": len(sensors),
                            "target": target,
                            "repeat": repeat + 1,
                            "fold": fold_idx + 1,
                            "subject": meta.iloc[global_i]["subject"],
                            "session": meta.iloc[global_i]["session"],
                            "y_true": float(y.iloc[global_i]),
                            "y_pred": float(y_pred[local_i]),
                        }

                        # Extract selected hyperparameters for auditability.
                        bp = search.best_params_
                        record["best_alpha"] = bp.get(
                            "regressor__model__alpha",
                            np.nan,
                        )
                        record["best_l1_ratio"] = bp.get(
                            "regressor__model__l1_ratio",
                            np.nan,
                        )

                        session_prediction_rows.append(record)
                        repeat_predictions.append(record)

                repeat_pred_df = pd.DataFrame(
                    repeat_predictions
                )

                sm = session_metrics(
                    repeat_pred_df["y_true"],
                    repeat_pred_df["y_pred"],
                )

                delta_df = make_delta_table(
                    repeat_pred_df
                )

                for _, drow in delta_df.iterrows():
                    delta_prediction_rows.append(
                        {
                            "configuration": config_name,
                            "sensors": ";".join(sensors),
                            "n_sensors": len(sensors),
                            "target": target,
                            "repeat": repeat + 1,
                            **drow.to_dict(),
                        }
                    )

                dm = delta_metrics(delta_df)

                performance_rows.append(
                    {
                        "configuration": config_name,
                        "sensors": ";".join(sensors),
                        "n_sensors": len(sensors),
                        "target": target,
                        "repeat": repeat + 1,
                        "n_participants": delta_df["subject"].nunique(),
                        "n_sessions": len(repeat_pred_df),
                        **sm,
                        **dm,
                    }
                )

    session_predictions = pd.DataFrame(
        session_prediction_rows
    )
    delta_predictions = pd.DataFrame(
        delta_prediction_rows
    )
    performance = pd.DataFrame(
        performance_rows
    )

    session_predictions.to_csv(
        OUT / "cv_session_predictions.csv",
        index=False,
    )

    delta_predictions.to_csv(
        OUT / "cv_longitudinal_predictions.csv",
        index=False,
    )

    performance.to_csv(
        OUT / "target_configuration_performance.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Average repeated-CV performance by config and target.
    # ---------------------------------------------------------
    metric_cols = [
        "session_mae",
        "session_rmse",
        "session_r2",
        "session_pearson_r",
        "session_ccc",
        "delta_mae",
        "delta_rmse",
        "delta_nrmse",
        "delta_r2",
        "delta_pearson_r",
        "delta_ccc",
        "direction_agreement",
    ]

    agg_dict = {
        "n_sensors": "first",
        "sensors": "first",
        "n_participants": "first",
    }

    for metric in metric_cols:
        agg_dict[metric] = "mean"

    target_summary = (
        performance
        .groupby(
            ["configuration", "target"],
            as_index=False,
        )
        .agg(agg_dict)
    )

    target_summary.to_csv(
        OUT / "target_configuration_performance.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Configuration-level summary across the 3 frozen targets.
    # Units differ, so aggregate only dimensionless metrics.
    # ---------------------------------------------------------
    config_summary = (
        target_summary
        .groupby("configuration", as_index=False)
        .agg(
            n_sensors=("n_sensors", "first"),
            sensors=("sensors", "first"),
            n_participants=("n_participants", "first"),
            mean_delta_ccc=("delta_ccc", "mean"),
            mean_delta_pearson_r=("delta_pearson_r", "mean"),
            mean_delta_nrmse=("delta_nrmse", "mean"),
            mean_direction_agreement=("direction_agreement", "mean"),
            mean_session_ccc=("session_ccc", "mean"),
        )
    )

    # Descriptive ranking score:
    # CCC rewards agreement; nRMSE penalizes error; direction agreement
    # captures whether worsening/improvement direction is preserved.
    # This is used ONLY for screening/ranking, not as a clinical endpoint.
    config_summary["screen_score"] = (
        config_summary["mean_delta_ccc"].fillna(-1.0)
        + (1.0 - config_summary["mean_delta_nrmse"].fillna(2.0))
        + config_summary["mean_direction_agreement"].fillna(0.0)
    ) / 3.0

    config_summary.to_csv(
        OUT / "configuration_summary.csv",
        index=False,
    )

    ranking = config_summary.sort_values(
        [
            "screen_score",
            "mean_delta_ccc",
            "mean_delta_nrmse",
            "n_sensors",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "rank",
        np.arange(1, len(ranking) + 1),
    )

    ranking.to_csv(
        OUT / "sensor_screen_ranking.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Human-readable summary.
    # ---------------------------------------------------------
    lines = [
        "WearGait-PD Primary Sensor Configuration Screen",
        "=" * 56,
        f"CORE reference participants: {len(core_subjects)}",
        f"Common body-sensor cohort used for fair comparison: {len(common_subjects)}",
        f"Configurations screened: {len(CONFIGS)}",
        f"Frozen gait targets: {', '.join(TARGETS)}",
        f"Repeated grouped CV: {N_REPEATS} x {N_OUTER_FOLDS}-fold",
        "",
        "Important:",
        "  s1 and s2 from the same participant always remain in the same fold.",
        "  Clinical variables are not model inputs.",
        "  All configurations are compared on the same participants.",
        "",
        "Top configurations by descriptive screening score:",
    ]

    for _, row in ranking.head(10).iterrows():
        lines.append(
            f"  #{int(row['rank'])} {row['configuration']} | "
            f"sensors={int(row['n_sensors'])} | "
            f"delta CCC={row['mean_delta_ccc']:.3f} | "
            f"delta nRMSE={row['mean_delta_nrmse']:.3f} | "
            f"direction={100*row['mean_direction_agreement']:.1f}% | "
            f"screen score={row['screen_score']:.3f}"
        )

    lines += [
        "",
        "Target-specific performance of the top 3 configurations:",
    ]

    for config_name in ranking.head(3)["configuration"]:
        lines.append(f"  {config_name}:")
        sub = target_summary[
            target_summary["configuration"] == config_name
        ]

        for _, row in sub.iterrows():
            lines.append(
                f"    {row['target']}: "
                f"delta CCC={row['delta_ccc']:.3f}, "
                f"r={row['delta_pearson_r']:.3f}, "
                f"nRMSE={row['delta_nrmse']:.3f}, "
                f"direction={100*row['direction_agreement']:.1f}%"
            )

    lines += [
        "",
        "NEXT:",
        "  Re-test the best single-sensor configuration and the strongest",
        "  multi-sensor comparator on their maximum available cohorts.",
        "  Add a secondary nonlinear model and bootstrap uncertainty.",
        "  Then construct the sensor-burden vs longitudinal-performance Pareto frontier.",
    ]

    summary = "\n".join(lines)

    (OUT / "sensor_screen_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print(f"\nOutputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
