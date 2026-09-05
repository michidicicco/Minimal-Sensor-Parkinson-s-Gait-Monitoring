r"""
09_validate_shortlisted_sensor_architectures.py

Final validation/sensitivity stage for the WearGait-PD minimal-sensor study.

This script validates a SHORTLIST chosen from the fair common-cohort screen.

Primary questions
-----------------
1. Does the best one-sensor architecture remain competitive when each
   configuration uses its maximum available CORE cohort?
2. Do conclusions persist under a nonlinear model?
3. What is the bootstrap uncertainty around longitudinal-change performance?
4. Which configurations sit on the hardware-burden/performance Pareto frontier?

Frozen reference targets
------------------------
- step_length_m
- cadence_steps_min
- swing_time_s

Shortlisted configurations
--------------------------
- Single_L_Ankle
- Single_R_Ankle
- Bilateral_Ankles
- Single_L_LatShank
- Single_R_DorsalFoot
- Single_R_Wrist
- Bilateral_Wrists
- LowerBack_plus_L_Ankle

Why these?
- L ankle: strongest overall one-sensor compromise in screen
- R ankle: side-sensitivity comparator
- bilateral ankles: symmetric 2-sensor comparator with full availability
- L shank: highest overall mean delta CCC among single sensors
- R dorsal foot: strong cadence specialist
- R wrist / bilateral wrists: strongest swing-time specialists
- lower back + L ankle: top overall 2-sensor screen score

Models
------
A. Elastic Net with grouped inner-CV tuning
B. Random Forest as a pre-specified nonlinear sensitivity model

Cross-validation
----------------
Repeated participant-grouped 5-fold CV.
Both sessions from a participant remain in the same fold.

Uncertainty
-----------
For each configuration/target/model:
- average the repeated out-of-fold prediction for each participant-session
- form predicted and true s2-s1 changes
- bootstrap participants with replacement
- report percentile 95% CIs for delta CCC, delta nRMSE, and direction agreement

IMPORTANT
---------
Clinical variables are not predictors in this primary engineering validation.
This script evaluates preservation of longitudinal gait change, not clinical
disease progression.

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
WearGait_PD_Longitudinal/project_sensor_validation/
    shortlisted_configuration_definitions.csv
    max_cohort_subjects.csv
    final_cv_session_predictions.csv
    final_longitudinal_predictions.csv
    final_validation_performance.csv
    bootstrap_uncertainty.csv
    architecture_summary.csv
    pareto_frontier.csv
    target_specialists.csv
    final_validation_summary.txt
"""

from pathlib import Path
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
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import RandomForestRegressor
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

OUT = ROOT / "project_sensor_validation"
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

CONFIGS = {
    "Single_L_Ankle": ["L_Ankle"],
    "Single_R_Ankle": ["R_Ankle"],
    "Bilateral_Ankles": ["R_Ankle", "L_Ankle"],
    "Single_L_LatShank": ["L_LatShank"],
    "Single_R_DorsalFoot": ["R_DorsalFoot"],
    "Single_R_Wrist": ["R_Wrist"],
    "Bilateral_Wrists": ["R_Wrist", "L_Wrist"],
    "LowerBack_plus_L_Ankle": ["LowerBack", "L_Ankle"],
}

N_REPEATS = 3
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
N_BOOT = 3000
RANDOM_SEED = 20260905

ELASTIC_GRID = {
    "regressor__model__alpha": [0.01, 0.1, 1.0, 2.0],
    "regressor__model__l1_ratio": [0.25, 0.75],
}


def ccc(y_true, y_pred):
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

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

    return float(2 * cov / denom)


def pearson_r(y_true, y_pred):
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(stats.pearsonr(x, y).statistic)


def nrmse(y_true, y_pred):
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

    return float(np.sqrt(mean_squared_error(x, y)) / sd)


def direction_agreement(y_true, y_pred):
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan

    return float(np.mean(np.sign(x) == np.sign(y)))


def randomized_group_folds(subjects, n_splits, repeat, seed):
    unique_subjects = np.array(sorted(pd.unique(subjects)))
    rng = np.random.default_rng(seed + repeat)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)

    fold_subjects = [
        set(shuffled[i::n_splits])
        for i in range(n_splits)
    ]

    subject_array = np.asarray(subjects)

    splits = []
    for fold_idx, test_subjects in enumerate(fold_subjects):
        test_mask = np.array(
            [s in test_subjects for s in subject_array],
            dtype=bool,
        )
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        splits.append((fold_idx, train_idx, test_idx))

    return splits


def config_feature_columns(columns, sensors):
    selected = []
    for sensor in sensors:
        prefix = f"{sensor}__"
        selected.extend(
            [c for c in columns if c.startswith(prefix)]
        )
    return sorted(set(selected))


def build_elastic():
    reg = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(
                    max_iter=20000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    return TransformedTargetRegressor(
        regressor=reg,
        transformer=StandardScaler(),
    )


def build_rf():
    # Pre-specified nonlinear sensitivity model.
    # No tuning here to avoid a large second model-selection search.
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_predict_model(
    model_name,
    X_train,
    y_train,
    groups_train,
    X_test,
):
    if model_name == "ElasticNet":
        inner_cv = GroupKFold(n_splits=N_INNER_FOLDS)

        search = GridSearchCV(
            estimator=build_elastic(),
            param_grid=ELASTIC_GRID,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )

        search.fit(
            X_train,
            y_train,
            groups=groups_train,
        )

        pred = search.predict(X_test)
        params = search.best_params_

        return pred, {
            "best_alpha": params.get(
                "regressor__model__alpha", np.nan
            ),
            "best_l1_ratio": params.get(
                "regressor__model__l1_ratio", np.nan
            ),
        }

    if model_name == "RandomForest":
        model = build_rf()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        return pred, {
            "best_alpha": np.nan,
            "best_l1_ratio": np.nan,
        }

    raise ValueError(model_name)


def delta_table_from_session_predictions(pred):
    """
    First average OOF prediction across repeats for each participant-session,
    then calculate one longitudinal delta per participant.
    """
    avg = (
        pred.groupby(
            ["subject", "session"],
            as_index=False,
        )
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
        )
    )

    rows = []

    for subject, sub in avg.groupby("subject"):
        s1 = sub[sub["session"] == "s1"]
        s2 = sub[sub["session"] == "s2"]

        if len(s1) != 1 or len(s2) != 1:
            continue

        rows.append(
            {
                "subject": subject,
                "true_s1": float(s1.iloc[0]["y_true"]),
                "true_s2": float(s2.iloc[0]["y_true"]),
                "pred_s1": float(s1.iloc[0]["y_pred"]),
                "pred_s2": float(s2.iloc[0]["y_pred"]),
                "true_delta": (
                    float(s2.iloc[0]["y_true"])
                    - float(s1.iloc[0]["y_true"])
                ),
                "pred_delta": (
                    float(s2.iloc[0]["y_pred"])
                    - float(s1.iloc[0]["y_pred"])
                ),
            }
        )

    return pd.DataFrame(rows)


def performance_from_delta(delta_df):
    y = delta_df["true_delta"].to_numpy(dtype=float)
    p = delta_df["pred_delta"].to_numpy(dtype=float)

    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]

    return {
        "n_participants": len(y),
        "delta_mae": (
            float(mean_absolute_error(y, p))
            if len(y) else np.nan
        ),
        "delta_rmse": (
            float(np.sqrt(mean_squared_error(y, p)))
            if len(y) else np.nan
        ),
        "delta_nrmse": nrmse(y, p),
        "delta_r2": (
            float(r2_score(y, p))
            if len(y) >= 3 else np.nan
        ),
        "delta_pearson_r": pearson_r(y, p),
        "delta_ccc": ccc(y, p),
        "direction_agreement": direction_agreement(y, p),
    }


def session_performance(pred):
    y = pred["y_true"].to_numpy(dtype=float)
    p = pred["y_pred"].to_numpy(dtype=float)

    # Average repeated prediction for each session before scoring.
    avg = (
        pred.groupby(
            ["subject", "session"],
            as_index=False,
        )
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
        )
    )

    y = avg["y_true"].to_numpy(dtype=float)
    p = avg["y_pred"].to_numpy(dtype=float)

    return {
        "session_mae": float(mean_absolute_error(y, p)),
        "session_rmse": float(
            np.sqrt(mean_squared_error(y, p))
        ),
        "session_r2": float(r2_score(y, p)),
        "session_pearson_r": pearson_r(y, p),
        "session_ccc": ccc(y, p),
    }


def bootstrap_delta_metrics(delta_df, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(delta_df)

    if n < 3:
        return {}

    results = {
        "delta_ccc": [],
        "delta_nrmse": [],
        "direction_agreement": [],
        "delta_pearson_r": [],
    }

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = delta_df.iloc[idx]

        y = boot["true_delta"].to_numpy(dtype=float)
        p = boot["pred_delta"].to_numpy(dtype=float)

        results["delta_ccc"].append(ccc(y, p))
        results["delta_nrmse"].append(nrmse(y, p))
        results["direction_agreement"].append(
            direction_agreement(y, p)
        )
        results["delta_pearson_r"].append(
            pearson_r(y, p)
        )

    rows = {}

    for metric, values in results.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]

        if len(arr) == 0:
            rows[f"{metric}_ci_low"] = np.nan
            rows[f"{metric}_ci_high"] = np.nan
        else:
            rows[f"{metric}_ci_low"] = float(
                np.percentile(arr, 2.5)
            )
            rows[f"{metric}_ci_high"] = float(
                np.percentile(arr, 97.5)
            )

    return rows


def pareto_mask(df):
    """
    Pareto-efficient if no other row has:
      - <= sensor count
      - >= mean delta CCC
      - <= mean delta nRMSE
      - >= mean direction agreement
    with at least one strict improvement.
    """
    efficient = []

    for i, row in df.iterrows():
        dominated = False

        for j, other in df.iterrows():
            if i == j:
                continue

            no_worse = (
                other["n_sensors"] <= row["n_sensors"]
                and other["mean_delta_ccc"] >= row["mean_delta_ccc"]
                and other["mean_delta_nrmse"] <= row["mean_delta_nrmse"]
                and other["mean_direction_agreement"]
                >= row["mean_direction_agreement"]
            )

            strictly_better = (
                other["n_sensors"] < row["n_sensors"]
                or other["mean_delta_ccc"] > row["mean_delta_ccc"]
                or other["mean_delta_nrmse"] < row["mean_delta_nrmse"]
                or other["mean_direction_agreement"]
                > row["mean_direction_agreement"]
            )

            if no_worse and strictly_better:
                dominated = True
                break

        efficient.append(not dominated)

    return np.array(efficient, dtype=bool)


def main():
    required = [
        IMU_FEATURE_FILE,
        AVAILABILITY_FILE,
        REFERENCE_FILE,
        ELIGIBILITY_FILE,
    ]

    for path in required:
        if not path.exists():
            raise SystemExit(f"Missing required input:\n{path}")

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

    core_ids = set(
        eligibility.loc[
            eligibility["paired_core"] == True,
            "subject",
        ]
    )

    avail_core = availability[
        availability["subject"].isin(core_ids)
    ].copy()

    # Save maximum cohort membership for each config.
    cohort_rows = []
    config_subjects = {}

    for config_name, sensors in CONFIGS.items():
        mask = np.ones(len(avail_core), dtype=bool)

        for sensor in sensors:
            mask &= avail_core[
                f"{sensor}_paired"
            ].astype(bool).to_numpy()

        ids = sorted(
            avail_core.loc[mask, "subject"].tolist()
        )
        config_subjects[config_name] = ids

        for subject in ids:
            cohort_rows.append(
                {
                    "configuration": config_name,
                    "n_sensors": len(sensors),
                    "sensors": ";".join(sensors),
                    "subject": subject,
                }
            )

    pd.DataFrame(cohort_rows).to_csv(
        OUT / "max_cohort_subjects.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "configuration": name,
                "n_sensors": len(sensors),
                "sensors": ";".join(sensors),
                "max_core_n": len(config_subjects[name]),
            }
            for name, sensors in CONFIGS.items()
        ]
    ).to_csv(
        OUT / "shortlisted_configuration_definitions.csv",
        index=False,
    )

    ref_cols = [
        "subject",
        "session",
        *TARGETS,
    ]

    merged = imu.merge(
        reference[ref_cols],
        on=["subject", "session"],
        how="inner",
        validate="one_to_one",
    )

    all_prediction_rows = []
    all_delta_rows = []
    performance_rows = []
    bootstrap_rows = []

    model_names = ["ElasticNet", "RandomForest"]

    total_jobs = (
        len(CONFIGS)
        * len(TARGETS)
        * len(model_names)
    )
    job = 0

    for config_name, sensors in CONFIGS.items():
        ids = config_subjects[config_name]

        data = merged[
            merged["subject"].isin(ids)
        ].copy()

        # keep participants with both sessions in merged data
        counts = (
            data.groupby("subject")["session"].nunique()
        )
        valid_ids = set(counts[counts == 2].index)

        data = data[
            data["subject"].isin(valid_ids)
        ].sort_values(
            ["subject", "session"]
        ).reset_index(drop=True)

        feature_cols = config_feature_columns(
            data.columns,
            sensors,
        )

        if not feature_cols:
            print(f"Skipping {config_name}: no features.")
            continue

        for target in TARGETS:
            target_numeric = pd.to_numeric(
                data[target],
                errors="coerce",
            )

            keep = target_numeric.notna()
            d = data.loc[keep].reset_index(drop=True)
            y = target_numeric.loc[keep].reset_index(drop=True)

            # Paired target requirement
            counts = d.groupby("subject")["session"].nunique()
            valid_ids = set(counts[counts == 2].index)
            keep2 = d["subject"].isin(valid_ids)

            d = d.loc[keep2].reset_index(drop=True)
            y = y.loc[keep2].reset_index(drop=True)

            X = d[feature_cols].copy()
            groups = d["subject"].to_numpy()

            for model_name in model_names:
                job += 1
                print(
                    f"[{job}/{total_jobs}] "
                    f"{config_name} | {target} | {model_name}"
                )

                pred_rows = []

                for repeat in range(N_REPEATS):
                    splits = randomized_group_folds(
                        groups,
                        n_splits=N_OUTER_FOLDS,
                        repeat=repeat,
                        seed=RANDOM_SEED,
                    )

                    for fold_idx, train_idx, test_idx in splits:
                        X_train = X.iloc[train_idx]
                        y_train = y.iloc[train_idx]
                        g_train = groups[train_idx]

                        X_test = X.iloc[test_idx]

                        pred, tuning = fit_predict_model(
                            model_name=model_name,
                            X_train=X_train,
                            y_train=y_train,
                            groups_train=g_train,
                            X_test=X_test,
                        )

                        for local_i, global_i in enumerate(test_idx):
                            row = {
                                "configuration": config_name,
                                "n_sensors": len(sensors),
                                "sensors": ";".join(sensors),
                                "target": target,
                                "model": model_name,
                                "repeat": repeat + 1,
                                "fold": fold_idx + 1,
                                "subject": d.iloc[global_i]["subject"],
                                "session": d.iloc[global_i]["session"],
                                "y_true": float(y.iloc[global_i]),
                                "y_pred": float(pred[local_i]),
                                **tuning,
                            }

                            pred_rows.append(row)
                            all_prediction_rows.append(row)

                pred_df = pd.DataFrame(pred_rows)

                delta_df = delta_table_from_session_predictions(
                    pred_df
                )

                for _, dr in delta_df.iterrows():
                    all_delta_rows.append(
                        {
                            "configuration": config_name,
                            "n_sensors": len(sensors),
                            "sensors": ";".join(sensors),
                            "target": target,
                            "model": model_name,
                            **dr.to_dict(),
                        }
                    )

                perf = {
                    "configuration": config_name,
                    "n_sensors": len(sensors),
                    "sensors": ";".join(sensors),
                    "target": target,
                    "model": model_name,
                    **session_performance(pred_df),
                    **performance_from_delta(delta_df),
                }

                performance_rows.append(perf)

                boot = bootstrap_delta_metrics(
                    delta_df,
                    n_boot=N_BOOT,
                    seed=(
                        RANDOM_SEED
                        + sum(ord(c) for c in config_name)
                        + sum(ord(c) for c in target)
                        + (0 if model_name == "ElasticNet" else 100000)
                    ),
                )

                bootstrap_rows.append(
                    {
                        "configuration": config_name,
                        "target": target,
                        "model": model_name,
                        "n_participants": len(delta_df),
                        **boot,
                    }
                )

    pred_all = pd.DataFrame(all_prediction_rows)
    delta_all = pd.DataFrame(all_delta_rows)
    performance = pd.DataFrame(performance_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)

    pred_all.to_csv(
        OUT / "final_cv_session_predictions.csv",
        index=False,
    )

    delta_all.to_csv(
        OUT / "final_longitudinal_predictions.csv",
        index=False,
    )

    performance.to_csv(
        OUT / "final_validation_performance.csv",
        index=False,
    )

    bootstrap.to_csv(
        OUT / "bootstrap_uncertainty.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Architecture summary across frozen targets, by model.
    # ---------------------------------------------------------
    architecture = (
        performance
        .groupby(
            ["configuration", "model"],
            as_index=False,
        )
        .agg(
            n_sensors=("n_sensors", "first"),
            sensors=("sensors", "first"),
            n_participants=("n_participants", "min"),
            mean_delta_ccc=("delta_ccc", "mean"),
            mean_delta_pearson_r=("delta_pearson_r", "mean"),
            mean_delta_nrmse=("delta_nrmse", "mean"),
            mean_direction_agreement=("direction_agreement", "mean"),
            mean_session_ccc=("session_ccc", "mean"),
        )
    )

    architecture["validation_score"] = (
        architecture["mean_delta_ccc"].fillna(-1)
        + (1 - architecture["mean_delta_nrmse"].fillna(2))
        + architecture["mean_direction_agreement"].fillna(0)
    ) / 3

    architecture.to_csv(
        OUT / "architecture_summary.csv",
        index=False,
    )

    # Pareto frontier calculated separately by model.
    pareto_frames = []

    for model_name, sub in architecture.groupby("model"):
        sub = sub.copy()
        sub["pareto_efficient"] = pareto_mask(sub)
        pareto_frames.append(sub)

    pareto = pd.concat(
        pareto_frames,
        ignore_index=True,
    )

    pareto.to_csv(
        OUT / "pareto_frontier.csv",
        index=False,
    )

    # Target specialists.
    specialists = []

    for model_name in model_names:
        for target in TARGETS:
            sub = performance[
                (performance["model"] == model_name)
                & (performance["target"] == target)
            ].copy()

            if sub.empty:
                continue

            best_ccc = sub.sort_values(
                ["delta_ccc", "delta_nrmse", "n_sensors"],
                ascending=[False, True, True],
            ).iloc[0]

            best_error = sub.sort_values(
                ["delta_nrmse", "delta_ccc", "n_sensors"],
                ascending=[True, False, True],
            ).iloc[0]

            best_direction = sub.sort_values(
                ["direction_agreement", "delta_ccc", "n_sensors"],
                ascending=[False, False, True],
            ).iloc[0]

            for label, row in [
                ("best_delta_ccc", best_ccc),
                ("best_delta_nrmse", best_error),
                ("best_direction_agreement", best_direction),
            ]:
                specialists.append(
                    {
                        "model": model_name,
                        "target": target,
                        "criterion": label,
                        "configuration": row["configuration"],
                        "n_sensors": row["n_sensors"],
                        "n_participants": row["n_participants"],
                        "delta_ccc": row["delta_ccc"],
                        "delta_nrmse": row["delta_nrmse"],
                        "direction_agreement": row[
                            "direction_agreement"
                        ],
                    }
                )

    specialists = pd.DataFrame(specialists)

    specialists.to_csv(
        OUT / "target_specialists.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Summary text
    # ---------------------------------------------------------
    en = architecture[
        architecture["model"] == "ElasticNet"
    ].sort_values(
        ["validation_score", "n_sensors"],
        ascending=[False, True],
    )

    rf = architecture[
        architecture["model"] == "RandomForest"
    ].sort_values(
        ["validation_score", "n_sensors"],
        ascending=[False, True],
    )

    lines = [
        "WearGait-PD Shortlisted Sensor Architecture Validation",
        "=" * 62,
        "",
        "Maximum CORE cohort sizes:",
    ]

    for config_name, sensors in CONFIGS.items():
        lines.append(
            f"  {config_name}: "
            f"{len(config_subjects[config_name])} participants"
        )

    lines += [
        "",
        "Elastic Net architecture ranking:",
    ]

    for _, row in en.iterrows():
        lines.append(
            f"  {row['configuration']} | "
            f"sensors={int(row['n_sensors'])} | "
            f"N={int(row['n_participants'])} | "
            f"delta CCC={row['mean_delta_ccc']:.3f} | "
            f"delta nRMSE={row['mean_delta_nrmse']:.3f} | "
            f"direction={100*row['mean_direction_agreement']:.1f}% | "
            f"score={row['validation_score']:.3f}"
        )

    lines += [
        "",
        "Random Forest sensitivity ranking:",
    ]

    for _, row in rf.iterrows():
        lines.append(
            f"  {row['configuration']} | "
            f"sensors={int(row['n_sensors'])} | "
            f"N={int(row['n_participants'])} | "
            f"delta CCC={row['mean_delta_ccc']:.3f} | "
            f"delta nRMSE={row['mean_delta_nrmse']:.3f} | "
            f"direction={100*row['mean_direction_agreement']:.1f}% | "
            f"score={row['validation_score']:.3f}"
        )

    lines += [
        "",
        "Pareto-efficient configurations:",
    ]

    for model_name in model_names:
        lines.append(f"  {model_name}:")
        sub = pareto[
            (pareto["model"] == model_name)
            & (pareto["pareto_efficient"] == True)
        ]

        for _, row in sub.iterrows():
            lines.append(
                f"    {row['configuration']} | "
                f"{int(row['n_sensors'])} sensor(s) | "
                f"CCC={row['mean_delta_ccc']:.3f} | "
                f"nRMSE={row['mean_delta_nrmse']:.3f} | "
                f"direction={100*row['mean_direction_agreement']:.1f}%"
            )

    lines += [
        "",
        "Interpretation guardrail:",
        "  Strong cadence performance does not imply equally strong",
        "  preservation of step-length or swing-time change.",
        "  The final device recommendation should therefore consider",
        "  both overall architecture performance and target-specific tradeoffs.",
        "",
        "NEXT:",
        "  Inspect bootstrap confidence intervals and model agreement.",
        "  Freeze the Pareto-optimal architecture conclusion.",
        "  Then add baseline clinical covariate analyses as secondary validation.",
    ]

    summary = "\n".join(lines)

    (OUT / "final_validation_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print(f"\nOutputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
