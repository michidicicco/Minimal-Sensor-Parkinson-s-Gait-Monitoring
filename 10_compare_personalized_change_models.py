r"""
10_compare_personalized_change_models.py

Tests the second major WearGait-PD research question:

Does within-person baseline-normalized IMU modeling improve longitudinal
gait-change detection compared with population absolute-session modeling?

Three approaches
----------------
A. PopulationAbsolute
   Existing Script 09 approach:
   predict gait metric separately at s1 and s2, then subtract predictions.

B. DirectRawDelta
   Predict reference s2-s1 change directly from raw IMU feature differences:
       IMU_s2 - IMU_s1

C. DirectNormalizedDelta
   Predict reference s2-s1 change directly from within-person symmetric
   relative IMU change:
       2 * (s2 - s1) / (|s1| + |s2| + epsilon)

The symmetric normalization is bounded and avoids unstable conventional
percentage changes when baseline values are close to zero.

Frozen targets
--------------
- step_length_m
- cadence_steps_min
- swing_time_s

Shortlisted architectures
-------------------------
- Single_L_Ankle
- Single_R_Ankle
- Bilateral_Ankles
- Single_L_LatShank
- LowerBack_plus_L_Ankle

Models
------
- ElasticNet with nested tuning
- RandomForest as nonlinear sensitivity analysis

Cross-validation
----------------
Repeated 5-fold participant-level CV. Each participant contributes ONE
longitudinal row in the direct-change approaches.

Comparison
----------
For each configuration/target/model:
- delta CCC
- delta Pearson r
- delta nRMSE
- direction-of-change agreement
- participant-bootstrap paired difference versus PopulationAbsolute

A positive paired bootstrap difference means improvement for:
- CCC
- Pearson r
- direction agreement
- nRMSE benefit = PopulationAbsolute nRMSE - personalized nRMSE

IMPORTANT
---------
This is an engineering comparison of longitudinal gait-change modeling.
It does not establish clinical Parkinson's disease progression.

Expected inputs
---------------
WearGait_PD_Longitudinal/
    project_imu_features/
        paired_imu_feature_changes.csv
        paired_sensor_feature_availability.csv
    project_reference_gait_refined/
        longitudinal_reference_changes_core.csv
        paired_reference_eligibility.csv
    project_sensor_validation/
        final_longitudinal_predictions.csv

Outputs
-------
WearGait_PD_Longitudinal/project_personalized_models/
    personalized_cv_predictions.csv
    personalized_model_performance.csv
    paired_improvement_bootstrap.csv
    modeling_strategy_summary.csv
    personalized_model_summary.txt
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
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import GridSearchCV, KFold
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
REFINED_DIR = ROOT / "project_reference_gait_refined"
VALIDATION_DIR = ROOT / "project_sensor_validation"

OUT = ROOT / "project_personalized_models"
OUT.mkdir(parents=True, exist_ok=True)

IMU_CHANGE_FILE = IMU_DIR / "paired_imu_feature_changes.csv"
AVAILABILITY_FILE = IMU_DIR / "paired_sensor_feature_availability.csv"
REFERENCE_CHANGE_FILE = (
    REFINED_DIR / "longitudinal_reference_changes_core.csv"
)
ELIGIBILITY_FILE = REFINED_DIR / "paired_reference_eligibility.csv"
ABSOLUTE_PREDICTIONS_FILE = (
    VALIDATION_DIR / "final_longitudinal_predictions.csv"
)

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
    "LowerBack_plus_L_Ankle": ["LowerBack", "L_Ankle"],
}

DIRECT_STRATEGIES = [
    "DirectRawDelta",
    "DirectNormalizedDelta",
]

MODELS = ["ElasticNet", "RandomForest"]

N_REPEATS = 3
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
N_BOOT = 3000
RANDOM_SEED = 20260905
EPS = 1e-8

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

    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    denom = vx + vy + (mx - my) ** 2

    return float(2 * cov / denom) if denom > 0 else np.nan


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

    return float(
        np.sqrt(mean_squared_error(x, y)) / sd
    )


def direction_agreement(y_true, y_pred):
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return np.nan

    return float(np.mean(np.sign(x) == np.sign(y)))


def performance(y_true, y_pred):
    return {
        "delta_ccc": ccc(y_true, y_pred),
        "delta_pearson_r": pearson_r(y_true, y_pred),
        "delta_nrmse": nrmse(y_true, y_pred),
        "direction_agreement": direction_agreement(
            y_true, y_pred
        ),
    }


def randomized_participant_folds(subjects, n_splits, repeat, seed):
    subjects = np.array(sorted(subjects))
    rng = np.random.default_rng(seed + repeat)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)

    folds = [
        set(shuffled[i::n_splits])
        for i in range(n_splits)
    ]

    result = []
    for fold_idx, test_ids in enumerate(folds):
        test_mask = np.array(
            [s in test_ids for s in subjects],
            dtype=bool,
        )
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        result.append((fold_idx, train_idx, test_idx))

    return result


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


def fit_predict(
    model_name,
    X_train,
    y_train,
    X_test,
    repeat,
):
    if model_name == "ElasticNet":
        inner = KFold(
            n_splits=N_INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED + repeat,
        )

        search = GridSearchCV(
            estimator=build_elastic(),
            param_grid=ELASTIC_GRID,
            scoring="neg_mean_absolute_error",
            cv=inner,
            n_jobs=-1,
            refit=True,
        )

        search.fit(X_train, y_train)
        return search.predict(X_test)

    if model_name == "RandomForest":
        model = build_rf()
        model.fit(X_train, y_train)
        return model.predict(X_test)

    raise ValueError(model_name)


def sensor_base_features(columns, sensor):
    suffix = "__s1"
    prefix = f"{sensor}__"

    features = []

    for col in columns:
        if col.startswith(prefix) and col.endswith(suffix):
            base = col[:-len(suffix)]
            features.append(base)

    return sorted(set(features))


def make_config_matrix(change_df, sensors, strategy):
    pieces = {}

    for sensor in sensors:
        bases = sensor_base_features(
            change_df.columns,
            sensor,
        )

        for base in bases:
            s1_col = f"{base}__s1"
            s2_col = f"{base}__s2"
            delta_col = f"delta__{base}"

            s1 = pd.to_numeric(
                change_df[s1_col],
                errors="coerce",
            )
            s2 = pd.to_numeric(
                change_df[s2_col],
                errors="coerce",
            )

            if strategy == "DirectRawDelta":
                if delta_col in change_df.columns:
                    values = pd.to_numeric(
                        change_df[delta_col],
                        errors="coerce",
                    )
                else:
                    values = s2 - s1

                pieces[f"raw_delta__{base}"] = values

            elif strategy == "DirectNormalizedDelta":
                denom = s1.abs() + s2.abs() + EPS
                values = 2.0 * (s2 - s1) / denom

                # Physical feature magnitudes and spectral features should
                # yield bounded symmetric changes. Clip only numerical noise.
                values = values.clip(-2.0, 2.0)

                pieces[f"norm_delta__{base}"] = values

            else:
                raise ValueError(strategy)

    return pd.DataFrame(pieces, index=change_df.index)


def average_repeated_predictions(pred_df):
    return (
        pred_df.groupby("subject", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
        )
    )


def bootstrap_paired_improvement(
    aligned,
    n_boot,
    seed,
):
    """
    aligned columns:
      y_true
      pred_absolute
      pred_personalized

    Positive difference always means personalized approach is better.
    """
    rng = np.random.default_rng(seed)
    n = len(aligned)

    out = {
        "ccc_improvement": [],
        "pearson_improvement": [],
        "nrmse_improvement": [],
        "direction_improvement": [],
    }

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b = aligned.iloc[idx]

        y = b["y_true"].to_numpy(dtype=float)
        pa = b["pred_absolute"].to_numpy(dtype=float)
        pp = b["pred_personalized"].to_numpy(dtype=float)

        out["ccc_improvement"].append(
            ccc(y, pp) - ccc(y, pa)
        )
        out["pearson_improvement"].append(
            pearson_r(y, pp) - pearson_r(y, pa)
        )
        out["nrmse_improvement"].append(
            nrmse(y, pa) - nrmse(y, pp)
        )
        out["direction_improvement"].append(
            direction_agreement(y, pp)
            - direction_agreement(y, pa)
        )

    result = {}

    for metric, vals in out.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]

        if len(arr):
            result[f"{metric}_mean"] = float(np.mean(arr))
            result[f"{metric}_ci_low"] = float(
                np.percentile(arr, 2.5)
            )
            result[f"{metric}_ci_high"] = float(
                np.percentile(arr, 97.5)
            )
        else:
            result[f"{metric}_mean"] = np.nan
            result[f"{metric}_ci_low"] = np.nan
            result[f"{metric}_ci_high"] = np.nan

    return result


def main():
    required = [
        IMU_CHANGE_FILE,
        AVAILABILITY_FILE,
        REFERENCE_CHANGE_FILE,
        ELIGIBILITY_FILE,
        ABSOLUTE_PREDICTIONS_FILE,
    ]

    for path in required:
        if not path.exists():
            raise SystemExit(
                f"Missing required input:\n{path}"
            )

    change = pd.read_csv(IMU_CHANGE_FILE)
    availability = pd.read_csv(AVAILABILITY_FILE)
    reference = pd.read_csv(REFERENCE_CHANGE_FILE)
    eligibility = pd.read_csv(ELIGIBILITY_FILE)
    absolute = pd.read_csv(ABSOLUTE_PREDICTIONS_FILE)

    for frame in [
        change,
        availability,
        reference,
        eligibility,
        absolute,
    ]:
        frame["subject"] = (
            frame["subject"].astype(str).str.upper()
        )

    core_ids = set(
        eligibility.loc[
            eligibility["paired_core"] == True,
            "subject",
        ]
    )

    change = change[
        change["subject"].isin(core_ids)
    ].copy()

    availability = availability[
        availability["subject"].isin(core_ids)
    ].copy()

    # Reference target changes.
    target_cols = ["subject"] + [
        f"delta_{target}"
        for target in TARGETS
    ]

    analysis = change.merge(
        reference[target_cols],
        on="subject",
        how="inner",
        validate="one_to_one",
    )

    prediction_rows = []
    performance_rows = []
    bootstrap_rows = []

    total_jobs = (
        len(CONFIGS)
        * len(TARGETS)
        * len(MODELS)
        * len(DIRECT_STRATEGIES)
    )
    job = 0

    for config_name, sensors in CONFIGS.items():
        # Maximum paired sensor cohort.
        av = availability.copy()
        sensor_mask = np.ones(len(av), dtype=bool)

        for sensor in sensors:
            sensor_mask &= av[
                f"{sensor}_paired"
            ].astype(bool).to_numpy()

        ids = sorted(
            av.loc[sensor_mask, "subject"].tolist()
        )

        base_data = analysis[
            analysis["subject"].isin(ids)
        ].copy()

        base_data = base_data.sort_values(
            "subject"
        ).reset_index(drop=True)

        for target in TARGETS:
            target_col = f"delta_{target}"
            y_all = pd.to_numeric(
                base_data[target_col],
                errors="coerce",
            )

            for strategy in DIRECT_STRATEGIES:
                X_all = make_config_matrix(
                    base_data,
                    sensors,
                    strategy,
                )

                keep = y_all.notna()
                X = X_all.loc[keep].reset_index(drop=True)
                y = y_all.loc[keep].reset_index(drop=True)
                meta = base_data.loc[
                    keep, ["subject"]
                ].reset_index(drop=True)

                # Drop entirely empty columns, if any.
                X = X.loc[:, X.notna().any(axis=0)]

                for model_name in MODELS:
                    job += 1
                    print(
                        f"[{job}/{total_jobs}] "
                        f"{config_name} | {target} | "
                        f"{strategy} | {model_name}"
                    )

                    repeated_rows = []

                    subjects = meta["subject"].to_numpy()

                    for repeat in range(N_REPEATS):
                        splits = randomized_participant_folds(
                            subjects=subjects,
                            n_splits=N_OUTER_FOLDS,
                            repeat=repeat,
                            seed=RANDOM_SEED,
                        )

                        for fold_idx, train_idx, test_idx in splits:
                            pred = fit_predict(
                                model_name=model_name,
                                X_train=X.iloc[train_idx],
                                y_train=y.iloc[train_idx],
                                X_test=X.iloc[test_idx],
                                repeat=repeat,
                            )

                            for local_i, global_i in enumerate(test_idx):
                                row = {
                                    "configuration": config_name,
                                    "n_sensors": len(sensors),
                                    "sensors": ";".join(sensors),
                                    "target": target,
                                    "strategy": strategy,
                                    "model": model_name,
                                    "repeat": repeat + 1,
                                    "fold": fold_idx + 1,
                                    "subject": meta.iloc[global_i]["subject"],
                                    "y_true": float(y.iloc[global_i]),
                                    "y_pred": float(pred[local_i]),
                                }

                                repeated_rows.append(row)
                                prediction_rows.append(row)

                    pred_df = pd.DataFrame(repeated_rows)
                    avg = average_repeated_predictions(pred_df)

                    perf = performance(
                        avg["y_true"],
                        avg["y_pred"],
                    )

                    performance_rows.append(
                        {
                            "configuration": config_name,
                            "n_sensors": len(sensors),
                            "sensors": ";".join(sensors),
                            "target": target,
                            "strategy": strategy,
                            "model": model_name,
                            "n_participants": len(avg),
                            **perf,
                        }
                    )

                    # -------------------------------------------------
                    # Paired comparison with PopulationAbsolute
                    # using same configuration/model/target/participants.
                    # -------------------------------------------------
                    abs_sub = absolute[
                        (absolute["configuration"] == config_name)
                        & (absolute["target"] == target)
                        & (absolute["model"] == model_name)
                    ][
                        ["subject", "true_delta", "pred_delta"]
                    ].copy()

                    abs_sub = abs_sub.rename(
                        columns={
                            "true_delta": "y_true_absolute",
                            "pred_delta": "pred_absolute",
                        }
                    )

                    aligned = avg.merge(
                        abs_sub,
                        on="subject",
                        how="inner",
                        validate="one_to_one",
                    )

                    # True targets should be identical apart from tiny
                    # floating representation differences.
                    aligned = aligned.rename(
                        columns={
                            "y_true": "y_true",
                            "y_pred": "pred_personalized",
                        }
                    )

                    if len(aligned) >= 3:
                        boot = bootstrap_paired_improvement(
                            aligned[
                                [
                                    "y_true",
                                    "pred_absolute",
                                    "pred_personalized",
                                ]
                            ],
                            n_boot=N_BOOT,
                            seed=(
                                RANDOM_SEED
                                + sum(ord(c) for c in config_name)
                                + sum(ord(c) for c in target)
                                + sum(ord(c) for c in strategy)
                                + (0 if model_name == "ElasticNet" else 100000)
                            ),
                        )
                    else:
                        boot = {}

                    bootstrap_rows.append(
                        {
                            "configuration": config_name,
                            "target": target,
                            "strategy": strategy,
                            "model": model_name,
                            "n_aligned": len(aligned),
                            **boot,
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    direct_performance = pd.DataFrame(performance_rows)
    paired_boot = pd.DataFrame(bootstrap_rows)

    predictions.to_csv(
        OUT / "personalized_cv_predictions.csv",
        index=False,
    )

    direct_performance.to_csv(
        OUT / "personalized_model_performance.csv",
        index=False,
    )

    paired_boot.to_csv(
        OUT / "paired_improvement_bootstrap.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Add PopulationAbsolute point performance from Script 09
    # for comparison in one summary table.
    # ---------------------------------------------------------
    absolute_rows = []

    for (
        config_name,
        target,
        model_name
    ), sub in absolute.groupby(
        ["configuration", "target", "model"]
    ):
        if config_name not in CONFIGS:
            continue
        if target not in TARGETS:
            continue
        if model_name not in MODELS:
            continue

        y = sub["true_delta"].to_numpy(dtype=float)
        p = sub["pred_delta"].to_numpy(dtype=float)

        absolute_rows.append(
            {
                "configuration": config_name,
                "n_sensors": len(CONFIGS[config_name]),
                "sensors": ";".join(CONFIGS[config_name]),
                "target": target,
                "strategy": "PopulationAbsolute",
                "model": model_name,
                "n_participants": len(sub),
                **performance(y, p),
            }
        )

    combined = pd.concat(
        [
            pd.DataFrame(absolute_rows),
            direct_performance,
        ],
        ignore_index=True,
    )

    # Aggregate across the 3 frozen targets.
    strategy_summary = (
        combined
        .groupby(
            ["configuration", "strategy", "model"],
            as_index=False,
        )
        .agg(
            n_sensors=("n_sensors", "first"),
            n_participants=("n_participants", "min"),
            mean_delta_ccc=("delta_ccc", "mean"),
            mean_delta_pearson_r=("delta_pearson_r", "mean"),
            mean_delta_nrmse=("delta_nrmse", "mean"),
            mean_direction_agreement=(
                "direction_agreement", "mean"
            ),
        )
    )

    strategy_summary["comparison_score"] = (
        strategy_summary["mean_delta_ccc"].fillna(-1)
        + (
            1
            - strategy_summary["mean_delta_nrmse"].fillna(2)
        )
        + strategy_summary[
            "mean_direction_agreement"
        ].fillna(0)
    ) / 3

    strategy_summary.to_csv(
        OUT / "modeling_strategy_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Human-readable summary.
    # ---------------------------------------------------------
    lines = [
        "WearGait-PD Personalized Longitudinal Modeling Comparison",
        "=" * 66,
        "",
        "Strategies:",
        "  PopulationAbsolute = predict s1/s2 separately, then subtract",
        "  DirectRawDelta = predict gait change from raw IMU feature change",
        "  DirectNormalizedDelta = predict gait change from within-person",
        "                          symmetric normalized IMU change",
        "",
        "Architecture/model strategy rankings:",
    ]

    ranked = strategy_summary.sort_values(
        ["model", "comparison_score"],
        ascending=[True, False],
    )

    for model_name in MODELS:
        lines.append(f"  {model_name}:")
        sub = ranked[ranked["model"] == model_name]

        for _, row in sub.iterrows():
            lines.append(
                f"    {row['configuration']} | "
                f"{row['strategy']} | "
                f"N={int(row['n_participants'])} | "
                f"CCC={row['mean_delta_ccc']:.3f} | "
                f"nRMSE={row['mean_delta_nrmse']:.3f} | "
                f"direction={100*row['mean_direction_agreement']:.1f}% | "
                f"score={row['comparison_score']:.3f}"
            )

    # Count target/model/config comparisons where normalized strategy
    # has a bootstrap CI entirely above zero for each improvement metric.
    norm_boot = paired_boot[
        paired_boot["strategy"] == "DirectNormalizedDelta"
    ].copy()

    lines += [
        "",
        "Paired bootstrap evidence that normalized modeling improves",
        "PopulationAbsolute (95% CI entirely above zero):",
    ]

    for metric in [
        "ccc_improvement",
        "nrmse_improvement",
        "direction_improvement",
    ]:
        low_col = f"{metric}_ci_low"
        if low_col in norm_boot.columns:
            n_positive = int(
                (norm_boot[low_col] > 0).sum()
            )
            n_total = int(
                norm_boot[low_col].notna().sum()
            )
            lines.append(
                f"  {metric}: {n_positive}/{n_total} comparisons"
            )

    lines += [
        "",
        "Interpretation rule:",
        "  Do not claim personalization improves performance merely because",
        "  its point estimate is higher. Prefer paired-bootstrap evidence",
        "  and consistency across both Elastic Net and Random Forest.",
        "",
        "NEXT:",
        "  If baseline-normalized modeling is supported, freeze that as the",
        "  personalized digital-biomarker result.",
        "  Then use baseline MDS-UPDRS/medication/DBS only as secondary",
        "  clinical association/covariate analyses.",
    ]

    summary = "\n".join(lines)

    (OUT / "personalized_model_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)
    print(f"\nOutputs saved to:\n{OUT}")


if __name__ == "__main__":
    main()
