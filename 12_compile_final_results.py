r"""
12_compile_final_results.py

Compiles the completed WearGait-PD analyses into a frozen, manuscript-ready
results package.

This script DOES NOT fit any new models or run new hypothesis tests.
It only summarizes analyses already completed in Scripts 04-11.

Scientific hierarchy
--------------------
PRIMARY ARCHITECTURE COMPARISON
    Script 08 common-cohort screen (N=38)
    -> fair head-to-head comparison on identical participants

SENSITIVITY / VALIDATION
    Script 09 maximum-available cohort + Elastic Net / Random Forest
    -> tests robustness, nonlinear sensitivity, and Pareto efficiency

PERSONALIZED CHANGE MODELING
    Script 10
    -> tests raw and within-person normalized change modeling

SECONDARY CLINICAL ANALYSIS
    Script 11
    -> baseline clinical construct/prognostic associations only
    -> does NOT establish delta clinical progression

Expected project outputs
------------------------
WearGait_PD_Longitudinal/
    project_reference_analysis/
        core_longitudinal_stats.csv
        strict_longitudinal_stats.csv
        longitudinal_metric_ranking.csv
    project_sensor_screen/
        sensor_screen_ranking.csv
        target_configuration_performance.csv
    project_sensor_validation/
        final_validation_performance.csv
        architecture_summary.csv
        bootstrap_uncertainty.csv
        pareto_frontier.csv
    project_personalized_models/
        modeling_strategy_summary.csv
        paired_improvement_bootstrap.csv
    project_clinical_secondary/
        baseline_construct_validity.csv
        baseline_severity_vs_longitudinal_change.csv
        categorical_longitudinal_descriptives.csv
    project_clinical_linkage/
        baseline_clinical_linkage.csv

Outputs
-------
WearGait_PD_Longitudinal/project_final_results/
    analysis_freeze_summary.txt
    table_1_reference_targets.csv
    table_2_primary_sensor_screen.csv
    table_3_validation_architectures.csv
    table_4_personalization_supported_effects.csv
    table_5_clinical_secondary.csv
    figure_1_reference_target_changes.png
    figure_2_primary_sensor_screen.png
    figure_3_validation_pareto.png
    figure_4_personalization_step_length.png
    figure_5_clinical_correlations.png
"""

from pathlib import Path
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(
        "matplotlib is required.\nInstall with:\n"
        "    python -m pip install matplotlib"
    )

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "project_final_results"
OUT.mkdir(parents=True, exist_ok=True)

REF_ANALYSIS = ROOT / "project_reference_analysis"
SCREEN = ROOT / "project_sensor_screen"
VALIDATION = ROOT / "project_sensor_validation"
PERSONALIZED = ROOT / "project_personalized_models"
CLINICAL = ROOT / "project_clinical_secondary"
CLIN_LINK = ROOT / "project_clinical_linkage"

CORE_STATS = REF_ANALYSIS / "core_longitudinal_stats.csv"
STRICT_STATS = REF_ANALYSIS / "strict_longitudinal_stats.csv"
METRIC_RANKING = REF_ANALYSIS / "longitudinal_metric_ranking.csv"

SCREEN_RANKING = SCREEN / "sensor_screen_ranking.csv"
SCREEN_TARGET = SCREEN / "target_configuration_performance.csv"

FINAL_PERF = VALIDATION / "final_validation_performance.csv"
ARCH_SUMMARY = VALIDATION / "architecture_summary.csv"
BOOTSTRAP = VALIDATION / "bootstrap_uncertainty.csv"
PARETO = VALIDATION / "pareto_frontier.csv"

STRATEGY = PERSONALIZED / "modeling_strategy_summary.csv"
PAIRED_BOOT = PERSONALIZED / "paired_improvement_bootstrap.csv"

CLIN_CONSTRUCT = CLINICAL / "baseline_construct_validity.csv"
CLIN_PROGNOSIS = CLINICAL / "baseline_severity_vs_longitudinal_change.csv"
CLIN_CATEGORICAL = CLINICAL / "categorical_longitudinal_descriptives.csv"
BASELINE_LINK = CLIN_LINK / "baseline_clinical_linkage.csv"

FROZEN_TARGETS = [
    "step_length_m",
    "cadence_steps_min",
    "swing_time_s",
]

TARGET_LABELS = {
    "step_length_m": "Step length",
    "cadence_steps_min": "Cadence",
    "swing_time_s": "Swing time",
}

PRIMARY_SINGLE_SENSOR_CLASS = (
    "single distal-leg sensor"
)


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required completed-analysis output:\n{path}"
        )


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def positive_supported(row, metric):
    low = row.get(f"{metric}_ci_low", np.nan)
    return pd.notna(low) and float(low) > 0


def main():
    required = [
        CORE_STATS,
        STRICT_STATS,
        METRIC_RANKING,
        SCREEN_RANKING,
        SCREEN_TARGET,
        FINAL_PERF,
        ARCH_SUMMARY,
        BOOTSTRAP,
        PARETO,
        STRATEGY,
        PAIRED_BOOT,
        CLIN_CONSTRUCT,
        CLIN_PROGNOSIS,
        BASELINE_LINK,
    ]

    for path in required:
        require(path)

    core = pd.read_csv(CORE_STATS)
    strict = pd.read_csv(STRICT_STATS)
    metric_rank = pd.read_csv(METRIC_RANKING)

    screen_rank = pd.read_csv(SCREEN_RANKING)
    screen_target = pd.read_csv(SCREEN_TARGET)

    final_perf = pd.read_csv(FINAL_PERF)
    architecture = pd.read_csv(ARCH_SUMMARY)
    bootstrap = pd.read_csv(BOOTSTRAP)
    pareto = pd.read_csv(PARETO)

    strategy = pd.read_csv(STRATEGY)
    paired_boot = pd.read_csv(PAIRED_BOOT)

    clin_construct = pd.read_csv(CLIN_CONSTRUCT)
    clin_prog = pd.read_csv(CLIN_PROGNOSIS)
    baseline_link = pd.read_csv(BASELINE_LINK)

    categorical = None
    if CLIN_CATEGORICAL.exists():
        categorical = pd.read_csv(CLIN_CATEGORICAL)

    # ------------------------------------------------------------------
    # TABLE 1: Frozen reference targets
    # ------------------------------------------------------------------
    frozen_rows = []

    for target in FROZEN_TARGETS:
        source = core[core["metric"] == target].copy()

        if source.empty:
            # Fall back to a likely metric-name column if naming differs.
            candidate_cols = [
                c for c in core.columns
                if c.lower() in {"metric", "outcome", "variable"}
            ]
            if candidate_cols:
                source = core[
                    core[candidate_cols[0]] == target
                ].copy()

        if source.empty:
            continue

        row = source.iloc[0].to_dict()

        frozen_rows.append(
            {
                "target": target,
                "target_label": TARGET_LABELS[target],
                "n": row.get("n", np.nan),
                "s1_mean": row.get("s1_mean", np.nan),
                "s2_mean": row.get("s2_mean", np.nan),
                "mean_delta": row.get("mean_delta", np.nan),
                "hedges_gz": row.get(
                    "hedges_gz",
                    row.get("hedges_g_z", np.nan),
                ),
                "direction_consistency": row.get(
                    "direction_consistency",
                    row.get(
                        "direction_consistency_prop",
                        np.nan,
                    ),
                ),
                "wilcoxon_q_fdr": row.get(
                    "wilcoxon_q_fdr",
                    row.get(
                        "fdr_wilcoxon_q",
                        np.nan,
                    ),
                ),
            }
        )

    table1 = pd.DataFrame(frozen_rows)
    table1.to_csv(
        OUT / "table_1_reference_targets.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # TABLE 2: Primary fair common-cohort architecture screen
    # ------------------------------------------------------------------
    cols2 = [
        "rank",
        "configuration",
        "n_sensors",
        "sensors",
        "n_participants",
        "mean_delta_ccc",
        "mean_delta_pearson_r",
        "mean_delta_nrmse",
        "mean_direction_agreement",
        "screen_score",
    ]
    cols2 = [c for c in cols2 if c in screen_rank.columns]

    table2 = screen_rank[cols2].copy()
    table2.to_csv(
        OUT / "table_2_primary_sensor_screen.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # TABLE 3: Validation architecture sensitivity
    # ------------------------------------------------------------------
    table3 = architecture.copy()

    pareto_small = pareto[
        [
            "configuration",
            "model",
            "pareto_efficient",
        ]
    ].copy()

    table3 = table3.merge(
        pareto_small,
        on=["configuration", "model"],
        how="left",
        validate="one_to_one",
    )

    table3 = table3.sort_values(
        ["model", "validation_score"],
        ascending=[True, False],
    )

    table3.to_csv(
        OUT / "table_3_validation_architectures.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # TABLE 4: Only bootstrap-supported personalization effects
    # ------------------------------------------------------------------
    supported_rows = []

    for _, row in paired_boot.iterrows():
        if row.get("strategy") != "DirectNormalizedDelta":
            continue

        for metric in [
            "ccc_improvement",
            "pearson_improvement",
            "nrmse_improvement",
            "direction_improvement",
        ]:
            if positive_supported(row, metric):
                supported_rows.append(
                    {
                        "configuration": row["configuration"],
                        "target": row["target"],
                        "model": row["model"],
                        "strategy": row["strategy"],
                        "metric": metric,
                        "mean_improvement": row.get(
                            f"{metric}_mean",
                            np.nan,
                        ),
                        "ci_low": row.get(
                            f"{metric}_ci_low",
                            np.nan,
                        ),
                        "ci_high": row.get(
                            f"{metric}_ci_high",
                            np.nan,
                        ),
                        "n_aligned": row.get(
                            "n_aligned",
                            np.nan,
                        ),
                    }
                )

    table4 = pd.DataFrame(supported_rows)

    if not table4.empty:
        table4 = table4.sort_values(
            [
                "target",
                "configuration",
                "model",
                "metric",
            ]
        )

    table4.to_csv(
        OUT / "table_4_personalization_supported_effects.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # TABLE 5: Secondary clinical correlations
    # ------------------------------------------------------------------
    clinical_rows = []

    for _, row in clin_construct.iterrows():
        clinical_rows.append(
            {
                "analysis_family": "baseline_construct_validity",
                "clinical_variable": row.get(
                    "clinical_variable",
                    np.nan,
                ),
                "gait_variable": row.get(
                    "baseline_gait_metric",
                    np.nan,
                ),
                "n": row.get("n", np.nan),
                "spearman_rho": row.get(
                    "spearman_rho",
                    np.nan,
                ),
                "p_value": row.get("p_value", np.nan),
                "q_fdr": row.get("q_fdr", np.nan),
            }
        )

    for _, row in clin_prog.iterrows():
        clinical_rows.append(
            {
                "analysis_family": (
                    "baseline_severity_vs_later_gait_change"
                ),
                "clinical_variable": row.get(
                    "baseline_clinical_variable",
                    np.nan,
                ),
                "gait_variable": (
                    "delta_"
                    + str(
                        row.get(
                            "longitudinal_gait_target",
                            "",
                        )
                    )
                ),
                "n": row.get("n", np.nan),
                "spearman_rho": row.get(
                    "spearman_rho",
                    np.nan,
                ),
                "p_value": row.get("p_value", np.nan),
                "q_fdr": row.get("q_fdr", np.nan),
            }
        )

    table5 = pd.DataFrame(clinical_rows).sort_values(
        ["analysis_family", "q_fdr", "p_value"]
    )

    table5.to_csv(
        OUT / "table_5_clinical_secondary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # FIGURE 1: Reference longitudinal target changes
    # ------------------------------------------------------------------
    if not table1.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))

        labels = table1["target_label"].tolist()
        effects = pd.to_numeric(
            table1["hedges_gz"],
            errors="coerce",
        ).to_numpy()

        y = np.arange(len(labels))

        ax.barh(y, effects)
        ax.axvline(0, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Paired Hedges g$_z$")
        ax.set_title(
            "Longitudinal responsiveness of frozen reference targets"
        )
        ax.invert_yaxis()

        savefig(
            OUT / "figure_1_reference_target_changes.png"
        )

    # ------------------------------------------------------------------
    # FIGURE 2: Primary common-cohort sensor screen
    # ------------------------------------------------------------------
    if not screen_rank.empty:
        top = screen_rank.head(10).copy()
        top = top.sort_values(
            "screen_score",
            ascending=True,
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        labels = top["configuration"].tolist()
        scores = top["screen_score"].to_numpy(dtype=float)
        y = np.arange(len(top))

        ax.barh(y, scores)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Descriptive screening score")
        ax.set_title(
            "Primary sensor-ablation screen\n"
            "Common participant cohort"
        )

        savefig(
            OUT / "figure_2_primary_sensor_screen.png"
        )

    # ------------------------------------------------------------------
    # FIGURE 3: Pareto view of validated architectures
    # ------------------------------------------------------------------
    if not architecture.empty:
        for model_name in sorted(
            architecture["model"].dropna().unique()
        ):
            sub = architecture[
                architecture["model"] == model_name
            ].copy()

            fig, ax = plt.subplots(figsize=(7.5, 5.5))

            x = sub["mean_delta_nrmse"].to_numpy(dtype=float)
            y = sub["mean_delta_ccc"].to_numpy(dtype=float)

            sizes = (
                70
                + 70
                * (
                    sub["n_sensors"].to_numpy(dtype=float)
                    - 1
                )
            )

            ax.scatter(x, y, s=sizes)

            for _, row in sub.iterrows():
                ax.annotate(
                    row["configuration"],
                    (
                        row["mean_delta_nrmse"],
                        row["mean_delta_ccc"],
                    ),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

            ax.set_xlabel("Mean longitudinal nRMSE (lower is better)")
            ax.set_ylabel("Mean longitudinal CCC (higher is better)")
            ax.set_title(
                f"Validated architecture tradeoff: {model_name}"
            )

            safe_model = (
                str(model_name)
                .lower()
                .replace(" ", "_")
            )

            savefig(
                OUT
                / f"figure_3_validation_pareto_{safe_model}.png"
            )

    # ------------------------------------------------------------------
    # FIGURE 4: Supported step-length personalization effects
    # ------------------------------------------------------------------
    step_effects = table4[
        table4["target"] == "step_length_m"
    ].copy() if not table4.empty else pd.DataFrame()

    if not step_effects.empty:
        # Separate metrics because units/scales differ.
        metrics = [
            "ccc_improvement",
            "nrmse_improvement",
            "direction_improvement",
        ]

        for metric in metrics:
            sub = step_effects[
                step_effects["metric"] == metric
            ].copy()

            if sub.empty:
                continue

            sub["label"] = (
                sub["configuration"].astype(str)
                + " | "
                + sub["model"].astype(str)
            )

            sub = sub.sort_values(
                "mean_improvement",
                ascending=True,
            )

            fig, ax = plt.subplots(figsize=(8.5, 5.0))

            y = np.arange(len(sub))
            means = sub["mean_improvement"].to_numpy(dtype=float)
            low = sub["ci_low"].to_numpy(dtype=float)
            high = sub["ci_high"].to_numpy(dtype=float)

            xerr = np.vstack(
                [
                    means - low,
                    high - means,
                ]
            )

            ax.errorbar(
                means,
                y,
                xerr=xerr,
                fmt="o",
                capsize=3,
            )
            ax.axvline(0, linewidth=1)
            ax.set_yticks(y)
            ax.set_yticklabels(sub["label"])
            ax.set_xlabel(
                "Improvement vs PopulationAbsolute"
            )
            ax.set_title(
                "Bootstrap-supported personalization effect\n"
                f"Step length: {metric}"
            )

            safe_metric = metric.replace("_", "-")

            savefig(
                OUT
                / f"figure_4_personalization_step_length_{safe_metric}.png"
            )

    # ------------------------------------------------------------------
    # FIGURE 5: Clinical correlation effect sizes
    # ------------------------------------------------------------------
    if not table5.empty:
        # Show the strongest absolute-rho correlations only.
        plot = table5.dropna(
            subset=["spearman_rho"]
        ).copy()

        plot["abs_rho"] = plot["spearman_rho"].abs()
        plot = plot.sort_values(
            "abs_rho",
            ascending=False,
        ).head(12)

        plot["label"] = (
            plot["clinical_variable"].astype(str)
            + " vs "
            + plot["gait_variable"].astype(str)
        )

        plot = plot.sort_values(
            "spearman_rho",
            ascending=True,
        )

        fig, ax = plt.subplots(figsize=(9, 7))

        y = np.arange(len(plot))

        ax.barh(
            y,
            plot["spearman_rho"].to_numpy(dtype=float),
        )
        ax.axvline(0, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(plot["label"], fontsize=8)
        ax.set_xlabel("Spearman rho")
        ax.set_title(
            "Strongest secondary clinical associations\n"
            "(none survived FDR correction)"
        )

        savefig(
            OUT / "figure_5_clinical_correlations.png"
        )

    # ------------------------------------------------------------------
    # Baseline clinical timing / medication context for interpretation
    # ------------------------------------------------------------------
    core_link = baseline_link[
        baseline_link["paired_core"] == True
    ].copy()

    days_col = "Days since Part III Clinical Evaluation"
    same_day_n = np.nan
    days_known_n = np.nan

    if days_col in core_link.columns:
        days = pd.to_numeric(
            core_link[days_col],
            errors="coerce",
        )
        days_known_n = int(days.notna().sum())
        same_day_n = int((days == 0).sum())

    med_counts = {}
    if "motor_exam_med_state" in core_link.columns:
        med_counts = (
            core_link["motor_exam_med_state"]
            .value_counts(dropna=False)
            .to_dict()
        )

    # ------------------------------------------------------------------
    # Analysis freeze summary
    # ------------------------------------------------------------------
    primary_top = (
        screen_rank.iloc[0]
        if len(screen_rank)
        else None
    )

    en_pareto = pareto[
        (pareto["model"] == "ElasticNet")
        & (pareto["pareto_efficient"] == True)
    ]

    rf_pareto = pareto[
        (pareto["model"] == "RandomForest")
        & (pareto["pareto_efficient"] == True)
    ]

    n_personal_supported = len(table4)
    n_personal_step = (
        int((table4["target"] == "step_length_m").sum())
        if not table4.empty
        else 0
    )

    construct_sig = int(
        (clin_construct["q_fdr"] < 0.05).sum()
    )
    prog_sig = int(
        (clin_prog["q_fdr"] < 0.05).sum()
    )

    construct_trends = clin_construct.sort_values(
        "p_value"
    ).head(2)

    prognosis_trends = clin_prog.sort_values(
        "p_value"
    ).head(2)

    lines = [
        "WearGait-PD FINAL ANALYSIS FREEZE",
        "=" * 68,
        "",
        "PRIMARY RESEARCH QUESTION",
        "What is the minimum wearable IMU architecture that preserves",
        "longitudinal gait-change information in Parkinson's disease?",
        "",
        "PRIMARY ARCHITECTURE ANALYSIS",
        "  The fair head-to-head comparison is Script 08, which used the",
        "  same 38 participants for every body-sensor configuration.",
    ]

    if primary_top is not None:
        lines += [
            f"  Top descriptive screen: {primary_top['configuration']}",
            f"    sensors: {int(primary_top['n_sensors'])}",
            f"    mean delta CCC: {primary_top['mean_delta_ccc']:.3f}",
            f"    mean delta nRMSE: {primary_top['mean_delta_nrmse']:.3f}",
            f"    direction agreement: "
            f"{100*primary_top['mean_direction_agreement']:.1f}%",
            "",
        ]

    lines += [
        "  Frozen interpretation:",
        "    Single-sensor distal-leg architectures remain competitive",
        "    with multi-sensor systems and stay on the longitudinal",
        "    performance frontier.",
        "    Exact optimal placement is model-dependent; do not claim that",
        "    one specific left/right location is universally superior.",
        "",
        "VALIDATION / MODEL SENSITIVITY",
        "  Pareto-efficient Elastic Net configurations:",
    ]

    for _, row in en_pareto.iterrows():
        lines.append(
            f"    {row['configuration']} "
            f"({int(row['n_sensors'])} sensor(s)): "
            f"CCC={row['mean_delta_ccc']:.3f}, "
            f"nRMSE={row['mean_delta_nrmse']:.3f}, "
            f"direction={100*row['mean_direction_agreement']:.1f}%"
        )

    lines += [
        "",
        "  Pareto-efficient Random Forest configurations:",
    ]

    for _, row in rf_pareto.iterrows():
        lines.append(
            f"    {row['configuration']} "
            f"({int(row['n_sensors'])} sensor(s)): "
            f"CCC={row['mean_delta_ccc']:.3f}, "
            f"nRMSE={row['mean_delta_nrmse']:.3f}, "
            f"direction={100*row['mean_direction_agreement']:.1f}%"
        )

    lines += [
        "",
        "TARGET-SPECIFIC INTERPRETATION",
        "  Cadence/rhythmic change is the strongest preserved domain.",
        "  Step-length/spatial change is harder to estimate from population",
        "  absolute-session models.",
        "  Do not imply equal performance across all gait domains.",
        "",
        "WITHIN-PERSON CHANGE MODELING",
        f"  Bootstrap-supported normalized-model improvements: "
        f"{n_personal_supported}",
        f"  Supported effects occurring for step length: "
        f"{n_personal_step}/{n_personal_supported if n_personal_supported else 0}",
        "",
        "  Frozen interpretation:",
        "    Within-person normalization is NOT globally superior.",
        "    Its supported benefit is domain-specific and concentrated in",
        "    ankle-based spatial/step-length change estimation.",
        "",
        "SECONDARY CLINICAL ANALYSIS",
        f"  Baseline construct correlations surviving FDR: {construct_sig}",
        f"  Baseline severity vs later gait-change correlations surviving FDR: "
        f"{prog_sig}",
    ]

    if same_day_n is not np.nan:
        lines += [
            f"  Part III timing known for {days_known_n} CORE participants;",
            f"  same-day clinical evaluation: {same_day_n}/{days_known_n}.",
        ]

    if med_counts:
        med_text = ", ".join(
            f"{k}={v}"
            for k, v in med_counts.items()
        )
        lines.append(
            f"  Baseline motor-exam medication-state distribution: {med_text}"
        )

    lines += [
        "",
        "  Strongest uncorrected construct-validity trends:",
    ]

    for _, row in construct_trends.iterrows():
        lines.append(
            f"    {row['clinical_variable']} vs "
            f"{row['baseline_gait_metric']}: "
            f"rho={row['spearman_rho']:.3f}, "
            f"p={row['p_value']:.4g}, "
            f"q={row['q_fdr']:.4g}"
        )

    lines += [
        "",
        "  Strongest uncorrected prognostic trends:",
    ]

    for _, row in prognosis_trends.iterrows():
        lines.append(
            f"    {row['baseline_clinical_variable']} vs delta "
            f"{row['longitudinal_gait_target']}: "
            f"rho={row['spearman_rho']:.3f}, "
            f"p={row['p_value']:.4g}, "
            f"q={row['q_fdr']:.4g}"
        )

    lines += [
        "",
        "  Frozen interpretation:",
        "    Baseline clinical variables did not show FDR-significant",
        "    associations with baseline gait or subsequent gait change.",
        "    These null secondary results do not alter the primary",
        "    sensor-only architecture conclusion.",
        "",
        "CLINICAL CLAIM LIMIT",
        "  No Version 2 follow-up clinical table has been linked.",
        "  Therefore:",
        "    - do NOT calculate or claim delta MDS-UPDRS",
        "    - do NOT label gait change as proven clinical PD progression",
        "    - describe the endpoint as longitudinal gait change",
        "",
        "FINAL STUDY TAKEAWAY",
        "  A minimal distal-leg IMU architecture can retain meaningful",
        "  longitudinal gait-change information while avoiding the hardware",
        "  burden of a large multi-sensor system. Rhythmic gait change is",
        "  preserved most consistently. Within-person change modeling adds",
        "  selective value for the more difficult spatial step-length domain.",
        "",
        "ANALYSIS STATUS: FROZEN",
        "  Do not add new exploratory hypothesis tests unless requested by",
        "  a reviewer, committee member, or a clearly pre-specified",
        "  sensitivity question.",
    ]

    summary = "\n".join(lines)

    (OUT / "analysis_freeze_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )

    print(summary)
    print(f"\nFinal results package saved to:\n{OUT}")


if __name__ == "__main__":
    main()
