# Minimal-Sensor Parkinson's Gait Monitoring

A reproducible wearable-biomechanics pipeline for evaluating how much longitudinal gait-change information can be preserved using reduced IMU sensor configurations in Parkinson's disease.

This project uses the longitudinal **WearGait-PD** dataset to compare single- and multi-sensor wearable architectures against synchronized instrumented-walkway gait measures. The primary engineering goal is to identify a minimal wearable configuration that preserves meaningful within-person gait changes while reducing hardware burden.

## Research Question

**What is the minimum wearable IMU architecture that preserves longitudinal gait-change information in Parkinson's disease?**

Primary gait targets:

- Step length — spatial domain
- Cadence — rhythmic domain
- Swing time — gait-cycle phase

The primary comparison uses a common participant cohort so every candidate body-sensor configuration is evaluated on the same individuals.

## Study Design

- Longitudinal repeated walking assessments
- Primary task: `selfpace_mat`
- Straight walking only: `GeneralEvent == "Walk"`
- 11 body-mounted IMUs:
  - Forehead
  - Xiphoid
  - Lower back
  - Bilateral wrists
  - Bilateral lateral shanks
  - Bilateral dorsal feet
  - Bilateral ankles
- 2 insole inertial units retained as exploratory sensors
- Instrumented pressure walkway used as the reference system
- Primary reference cohort: 42 paired participants
- Common body-sensor cohort for the fair sensor-ablation screen: 38 participants
- 18 wearable sensor configurations evaluated
- Models:
  - Elastic Net
  - Random Forest sensitivity analysis
- Participant-grouped cross-validation
- Longitudinal endpoint: session 2 − session 1

> This repository evaluates preservation of longitudinal gait change. It does **not** claim that the observed gait changes represent clinical Parkinson's disease progression.

---

## Repository Structure

```text
WearGait_PD_Longitudinal/
│
├── scripts/
│   ├── 01_audit_weargait_longitudinal.py
│   ├── 02_build_paired_longitudinal_cohort_v2.py
│   ├── 03_extract_reference_gait_metrics.py
│   ├── 03b_refine_reference_qc.py
│   ├── 04_analyze_longitudinal_reference.py
│   ├── 05_audit_clinical_metadata.py
│   ├── 06_build_baseline_clinical_linkage.py
│   ├── 07_extract_imu_features.py
│   ├── 08_screen_sensor_configurations.py
│   ├── 09_validate_shortlisted_sensor_architectures.py
│   ├── 10_compare_personalized_change_models.py
│   ├── 11_secondary_clinical_analysis.py
│   └── 12_compile_final_results.py
│
├── PD Participants/
├── Clinical_Metadata/
│
├── project_audit/
├── project_cohort_v2/
├── project_reference_gait/
├── project_reference_gait_refined/
├── project_reference_analysis/
├── project_clinical_audit/
├── project_clinical_linkage/
├── project_imu_features/
├── project_sensor_screen/
├── project_sensor_validation/
├── project_personalized_models/
├── project_clinical_secondary/
└── project_final_results/
```

Raw WearGait-PD data are **not included** in this repository. Users should obtain the dataset from the official WearGait-PD/Synapse source and follow its access and data-use requirements.

---

# Analysis Scripts

## 01 — Dataset Audit

### `01_audit_weargait_longitudinal.py`

Performs an initial audit of the downloaded WearGait-PD longitudinal dataset.

Main tasks:

- Inventories downloaded files
- Detects participant/session/task structure
- Summarizes available CSV files and columns
- Identifies candidate clinical/metadata files
- Creates participant/session and task-level summaries

Primary outputs:

```text
project_audit/
├── file_inventory.csv
├── csv_inventory.csv
├── participant_session_summary.csv
├── participant_task_summary.csv
├── task_summary.csv
├── column_inventory.csv
├── candidate_clinical_files.csv
└── audit_summary.txt
```

---

## 02 — Build the Paired Longitudinal Cohort

### `02_build_paired_longitudinal_cohort_v2.py`

Builds the corrected Version 2 longitudinal cohort.

Important features:

- Recognizes both `NLS###` and `WPD###` participant IDs
- Uses `selfpace_mat` as the primary synchronized straight-walking file
- Does not globally exclude a participant because one sensor is missing
- Tracks paired availability separately for each wearable location
- Defines primary reference eligibility from paired sessions and usable walkway data

Primary outputs:

```text
project_cohort_v2/
├── session_file_map.csv
├── selfpace_mat_session_qc.csv
├── paired_longitudinal_cohort.csv
├── paired_sensor_availability.csv
├── sensor_availability_summary.csv
└── cohort_summary.txt
```

---

## 03 — Extract Instrumented-Walkway Reference Gait Metrics

### `03_extract_reference_gait_metrics.py`

Extracts reference spatiotemporal gait measures from the synchronized pressure walkway.

Reference construction includes:

- Straight walking only
- Pressure-walkway foot-contact events
- Spatial footfall position
- Step and stride timing
- Step and stride length
- Stance and swing time
- Cadence
- Gait speed
- Variability measures
- Bilateral asymmetry measures

Primary outputs:

```text
project_reference_gait/
├── session_reference_gait_metrics.csv
├── reference_metric_qc.csv
├── longitudinal_reference_changes.csv
└── reference_summary.txt
```

---

## 03b — Refine Reference QC

### `03b_refine_reference_qc.py`

Separates reference-quality requirements into two pre-specified tiers rather than using one all-or-nothing QC threshold.

### CORE QC

Used for mean spatiotemporal gait measures.

Requires:

- At least 2 usable straight-walking bouts
- At least 5 valid steps
- At least 3 valid strides

### STRICT QC

Used for variability and asymmetry measures.

Requires:

- At least 2 usable straight-walking bouts
- At least 6 valid steps
- At least 4 valid strides

Primary outputs:

```text
project_reference_gait_refined/
├── reference_metric_eligibility.csv
├── paired_reference_eligibility.csv
├── longitudinal_reference_changes_core.csv
├── longitudinal_reference_changes_strict.csv
├── qc_sensitivity_summary.csv
└── refined_reference_summary.txt
```

---

## 04 — Analyze Longitudinal Reference Changes

### `04_analyze_longitudinal_reference.py`

Characterizes session-2 minus session-1 changes in the reference gait measures before any IMU sensor-ablation modeling.

Analyses include:

- Baseline and follow-up descriptive statistics
- Mean and median longitudinal change
- Bootstrap 95% confidence intervals
- Paired standardized effect size
- Paired t-tests
- Wilcoxon signed-rank tests
- Benjamini-Hochberg FDR correction
- Direction-of-change consistency
- Paired-change visualizations

Primary outputs:

```text
project_reference_analysis/
├── core_longitudinal_stats.csv
├── strict_longitudinal_stats.csv
├── longitudinal_metric_ranking.csv
├── longitudinal_analysis_summary.txt
└── figures/
```

The final frozen primary targets are:

```text
step_length_m
cadence_steps_min
swing_time_s
```

---

## 05 — Audit Clinical Metadata

### `05_audit_clinical_metadata.py`

Searches downloaded metadata files for clinically relevant variables.

Examples include:

- MDS-UPDRS
- Hoehn & Yahr
- Medication state
- DBS status
- Disease duration
- Age
- Sex
- Visit/session timing

This stage was used to determine what clinical data were available before making any clinical-progression claims.

Primary outputs are written to:

```text
project_clinical_audit/
```

including metadata inventories, candidate rankings, previews, and a clinical audit summary.

---

## 06 — Build Baseline Clinical Linkage

### `06_build_baseline_clinical_linkage.py`

Links the Version 1 PD demographic/clinical table to the Version 2 longitudinal cohort.

Important implementation detail:

```python
pd.read_csv(..., header=1)
```

The clinical table uses a two-row header, with the second row containing the actual column names.

This script also:

- Normalizes participant IDs
- Audits clinical-field completeness
- Calculates MDS-UPDRS Part III totals when item-level data are complete
- Preserves medication-state information
- Links baseline clinical variables to CORE and STRICT gait cohorts

Primary outputs:

```text
project_clinical_linkage/
├── baseline_clinical_linkage.csv
├── clinical_completeness.csv
├── mds_updrs_part3_item_qc.csv
└── clinical_linkage_summary.txt
```

Only baseline clinical information was available for this analysis. No follow-up clinical table was used to calculate longitudinal MDS-UPDRS change.

---

## 07 — Extract IMU Features

### `07_extract_imu_features.py`

Extracts orientation-robust IMU features from straight-walking bouts.

Body sensors use:

- Free-acceleration magnitude
- Gyroscope magnitude

The two insole IMUs are treated as exploratory because their channel structure differs from the body-mounted IMUs.

Features per sensor include:

```text
Acceleration
- RMS
- SD
- IQR
- 95th percentile
- jerk RMS
- dominant gait-band frequency
- spectral entropy
- periodicity

Gyroscope
- RMS
- SD
- IQR
- 95th percentile
- jerk RMS
- dominant gait-band frequency
- spectral entropy
- periodicity
```

Processing principles:

- Straight walking only
- Continuous walking bouts analyzed independently
- Session feature = median across usable bouts
- Sensor-session QC requires at least 2 usable bouts

Primary outputs:

```text
project_imu_features/
├── session_imu_features.csv
├── sensor_session_qc.csv
├── paired_imu_feature_changes.csv
├── paired_sensor_feature_availability.csv
└── imu_feature_summary.txt
```

---

## 08 — Primary Sensor-Ablation Screen

### `08_screen_sensor_configurations.py`

Performs the primary fair head-to-head sensor comparison.

The script evaluates 18 pre-specified body-IMU configurations against the three frozen gait targets.

Key safeguards:

- Same common 38-participant cohort for every configuration
- Participant-grouped repeated cross-validation
- Session 1 and session 2 from the same participant remain in the same fold
- Scaling and imputation are performed inside training folds
- Elastic Net hyperparameters are selected inside training data
- No clinical variables are used as model inputs
- Insoles are excluded from the primary hardware screen

Performance metrics include:

- Concordance correlation coefficient (CCC)
- Pearson correlation
- MAE
- RMSE
- normalized RMSE
- direction-of-change agreement

A descriptive screening score is used only to rank candidate architectures for follow-up validation.

Primary outputs:

```text
project_sensor_screen/
├── sensor_configuration_definitions.csv
├── common_cohort_subjects.csv
├── cv_session_predictions.csv
├── cv_longitudinal_predictions.csv
├── target_configuration_performance.csv
├── configuration_summary.csv
├── sensor_screen_ranking.csv
└── sensor_screen_summary.txt
```

---

## 09 — Validate Shortlisted Sensor Architectures

### `09_validate_shortlisted_sensor_architectures.py`

Re-tests shortlisted configurations using each architecture's maximum available CORE cohort.

Shortlisted architectures include:

```text
Single_L_Ankle
Single_R_Ankle
Bilateral_Ankles
Single_L_LatShank
Single_R_DorsalFoot
Single_R_Wrist
Bilateral_Wrists
LowerBack_plus_L_Ankle
```

Models:

- Elastic Net
- Random Forest

This stage evaluates:

- Robustness to model family
- Maximum available sample size
- Bootstrap uncertainty
- Hardware burden versus performance
- Pareto-efficient configurations
- Target-specific specialists

Primary outputs:

```text
project_sensor_validation/
├── shortlisted_configuration_definitions.csv
├── max_cohort_subjects.csv
├── final_cv_session_predictions.csv
├── final_longitudinal_predictions.csv
├── final_validation_performance.csv
├── bootstrap_uncertainty.csv
├── architecture_summary.csv
├── pareto_frontier.csv
├── target_specialists.csv
└── final_validation_summary.txt
```

---

## 10 — Compare Within-Person Change Models

### `10_compare_personalized_change_models.py`

Tests whether within-person change modeling improves longitudinal gait-change estimation.

Three strategies are compared:

### A. Population Absolute

Predict each gait measure separately at session 1 and session 2, then calculate:

```text
predicted change = predicted session 2 - predicted session 1
```

### B. Direct Raw Delta

Predict reference longitudinal change directly from:

```text
IMU session 2 - IMU session 1
```

### C. Direct Normalized Delta

Uses symmetric within-person normalization:

```text
2 × (session2 - session1) / (|session1| + |session2| + epsilon)
```

Models:

- Elastic Net
- Random Forest

Comparison metrics:

- CCC
- Pearson r
- normalized RMSE
- direction agreement
- participant-bootstrap paired improvement versus Population Absolute

Primary outputs:

```text
project_personalized_models/
├── personalized_cv_predictions.csv
├── personalized_model_performance.csv
├── paired_improvement_bootstrap.csv
├── modeling_strategy_summary.csv
└── personalized_model_summary.txt
```

The supported benefit of within-person normalization was selective rather than universal and was concentrated in ankle-based step-length estimation.

---

## 11 — Secondary Clinical Analysis

### `11_secondary_clinical_analysis.py`

Uses available **baseline** clinical information for secondary context.

Analyses include:

1. Baseline clinical construct validity  
2. Baseline severity versus subsequent gait change  
3. Descriptive medication/DBS/sex sensitivity analyses  

Primary outputs:

```text
project_clinical_secondary/
├── baseline_construct_validity.csv
├── baseline_severity_vs_longitudinal_change.csv
├── categorical_longitudinal_descriptives.csv
├── clinical_analysis_dataset.csv
└── clinical_secondary_summary.txt
```

Important limitation:

There is no linked Version 2 follow-up clinical table in this analysis, so this script does not calculate longitudinal MDS-UPDRS change and does not establish clinical disease progression.

---

## 12 — Compile and Freeze Final Results

### `12_compile_final_results.py`

Collects the completed outputs from the reference, sensor-ablation, validation, personalized-modeling, and clinical analyses into a manuscript-ready final-results package.

This script does **not** fit new models or perform new hypothesis tests.

Primary outputs:

```text
project_final_results/
├── analysis_freeze_summary.txt
├── table_1_reference_targets.csv
├── table_2_primary_sensor_screen.csv
├── table_3_validation_architectures.csv
├── table_4_personalization_supported_effects.csv
├── table_5_clinical_secondary.csv
├── figure_1_reference_target_changes.png
├── figure_2_primary_sensor_screen.png
├── figure_3_validation_pareto.png
├── figure_4_personalization_step_length.png
└── figure_5_clinical_correlations.png
```

This folder represents the frozen analysis used for manuscript interpretation.

---

# Recommended Execution Order

Run scripts from the project root or from the `scripts` directory in numerical order:

```powershell
python scripts\01_audit_weargait_longitudinal.py
python scripts\02_build_paired_longitudinal_cohort_v2.py
python scripts\03_extract_reference_gait_metrics.py
python scripts\03b_refine_reference_qc.py
python scripts\04_analyze_longitudinal_reference.py
python scripts\05_audit_clinical_metadata.py
python scripts\06_build_baseline_clinical_linkage.py
python scripts\07_extract_imu_features.py
python scripts\08_screen_sensor_configurations.py
python scripts\09_validate_shortlisted_sensor_architectures.py
python scripts\10_compare_personalized_change_models.py
python scripts\11_secondary_clinical_analysis.py
python scripts\12_compile_final_results.py
```

The core engineering pipeline is:

```text
01 → 02 → 03 → 03b → 04 → 07 → 08 → 09 → 10 → 12
```

Clinical context is added through:

```text
05 → 06 → 11 → 12
```

---

# Python Dependencies

The scripts use standard scientific Python packages:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
```

Install with:

```powershell
python -m pip install numpy pandas scipy scikit-learn matplotlib
```

A virtual environment is recommended.

Example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn matplotlib
```

---

# Key Engineering Findings

The final analysis supports several design-level conclusions:

- Single-sensor distal-leg architectures remained competitive with multi-sensor configurations.
- Adding more sensors did not consistently improve longitudinal performance.
- Exact optimal placement was model-dependent rather than universal.
- Cadence/rhythmic change was the most consistently preserved target.
- Step-length/spatial change was more difficult to estimate.
- Within-person normalization showed selective benefit for ankle-based step-length change rather than universal improvement.
- Pareto analysis favored minimal architectures that balanced predictive performance and hardware burden.

The central design principle is:

> **Maximize longitudinal information per sensor.**

---

# Reproducibility Notes

- The main sensor comparison uses participant-level grouping to prevent leakage between repeated sessions.
- Session 1 and session 2 from the same participant remain in the same cross-validation fold.
- Clinical variables are excluded from the primary engineering models.
- The primary hardware screen excludes insole sensors.
- Reduced-sensor architectures are compared against synchronized instrumented-walkway reference measures.
- The common-cohort screen and maximum-cohort validation are intentionally separate analyses.
- Screening scores are descriptive ranking tools, not clinical endpoints.

---

# Interpretation Guardrails

Please use the following language when interpreting this repository:

### Supported

- longitudinal gait change
- longitudinal gait-change preservation
- minimal wearable architecture
- sensor-ablation analysis
- within-person change modeling
- performance–hardware tradeoff

### Not supported by this analysis

- prediction of Parkinson's disease progression
- prediction of UCL/clinical injury
- proof that one left/right sensor location is universally optimal
- direct longitudinal MDS-UPDRS progression
- clinical diagnostic claims

---

# Data Availability

This repository contains analysis code only.

WearGait-PD data should be obtained from the official dataset source. Raw participant data should not be committed to the public GitHub repository.

Recommended `.gitignore` entries include:

```gitignore
PD Participants/
Clinical_Metadata/
project_*/
*.csv
*.tsv
*.mat
__pycache__/
*.pyc
.venv/
```

If selected result tables or figures are intended for public release, place them in a dedicated `results/` or `figures/` directory and remove those paths from the ignore rules.

---

# Author

**Michelle Liete Di Cicco**  
Department of Biomedical Engineering  
University of North Dakota

Research interests: wearable sensing, biomechanics, digital health, medical devices, and machine learning.

---

## Project Title

**Minimal-Sensor Parkinson's Gait Monitoring**

Long-form manuscript title:

**Minimal-Sensor Wearable Architecture for Longitudinal Gait Monitoring in Parkinson's Disease: Sensor Ablation and Within-Person Change Modeling**
