# Minimal-Sensor Parkinson's Gait Monitoring

**Longitudinal wearable biomechanics, sensor ablation, and within-person change modeling in Parkinson's disease**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Wearable inertial measurement units (IMUs) can provide objective gait measurements in Parkinson's disease (PD), but large multi-sensor systems increase hardware burden and may be less practical for repeated longitudinal monitoring.

This project evaluates whether **reduced wearable sensor configurations can preserve meaningful within-person gait-change information over time**.

Using longitudinal **WearGait-PD** data, the analysis compares 18 pre-specified wearable sensor architectures against synchronized instrumented-walkway reference measures. The pipeline includes reference gait extraction, quality control, IMU feature engineering, sensor ablation, participant-grouped machine learning, model sensitivity analysis, Pareto evaluation, within-person normalization, and secondary clinical analyses.

The central engineering question is:

> **What is the minimum wearable IMU architecture that preserves longitudinal gait-change information in Parkinson's disease?**

The design principle that emerged from the final analysis is:

> **Maximize longitudinal information per sensor.**

---

## Project Highlights

- Longitudinal repeated walking assessments
- 42 participants with valid paired reference gait data
- 38 participants in the common cohort used for the primary fair sensor comparison
- 11 body-mounted IMU locations
- 2 exploratory insole inertial units
- 18 wearable sensor configurations evaluated
- Instrumented pressure walkway used as the reference system
- Three frozen longitudinal gait targets:
  - **Step length** — spatial domain
  - **Cadence** — rhythmic domain
  - **Swing time** — gait-cycle phase
- Elastic Net as the primary linear model
- Random Forest as a nonlinear sensitivity model
- Participant-grouped repeated cross-validation
- Bootstrap uncertainty analysis
- Pareto analysis of performance versus sensor burden
- Within-person normalized change modeling
- Secondary baseline clinical analyses

---

## Main Findings

The final analysis supports several engineering-level conclusions:

- **Single-sensor distal-leg architectures remained competitive with multi-sensor systems.**
- Adding more sensors did **not** consistently improve longitudinal gait-change performance.
- Exact optimal placement was **model-dependent** rather than universal.
- **Cadence** was the most consistently preserved longitudinal gait domain.
- **Step-length change** was more difficult to estimate.
- Within-person normalization was **not universally superior**, but showed selective benefit for **ankle-based step-length estimation**.
- Pareto analysis favored compact sensor architectures that balanced predictive performance with hardware burden.

These results support an outcome-specific approach to wearable design rather than assuming that more sensors automatically produce better longitudinal monitoring.

---

## Important Interpretation Guardrail

This repository evaluates **preservation of longitudinal gait change**.

It does **not** establish that the observed gait changes represent clinical Parkinson's disease progression.

The available clinical linkage used baseline clinical information only. A linked Version 2 follow-up clinical table was not available for calculation of longitudinal MDS-UPDRS change.

Supported language includes:

- longitudinal gait change
- longitudinal gait-change preservation
- minimal wearable architecture
- sensor-ablation analysis
- within-person change modeling
- hardware-performance tradeoff

This analysis does **not** support claims of:

- direct prediction of Parkinson's disease progression
- longitudinal MDS-UPDRS progression
- universal superiority of one left/right sensor location
- clinical diagnostic performance

---

# Repository Structure

```text
Minimal-Sensor-Parkinson-s-Gait-Monitoring/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── CITATION.cff
├── run_pipeline.ps1
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
├── results/              # Optional public aggregate outputs
└── figures/              # Optional public figures
```

The raw WearGait-PD dataset is **not included** in this repository.

---

# Data

## WearGait-PD

The analysis uses longitudinal WearGait-PD data containing synchronized wearable-sensor and instrumented-walkway recordings.

The synchronized system includes:

### Body-mounted IMUs

Eleven body-mounted IMU locations:

- Forehead
- Xiphoid
- Lower back
- Left wrist
- Right wrist
- Left lateral shank
- Right lateral shank
- Left dorsal foot
- Right dorsal foot
- Left ankle
- Right ankle

### Exploratory insole inertial units

- Left insole
- Right insole

The insole inertial units are retained for exploratory feature extraction but are excluded from the primary body-sensor hardware screen.

### Reference system

An instrumented pressure walkway provides synchronized gait-reference measurements used to derive temporal and spatial gait outcomes.

### Primary task

The primary analysis uses:

```text
selfpace_mat
```

Only straight-walking segments are included:

```text
GeneralEvent == "Walk"
```

Turning trials are not part of the primary sensor-ablation analysis.

---

# Analysis Pipeline

The full workflow is:

```text
Dataset audit
    ↓
Paired longitudinal cohort
    ↓
Reference gait extraction
    ↓
Reference QC refinement
    ↓
Longitudinal reference analysis
    ↓
IMU feature extraction
    ↓
Primary sensor-ablation screen
    ↓
Shortlisted architecture validation
    ↓
Within-person change modeling
    ↓
Secondary clinical analysis
    ↓
Final analysis freeze
```

Clinical metadata auditing and linkage are performed in parallel before the secondary clinical analyses.

---

# Scripts

## 01 — Audit the Longitudinal Dataset

### `01_audit_weargait_longitudinal.py`

Performs the initial dataset audit.

Main tasks:

- inventories downloaded files
- detects participant, session, and task structure
- inventories CSV columns
- summarizes participant/session coverage
- identifies candidate clinical or metadata files

Primary output directory:

```text
project_audit/
```

Typical outputs include:

```text
file_inventory.csv
csv_inventory.csv
participant_session_summary.csv
participant_task_summary.csv
task_summary.csv
column_inventory.csv
candidate_clinical_files.csv
audit_summary.txt
```

---

## 02 — Build the Corrected Paired Longitudinal Cohort

### `02_build_paired_longitudinal_cohort_v2.py`

Builds the corrected Version 2 cohort used by the downstream analysis.

Key features:

- recognizes both `NLS###` and `WPD###` participant IDs
- uses `selfpace_mat` as the primary synchronized walking file
- evaluates reference-data usability
- tracks paired sensor availability separately for each location
- does not globally remove a participant because one non-required sensor is missing

Primary output directory:

```text
project_cohort_v2/
```

Outputs include:

```text
session_file_map.csv
selfpace_mat_session_qc.csv
paired_longitudinal_cohort.csv
paired_sensor_availability.csv
sensor_availability_summary.csv
cohort_summary.txt
```

---

## 03 — Extract Reference Gait Metrics

### `03_extract_reference_gait_metrics.py`

Derives instrumented-walkway spatiotemporal gait measures from synchronized walking trials.

Metrics include:

- gait speed
- cadence
- step time
- stride time
- stance time
- swing time
- double-support percentage
- step length
- stride length
- variability metrics
- bilateral asymmetry metrics

Reference construction uses:

- pressure-walkway left/right foot contacts
- longitudinal footfall position
- straight-walking segments only
- exclusion of annotation-boundary partial contacts

Primary output directory:

```text
project_reference_gait/
```

Outputs include:

```text
session_reference_gait_metrics.csv
reference_metric_qc.csv
longitudinal_reference_changes.csv
reference_summary.txt
```

---

## 03b — Refine Reference Quality Control

### `03b_refine_reference_qc.py`

Creates two pre-specified QC tiers so simple mean gait measures are not subjected to unnecessarily strict variability requirements.

### CORE QC

Used for mean spatiotemporal gait measures.

Requires:

- at least 2 usable straight-walking bouts
- at least 5 valid steps
- at least 3 valid strides

### STRICT QC

Used for variability and asymmetry measures.

Requires:

- at least 2 usable straight-walking bouts
- at least 6 valid steps
- at least 4 valid strides

Primary output directory:

```text
project_reference_gait_refined/
```

Outputs include:

```text
reference_metric_eligibility.csv
paired_reference_eligibility.csv
longitudinal_reference_changes_core.csv
longitudinal_reference_changes_strict.csv
qc_sensitivity_summary.csv
refined_reference_summary.txt
```

---

## 04 — Analyze Longitudinal Reference Changes

### `04_analyze_longitudinal_reference.py`

Characterizes session-2 minus session-1 changes before any wearable-sensor modeling.

Analyses include:

- baseline and follow-up descriptive statistics
- mean and median change
- bootstrap 95% confidence intervals
- paired standardized effect size
- paired t-test
- Wilcoxon signed-rank test
- Benjamini-Hochberg FDR correction
- direction-of-change consistency
- paired-change plots

Primary output directory:

```text
project_reference_analysis/
```

Outputs include:

```text
core_longitudinal_stats.csv
strict_longitudinal_stats.csv
longitudinal_metric_ranking.csv
longitudinal_analysis_summary.txt
figures/
```

The frozen primary targets used downstream are:

```text
step_length_m
cadence_steps_min
swing_time_s
```

---

## 05 — Audit Clinical Metadata

### `05_audit_clinical_metadata.py`

Searches available non-task metadata files for clinically relevant variables.

Examples include:

- MDS-UPDRS
- Hoehn & Yahr
- medication state
- medication timing
- DBS status
- disease duration
- age
- sex
- visit/session timing

Primary output directory:

```text
project_clinical_audit/
```

This stage is used to determine what clinical variables can be linked without making unsupported longitudinal progression claims.

---

## 06 — Build Baseline Clinical Linkage

### `06_build_baseline_clinical_linkage.py`

Links the Version 1 PD demographic/clinical table with the Version 2 longitudinal participant IDs.

The source clinical CSV uses a two-row header, so the script reads it using:

```python
pd.read_csv(..., header=1)
```

This script:

- normalizes participant IDs
- links baseline clinical variables
- audits clinical-field completeness
- calculates MDS-UPDRS Part III totals when item-level data are complete
- preserves medication-state information
- flags CORE and STRICT cohort membership

Primary output directory:

```text
project_clinical_linkage/
```

Outputs include:

```text
baseline_clinical_linkage.csv
clinical_completeness.csv
mds_updrs_part3_item_qc.csv
clinical_linkage_summary.txt
```

---

## 07 — Extract IMU Features

### `07_extract_imu_features.py`

Extracts device-oriented features from each wearable sensor during straight walking.

Body-mounted IMUs use:

- free-acceleration magnitude
- gyroscope magnitude

Features are designed to be relatively orientation-robust.

### Acceleration features

- RMS
- standard deviation
- IQR
- 95th percentile
- jerk RMS
- dominant gait-band frequency
- spectral entropy
- periodicity

### Gyroscope features

- RMS
- standard deviation
- IQR
- 95th percentile
- jerk RMS
- dominant gait-band frequency
- spectral entropy
- periodicity

Processing rules:

- straight walking only
- continuous walking bouts processed independently
- session value = median across usable bouts
- sensor-session QC requires at least 2 usable bouts

Primary output directory:

```text
project_imu_features/
```

Outputs include:

```text
session_imu_features.csv
sensor_session_qc.csv
paired_imu_feature_changes.csv
paired_sensor_feature_availability.csv
imu_feature_summary.txt
```

---

## 08 — Primary Sensor-Ablation Screen

### `08_screen_sensor_configurations.py`

Performs the primary fair head-to-head sensor-ablation analysis.

The three frozen reference targets are:

```text
step_length_m
cadence_steps_min
swing_time_s
```

### Primary design safeguards

- paired CORE reference cohort only
- common body-sensor cohort for fair comparison
- same participants used for every screened architecture
- participant-grouped repeated cross-validation
- both sessions from the same participant remain in the same fold
- preprocessing is fit within training folds
- Elastic Net tuning is performed within training data
- clinical variables are excluded
- insole sensors are excluded from the primary hardware comparison

### Sensor architectures

The 18 screened configurations include:

#### Eleven single-sensor configurations

1. Forehead
2. Xiphoid
3. Lower back
4. Left wrist
5. Right wrist
6. Left lateral shank
7. Right lateral shank
8. Left dorsal foot
9. Right dorsal foot
10. Left ankle
11. Right ankle

#### Seven multi-sensor configurations

12. Bilateral wrists  
13. Bilateral lateral shanks  
14. Bilateral dorsal feet  
15. Bilateral ankles  
16. Lower back + left ankle  
17. Lower back + right ankle  
18. Lower back + bilateral ankles  

### Performance metrics

- concordance correlation coefficient (CCC)
- Pearson correlation
- MAE
- RMSE
- normalized RMSE
- direction-of-change agreement

A descriptive screening score is used only to rank configurations for validation.

Primary output directory:

```text
project_sensor_screen/
```

Outputs include:

```text
sensor_configuration_definitions.csv
common_cohort_subjects.csv
cv_session_predictions.csv
cv_longitudinal_predictions.csv
target_configuration_performance.csv
configuration_summary.csv
sensor_screen_ranking.csv
sensor_screen_summary.txt
```

---

## 09 — Validate Shortlisted Sensor Architectures

### `09_validate_shortlisted_sensor_architectures.py`

Re-tests shortlisted configurations using each architecture's maximum available CORE cohort.

Shortlisted configurations:

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

- maximum available participant sample
- model-family robustness
- bootstrap uncertainty
- target-specific performance
- sensor burden
- Pareto efficiency

Primary output directory:

```text
project_sensor_validation/
```

Outputs include:

```text
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
```

---

## 10 — Compare Within-Person Change Models

### `10_compare_personalized_change_models.py`

Tests whether direct within-person change modeling improves longitudinal estimation compared with population absolute-session modeling.

Three modeling strategies are compared.

### A. Population Absolute

Predict session 1 and session 2 separately:

```text
predicted change = predicted session 2 - predicted session 1
```

### B. Direct Raw Delta

Predict reference change directly from:

```text
IMU session 2 - IMU session 1
```

### C. Direct Normalized Delta

Use symmetric within-person normalized feature change:

```text
2 × (session2 - session1) /
(|session1| + |session2| + epsilon)
```

Models:

- Elastic Net
- Random Forest

Evaluation includes:

- CCC
- Pearson r
- normalized RMSE
- direction agreement
- participant-bootstrap paired improvement relative to Population Absolute

Primary output directory:

```text
project_personalized_models/
```

Outputs include:

```text
personalized_cv_predictions.csv
personalized_model_performance.csv
paired_improvement_bootstrap.csv
modeling_strategy_summary.csv
personalized_model_summary.txt
```

The final analysis found that personalization was **selectively useful**, with supported benefits concentrated in ankle-based step-length estimation.

---

## 11 — Secondary Clinical Analysis

### `11_secondary_clinical_analysis.py`

Uses available baseline clinical variables for secondary interpretation.

Questions include:

1. Are baseline gait measures associated with baseline motor severity?
2. Is baseline severity associated with subsequent gait change?
3. Do longitudinal gait changes differ descriptively by baseline medication, DBS, or sex groups?

Primary output directory:

```text
project_clinical_secondary/
```

Outputs include:

```text
baseline_construct_validity.csv
baseline_severity_vs_longitudinal_change.csv
categorical_longitudinal_descriptives.csv
clinical_analysis_dataset.csv
clinical_secondary_summary.txt
```

These analyses are secondary and do not alter the primary sensor-only architecture conclusion.

---

## 12 — Compile and Freeze Final Results

### `12_compile_final_results.py`

Compiles completed outputs into a manuscript-ready final-results package.

This script:

- does not fit new models
- does not run new hypothesis tests
- collects already completed analyses
- creates final summary tables
- creates publication-oriented figures
- writes the final analysis-freeze summary

Primary output directory:

```text
project_final_results/
```

Outputs include:

```text
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
```

---

# Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/michidicco/Minimal-Sensor-Parkinson-s-Gait-Monitoring.git
cd Minimal-Sensor-Parkinson-s-Gait-Monitoring
```

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Core dependencies are:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
```

## 4. Add the WearGait-PD data locally

The expected local project structure includes:

```text
PD Participants/
Clinical_Metadata/
```

These directories are ignored by Git and should **not** be committed.

## 5. Run the pipeline

On Windows PowerShell:

```powershell
.\run_pipeline.ps1
```

The runner executes the analysis scripts in numerical order and stops if a stage fails.

---

# Manual Execution

Scripts can also be run individually:

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

The core engineering sequence is:

```text
01 → 02 → 03 → 03b → 04 → 07 → 08 → 09 → 10 → 12
```

The clinical-context branch is:

```text
05 → 06 → 11 → 12
```

---

# Reproducibility

Several safeguards are built into the analysis:

- repeated sessions from the same participant are kept in the same CV fold
- the primary architecture screen uses the same participants for all candidate configurations
- feature preprocessing occurs within training folds
- clinical variables are excluded from primary wearable models
- the common-cohort screen and maximum-cohort validation are kept separate
- Random Forest is used as a pre-specified nonlinear sensitivity analysis
- participant bootstrap is used for uncertainty estimation
- the screening score is descriptive and is not treated as a clinical endpoint
- the final compilation script does not introduce new exploratory tests

---

# `.gitignore`

The repository is configured to ignore:

- raw WearGait-PD participant data
- clinical metadata
- generated analysis folders
- virtual environments
- Python cache files
- credentials and environment files

Do not commit:

```text
PD Participants/
Clinical_Metadata/
.env
.synapseConfig
access tokens
personal access tokens
participant-level raw data
```

---

# Limitations

This study should be interpreted within several constraints:

- modest longitudinal sample size
- controlled walking task rather than free-living monitoring
- heterogeneous baseline medication/treatment state
- variable paired sensor availability
- no linked follow-up clinical table for longitudinal MDS-UPDRS change
- no external validation cohort
- orientation-robust magnitude features trade some directional information for deployment robustness
- exact optimal left/right placement may depend on participant asymmetry and model choice

---

# Project Title

Short project title:

**Minimal-Sensor Parkinson's Gait Monitoring**

Long-form research title:

**Minimal-Sensor Wearable Architecture for Longitudinal Gait Monitoring in Parkinson's Disease: Sensor Ablation and Within-Person Change Modeling**

---

# Author

**Michelle Liete Di Cicco**  
Department of Biomedical Engineering  
University of North Dakota

ORCID: [0009-0001-6618-0783](https://orcid.org/0009-0001-6618-0783)

GitHub: [michidicicco](https://github.com/michidicicco)

Repository:

[Minimal-Sensor-Parkinson-s-Gait-Monitoring](https://github.com/michidicco/Minimal-Sensor-Parkinson-s-Gait-Monitoring)

---

# Citation

If you use or build on this code, please cite the repository using the metadata in:

```text
CITATION.cff
```


# License

This repository is released under the **MIT License**.

See:

```text
LICENSE
```

for the full license text.

---

# Acknowledgment

This project analyzes the WearGait-PD dataset. The dataset authors and original data contributors should be cited according to the official WearGait-PD data-use and citation guidance.

Raw WearGait-PD data are not redistributed in this repository.
