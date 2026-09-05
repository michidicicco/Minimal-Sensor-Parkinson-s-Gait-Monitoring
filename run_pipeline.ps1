# WearGait-PD analysis pipeline
# Run from the repository/project root in PowerShell.
# Stops immediately if any script fails.

$ErrorActionPreference = "Stop"

$scripts = @(
    "01_audit_weargait_longitudinal.py",
    "02_build_paired_longitudinal_cohort_v2.py",
    "03_extract_reference_gait_metrics.py",
    "03b_refine_reference_qc.py",
    "04_analyze_longitudinal_reference.py",
    "05_audit_clinical_metadata.py",
    "06_build_baseline_clinical_linkage.py",
    "07_extract_imu_features.py",
    "08_screen_sensor_configurations.py",
    "09_validate_shortlisted_sensor_architectures.py",
    "10_compare_personalized_change_models.py",
    "11_secondary_clinical_analysis.py",
    "12_compile_final_results.py"
)

Write-Host ""
Write-Host "WearGait-PD Minimal-Sensor Pipeline" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

foreach ($script in $scripts) {
    $path = Join-Path "scripts" $script

    if (-not (Test-Path $path)) {
        throw "Missing script: $path"
    }

    Write-Host "Running $script ..." -ForegroundColor Yellow
    python $path

    if ($LASTEXITCODE -ne 0) {
        throw "$script failed with exit code $LASTEXITCODE"
    }

    Write-Host "Completed $script" -ForegroundColor Green
    Write-Host ""
}

Write-Host "All WearGait-PD analyses completed successfully." -ForegroundColor Green
Write-Host "Final compiled outputs should be in: project_final_results\" -ForegroundColor Cyan
