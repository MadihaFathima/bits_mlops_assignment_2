# Builds the final submission zip: source code, configs, and model artifacts,
# excluding venv/, the raw/processed datasets, DVC's local cache, and .git.
# Run from the project root: .\scripts\build_submission_zip.ps1

$ErrorActionPreference = "Stop"
$root = (Get-Location).Path
$staging = Join-Path $env:TEMP "bits_mlops_assignment_2_staging"
$zipPath = Join-Path (Split-Path $root -Parent) "bits_mlops_assignment_2_submission.zip"

if (Test-Path $staging) { Remove-Item -Recurse -Force $staging }
New-Item -ItemType Directory -Force -Path $staging | Out-Null

# Top-level files
Copy-Item README.md, RUNBOOK.md, requirements.txt, .gitignore, .dockerignore, .dvcignore, mlflow.db -Destination $staging
Copy-Item "2024ad05394_Assignment_2_Report.pdf" -Destination $staging

# Folders to copy wholesale
foreach ($dir in @("src", "tests", "docker", "k8s", "monitoring", ".github", "models", "outputs")) {
    robocopy $dir (Join-Path $staging $dir) /E /NFL /NDL /NJH /NJS | Out-Null
}

# scripts/: excludes screenshots/ (report evidence images, not source code)
robocopy scripts (Join-Path $staging "scripts") /E /XD screenshots /NFL /NDL /NJH /NJS | Out-Null

# mlruns/: only the successful training run's own tracked artifacts (confusion matrix,
# loss curve, model_metadata.json, model file). Excludes mlruns/1/models/ (a ~12MB
# duplicate of the model already shipped at models/cnn_baseline.pt) and the two failed
# debug runs from early MLflow troubleshooting (see RUNBOOK) -- neither adds evidence,
# both just bulk up the zip.
$successfulRun = "1\9a622c5b19bb4e8dab8280c7fb374abf"
robocopy "mlruns\$successfulRun" (Join-Path $staging "mlruns\$successfulRun") /E /NFL /NDL /NJH /NJS | Out-Null

# .dvc config only, not cache/tmp
robocopy .dvc (Join-Path $staging ".dvc") /E /XD cache tmp /NFL /NDL /NJH /NJS | Out-Null

# data/: DVC pointer files + tiny license/readme, not the actual images
New-Item -ItemType Directory -Force -Path (Join-Path $staging "data\raw") | Out-Null
Copy-Item data\.gitignore -Destination (Join-Path $staging "data")
Copy-Item data\processed.dvc -Destination (Join-Path $staging "data")
Copy-Item data\raw\.gitignore -Destination (Join-Path $staging "data\raw")
Copy-Item -LiteralPath "data\raw\MSR-LA - 3467.docx" -Destination (Join-Path $staging "data\raw")
Copy-Item -LiteralPath "data\raw\readme[1].txt" -Destination (Join-Path $staging "data\raw")

if (Test-Path $zipPath) { Remove-Item -Force $zipPath }
Compress-Archive -Path "$staging\*" -DestinationPath $zipPath

Remove-Item -Recurse -Force $staging

Write-Host "Built: $zipPath"
Write-Host "Size: $([math]::Round((Get-Item $zipPath).Length / 1MB, 1)) MB"
