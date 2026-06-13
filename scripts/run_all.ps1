$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -Path $repo
python -m experiments.run_benchmark --preset full
python -m experiments.run_v3_evidence
