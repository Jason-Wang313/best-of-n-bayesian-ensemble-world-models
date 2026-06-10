$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -Path $repo
python -m experiments.run_benchmark --preset smoke
