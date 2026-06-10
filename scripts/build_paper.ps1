$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -Path $repo

if (-not (Test-Path -LiteralPath "results\full\generated_results.tex")) {
  python -m experiments.run_benchmark --preset full
}

Copy-Item -LiteralPath "results\full\generated_results.tex" -Destination "paper\generated_results.tex" -Force

Push-Location -Path "paper"
$latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
$perl = Get-Command perl -ErrorAction SilentlyContinue
if ($latexmk -and $perl) {
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
}
if (-not $latexmk -or -not $perl -or $LASTEXITCODE -ne 0) {
  Write-Host "latexmk unavailable or failed; falling back to direct pdflatex/bibtex passes."
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
}
Pop-Location

New-Item -ItemType Directory -Force -Path "paper\final" | Out-Null
Copy-Item -LiteralPath "paper\main.pdf" -Destination "paper\final\iclr_submission.pdf" -Force
Copy-Item -LiteralPath "paper\final\iclr_submission.pdf" -Destination "$HOME\Downloads\iclr_submission_bayesian_ensemble_world_models.pdf" -Force
Write-Host "paper\final\iclr_submission.pdf"
Write-Host "$HOME\Downloads\iclr_submission_bayesian_ensemble_world_models.pdf"
