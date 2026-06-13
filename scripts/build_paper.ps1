param(
  [switch]$FinalDesktop
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -Path $repo

if (-not (Test-Path -LiteralPath "results\full\generated_results.tex")) {
  python -m experiments.run_benchmark --preset full
}
if (-not (Test-Path -LiteralPath "results\v3\generated_results.tex")) {
  python -m experiments.run_v3_evidence
}

Copy-Item -LiteralPath "results\full\generated_results.tex" -Destination "paper\generated_results.tex" -Force
Copy-Item -LiteralPath "results\v3\generated_results.tex" -Destination "paper\generated_v3_results.tex" -Force

$desktop = Join-Path $HOME "OneDrive\Desktop"
if (-not (Test-Path -LiteralPath $desktop)) {
  $desktop = Join-Path $HOME "Desktop"
}
$desktopPdf = Join-Path $desktop "best of n bayesian ensemble world models-v3.pdf"

Push-Location -Path "paper"
$latexArtifacts = @("main.aux", "main.bbl", "main.blg", "main.log", "main.out", "main.pdf")
foreach ($artifact in $latexArtifacts) {
  if (Test-Path -LiteralPath $artifact) {
    Remove-Item -LiteralPath $artifact -Force
  }
}
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
$finalPdf = "paper\final\best of n bayesian ensemble world models-v3.pdf"
Copy-Item -LiteralPath "paper\main.pdf" -Destination $finalPdf -Force
Write-Host $finalPdf
if ($FinalDesktop) {
  Copy-Item -LiteralPath $finalPdf -Destination $desktopPdf -Force
  Write-Host $desktopPdf
}
