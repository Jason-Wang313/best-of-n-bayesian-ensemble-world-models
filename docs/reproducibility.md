# Reproducibility Notes

## Environment

- Python 3.10 or newer
- `numpy`
- `matplotlib`
- `pytest`
- Optional for paper build: MiKTeX or another LaTeX distribution with `pdflatex`, `bibtex`, and preferably `latexmk`

## Commands

```powershell
python -m pip install -e .
pytest -q
python -m experiments.run_benchmark --preset smoke
python -m experiments.run_benchmark --preset full
python -m experiments.run_v3_evidence
powershell -ExecutionPolicy Bypass -File scripts\build_paper.ps1
```

## Determinism

The synthetic benchmark uses fixed integer seeds. Each problem samples a shared candidate set up to the maximum budget and evaluates lower budgets by prefixing the same set. This avoids comparing different candidate distributions across N.

## No Test-Candidate Leakage

Only the `oracle` selector uses true returns. The calibrated pessimistic selector fits a scalar residual-to-std multiplier on separate calibration candidates and then uses only ensemble predictions on test candidates.

## Expanded Evidence Suite

`experiments.run_v3_evidence` is designed to stay CPU/RAM bounded. It sets
thread counts to one, uses fixed seeds, evaluates 9 posterior stress conditions,
adds a lightweight bootstrap ridge return-ensemble suite under evaluation shift,
and writes paired-effect, calibration-coverage, manifest, figure, and
claim-evidence artifacts under `results/v3` and `figures/v3`.
