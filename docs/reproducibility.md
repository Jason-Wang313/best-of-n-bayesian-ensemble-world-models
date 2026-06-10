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
powershell -ExecutionPolicy Bypass -File scripts\build_paper.ps1
```

## Determinism

The synthetic benchmark uses fixed integer seeds. Each problem samples a shared candidate set up to the maximum budget and evaluates lower budgets by prefixing the same set. This avoids comparing different candidate distributions across N.

## No Test-Candidate Leakage

Only the `oracle` selector uses true returns. The calibrated pessimistic selector fits a scalar residual-to-std multiplier on separate calibration candidates and then uses only ensemble predictions on test candidates.
