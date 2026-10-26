# KFM prediction — delivery

Predicting knee flexion moment (KFM) from gait parameters, trained on
`kfm_steps.csv` (1664 QC-clean steps from 22 subjects).

Two targets, both normalised by body mass:

| target | unit |
|---|---|
| `peak_kfm_Nm_per_kg` | N·m/kg — peak KFM during stance |
| `kfm_impulse_Nms_per_kg` | N·m·s/kg — KFM integrated over stance |

## Headline results

Leave-one-subject-out (LOSO) cross-validation — every subject is held out
entirely, so these are numbers for a person the model has never seen.

| target | no calibration | + 2 calibration steps | noise ceiling |
|---|---|---|---|
| peak KFM | R² = 0.334, MAE = 0.300 | **R² = 0.677**, MAE = 0.205 | 0.799 |
| KFM impulse | R² = 0.389, MAE = 0.119 | **R² = 0.671**, MAE = 0.089 | 0.750 |

The winner for both targets is **Ridge regression on quadratic features**
(`alpha = 10`). Every nonlinear model tested — SVR, HistGradientBoosting,
random forest, kNN, MLP, GPR — scored lower. See `results/tuning2.csv` for
all 22 configurations.

Two limits worth knowing before using this:

- **The ceiling is R² ≈ 0.8, not 1.0.** Even knowing the subject and the
  slope exactly only reaches 0.799 / 0.750; the remaining 20–25% is genuine
  step-to-step variability.
- **Individual offsets are not predictable from body mass** (LOSO R² −0.06 /
  −0.16). Without calibration steps, expect population-average accuracy only.

That is why calibration matters more than model choice: two measured steps
take R² from 0.33 to 0.68.

## Layout

```
kfm_delivery/
├── kfm_steps.csv              dataset (copy; original is one level up)
├── train_final.py             evaluate, refit, export models + report
├── tune_kfm2.py               hyperparameter sweep -> results/tuning2.csv
├── predict_calibrated.py      command-line inference, optional calibration
├── kfm_model/
│   ├── peak_kfm_final.joblib        trained winners (drop-in for predict_kfm.py)
│   ├── kfm_impulse_final.joblib
│   └── *_final.json                 params, valid ranges, metrics
├── results/
│   ├── report.md                    evaluation tables
│   ├── final_metrics.csv            same numbers, machine-readable
│   ├── tuning2.csv                  full hyperparameter sweep
│   └── benchmark.csv                earlier comparison incl. feature sets
└── archive/                   superseded scripts, kept for reference
```

## Requirements

The project virtualenv is one level up and is the only interpreter with the
dependencies installed (the system `python3` has none):

```bash
cd .. && .venv/bin/python kfm_delivery/train_final.py
```

Or activate it and run from anywhere:

```bash
source ../.venv/bin/activate
python kfm_delivery/train_final.py
```

Dependencies: numpy, pandas, scikit-learn, joblib (numpy 2.5.4 / pandas 3.1.0
/ sklearn 1.9.1 as tested).

## Usage

Predict for a few gaits (population model, no calibration):

```bash
python predict_calibrated.py --slope 10 12 --speed 1.2 1.1 --step-length 0.60 0.62
```

Add `--mass 74.8` to also get absolute N·m / N·m·s columns, and `--json` for
machine-readable output.

Predict for a subject you have measured steps for:

```bash
python predict_calibrated.py --calibration subject_steps.csv \
    --slope 12 --speed 1.1 --step-length 0.62
```

`subject_steps.csv` needs `slope_deg`, `speed_mps`, `step_length_m` plus one
or both target columns. Two to five steps suffice; the mean residual on those
steps becomes the offset applied to all predictions.

Input validity ranges (outside them, predictions are extrapolation and should
not be trusted): slope 5.2–18°, speed 0.16–1.57 m/s, step length 0.21–0.79 m.
The dense region is slope 5–18°, speed 0.75–1.42 m/s, step length 0.44–0.71 m.

## Reproducing

```bash
python tune_kfm2.py    # ~35 min: the sweep (sequential; this machine has 4 cores)
python train_final.py  # ~1 min: reads the sweep, evaluates, exports
```

Notes for anyone re-running this:

- Score with **pooled** MSE across LOSO folds. Averaging per-fold R² is wrong:
  within one subject the variance is small, so a single offset error sends
  that fold to R² ≈ −10 and the average becomes meaningless.
- `GridSearchCV` / `RandomizedSearchCV` with nested joblib parallelism hangs
  or crawls on a 4-core machine. `tune_kfm2.py` is sequential by design.
- `GaussianProcessRegressor` with an optimised kernel degenerates here — it
  drives the speed and step-length length scales to the lower bound and
  predicts near the mean, and takes 30–60 min per LOSO pass. If you need GPR,
  pass `optimizer=None` with a fixed kernel.
- Drop rows where `peak_at_window_end == True` before training (29 rows;
  these trials never reached a real peak).

## Dropping into the existing model

`kfm_model/*_final.joblib` are sklearn pipelines taking exactly the three
columns in `slope_deg, speed_mps, step_length_m` order, so they replace the
old SVR/MLP files in `../kfm_model/predict_kfm.py` without code changes:

```python
peak_model = joblib.load("peak_kfm_final.joblib")
impulse_model = joblib.load("kfm_impulse_final.joblib")
```

The old scripts must be run from the folder holding the `.joblib` files, since
they load by bare filename.
