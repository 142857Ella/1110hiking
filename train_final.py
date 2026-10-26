"""
Final KFM model training, evaluation and export.

Pipeline
--------
1. Load kfm_steps.csv, drop QC-flagged rows (peak_at_window_end == True).
2. Read the best configuration per target from results/tuning.csv.
3. Evaluate with leave-one-subject-out (LOSO) CV:
      - constant baseline
      - uncalibrated model
      - model + per-subject offset calibration on k known steps
4. Refit on all subjects and export .joblib files that drop straight into
   predict_kfm.py (StandardScaler + estimator, 3 features, same order).

Run:
    .venv/bin/python train_final.py
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

TARGETS = {
    "peak_kfm_Nm_per_kg": "peak_kfm",
    "kfm_impulse_Nms_per_kg": "kfm_impulse",
}
FEATURES = ["slope_deg", "speed_mps", "step_length_m"]  # must match predict_kfm.py
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "kfm_model")
REPORT = os.path.join(HERE, "results", "report.md")
DATA = os.path.join(HERE, "kfm_steps.csv")
TUNING = os.path.join(HERE, "results", "tuning2.csv")


def instantiate(model_name: str, config: str):
    """Rebuild the winning estimator from a tuning row."""
    if model_name == "Ridge":
        alpha = {"a=1": 1.0, "a=10": 10.0, "a=100": 100.0}[config]
        return make_pipeline(
            PolynomialFeatures(2, include_bias=False), StandardScaler(),
            Ridge(alpha=alpha))
    raise ValueError(f"unmapped winner: {model_name}/{config}")


def loso_from_factory(factory, X, y, groups):
    pred = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = factory().fit(X[tr], y[tr]).predict(X[te])
    return pred


def calibrated(y, pred, X, factory, groups, k):
    """Refit per fold, then shift the held-out subject's predictions by the
    mean residual measured on k calibration steps spread over its slope range."""
    out = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = factory().fit(X[tr], y[tr])
        p = m.predict(X[te])
        if k > 0:
            idx = te[np.linspace(0, len(te) - 1, num=min(k, len(te)), dtype=int)]
            p = p + np.mean(y[idx] - m.predict(X[idx]))
        out[te] = p
    return out


def metrics(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=mean_squared_error(y, p) ** 0.5)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

    raw = pd.read_csv(DATA)
    d = raw[raw.peak_at_window_end != True].reset_index(drop=True)  # noqa: E712
    X = d[FEATURES].values
    groups = d.subject.values
    print(f"rows {len(raw)} -> {len(d)} | subjects {d.subject.nunique()}")

    tuning = pd.read_csv(TUNING)
    tuning = tuning.rename(columns={"config": "params"})

    report, summary_rows = [], []
    report.append("# KFM prediction - model comparison (leave-one-subject-out)\n")
    report.append(f"Data: {len(d)} steps, {d.subject.nunique()} subjects, "
                  f"{len(raw) - len(d)} rows dropped by QC.\n")
    report.append(f"Features: `{', '.join(FEATURES)}` (the inputs a user can set).\n")

    for target, short in TARGETS.items():
        y = d[target].values
        best = tuning[tuning.target == target].sort_values("cv_rmse").iloc[0]
        name, params = best.model, best.params
        factory = (lambda n=name, p=params: instantiate(n, p))

        print(f"\n=== {target}  ->  {name}")
        report.append(f"\n## {target}\n\nSelected model: **{name}**  \n"
                      f"params: `{params}`\n")

        base = np.full(len(y), y[[0]].mean())
        base_pred = np.full(len(y), np.nan)
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            base_pred[te] = y[tr].mean()

        p_raw = loso_from_factory(factory, X, y, groups)

        lines = ["| setting | R2 | MAE | RMSE |", "|---|---|---|---|"]
        for label, p in [("constant mean (baseline)", base_pred),
                         (f"{name}, no calibration", p_raw)]:
            m = metrics(y, p)
            lines.append(f"| {label} | {m['R2']:+.3f} | {m['MAE']:.3f} | {m['RMSE']:.3f} |")
            summary_rows.append(dict(target=target, setting=label, model=name,
                                     k=0, **m))
        for k in (1, 2, 3, 5):
            p = calibrated(y, p_raw, X, factory, groups, k)
            m = metrics(y, p)
            lines.append(f"| {name}, calibrated on {k} step(s) | "
                         f"{m['R2']:+.3f} | {m['MAE']:.3f} | {m['RMSE']:.3f} |")
            summary_rows.append(dict(target=target, setting=f"calibrated k={k}",
                                     model=name, k=k, **m))
        report.append("\n".join(lines) + "\n")
        print("\n".join(lines))

        # refit on everything and export
        final = factory().fit(X, y)
        fname = f"{short}_final.joblib"
        joblib.dump(final, os.path.join(OUT_DIR, fname))
        meta = dict(target=target, model=name, params=params,
                    features=FEATURES,
                    valid_range={c: [float(d[c].min()), float(d[c].max())]
                                 for c in FEATURES},
                    loso_no_calibration=metrics(y, p_raw),
                    loso_calibrated_k2=metrics(
                        y, calibrated(y, p_raw, X, factory, groups, 2)),
                    noise_ceiling_explained=float(
                        r2_score(y, d.groupby(["subject", "slope_deg"])[target]
                                 .transform("mean").values)))
        with open(os.path.join(OUT_DIR, f"{short}_final.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  saved {OUT_DIR}/{fname}")

    metrics_csv = os.path.join(HERE, "results", "final_metrics.csv")
    pd.DataFrame(summary_rows).to_csv(metrics_csv, index=False)
    with open(REPORT, "w") as f:
        f.write("\n".join(report))
    print(f"\nwrote {REPORT} and {metrics_csv}")


if __name__ == "__main__":
    main()
