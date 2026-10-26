"""
Predict knee flexion moment (KFM) from gait parameters.

Two modes
---------
1. Population model (no calibration) -- same inputs as predict_kfm.py:
       slope_deg, speed_mps, step_length_m
   Accurate to ~R2 0.33 (peak) / 0.38 (impulse) on an unseen subject.

2. Calibrated model -- give a few measured steps from the subject first.
   The mean residual on those steps becomes a per-subject offset. Two steps
   lift R2 to ~0.65; five steps to ~0.70.

Usage
-----
    # population prediction for a table of gaits
    python predict_calibrated.py --slope 10 --speed 1.2 --step-length 0.6

    # calibrate on measured steps, then predict
    python predict_calibrated.py --calibration calibration.csv \
        --slope 8 12 --speed 1.0 1.1 --step-length 0.6 0.65

`calibration.csv` needs the same three columns plus `peak_kfm_Nm_per_kg`
and/or `kfm_impulse_Nms_per_kg`.
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "kfm_model")
FEATURES = ["slope_deg", "speed_mps", "step_length_m"]

MODELS = {
    "peak_kfm_Nm_per_kg": "peak_kfm_final.joblib",
    "kfm_impulse_Nms_per_kg": "kfm_impulse_final.joblib",
}


def load_pair(model_dir=MODEL_DIR):
    out = {}
    for target, fname in MODELS.items():
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing -- run `python train_final.py` first")
        out[target] = joblib.load(path)
    return out


def predict(models, X: pd.DataFrame, offsets=None):
    Xf = X[FEATURES]
    res = X.copy()
    for target, model in models.items():
        p = model.predict(Xf)
        if offsets and target in offsets:
            p = p + offsets[target]
        res[target] = p
        # absolute units: multiply by body mass if supplied
        if "body_mass_kg" in X:
            res[target.replace("_per_kg", "")] = p * X["body_mass_kg"]
    return res


def calibration_offsets(models, calib: pd.DataFrame, cols=None):
    """Mean residual per target on the calibration steps."""
    offsets, report = {}, {}
    for target, model in models.items():
        if target not in calib:
            continue
        use = calib.dropna(subset=[target])
        if cols is not None:
            use = use[cols]
        if not len(use):
            continue
        resid = use[target].values - model.predict(use[FEATURES])
        offsets[target] = float(np.mean(resid))
        report[target] = dict(n_steps=len(use), offset=offsets[target],
                              resid_sd=float(np.std(resid, ddof=1)) if len(use) > 1 else float("nan"))
    return offsets, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slope", nargs="+", type=float, required=True,
                    help="slope in degrees (5.2-18)")
    ap.add_argument("--speed", nargs="+", type=float, required=True,
                    help="walking speed m/s (0.75-1.42 to stay in-range)")
    ap.add_argument("--step-length", nargs="+", type=float, required=True,
                    help="step length in m (0.44-0.71 to stay in-range)")
    ap.add_argument("--mass", type=float, default=None,
                    help="body mass in kg; adds absolute N*m / N*m*s columns")
    ap.add_argument("--calibration", default=None,
                    help="CSV of measured steps for this subject")
    ap.add_argument("--calib-cols", nargs="*", default=None,
                    help="optional column subset of the calibration CSV to use")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args()

    n = max(len(args.slope), len(args.speed), len(args.step_length))
    def br(v):
        return v * n if len(v) == 1 else v
    X = pd.DataFrame({
        "slope_deg": br(args.slope),
        "speed_mps": br(args.speed),
        "step_length_m": br(args.step_length),
    })
    if args.mass is not None:
        X["body_mass_kg"] = args.mass

    models = load_pair(args.model_dir)
    offsets, cal_report = None, None
    if args.calibration:
        calib = pd.read_csv(args.calibration)
        cols = [c for c in (args.calib_cols or []) if c in calib] or None
        offsets, cal_report = calibration_offsets(models, calib, cols)
        if not offsets:
            print("warning: calibration file had no usable target column")

    out = predict(models, X, offsets)

    meta = {}
    for target in MODELS:
        p = os.path.join(args.model_dir,
                         MODELS[target].replace(".joblib", ".json"))
        if os.path.exists(p):
            with open(p) as f:
                meta[target] = json.load(f)

    if args.json:
        print(json.dumps({
            "predictions": out.to_dict(orient="records"),
            "calibration": cal_report,
            "model_meta": meta,
        }, indent=2))
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(out.to_string(index=False))
        if cal_report:
            print("\ncalibration offsets applied:")
            for t, r in cal_report.items():
                print(f"  {t:26s} {r['offset']:+.3f}  (n={r['n_steps']}, sd={r['resid_sd']:.3f})")
        else:
            print("\n(no calibration -- predictions are population averages, "
                  "expect R2 ~0.33-0.38 on a new subject)")
        for t, m in meta.items():
            if m.get("loso_no_calibration"):
                print(f"  {t}: LOSO R2={m['loso_no_calibration']['R2']:.3f} "
                      f"MAE={m['loso_no_calibration']['MAE']:.3f}")


if __name__ == "__main__":
    main()
