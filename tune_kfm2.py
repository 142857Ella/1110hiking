"""Lean sequential LOSO hyperparameter sweep (4-core machine, no nested parallelism)."""
import json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

d = pd.read_csv(os.path.join(HERE, "kfm_steps.csv"))
d = d[d.peak_at_window_end != True].reset_index(drop=True)
F = ["slope_deg", "speed_mps", "step_length_m"]
X = d[F].values; g = d.subject.values
SPLITS = list(LeaveOneGroupOut().split(X, np.zeros(len(X)), g))

def loso_mse(factory, y):
    p = np.full(len(y), np.nan)
    for tr, te in SPLITS:
        p[te] = factory().fit(X[tr], y[tr]).predict(X[te])
    return float(np.mean((y - p) ** 2)), p

def P(**kw):  # poly2 + scaler + ridge
    return lambda: make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(**kw))
def S(**kw):
    return lambda: make_pipeline(StandardScaler(), SVR(**kw))

CAND = {
 "Ridge":     [("a=1",P(alpha=1)), ("a=10",P(alpha=10)), ("a=100",P(alpha=100))],
 "SVR":       [("C1g.1e.03",S(C=1,gamma=0.1,epsilon=0.03)), ("C3g.1e.03",S(C=3,gamma=0.1,epsilon=0.03)),
               ("C10g.1e.03",S(C=10,gamma=0.1,epsilon=0.03)), ("C3g.3e.03",S(C=3,gamma=0.3,epsilon=0.03)),
               ("C10g.3e.01",S(C=10,gamma=0.3,epsilon=0.01)), ("C30g.1e.03",S(C=30,gamma=0.1,epsilon=0.03))],
 "kNN":       [("k=10d",lambda: make_pipeline(StandardScaler(),KNeighborsRegressor(10,weights="distance"))),
               ("k=20d",lambda: make_pipeline(StandardScaler(),KNeighborsRegressor(20,weights="distance"))),
               ("k=40d",lambda: make_pipeline(StandardScaler(),KNeighborsRegressor(40,weights="distance"))),
               ("k=25u",lambda: make_pipeline(StandardScaler(),KNeighborsRegressor(25,weights="uniform")))],
 "HistGBR":   [("l4lr.06",lambda: HistGradientBoostingRegressor(max_iter=300,learning_rate=0.06,max_leaf_nodes=4,min_samples_leaf=20,random_state=0)),
               ("l7lr.06",lambda: HistGradientBoostingRegressor(max_iter=300,learning_rate=0.06,max_leaf_nodes=7,min_samples_leaf=20,random_state=0)),
               ("l7lr.03l2",lambda: HistGradientBoostingRegressor(max_iter=400,learning_rate=0.03,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1.0,random_state=0)),
               ("l15lr.03l2",lambda: HistGradientBoostingRegressor(max_iter=400,learning_rate=0.03,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=10.0,random_state=0))],
 "RF":        [("leaf10",lambda: RandomForestRegressor(n_estimators=200,min_samples_leaf=10,n_jobs=1,random_state=0)),
               ("leaf20",lambda: RandomForestRegressor(n_estimators=200,min_samples_leaf=20,n_jobs=1,random_state=0)),
               ("leaf5",lambda: RandomForestRegressor(n_estimators=200,min_samples_leaf=5,n_jobs=1,random_state=0))],
 "MLP":       [("32a1e-3",lambda: make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(32,),alpha=1e-3,max_iter=1500,random_state=0))),
               ("64-32a1e-2",lambda: make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(64,32),alpha=1e-2,max_iter=1500,random_state=0))),
               ("16a1e-1",lambda: make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(16,),alpha=1e-1,max_iter=1500,random_state=0)))],
}

rows = []
for tgt in ["peak_kfm_Nm_per_kg", "kfm_impulse_Nms_per_kg"]:
    y = d[tgt].values
    print(f"\n=== {tgt}", flush=True)
    for name, cands in CAND.items():
        for label, fac in cands:
            t0 = time.time()
            mse, p = loso_mse(fac, y)
            rows.append(dict(target=tgt, model=name, config=label, cv_rmse=mse**0.5,
                             cv_r2=r2_score(y, p), secs=time.time()-t0))
            print(f"  {name:9s} {label:12s} RMSE={mse**0.5:.4f} R2={r2_score(y,p):+.3f} ({time.time()-t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "tuning2.csv"), index=False)
print("\nBEST PER TARGET")
df = pd.DataFrame(rows)
print(df.loc[df.groupby("target").cv_rmse.idxmin()].to_string(index=False))
