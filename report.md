# KFM prediction - model comparison (leave-one-subject-out)

Data: 1664 steps, 22 subjects, 29 rows dropped by QC.

Features: `slope_deg, speed_mps, step_length_m` (the inputs a user can set).


## peak_kfm_Nm_per_kg

Selected model: **Ridge**  
params: `a=10`

| setting | R2 | MAE | RMSE |
|---|---|---|---|
| constant mean (baseline) | -0.049 | 0.375 | 0.459 |
| Ridge, no calibration | +0.334 | 0.300 | 0.366 |
| Ridge, calibrated on 1 step(s) | +0.527 | 0.236 | 0.308 |
| Ridge, calibrated on 2 step(s) | +0.677 | 0.205 | 0.255 |
| Ridge, calibrated on 3 step(s) | +0.694 | 0.197 | 0.248 |
| Ridge, calibrated on 5 step(s) | +0.709 | 0.192 | 0.242 |


## kfm_impulse_Nms_per_kg

Selected model: **Ridge**  
params: `a=10`

| setting | R2 | MAE | RMSE |
|---|---|---|---|
| constant mean (baseline) | -0.036 | 0.157 | 0.199 |
| Ridge, no calibration | +0.389 | 0.119 | 0.153 |
| Ridge, calibrated on 1 step(s) | +0.617 | 0.095 | 0.121 |
| Ridge, calibrated on 2 step(s) | +0.671 | 0.089 | 0.112 |
| Ridge, calibrated on 3 step(s) | +0.703 | 0.084 | 0.107 |
| Ridge, calibrated on 5 step(s) | +0.709 | 0.083 | 0.105 |
