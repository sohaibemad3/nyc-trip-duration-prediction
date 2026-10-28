# Experiment Tracking & Benchmark Registry

This document records the experimental results across all evaluated models, comparing historical notebook findings with independently verified results.

---

## 1. Historical Notebook Experiments

The historical experiments recorded in `EDA_NYC.ipynb` evaluated Ridge, XGBoost, and MLP on the validation set, with Ridge and XGBoost evaluated on the test set.

### Historical Validation Set Results (`val.csv`, $N = 224,303$)

| Model | Hyperparameters | Log R² | Log RMSE (RMSLE) | Verification Status |
|---|---|---|---|---|
| **Ridge Regression** | $\alpha = 1.0$ | **0.702169** | **0.409291** | **Independently Verified** |
| **XGBoost Regressor** | $n\_estimators=200, \eta=0.05, d_{\max}=5$ | **0.778113** | **0.353275** | **Independently Verified** |
| **MLP Regressor** | $(64, 32)\text{ units}, \text{max\_iter}=1000$ | **0.809167** | **0.327622** | Historical Record (Notebook Cell 29) |

### Historical Test Set Results (`test.csv`, $N = 226,182$)

| Model | Hyperparameters | Log R² | Log RMSE (RMSLE) | Verification Status |
|---|---|---|---|---|
| **Ridge Regression** | $\alpha = 1.0$ | **0.634190** | **0.476834** | **Independently Verified** |
| **XGBoost Regressor** | $n\_estimators=200, \eta=0.05, d_{\max}=5$ | **0.685824** | **0.441902** | **Independently Verified** |
| **MLP Regressor** | $(64, 32)\text{ units}, \text{max\_iter}=1000$ | N/A | N/A | Interrupted in original notebook |

---

## 2. Verified Experimental Benchmark

The results below were independently reproduced and verified using the modular pipeline (`scripts/train.py`, `scripts/evaluate.py`) on Ubuntu Linux with Python 3.10.12.

### Comprehensive Benchmark Table

| Model | Split | Evaluated Rows | Log R² | Log RMSE (RMSLE) | Raw MAE (min) | Raw RMSE (min) | Training Time (s) |
|---|---|---|---|---|---|---|---|
| **Ridge** | Validation | 224,303 | **0.7022** | **0.4093** | 3.78 min | 5.86 min | 0.15s |
| **Ridge** | Test Set | 226,182 | **0.6342** | **0.4768** | 5.84 min | 84.81 min | 0.15s |
| **XGBoost** | Validation | 224,303 | **0.7781** | **0.3533** | 3.16 min | 4.97 min | 9.27s |
| **XGBoost** | Test Set | 226,182 | **0.6858** | **0.4419** | 5.21 min | 84.75 min | 9.27s |

> [!NOTE]
> **Observation on Raw Scale RMSE**:
> While both Ridge and XGBoost achieve low Mean Absolute Error on the test set (~5.2 to 5.8 minutes), the raw scale RMSE is higher on test (84.8 minutes). This is a direct consequence of the test set not being filtered for extreme duration outliers (trips $>120$ minutes or extreme traffic delays that exist in raw test records), whereas the validation set was cleaned. In log space, extreme outliers are compressed, yielding stable RMSLE ($0.4419$).

---

## 3. Ridge Linear Model Weight Interpretation

Inspecting the learned coefficients of the Ridge model reveals the relative linear influence of each normalized feature:

| Feature Name | Coefficient ($\mathbf{w}$) | Interpretation |
|---|---|---|
| `distance_km` | **+3.4520** | Dominant positive predictor: longer travel distance directly increases duration. |
| `is_holiday` | **-0.1118** | Official holidays reduce traffic congestion, leading to shorter travel times. |
| `is_weekend` | **-0.0499** | Weekend trips encounter less commercial and commuter traffic. |
| `is_rush_hour` | **+0.0831** | Rush hour periods significantly increase expected trip duration. |
| `is_late_night` | **+0.0720** | Moderate increase due to longer late-night inter-borough trips. |
| `hour_cos` | **-0.4078** | Cyclical diurnal peak/trough modulation. |
| `hour_sin` | **-0.2490** | Cyclical morning vs evening phase shift. |
| `dropoff_latitude` | **-0.2997** | Northbound trips in Manhattan (uptown/Bronx). |
| `dropoff_longitude` | **-0.2887** | Eastbound trips (Queens/Brooklyn). |
| `vendor_id_2` | **+0.0028** | Negligible recording discrepancy between hardware vendors. |
| *Bias (Intercept)* | **+5.0589** | Base log trip duration ($\approx 156\text{ seconds}$ / $2.6\text{ minutes}$). |

---

## 4. Hardware Environment for Verification

- **Processor**: 12 vCPUs (x86_64)
- **RAM**: 15 GiB
- **Operating System**: Linux (Ubuntu 22.04 LTS kernel)
- **Python**: 3.10.12
- **Core Libraries**: Scikit-Learn 1.7.2, XGBoost 3.2.0, NumPy 2.2.6, Pandas 2.3.3
