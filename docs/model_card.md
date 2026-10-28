# Model Card: NYC Taxi Trip Duration Predictor

Following the Model Cards for Model Reporting framework (Mitchell et al., 2019).

---

## 1. Model Details

- **Model Name**: NYC Taxi Trip Duration XGBoost Regressor
- **Model Version**: 0.1.0
- **Model Type**: Gradient Boosted Decision Tree Ensemble (`XGBRegressor`)
- **Developer**: Sohaib (Machine Learning Engineer)
- **Framework**: XGBoost 3.2.0, Scikit-learn 1.7.2, Python 3.10
- **Hyperparameters**:
  - `n_estimators`: 200
  - `learning_rate`: 0.05
  - `max_depth`: 5
  - `random_state`: 42
  - `n_jobs`: -1
- **License**: MIT License

---

## 2. Intended Use

### Primary Intended Uses
- Real-time estimated time of arrival (ETA) prediction for taxi trips in the New York City metropolitan area.
- Dispatch optimization, fleet allocation, and passenger trip duration forecasting.
- Transportation research and historical traffic pattern benchmarking.

### Out-of-Scope & Unintended Uses
- Inter-city travel beyond the NYC metropolitan boundary ($[-74.03, -73.77]$ lon, $[40.63, 40.85]$ lat).
- Emergency medical vehicle or police response dispatching (requires street-level routing and live sirens).
- Autonomous vehicle motion planning or dynamic safety-critical systems.

---

## 3. Factors, Data & Preprocessing

### Input Features (16)
1. `pickup_longitude` (float)
2. `pickup_latitude` (float)
3. `dropoff_longitude` (float)
4. `dropoff_latitude` (float)
5. `distance_km` (float, Yeo-Johnson power-transformed and standardized)
6. `hour_sin` (float, $\sin(2\pi \cdot \text{hour} / 24)$)
7. `hour_cos` (float, $\cos(2\pi \cdot \text{hour} / 24)$)
8. `weekday_sin` (float, $\sin(2\pi \cdot \text{weekday} / 7)$)
9. `weekday_cos` (float, $\cos(2\pi \cdot \text{weekday} / 7)$)
10. `pickup_day` (int, 1-31)
11. `pickup_month` (int, 1-6)
12. `is_weekend` (binary, $\{0, 1\}$)
13. `is_rush_hour` (binary, $\{0, 1\}$)
14. `is_late_night` (binary, $\{0, 1\}$)
15. `is_holiday` (binary, $\{0, 1\}$)
16. `vendor_id_2` (binary, $\{0, 1\}$)

### Training Data
- **Source**: New York City Taxi and Limousine Commission (TLC) Yellow Taxi trip records (January 1 - June 30, 2016).
- **Records**: 977,490 cleaned records (after removing outside-NYC trips, zero-distance records, implausible speeds, and trips $>2$ hours).

---

## 4. Quantitative Metrics

Performance evaluated on log-transformed target $\log(1 + \text{trip\_duration})$:

| Metric | Validation Set | Test Set (Holdout) | Description |
|---|---|---|---|
| **$R^2$ (Log Scale)** | **0.7781** | **0.6858** | Variance explained in log-duration space |
| **RMSE (Log Scale / RMSLE)** | **0.3533** | **0.4419** | Root Mean Squared Logarithmic Error |
| **MAE (Real Scale - Minutes)** | **3.16 min** | **5.21 min** | Average absolute prediction error |
| **RMSE (Real Scale - Minutes)**| **4.97 min** | **84.75 min** | Real scale error (sensitive to test outliers) |

---

## 5. Limitations & Bias Considerations

1. **Geographic Concentration**: The overwhelming majority of trips originate in Manhattan and at the two major airports (JFK and LaGuardia). Trips originating in Staten Island or outer Bronx neighborhoods have sparser representation.
2. **Missing Weather Signals**: Extreme weather events (such as the historic January 22–24, 2016 Winter Storm Jonas blizzard in NYC) caused severe transit shutdowns not captured by calendar features alone.
3. **Temporal Drift**: Urban infrastructure, bike lanes, bus lanes, dynamic tolling, and ride-hailing volumes have evolved substantially since 2016. Direct application to current traffic conditions requires fine-tuning on modern TLC data.
