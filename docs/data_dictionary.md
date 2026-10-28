# Data Dictionary: NYC Taxi Trip Duration

This document provides complete technical specifications for all original, derived, and target features in the NYC Taxi Trip Duration dataset.

---

## 1. Raw Dataset Schema

The raw dataset contains records of yellow taxi trips operating in New York City from **January 1, 2016 through June 30, 2016**.

| Column Name | Raw Dtype | Physical Meaning | Example | Range / Domain | Notes |
|---|---|---|---|---|---|
| `id` | `object` / `str` | Unique alphanumeric trip identifier | `"id2793718"` | Non-null unique keys | Used for tracking; dropped prior to model fitting. |
| `vendor_id` | `int64` | In-cab technology provider code | `1`, `2` | $\{1, 2\}$ | $1$: Creative Mobile Technologies (CMT); $2$: VeriFone Inc. (VTS). Encoded as dummy `vendor_id_2`. |
| `pickup_datetime` | `object` (ISO-8601) | Date and time when the taximeter was engaged | `"2016-03-14 17:24:55"` | 2016-01-01 to 2016-06-30 | Local NYC timestamp (Eastern Time). |
| `passenger_count` | `int64` | Number of passengers in vehicle | `1` | $[0, 9]$ | Driver-entered value. Trips with 0 passengers represent unaccompanied parcel/package delivery runs or entry errors. |
| `pickup_longitude` | `float64` | Longitude of pickup location | `-73.982155` | Valid NYC: $[-74.03, -73.77]$ | WGS84 decimal degrees. Sensor noise or inter-state trips outside bounds are filtered. |
| `pickup_latitude` | `float64` | Latitude of pickup location | `40.767937` | Valid NYC: $[40.63, 40.85]$ | WGS84 decimal degrees. |
| `dropoff_longitude`| `float64` | Longitude of dropoff location | `-73.964630` | Valid NYC: $[-74.03, -73.77]$ | WGS84 decimal degrees. |
| `dropoff_latitude` | `float64` | Latitude of dropoff location | `40.765602` | Valid NYC: $[40.63, 40.85]$ | WGS84 decimal degrees. |
| `store_and_fwd_flag`| `object` | Store-and-forward indicator | `"N"` | `{"Y", "N"}` | `"Y"` indicates vehicle lost wireless connection and stored record in flash memory before transmitting to server. |
| `trip_duration` | `int64` | **Target Variable**: Total trip elapsed time | `827` | $[1, 2,227,612]$ sec | Measured in integer seconds by taximeter. |

---

## 2. Derived & Engineered Features

| Feature Name | Derived Type | Mathematical Formula / Derivation | Range | Purpose & Motivation |
|---|---|---|---|---|
| `distance_km` | `float64` | Haversine great-circle distance between pickup and dropoff coordinates using $R = 6371\text{ km}$. | $[0, \infty)$ | Strongest linear and non-linear predictor of travel duration. Power-transformed via Yeo-Johnson during preprocessing. |
| `duration_minutes` | `float64` | $\text{trip\_duration} / 60.0$ | $[0.01, \infty)$ | Human-interpretable duration scale used strictly for EDA and outlier filtering. |
| `real_speed_kmh` | `float64` | $\text{distance\_km} / (\text{duration\_minutes} / 60.0)$ | $[0, \infty)$ | Implied average trip speed. Used exclusively for filtering impossible speeds ($\le 1\text{ km/h}$ or $\ge 160\text{ km/h}$) on training data. Never passed as an input feature (avoids target leakage). |
| `pickup_month` | `int32` | $\text{month}(t_{\text{pickup}})$ | $[1, 6]$ | Seasonal volume variation across Winter and Spring 2016. |
| `pickup_day` | `int32` | $\text{day}(t_{\text{pickup}})$ | $[1, 31]$ | Day-of-month calendar effect. |
| `pickup_hour` | `int32` | $\text{hour}(t_{\text{pickup}})$ | $[0, 23]$ | Diurnal traffic cycle. |
| `pickup_weekday` | `int32` | $\text{weekday}(t_{\text{pickup}})$ (0 = Monday) | $[0, 6]$ | Workweek vs weekend travel patterns. |
| `hour_sin` | `float64` | $\sin(2\pi \cdot \text{hour} / 24)$ | $[-1.0, 1.0]$ | Continuous harmonic encoding preserving the circular topology of time (hour 23 is adjacent to hour 0). |
| `hour_cos` | `float64` | $\cos(2\pi \cdot \text{hour} / 24)$ | $[-1.0, 1.0]$ | Orthogonal component for 24-hour circular representation. |
| `weekday_sin` | `float64` | $\sin(2\pi \cdot \text{weekday} / 7)$ | $[-1.0, 1.0]$ | Harmonic encoding preserving the 7-day weekly cycle (Sunday is adjacent to Monday). |
| `weekday_cos` | `float64` | $\cos(2\pi \cdot \text{weekday} / 7)$ | $[-1.0, 1.0]$ | Orthogonal component for 7-day circular representation. |
| `is_weekend` | `int64` | $\mathbb{I}(\text{weekday} \ge 5)$ | $\{0, 1\}$ | Binary indicator for Saturday and Sunday. |
| `is_rush_hour` | `int64` | Weekday and (7-10 AM or 4-7 PM) | $\{0, 1\}$ | Indicator for peak congestion periods when traffic speeds plummet. |
| `is_late_night` | `int64` | $\mathbb{I}(\text{hour} \ge 23 \lor \text{hour} < 5)$ | $\{0, 1\}$ | Indicator for low-traffic overnight hours where average speeds peak. |
| `is_holiday` | `int64` | US federal and New York state official holidays | $\{0, 1\}$ | Accounts for holiday traffic patterns (e.g. New Year's Day, Memorial Day, Martin Luther King Jr. Day). |
| `vendor_id_2` | `int64` | Dummy encoding: $\mathbb{I}(\text{vendor\_id} == 2)$ | $\{0, 1\}$ | Captures systematic recording differences between CMT and VeriFone hardware. |
| `log_trip_duration` | `float64` | $\log(1 + \text{trip\_duration})$ | $[0, \infty)$ | **Transformed Target**: Log1p-transformed duration in seconds. Optimizing with RMSE equals minimizing RMSLE on raw seconds. |

---

## 3. Data Cleaning Criteria & Boundaries

The following filters are applied to training and validation splits:

1. **Geographic Metro NYC Box**:
   - `pickup_longitude` $\in [-74.03, -73.77]$
   - `pickup_latitude` $\in [40.63, 40.85]$
   - `dropoff_longitude` $\in [-74.03, -73.77]$
   - `dropoff_latitude` $\in [40.63, 40.85]$
2. **Zero-Distance Anomaly**:
   - $\text{distance\_km} > 0.0$
3. **Plausible Speed Window**:
   - $1.0\text{ km/h} \le \text{real\_speed\_kmh} \le 160.0\text{ km/h}$
4. **Trip Duration Upper Cap**:
   - $\text{duration\_minutes} \le 120.0$ (2 hours)
