# Project Overview: NYC Taxi Trip Duration Prediction

## 1. Executive Summary

Accurate travel time estimation is a foundational capability for intelligent transportation systems, ride-hailing platforms, and municipal traffic management. This project implements a production-grade Machine Learning system to predict taxi trip duration in New York City using the NYC Taxi and Limousine Commission (TLC) Yellow Taxi trip records.

The repository transforms exploratory analytical research into a clean, reproducible, and tested ML architecture. It features modular data validation, physical anomaly filtering, Haversine geospatial feature engineering, harmonic temporal encodings, power transformations, model comparison (Ridge Regression, Gradient Boosted Trees via XGBoost, and Multilayer Perceptron), and reproducible inference scripts.

---

## 2. Problem Formulation

Given a taxi ride record initiated in New York City:
- **Input Features ($X$)**:
  - Pickup location: latitude $\phi_1$, longitude $\lambda_1$
  - Dropoff location: latitude $\phi_2$, longitude $\lambda_2$
  - Timestamp: pickup date and time $t_{\text{pickup}}$
  - Technology vendor: $V \in \{1, 2\}$
  - Passenger count: $N_{\text{passengers}}$
- **Target ($y$)**:
  - Trip duration in seconds: $T \in \mathbb{R}^+$
- **Objective**:
  - Learn a mapping $f(X) \to \log(1 + T)$ minimizing Root Mean Squared Logarithmic Error (RMSLE).

### Why Predict Log-Duration?
Taxi trip durations span multiple orders of magnitude—from short 2-minute neighborhood hops to 90-minute cross-borough airport transits—and exhibit severe positive skewness. Minimizing Root Mean Squared Error (RMSE) directly on raw seconds would cause estimators to disproportionately penalize errors on long trips while underfitting short trips. Optimizing in log-space ($\log(1 + y)$):
1. Normalizes the target distribution, aligning with Gaussian assumptions in linear models and neural networks.
2. Equalizes relative percentage errors across short and long trips.
3. Formulates the standard Kaggle evaluation metric (RMSLE) as standard RMSE in log-transformed space.

---

## 3. Scope and Architectural Workflow

```
                                  +-----------------------+
                                  | Raw TLC Trip Records  |
                                  | (train, val, test)    |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | Schema Validation &   |
                                  | Geographic Bounding   |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | Outlier Filtering     |
                                  | (Train / Val Only)    |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | Feature Engineering   |
                                  | - Haversine Distance  |
                                  | - Cyclical Timestamps |
                                  | - NY State Holidays   |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | Preprocessing Pipeline|
                                  | - PowerTransformer   |
                                  | - One-Hot Encoding    |
                                  | - MinMaxScaler        |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | Estimator Training    |
                                  | - Ridge Regression    |
                                  | - XGBoost Regressor   |
                                  | - MLP Regressor       |
                                  +-----------+-----------+
                                              |
                                              v
                                  +-----------------------+
                                  | Evaluation & Metrics  |
                                  | (Log R², RMSLE, MAE)  |
                                  +-----------------------+
```

---

## 4. Key Engineering Standards

- **Strict Contamination Prevention**: Preprocessors (PowerTransformer, MinMaxScaler) and outlier filters are fitted exclusively on training data and subsequently applied to validation and test holdouts.
- **Reproducibility**: Global random seeds, deterministic sampling, pinned package specifications, and standard Scikit-learn pipelines.
- **Portability**: Relative path resolutions independent of current working directory or absolute filesystem paths.
- **Test-Driven Reliability**: Comprehensive test suite covering boundary checks, feature calculations, transformer invariants, and inference workflows.
