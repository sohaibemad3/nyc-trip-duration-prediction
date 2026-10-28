# Modeling Methodology: NYC Taxi Trip Duration

## 1. Problem Formulation & Splitting Strategy

The goal is to predict the continuous trip duration of NYC yellow taxi rides. The project dataset consists of 1,458,641 records divided into three fixed, disjoint partitions:

- **Training Set (`train.csv`)**: 1,000,000 trips (~68.5%)
  Used for fitting feature transformers (PowerTransformer, MinMaxScaler) and estimator parameters.
- **Validation Set (`val.csv`)**: 229,319 trips (~15.7%)
  Used for model selection, hyperparameter tuning, and cross-model comparison.
- **Test Set (`test.csv`)**: 229,322 trips (~15.7%)
  Held out strictly until final model benchmarking.

---

## 2. Leakage Prevention Principles

Data leakage is a primary risk in tabular spatio-temporal problems. We employ three strict architectural safeguards:

1. **Target-Dependent Filters Isolated to Training**:
   - `real_speed_kmh` and `duration_minutes` require the ground-truth target `trip_duration`.
   - These filters are used **only** to cleanse training and validation sets of gross sensor errors (e.g., recorded speeds $>160\text{ km/h}$ or durations $>2\text{ hours}$).
   - They are **never** applied to the test set or during live inference.
2. **Transformer Fitting Scope**:
   - `PowerTransformer` (Yeo-Johnson) and `MinMaxScaler` are fitted **strictly** on the training set $X_{\text{train}}$.
   - Validation and test splits are transformed using the fitted state without updating parameters.
3. **No Target Leakage in Features**:
   - Derived features are calculated strictly from pickup/dropoff coordinates and timestamps. No out-of-fold target encodings or post-trip features are computed.

---

## 3. Mathematical Feature Engineering

### 3.1 Geospatial Distance (Haversine Formula)
Because NYC coordinates are given as latitude $\phi$ and longitude $\lambda$ on an oblate spheroid, Euclidean distances in degrees distort physical distance due to latitude convergence. We apply the great-circle Haversine formula:

$$\Delta\phi = \phi_2 - \phi_1, \quad \Delta\lambda = \lambda_2 - \lambda_1$$
$$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)$$
$$c = 2 \arcsin\left(\min(1, \sqrt{a})\right)$$
$$d = R_{\text{earth}} \cdot c \quad (R_{\text{earth}} = 6371.0\text{ km})$$

### 3.2 Yeo-Johnson Power Transformation
The distribution of `distance_km` is heavily right-skewed. To stabilize variance and linearize relationships with log-duration, we apply the parametric Yeo-Johnson transformation:

$$\psi(\lambda, y) = \begin{cases} ((y + 1)^\lambda - 1)/\lambda & \text{if } \lambda \neq 0, y \ge 0 \\ \ln(y + 1) & \text{if } \lambda = 0, y \ge 0 \end{cases}$$

The parameter $\lambda$ is estimated via maximum likelihood on $X_{\text{train}}$, followed by zero-mean unit-variance standardization.

### 3.3 Harmonic Cyclical Encodings
Standard integer representations of hours ($0 \dots 23$) introduce an artificial discontinuity between 23:59 and 00:00. We project temporal variables onto a 2D unit circle:

$$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
$$\text{weekday\_sin} = \sin\left(\frac{2\pi \cdot \text{weekday}}{7}\right), \quad \text{weekday\_cos} = \cos\left(\frac{2\pi \cdot \text{weekday}}{7}\right)$$

This preserves continuous cyclical proximity for linear models, neural networks, and decision trees.

---

## 4. Model Architectures & Selection

We evaluate three complementary model families:

1. **Ridge Regression ($L_2$ Regularization)**:
   $$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$
   - Baseline linear estimator.
   - Closed-form solution with high computational efficiency.
   - Provides interpretable linear coefficients indicating feature directionality.

2. **XGBoost (Extreme Gradient Boosted Trees)**:
   - Ensembles shallow regression trees ($T = 200, d_{\max} = 5, \eta = 0.05$).
   - Learns non-linear spatial interactions (e.g. crossing congested bridge bottlenecks between Manhattan and Brooklyn/Queens) and traffic congestion non-linearities.
   - Robust to monotonic scaling and outliers.

3. **Multilayer Perceptron (MLP)**:
   - Fully connected feedforward neural architecture with 2 hidden layers: $(64, 32)$ units with ReLU activations.
   - Optimized using Adam with $L_2$ penalty and standard backpropagation.
   - Requires MinMax normalized inputs for gradient stability.

---

## 5. Evaluation Protocol

The evaluation is conducted on the log-transformed scale:
$$y = \log(1 + \text{trip\_duration})$$

### Primary Metrics:
1. **$R^2$ (Coefficient of Determination on Log Scale)**:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   Measures proportion of log-duration variance explained by the model.

2. **Log RMSE / RMSLE**:
   $$\text{RMSE}_{\text{log}} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( \log(y_i + 1) - \log(\hat{y}_i + 1) \right)^2} \equiv \text{RMSLE}$$

### Inverted Practical Metrics:
To provide real-world operational interpretation, predictions are inverted back to seconds ($\hat{T} = \exp(\hat{y}) - 1$) and minutes ($\hat{T} / 60$):
- **Raw MAE (minutes)**: Mean absolute prediction error in minutes.
- **Raw RMSE (minutes)**: Root mean squared prediction error in minutes.
