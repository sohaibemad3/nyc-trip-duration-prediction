# Reproducibility Guide & Environment Instructions

This guide provides exact step-by-step instructions to replicate all experimental results, figures, and artifacts reported in this repository.

---

## 1. System Requirements & Pinned Environment

- **Supported Python**: `Python 3.10.x` (tested on 3.10.12)
- **Operating System**: Linux (Ubuntu 20.04/22.04 LTS recommended), macOS, or Windows (WSL2 recommended)
- **Hardware Minimum**: 8 GB RAM, 4 CPU cores (16 GB RAM recommended for multi-threaded tree building)

---

## 2. Virtual Environment Setup

From the repository root:

```bash
# 1. Create a clean virtual environment
python3 -m venv .venv

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install the package in editable mode
pip install -e .
```

---

## 3. Dataset Verification & Placement

Verify that `train.csv`, `val.csv`, and `test.csv` are positioned in either the repository root or inside `data/raw/`:

```bash
ls -lh train.csv val.csv test.csv
```

Expected row counts:
- `train.csv`: 1,000,001 lines (1 header + 1,000,000 records)
- `val.csv`:   229,320 lines (1 header + 229,319 records)
- `test.csv`:  229,323 lines (1 header + 229,322 records)

---

## 4. Running the Automated Test Suite

Run pytest to ensure all data validators, distance calculations, transformers, and regression metrics pass:

```bash
pytest -v
```

All 22 unit and integration tests should pass within ~1-2 seconds.

---

## 5. Training Models & Reproducing Historical Benchmarks

### Reproducing Ridge Regression Baseline
```bash
python scripts/train.py --model ridge --evaluate-test
```

**Expected Exact Output**:
- Validation: $R^2 = 0.7022$, $\text{RMSE} = 0.4093$
- Test Set:   $R^2 = 0.6342$, $\text{RMSE} = 0.4768$

### Reproducing XGBoost Regressor
```bash
python scripts/train.py --model xgboost --evaluate-test
```

**Expected Exact Output**:
- Validation: $R^2 = 0.7781$, $\text{RMSE} = 0.3533$
- Test Set:   $R^2 = 0.6858$, $\text{RMSE} = 0.4419$

---

## 6. Running Independent Evaluation

To evaluate a previously saved model artifact against any data split:

```bash
# Evaluate XGBoost on the test set
python scripts/evaluate.py --model-path models/xgboost_model.joblib --split test

# Evaluate Ridge on the validation set
python scripts/evaluate.py --model-path models/ridge_model.joblib --split val
```

---

## 7. Generating Single or Batch Predictions

### Single-Trip Inference
```bash
python scripts/predict.py \
    --model-path models/xgboost_model.joblib \
    --pickup-datetime "2016-03-15 18:30:00" \
    --pickup-longitude -73.982 \
    --pickup-latitude 40.751 \
    --dropoff-longitude -73.965 \
    --dropoff-latitude 40.768 \
    --vendor-id 2
```

### Batch Inference on CSV
```bash
python scripts/predict.py \
    --model-path models/xgboost_model.joblib \
    --input-csv test.csv \
    --output-csv reports/results/test_predictions.csv
```

---

## 8. Determinism and Random Seeds

- Preprocessing, data sampling, and model builders all consume `RANDOM_SEED = 42`.
- XGBoost multi-threading (`n_jobs=-1`) is deterministic on a fixed CPU architecture.
