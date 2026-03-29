# Titanic Survival Prediction — Random Forest Pipeline

A Jupyter notebook implementing and benchmarking machine learning pipelines for the classic **Titanic survival prediction** task. The notebook progresses from a clean baseline to a multi-pipeline comparison with performance profiling.

---

## Overview

The notebook is structured in two main sections:

1. **Baseline Random Forest Pipeline** — A clean, well-engineered Random Forest classifier with rich feature engineering and 10-fold cross-validation.
2. **Multi-Pipeline Benchmark** — A head-to-head comparison of three approaches:
   - **Pipeline A**: Baseline Random Forest
   - **Pipeline B**: JIT-accelerated preprocessing (Numba) + Random Forest
   - **Pipeline C**: Gradient boosting (XGBoost)

---

## Dataset

| Split | Shape |
|-------|-------|
| Train | 891 rows × 12 columns |
| Test  | 418 rows × 11 columns |

- **Survival rate (train):** 38.4%
- **Source:** Kaggle Titanic dataset

---

## Feature Engineering

The `engineer()` function transforms raw Titanic data into **21 features**:

| Feature | Description |
|---------|-------------|
| `Title` | Extracted from passenger name, mapped to 5 categories (Mr, Mrs, Miss, Master, Rare) |
| `Deck` | Derived from the Cabin column (A–G, T → 1–8; unknown → 0) |
| `Cabin_Known` | Binary flag: whether cabin info is available |
| `Cabin_Count` | Number of cabins assigned to the passenger |
| `Family_Size` | SibSp + Parch + 1 |
| `Is_Alone` | Binary flag: family size == 1 |
| `Family_Group` | Categorized family size (0=alone, 1=small, 2=large) |
| `Fare_Per_Person` | Fare divided by family size |
| `Fare_Log` | Log-transformed fare (log1p) |
| `Fare_Bin` | Fare quartile bin (0–3) |
| `Age` | Missing values imputed by median within Pclass × Title groups |
| `Age_Band` | Age bucketed into life stages (child, teen, adult, middle-aged, senior) |
| `Age_x_Class` | Interaction term: Age × Pclass |
| `Sex` | Encoded as binary (female=1, male=0) |
| `Embarked` | Encoded as integer (C=0, Q=1, S=2); missing filled with 'S' |
| `Ticket_Freq` | Frequency of each ticket number |
| `Pclass_Sex` | Interaction term: Pclass × 10 + Sex |

---

## Model Configuration

### Random Forest (Pipelines A & B)

```python
RandomForestClassifier(
    n_estimators=500,
    max_depth=7,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

### XGBoost (Pipeline C)

```python
XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method='hist',
    scale_pos_weight=<imbalance_ratio>
)
```

---

## Results

### Cross-Validation (10-Fold Stratified)

| Pipeline | CV Accuracy | Std | Train Accuracy | Overfitting Gap | Total Time | Speedup |
|----------|-------------|-----|----------------|-----------------|------------|---------|
| A — Baseline RF | **0.8282** | 0.0346 | 0.8878 | 0.0595 (healthy) | 7.87s | 1.00× |
| B — JIT + RF | **0.8282** | 0.0346 | 0.8878 | 0.0595 (healthy) | 5.79s | **1.36×** |
| C — XGBoost | 0.8204 | 0.0273 | 0.9315 | 0.1112 (**OVERFIT**) | 1.38s | **5.68×** |

### Key Findings

- **Best CV accuracy**: Baseline RF and JIT RF both achieve **82.82%** accuracy.
- **Fastest pipeline**: XGBoost runs **5.68× faster** than the baseline RF, but overfits significantly (gap > 10%).
- **JIT preprocessing**: Numba-accelerated z-score standardization reduces total wall-clock time by ~26% while preserving identical accuracy.
- **Best submission**: Pipeline A (Baseline RF) is selected, predicting a survival rate of **40.4%**.

---

## Pipeline Execution Summary

```
Total wall-clock time: ~23s (across all three pipelines)

Detailed timing:
  Data loading:           ~0.015s
  Feature engineering:    ~0.038s
  JIT compilation:        ~1.54s (one-time cost)
  Pipeline A (CV + fit):  ~7.87s
  Pipeline B (CV + fit):  ~5.79s
  Pipeline C (CV + fit):  ~1.38s
```

---

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `xgboost`
- `numba`

---

## Output

- `submission.csv` — Predictions for the 418 test passengers, formatted for Kaggle submission.

---

## Notes

- All missing values are handled during feature engineering (Age, Fare, Embarked).
- No missing values remain in either the training or test feature matrices after engineering.
- Numba JIT functions are compiled once at notebook startup; subsequent calls are near-instant.