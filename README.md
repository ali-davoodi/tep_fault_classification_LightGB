# TEP Fault Classification

Multiclass fault classifier for the [Tennessee Eastman Process (TEP) dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1), using LightGBM to identify 21 fault types and normal operation from process measurements.

## Problem

Given 52 process variables sampled every 3 minutes, classify each observation as one of 22 classes:

- **Class 0** — normal operation
- **Classes 1–21** — fault types IDV 1–21 (injected at simulation hour 8)

## Dataset

| File | Description | Rows |
|------|-------------|------|
| `ff_train.parquet` | Normal operation, training | 250,000 |
| `ff_test.parquet` | Normal operation, testing | 480,000 |
| `f_train.parquet` | 21 fault types, training | 5,000,000 |
| `f_test.parquet` | 21 fault types, testing | 9,600,000 |

Only **post-injection rows** (sample > 160) are used from the faulty files, where fault signatures are present.

## Notebook Contents

`main.ipynb` walks through:

1. **Build Training Set** — normal samples from `ff_train` + post-injection faulty samples from `f_train` (loaded fault-by-fault to keep RAM ~30 MB/iter)
2. **Build Test Set** — normal samples from `ff_test` + post-injection faulty samples from `f_test`
3. **Train LightGBM** — 22-class multiclass with early stopping
4. **Evaluate** — overall accuracy, per-fault classification report, confusion matrix, feature importances

## Setup

```bash
pip install lightgbm pyarrow pandas numpy matplotlib seaborn scikit-learn
```

Place the parquet files in `../datasets/TEP/Harvard/` relative to this repo, then run `main.ipynb`.
