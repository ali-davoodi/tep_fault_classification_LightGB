# TEP Fault Classification

## Project Goal
Train and evaluate a LightGBM multiclass classifier on the Tennessee Eastman Process (TEP) dataset to identify fault types (IDV 1–21) from process measurements.

## Project Diagram

```
../datasets/TEP/Harvard/
├── ff_train.parquet   (normal operation, training)
├── ff_test.parquet    (normal operation, testing)
├── f_train.parquet    (21 fault types, training)
└── f_test.parquet     (21 fault types, testing)

                        │
                        ▼ (pyarrow / pandas)

┌──────────────────────────────────────────────────────────┐
│                 main.ipynb                               │
│                                                          │
│  1. Load Data      post-injection rows only              │
│         │          fault-by-fault for f_train            │
│  2. Train          LightGBM multiclass                   │
│         │          target: faultNumber (0–21)            │
│  3. Evaluate       confusion matrix, per-fault           │
│                    accuracy, feature importances         │
└──────────────────────────────────────────────────────────┘
```

## Dataset

See `../fault_dataset_exploring/CLAUDE.md` for full schema and memory-loading notes.

- **52 features**: `xmeas_1`–`xmeas_41`, `xmv_1`–`xmv_11` (all float32)
- **Target**: `faultNumber` — 0 (normal) + 1–21 (fault types) = 22 classes
- **Fault injection**: sample 160; only post-injection rows used for training
- **Train/test split**: `f_train.parquet` → train, `f_test.parquet` → test (natural split)

## Environment

- Python 3 in Jupyter notebook (`main.ipynb`), kernel: `ml_embedded`
- Key libraries: `lightgbm`, `pyarrow`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- Data path: `../datasets/TEP/Harvard/`

## Memory-Efficient Loading

Large parquet files are never loaded fully. Load post-injection rows only:
- `f_train`: fault-by-fault loop with `filters=[("faultNumber", "==", fnum), ("sample", ">", 160)]`
- `f_test`: single read with `filters=[("sample", ">", 160)]` + column projection

## Notes
- Normal operation (faultNumber == 0) comes from ff_train / ff_test, not f_train / f_test.
- Include normal samples in training to make it a 22-class problem (0 = normal, 1–21 = faults).
- LightGBM label requirement: classes must be 0-indexed integers — map faultNumber to int.
