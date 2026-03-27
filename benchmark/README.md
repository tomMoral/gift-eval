# GIFT-Eval Benchopt Benchmark

This directory contains a [benchopt](https://benchopt.github.io/) benchmark
for evaluating time series foundation models on the
[GIFT-Eval](https://github.com/SalesforceAIResearch/gift-eval) benchmark.

## Overview

| Component | Description |
|-----------|-------------|
| `objective.py` | Computes the full suite of GIFT-Eval metrics via GluonTS |
| `datasets/gift_eval.py` | Loads GIFT-Eval datasets using `gift_eval.data.Dataset` |
| `solvers/chronos2.py` | Amazon Chronos-2 zero-shot forecaster |
| `solvers/timesfm2p5.py` | Google TimesFM-2.5 (200 M) zero-shot forecaster |

The default dataset is **`m4_weekly / short`** – the smallest dataset in
GIFT-Eval (359 weekly univariate series, prediction length = 8).

## Prerequisites

### 1. GIFT-Eval data

Download the dataset and set the environment variable:

```bash
huggingface-cli download Salesforce/GiftEval \
    --repo-type=dataset --local-dir /path/to/data

echo "GIFT_EVAL=/path/to/data" >> .env   # repo root .env is loaded automatically
```

### 2. Python environment

```bash
# Core dependencies (from the repo root)
pip install -e .

# Chronos-2
pip install "chronos-forecasting>=2.1"

# TimesFM-2.5 (requires Python ≥ 3.11)
git clone https://github.com/google-research/timesfm.git
cd timesfm && pip install -e . && cd ..
```

### 3. benchopt

```bash
pip install benchopt
```

## Running the benchmark

```bash
# From the repo root, run both solvers on the default dataset (m4_weekly/short)
benchopt run benchmark/

# Run only one solver
benchopt run benchmark/ -s Chronos-2

# Change dataset (e.g. hospital with short term)
benchopt run benchmark/ --config benchmark/config.yml
```

## Structure

```
benchmark/
├── objective.py          # Objective: GIFT-Eval metrics via evaluate_forecasts
├── datasets/
│   └── gift_eval.py      # Dataset: wraps gift_eval.data.Dataset
└── solvers/
    ├── chronos2.py       # Solver: Chronos-2 pipeline
    └── timesfm2p5.py     # Solver: TimesFM-2.5 200M
```

## Adding new datasets

Edit `datasets/gift_eval.py` and extend the `parameters` dict:

```python
parameters = {
    "dataset_name": ["m4_weekly", "hospital", "electricity/H"],
    "term": ["short"],
}
```

For datasets not yet in `_DATASET_PROPERTIES`, add an entry there as well.

## Adding new solvers

Create a new file under `solvers/` that subclasses `benchopt.BaseSolver`
with `sampling_strategy = "run_once"`.  The solver receives
`gift_eval_dataset` (a `gift_eval.data.Dataset` instance) via
`set_objective` and must return `{"forecasts": List[gluonts.model.Forecast]}`
from `get_result`.

## Metrics

The objective reports all standard GIFT-Eval metrics:
`MASE[0.5]` (primary), `MSE[mean]`, `MSE[0.5]`, `MAE[0.5]`, `MAPE[0.5]`,
`sMAPE[0.5]`, `MSIS`, `RMSE[mean]`, `NRMSE[mean]`, `ND[0.5]`,
`mean_weighted_sum_quantile_loss`.
