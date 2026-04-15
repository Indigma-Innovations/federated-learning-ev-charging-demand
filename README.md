# Federated Learning for EV Fleets
This repository is the **companion codebase** for the study by Indigma on **early EV charging demand prediction** with **federated learning (FL)**.

In short, the paper investigates how accurately we can estimate a charging session's final delivered energy using only:
- metadata available at plug-in time, and
- the first minutes of charging behavior.

The workflow focuses on ACN Caltech sessions, engineers tabular early-window features (user intent proxies, temporal context, and initial charging signals), and compares centralized versus federated model training. The key finding is that federated training can approach centralized predictive performance while keeping raw data local to each operational partition (i.e., station-level clients), making FL a practical privacy-aware option for distributed EV charging analytics.

## Setup
This project uses the `uv` package manager.

If you don't have uv installed, install it first:
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Or, if you are using Windows:
```bash
irm https://astral.sh/uv/install.ps1 | iex
```

After installing uv, install the project dependencies with:
```bash
uv sync
```

## Data download and preparation

This project uses the official [ACN-Data](https://ev.caltech.edu/dataset) API.

Before downloading the dataset, you need to:

1. Register on the official ACN-Data website.
2. Obtain your API token.
3. Add the token to your `.env` file:

```env
ACN_TOKEN=your-token-here
```

To download the ACN dataset, run:
```bash
uv run python -m federated_acn.data_dl.download_acn_data --outdir ./acn_fl_data
```

If the download is interrupted, the script can resume from saved progress.

### Early session dataset

This step transforms raw ACN data into an ML-ready dataset where:
- Each row is one charging session
- Input is early-session behavior (first N minutes)
- Output is total delivered energy

To build the dataset:

```bash
uv run python -m federated_acn.data_dl.build_early_session_dataset --data-dir ./acn_fl_data --early-window-min 10 --site caltech
```

### Produced files

- `raw/sessions/*.jsonl`
- `raw/timeseries/*.jsonl`
- `processed/dataset_early_session_{site}.parquet`
- `processed/dataset_early_session_{site}.csv`

## Reproducibility

- A centralized multi-seed launcher exists at `scripts/train_centralized_multiseed.py`
- Federated multi-seed launcher exists at `scripts/train_federated_multiseed.py`.

## Training

### Centralized baseline

```bash
uv run python -m federated_acn.ml.train_centralized \
  --data-path ./acn_fl_data/processed/dataset_early_session_caltech.parquet \
  --model mlp \
  --embedding \
  --loss mae \
  --epochs 40 \
  --batch-size 128 \
  --outdir ./outputs/centralized
```

### Federated simulation

```bash
uv run python -m federated_acn.fl.run_simulation \
  --data-path ./acn_fl_data/processed/dataset_early_session_caltech.parquet \
  --model-name gru \
  --embedding \
  --loss mae \
  --batch-size 128 \
  --local-epochs 3 \
  --num-server-rounds 40 \
  --learning-rate 1e-3 \
  --fraction-train 0.2 \
  --fraction-evaluate 1.0
```

## Federated EDA (station-level)

Generate exploratory analysis for station-level clients, including:

- Number of clients
- Number of samples per client
- Per-client `mean`, `min`, `max`, `std` for the target
- Distribution plots for these statistics
- Non-IID decision based on weighted Jensen-Shannon divergence

```bash
uv run python -m federated_acn.fl_eda \
  --data-path ./acn_fl_data/processed/dataset_early_session_caltech.parquet \
  --outdir ./outputs/federated_eda
```

Artifacts are generated in:

- `outputs/federated_eda/station/*`

## Supported Models
- `mlp` (tabular MLP, uses embeddings when enabled)
- `transformer` (tabular Transformer, uses embeddings when enabled)
- `cnn1d` (lightweight 1D CNN over feature tokens)
- `gru` (lightweight GRU over feature tokens)
- `linear_regression` (scikit-learn LinearRegression)
- `xgboost` (XGBoost regressor, requires `xgboost` dependency)

### Additional models not reported in the paper
- `random_forest` (scikit-learn RandomForestRegressor)
- `res-mlp` (tabular residual MLP)
- `dcn` (Deep & Cross Network for tabular regression)

## Resource profiling

Use the resource profiling script to compare memory usage and model artifact sizes
across all centralized model variants:

```bash
uv run python scripts/profile_model_resources.py \
  --data-path ./acn_fl_data/processed/dataset_early_session_caltech.parquet \
  --batch-size 128 \
  --seed 42 \
  --output-dir ./outputs/resource_profile
```

What it does:

- Profiles models: `linear_regression`, `random_forest`, `xgboost`, `mlp`,
  `res-mlp`, `dcn`, `cnn1d`, `gru`, `transformer`.
- Reuses centralized preprocessing and split logic.
- Measures peak CPU RAM (RSS) and peak GPU RAM (if CUDA is available) for both:
  - `train` phase
  - `infer` phase (fixed batch pass)
- Saves trained artifacts to `outputs/resource_profile/artifacts/`.
- Writes summary files:
  - `outputs/resource_profile/resource_summary.csv`
  - `outputs/resource_profile/resource_summary.json`

Summary fields:

- `model`, `phase`, `peak_cpu_bytes`, `peak_gpu_bytes`, `artifact_bytes`,
  `batch_size`, `seed`, `n_features`, `n_rows`

### Centralized vs Federated applicability

This script is **for centralized profiling**. It does **not** run Flower
federated rounds or include federated-system costs (client/server orchestration,
network transfer, serialization, per-client heterogeneity, etc.).

You can still use the outputs as a rough per-model baseline for expected local
train/inference memory footprint on a single process/device, but they are not a
full federated resource profile.

## License

This project is licensed under the Apache-2.0 License. See the [`LICENSE`](./LICENSE) file for details.
