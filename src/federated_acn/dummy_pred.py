import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from federated_acn.fl.task import (
    build_client_id_columns,
    get_client_ids,
    load_dataframe,
    split_df,
)
from federated_acn.metrics import compute_regression_metrics, summarize_target
from federated_acn.ml.dataset import TARGET_COL


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def gaussian_dummy_predictions(
    num_samples: int,
    mean: float,
    std: float,
    seed: int,
) -> np.ndarray:
    effective_std = max(float(std), 1e-8)
    rng = np.random.default_rng(seed)
    pred = rng.normal(loc=float(mean), scale=effective_std, size=num_samples)
    return np.clip(pred, a_min=0.0, a_max=None)


def evaluate_split(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = compute_regression_metrics(y_true=y_true, y_pred=y_pred)
    metrics["num_samples"] = int(y_true.shape[0])
    return metrics


def evaluate_centralized(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> Dict[str, Any]:
    y_train = train_df[TARGET_COL].to_numpy(dtype=np.float64)
    y_val = val_df[TARGET_COL].to_numpy(dtype=np.float64)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.float64)

    train_mean = float(np.mean(y_train))
    train_std = float(np.std(y_train))

    mean_only = {
        "train": np.full_like(y_train, fill_value=train_mean, dtype=np.float64),
        "val": np.full_like(y_val, fill_value=train_mean, dtype=np.float64),
        "test": np.full_like(y_test, fill_value=train_mean, dtype=np.float64),
    }

    gaussian = {
        "train": gaussian_dummy_predictions(len(y_train), train_mean, train_std, seed=seed),
        "val": gaussian_dummy_predictions(len(y_val), train_mean, train_std, seed=seed + 1),
        "test": gaussian_dummy_predictions(len(y_test), train_mean, train_std, seed=seed + 2),
    }

    return {
        "setup": {
            "mode": "centralized",
            "train_target_mean": train_mean,
            "train_target_std": train_std,
            "train_target_summary": summarize_target(train_df),
            "val_target_summary": summarize_target(val_df),
            "test_target_summary": summarize_target(test_df),
        },
        "predictors": {
            "mean_only": {
                "train": evaluate_split(y_train, mean_only["train"]),
                "val": evaluate_split(y_val, mean_only["val"]),
                "test": evaluate_split(y_test, mean_only["test"]),
            },
            "mean_std_gaussian": {
                "train": evaluate_split(y_train, gaussian["train"]),
                "val": evaluate_split(y_val, gaussian["val"]),
                "test": evaluate_split(y_test, gaussian["test"]),
            },
        },
    }


def evaluate_federated(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    partition_by: str,
    min_total_sessions: int,
) -> Dict[str, Any]:
    if partition_by != "station":
        raise ValueError(f"Unsupported partition_by={partition_by}; only 'station' is supported.")
    train_df = build_client_id_columns(train_df)
    client_col = "client_station"
    client_ids = get_client_ids(
        train_df,
        partition_by=partition_by,
        min_total_sessions=min_total_sessions,
    )
    if not client_ids:
        raise ValueError(
            "No eligible clients found. Try lowering --min-total-sessions."
        )

    y_train = train_df[TARGET_COL].to_numpy(dtype=np.float64)
    y_val = val_df[TARGET_COL].to_numpy(dtype=np.float64)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.float64)

    mean_only_train_preds = []
    mean_only_val_preds = []
    mean_only_test_preds = []

    gaussian_train_preds = []
    gaussian_val_preds = []
    gaussian_test_preds = []

    client_summaries = []

    for idx, client_id in enumerate(client_ids):
        client_rows = train_df[train_df[client_col].astype(str) == client_id]
        y_client = client_rows[TARGET_COL].to_numpy(dtype=np.float64)
        client_mean = float(np.mean(y_client))
        client_std = float(np.std(y_client))

        mean_only_train_preds.append(np.full_like(y_train, fill_value=client_mean, dtype=np.float64))
        mean_only_val_preds.append(np.full_like(y_val, fill_value=client_mean, dtype=np.float64))
        mean_only_test_preds.append(np.full_like(y_test, fill_value=client_mean, dtype=np.float64))

        gaussian_train_preds.append(
            gaussian_dummy_predictions(len(y_train), client_mean, client_std, seed=seed + idx * 13)
        )
        gaussian_val_preds.append(
            gaussian_dummy_predictions(len(y_val), client_mean, client_std, seed=seed + idx * 13 + 1)
        )
        gaussian_test_preds.append(
            gaussian_dummy_predictions(len(y_test), client_mean, client_std, seed=seed + idx * 13 + 2)
        )

        client_summaries.append(
            {
                "client_id": client_id,
                "num_train_samples": int(len(y_client)),
                "target_mean": client_mean,
                "target_std": client_std,
            }
        )

    mean_only = {
        "train": np.mean(np.stack(mean_only_train_preds), axis=0),
        "val": np.mean(np.stack(mean_only_val_preds), axis=0),
        "test": np.mean(np.stack(mean_only_test_preds), axis=0),
    }
    gaussian = {
        "train": np.mean(np.stack(gaussian_train_preds), axis=0),
        "val": np.mean(np.stack(gaussian_val_preds), axis=0),
        "test": np.mean(np.stack(gaussian_test_preds), axis=0),
    }

    return {
        "setup": {
            "mode": "federated",
            "partition_by": partition_by,
            "min_total_sessions": int(min_total_sessions),
            "num_clients": int(len(client_ids)),
            "train_target_summary": summarize_target(train_df),
            "val_target_summary": summarize_target(val_df),
            "test_target_summary": summarize_target(test_df),
            "clients": client_summaries,
        },
        "predictors": {
            "mean_only": {
                "train": evaluate_split(y_train, mean_only["train"]),
                "val": evaluate_split(y_val, mean_only["val"]),
                "test": evaluate_split(y_test, mean_only["test"]),
            },
            "mean_std_gaussian": {
                "train": evaluate_split(y_train, gaussian["train"]),
                "val": evaluate_split(y_val, gaussian["val"]),
                "test": evaluate_split(y_test, gaussian["test"]),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dummy predictors for ACN data. Supports centralized mode and federated mode "
            "(client-level dummy predictors averaged into global predictions)."
        )
    )
    parser.add_argument("--data-path", type=str, default="./acn_fl_data/processed/dataset_early_session_caltech.parquet")
    parser.add_argument("--site-name", type=str, default="caltech")
    parser.add_argument("--federated", type=str2bool, default=False)
    parser.add_argument("--min-total-sessions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cyclical", type=str2bool, default=True)
    parser.add_argument("--outdir", type=str, default="outputs/centralized")
    parser.add_argument("--output-name", type=str, default="metrics_dummy.json")
    args = parser.parse_args()

    df = load_dataframe(data_path=args.data_path, site_name=args.site_name, cyclical=args.cyclical)
    train_df, val_df, test_df = split_df(df, seed=args.seed)

    if args.federated:
        results = evaluate_federated(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            seed=args.seed,
            partition_by="station",
            min_total_sessions=args.min_total_sessions,
        )
    else:
        results = evaluate_centralized(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            seed=args.seed,
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / args.output_name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved dummy predictor metrics to: {output_path}")


if __name__ == "__main__":
    main()
