#!/usr/bin/env python3
"""Profile centralized model resource usage (CPU/GPU memory and artifact size)."""
import argparse
import json
import pickle
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from federated_acn.ml.features import apply_cyclical_time_features
from federated_acn.ml.train_centralized import (
    build_model,
    build_sklearn_features,
    filter_caltech_only,
    get_loss_criterion,
    make_loaders,
)

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency for model selection
    xgb = None


MODELS = [
    "linear_regression",
    "random_forest",
    "xgboost",
    "mlp",
    "res-mlp",
    "dcn",
    "cnn1d",
    "gru",
    "transformer",
]


@dataclass
class PeakMemory:
    cpu_peak_bytes: int
    gpu_peak_bytes: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MemoryPoller:
    """Background RSS poller used to track CPU peak during long-running sections."""

    def __init__(self, interval_sec: float = 0.01) -> None:
        self.process = psutil.Process()
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_rss = self.process.memory_info().rss

    def _run(self) -> None:
        while not self._stop_event.is_set():
            rss = self.process.memory_info().rss
            if rss > self._peak_rss:
                self._peak_rss = rss
            self._stop_event.wait(self.interval_sec)

    def start(self) -> None:
        self._stop_event.clear()
        self._peak_rss = self.process.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._peak_rss = max(self._peak_rss, self.process.memory_info().rss)
        return int(self._peak_rss)


def measure_peak_memory(operation: Callable[[], None], device: torch.device) -> PeakMemory:
    poller = MemoryPoller()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    poller.start()
    operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    cpu_peak = poller.stop()
    gpu_peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    return PeakMemory(cpu_peak_bytes=cpu_peak, gpu_peak_bytes=gpu_peak)


def run_torch_train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    model.train(True)
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch["y"]
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()


def run_torch_inference_batch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(batch)
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile model resource usage for all centralized model names."
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="./outputs/resource_profile")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="Execution device: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        default="standardization",
        choices=["standardization", "minmax", "none"],
    )
    parser.add_argument(
        "--cyclical",
        dest="cyclical",
        action="store_true",
        help="Enable cyclical date-derived features (default).",
    )
    parser.add_argument(
        "--no-cyclical",
        dest="cyclical",
        action="store_false",
        help="Disable cyclical date-derived features.",
    )
    parser.set_defaults(cyclical=True)
    parser.add_argument(
        "--embedding",
        dest="use_embeddings",
        action="store_true",
        help="Enable categorical embeddings (default).",
    )
    parser.add_argument(
        "--no-embedding",
        dest="use_embeddings",
        action="store_false",
        help="Disable categorical embeddings.",
    )
    parser.set_defaults(use_embeddings=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {data_path}")

    outdir = Path(args.output_dir).expanduser().resolve()
    artifacts_dir = outdir / "artifacts"
    outdir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    df = filter_caltech_only(df)
    df = apply_cyclical_time_features(df, cyclical=args.cyclical)

    train_loader, val_loader, test_loader, preprocessor, train_df, val_df, test_df = (
        make_loaders(
            df,
            batch_size=args.batch_size,
            target_log1p=False,
            preprocessing=args.preprocessing,
        )
    )

    n_rows = int(len(df))
    categorical_cardinalities = preprocessor.categorical_cardinalities()
    num_numerical_features = len(preprocessor.numerical_cols)
    n_features = int(
        num_numerical_features
        + (len(categorical_cardinalities) if args.use_embeddings else 0)
    )

    x_train = build_sklearn_features(train_df, preprocessor)
    x_test = build_sklearn_features(test_df, preprocessor)
    y_train = train_df["kwh_delivered"].to_numpy(dtype=np.float32)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = get_loss_criterion("mae")

    rows: list[dict[str, int | str]] = []

    for model_name in MODELS:
        print(f"\nProfiling model: {model_name}")

        if model_name in {"linear_regression", "random_forest", "xgboost"}:
            if model_name == "linear_regression":
                model = LinearRegression()
            elif model_name == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=300,
                    criterion="absolute_error",
                    random_state=args.seed,
                    n_jobs=-1,
                )
            else:
                if xgb is None:
                    raise ImportError(
                        "xgboost is not installed; install it to profile xgboost model."
                    )
                model = xgb.XGBRegressor(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:absoluteerror",
                    eval_metric="mae",
                    random_state=args.seed,
                    n_jobs=4,
                )

            train_peak = measure_peak_memory(lambda: model.fit(x_train, y_train), device)

            inference_x = x_test[: args.batch_size]
            infer_peak = measure_peak_memory(lambda: model.predict(inference_x), device)

            artifact_path = artifacts_dir / f"{model_name}.pkl"
            with open(artifact_path, "wb") as f:
                pickle.dump(model, f)
        else:
            model = build_model(
                model_name=model_name,
                num_numerical_features=num_numerical_features,
                categorical_cardinalities=categorical_cardinalities,
                use_embeddings=args.use_embeddings,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

            def train_op() -> None:
                for _ in range(args.epochs):
                    run_torch_train_epoch(
                        model=model,
                        train_loader=train_loader,
                        criterion=criterion,
                        optimizer=optimizer,
                        device=device,
                    )

            train_peak = measure_peak_memory(train_op, device)
            infer_peak = measure_peak_memory(
                lambda: run_torch_inference_batch(model, test_loader, device),
                device,
            )

            artifact_path = artifacts_dir / f"{model_name}.pt"
            torch.save(model.state_dict(), artifact_path)

        artifact_bytes = int(artifact_path.stat().st_size)
        for phase, peak in (("train", train_peak), ("infer", infer_peak)):
            rows.append(
                {
                    "model": model_name,
                    "phase": phase,
                    "peak_cpu_bytes": int(peak.cpu_peak_bytes),
                    "peak_gpu_bytes": int(peak.gpu_peak_bytes),
                    "artifact_bytes": artifact_bytes,
                    "batch_size": int(args.batch_size),
                    "seed": int(args.seed),
                    "n_features": n_features,
                    "n_rows": n_rows,
                }
            )

    summary_df = pd.DataFrame(rows)
    csv_path = outdir / "resource_summary.csv"
    json_path = outdir / "resource_summary.json"
    summary_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nWrote summary CSV:  {csv_path}")
    print(f"Wrote summary JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
