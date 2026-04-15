import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from federated_acn.metrics import compute_regression_metrics, summarize_target
from federated_acn.ml.dataset import (
    EarlySessionDataset,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TabularPreprocessor,
)
from federated_acn.ml.dcn import DeepCrossRegressor
from federated_acn.ml.cnn1d import TabularCNN1D
from federated_acn.ml.features import (
    CYCLICAL_FEATURE_COLS,
    apply_cyclical_time_features,
)
from federated_acn.ml.gru import TabularGRU
from federated_acn.ml.losses import physics_constrained_loss
from federated_acn.ml.mlp import TabularMLP, TabularResMLP
from federated_acn.ml.target_transform import (
    inverse_transform_target_numpy,
    inverse_transform_target_tensor,
)
from federated_acn.ml.transformer import TabularTransformer

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency for model selection
    xgb = None


def get_loss_criterion(loss_name: str) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()
    return train_df, val_df, test_df


def filter_caltech_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["site_name"].astype(str).str.lower() == "caltech"].copy()


def make_loaders(
    df: pd.DataFrame,
    batch_size: int,
    target_log1p: bool,
    preprocessing: str,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    TabularPreprocessor,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    train_df, val_df, test_df = train_val_test_split(df)

    preprocessor = TabularPreprocessor(
        numerical_cols=[c for c in NUMERICAL_COLS if c in df.columns],
        categorical_cols=[c for c in CATEGORICAL_COLS if c in df.columns],
        passthrough_numerical_cols=[
            c for c in CYCLICAL_FEATURE_COLS if c in df.columns
        ],
        preprocessing=preprocessing,
    )
    preprocessor.fit(train_df)

    train_ds = EarlySessionDataset(train_df, preprocessor, target_log1p=target_log1p)
    val_ds = EarlySessionDataset(val_df, preprocessor, target_log1p=target_log1p)
    test_ds = EarlySessionDataset(test_df, preprocessor, target_log1p=target_log1p)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        preprocessor,
        train_df,
        val_df,
        test_df,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_log1p: bool,
    lambda_early: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_base_loss = 0.0
    total_loss_early = 0.0
    y_true_batches = []
    y_pred_batches = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y = batch["y"]

        if training:
            optimizer.zero_grad()

        pred = model(batch)
        loss, base_loss, loss_early = physics_constrained_loss(
            pred=pred,
            target=y,
            base_loss_fn=criterion,
            early_energy_kwh=batch["early_energy_kwh"],
            target_log1p=target_log1p,
            lambda_early=lambda_early,
        )

        if training:
            loss.backward()
            optimizer.step()

        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_base_loss += float(base_loss.item()) * bs
        total_loss_early += float(loss_early.item()) * bs
        y_true_batches.append(y.detach().cpu().numpy())
        y_pred_batches.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)
    y_true_eval = inverse_transform_target_numpy(y_true, use_log1p=target_log1p)
    y_pred_eval = inverse_transform_target_numpy(y_pred, use_log1p=target_log1p)
    metrics = compute_regression_metrics(y_true=y_true_eval, y_pred=y_pred_eval)
    metrics["num_samples"] = int(y_true_eval.shape[0])
    metrics["target_min"] = float(np.min(y_true_eval))
    metrics["target_max"] = float(np.max(y_true_eval))
    metrics["target_mean"] = float(np.mean(y_true_eval))
    metrics["target_std"] = float(np.std(y_true_eval))
    metrics["avg_batch_loss"] = total_loss / max(y_true_eval.shape[0], 1)
    metrics["loss"] = metrics["avg_batch_loss"]
    metrics["base_loss"] = total_base_loss / max(y_true_eval.shape[0], 1)
    metrics["loss_early"] = total_loss_early / max(y_true_eval.shape[0], 1)
    return metrics


def build_model(
    model_name: str,
    num_numerical_features: int,
    categorical_cardinalities: Dict[str, int],
    use_embeddings: bool,
) -> nn.Module:
    model_categorical_cardinalities = (
        categorical_cardinalities if use_embeddings else {}
    )

    if model_name == "mlp":
        return TabularMLP(
            num_numerical_features=num_numerical_features,
            categorical_cardinalities=model_categorical_cardinalities,
            hidden_dim=128,
            dropout=0.2,
        )
    if model_name == "res-mlp":
        return TabularResMLP(
            num_numerical_features=num_numerical_features,
            categorical_cardinalities=model_categorical_cardinalities,
            hidden_dim=128,
            dropout=0.2,
            num_blocks=2,
        )
    if model_name == "dcn":
        return DeepCrossRegressor(
            num_numerical_features=num_numerical_features,
            categorical_cardinalities=model_categorical_cardinalities,
            cross_layers=3,
            deep_hidden_dims=[128, 64],
            dropout=0.1,
            positive_output=True,
        )
    if model_name == "transformer":
        return TabularTransformer(
            num_numerical_features=num_numerical_features,
            categorical_cardinalities=model_categorical_cardinalities,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.1,
        )
    if model_name == "cnn1d":
        return TabularCNN1D(
            num_numerical_features=num_numerical_features,
            categorical_cardinalities=model_categorical_cardinalities,
            token_dim=16,
            hidden_channels=32,
            dropout=0.1,
        )
    if model_name == "gru":
        return TabularGRU(
            num_numerical_features=num_numerical_features,
            categorical_cardinalities=model_categorical_cardinalities,
            token_dim=16,
            hidden_dim=32,
            dropout=0.1,
        )
    raise ValueError(f"Unknown model_name={model_name}")


def build_sklearn_features(
    df: pd.DataFrame,
    preprocessor: TabularPreprocessor,
) -> np.ndarray:
    x_num = preprocessor.transform_numerical(df)
    x_cat = preprocessor.transform_categorical(df)
    cat_arrays = [x_cat[col].astype(np.float32) for col in preprocessor.categorical_cols]
    if cat_arrays:
        x_cat_arr = np.stack(cat_arrays, axis=1)
        return np.concatenate([x_num, x_cat_arr], axis=1)
    return x_num


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=[
            "mlp",
            "res-mlp",
            "transformer",
            "cnn1d",
            "gru",
            "dcn",
            "linear_regression",
            "random_forest",
            "xgboost",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"])
    parser.add_argument("--lambda-early", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--embedding",
        dest="use_embeddings",
        action="store_true",
        help="Enable categorical embeddings for station/cluster features.",
    )
    parser.add_argument(
        "--no-embedding",
        dest="use_embeddings",
        action="store_false",
        help="Disable categorical embeddings (default).",
    )
    parser.set_defaults(use_embeddings=True)
    parser.add_argument(
        "--cyclical",
        dest="cyclical",
        action="store_true",
        help="Enable cyclical (sin/cos) encoding for date-derived features (default).",
    )
    parser.add_argument(
        "--no-cyclical",
        dest="cyclical",
        action="store_false",
        help="Disable cyclical encoding and keep raw date-derived integer features.",
    )
    parser.set_defaults(cyclical=True)
    parser.add_argument(
        "--target-log1p",
        dest="target_log1p",
        action="store_true",
        help="Train on log1p(target) and inverse-transform predictions for evaluation metrics.",
    )
    parser.add_argument(
        "--no-target-log1p",
        dest="target_log1p",
        action="store_false",
        help="Train/evaluate directly on the raw target (default).",
    )
    parser.set_defaults(target_log1p=False)
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-n-estimators", type=int, default=400)
    parser.add_argument(
        "--preprocessing",
        type=str,
        default="standardization",
        choices=["standardization", "minmax", "none"],
        help="Numerical preprocessing for non-cyclical features.",
    )
    parser.add_argument("--outdir", type=str, default="./outputs/centralized")
    args = parser.parse_args()

    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path).expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset parquet not found: {data_path}")

    df = pd.read_parquet(data_path)
    df = filter_caltech_only(df)
    df = apply_cyclical_time_features(df, cyclical=args.cyclical)

    print(f"Loaded rows (Caltech only): {len(df):,}")
    print("First preprocessed row (centralized):")
    print(df.head(1).to_string(index=False))

    print(f"Loaded rows (Caltech only): {len(df):,}")

    train_loader, val_loader, test_loader, preprocessor, train_df, val_df, test_df = (
        make_loaders(
            df,
            batch_size=args.batch_size,
            target_log1p=args.target_log1p,
            preprocessing=args.preprocessing,
        )
    )
    print("Split and target summary:")
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        split_summary = summarize_target(split_df)
        print(
            f"  {split_name:>5}: n={split_summary['num_samples']:,}, "
            f"min={split_summary['target_min']:.4f}, max={split_summary['target_max']:.4f}, "
            f"mean={split_summary['target_mean']:.4f}, std={split_summary['target_std']:.4f}"
        )

    categorical_cardinalities = preprocessor.categorical_cardinalities()
    num_numerical_features = len(preprocessor.numerical_cols)

    if args.model in {"linear_regression", "random_forest", "xgboost"}:
        x_train = build_sklearn_features(train_df, preprocessor)
        x_val = build_sklearn_features(val_df, preprocessor)
        x_test = build_sklearn_features(test_df, preprocessor)
        y_train = train_df["kwh_delivered"].to_numpy(dtype=np.float32)
        y_val = val_df["kwh_delivered"].to_numpy(dtype=np.float32)
        y_test = test_df["kwh_delivered"].to_numpy(dtype=np.float32)

        if args.model == "linear_regression":
            model = LinearRegression()
        elif args.model == "random_forest":
            model = RandomForestRegressor(
                n_estimators=args.rf_n_estimators,
                criterion="absolute_error" if args.loss == "mae" else "squared_error",
                random_state=args.seed,
                n_jobs=-1,
            )
        else:
            if xgb is None:
                raise ImportError(
                    "xgboost is not installed. Install project dependencies including xgboost."
                )
            model = xgb.XGBRegressor(
                n_estimators=args.xgb_n_estimators,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:absoluteerror" if args.loss == "mae" else "reg:squarederror",
                eval_metric="mae" if args.loss == "mae" else "rmse",
                random_state=args.seed,
                n_jobs=4,
            )

        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        test_pred = model.predict(x_test)
        train_pred = np.clip(train_pred, a_min=0.0, a_max=None)
        val_pred = np.clip(val_pred, a_min=0.0, a_max=None)
        test_pred = np.clip(test_pred, a_min=0.0, a_max=None)

        train_metrics = compute_regression_metrics(y_true=y_train, y_pred=train_pred)
        val_metrics = compute_regression_metrics(y_true=y_val, y_pred=val_pred)
        test_metrics = compute_regression_metrics(y_true=y_test, y_pred=test_pred)
        best_val_rmse = float(val_metrics["rmse"])
        best_epoch = 1
        history = [
            {
                "epoch": 1,
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "train_mape": train_metrics["mape"],
                "train_mase": train_metrics["mase"],
                "train_r2": train_metrics["r2"],
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_mape": val_metrics["mape"],
                "val_mase": val_metrics["mase"],
                "val_r2": val_metrics["r2"],
                "preprocessing": args.preprocessing,
            }
        ]
        embedding_tag = "embedding" if args.use_embeddings else "no_embedding"
        cyclical_tag = "cyclical" if args.cyclical else "no_cyclical"
        target_tag = "target_log1p" if args.target_log1p else "target_raw"
        target_tag = "target_log1p" if args.target_log1p else "target_raw"
        preprocessing_tag = f"preproc_{args.preprocessing}"
        model_tag = (
            f"{args.model}_{embedding_tag}_{cyclical_tag}_{target_tag}_{preprocessing_tag}_"
            f"{args.loss}_{args.seed}_lambdaearly_{args.lambda_early:g}"
        )
        pd.DataFrame(history).to_csv(outdir / f"history_{model_tag}.csv", index=False)
        with open(outdir / f"metrics_{model_tag}.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": args.model,
                    "use_embeddings": bool(args.use_embeddings),
                    "cyclical": bool(args.cyclical),
                    "target_log1p": bool(args.target_log1p),
                    "preprocessing": args.preprocessing,
                    "seed": int(args.seed),
                    "loss": args.loss,
                    "lambda_early": args.lambda_early,
                    "best_val_rmse": best_val_rmse,
                    "best_epoch": best_epoch,
                    "splits": {
                        "train": train_metrics,
                        "val": val_metrics,
                        "test": test_metrics,
                    },
                    "test": test_metrics,
                    "model": args.model,
                    "num_rows_caltech": int(len(df)),
                    "num_numerical_features": num_numerical_features,
                    "categorical_cardinalities": categorical_cardinalities,
                },
                f,
                indent=2,
            )
        print("\nBest validation RMSE:", best_val_rmse)
        print("Test metrics:", test_metrics)
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name=args.model,
        num_numerical_features=num_numerical_features,
        categorical_cardinalities=categorical_cardinalities,
        use_embeddings=args.use_embeddings,
    ).to(device)

    criterion = get_loss_criterion(args.loss)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_rmse = float("inf")
    best_epoch = 0
    embedding_tag = "embedding" if args.use_embeddings else "no_embedding"
    cyclical_tag = "cyclical" if args.cyclical else "no_cyclical"
    target_tag = "target_log1p" if args.target_log1p else "target_raw"
    preprocessing_tag = f"preproc_{args.preprocessing}"
    model_tag = (
        f"{args.model}_{embedding_tag}_{cyclical_tag}_{target_tag}_{preprocessing_tag}_"
        f"{args.loss}_{args.seed}_lambdaearly_{args.lambda_early:g}"
    )
    best_path = outdir / f"best_{model_tag}.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            target_log1p=args.target_log1p,
            lambda_early=args.lambda_early,
            optimizer=optimizer,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            target_log1p=args.target_log1p,
            lambda_early=args.lambda_early,
            optimizer=None,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_base_loss": train_metrics["base_loss"],
            "train_loss_early": train_metrics["loss_early"],
            "train_rmse": train_metrics["rmse"],
            "train_mape": train_metrics["mape"],
            "train_mase": train_metrics["mase"],
            "train_r2": train_metrics["r2"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_base_loss": val_metrics["base_loss"],
            "val_loss_early": val_metrics["loss_early"],
            "val_rmse": val_metrics["rmse"],
            "val_mape": val_metrics["mape"],
            "val_mase": val_metrics["mase"],
            "val_r2": val_metrics["r2"],
            "preprocessing": args.preprocessing,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_rmse={train_metrics['rmse']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | "
            f"val_mae={val_metrics['mae']:.4f} | "
            f"val_mape={val_metrics['mape']:.4f} | "
            f"val_r2={val_metrics['r2']:.4f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    train_metrics = run_epoch(
        model,
        train_loader,
        criterion,
        device,
        target_log1p=args.target_log1p,
        lambda_early=args.lambda_early,
        optimizer=None,
    )
    val_metrics = run_epoch(
        model,
        val_loader,
        criterion,
        device,
        target_log1p=args.target_log1p,
        lambda_early=args.lambda_early,
        optimizer=None,
    )
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        target_log1p=args.target_log1p,
        lambda_early=args.lambda_early,
        optimizer=None,
    )

    print("\nBest validation RMSE:", best_val_rmse)
    print("Test metrics:", test_metrics)

    pd.DataFrame(history).to_csv(outdir / f"history_{model_tag}.csv", index=False)

    with open(outdir / f"metrics_{model_tag}.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model,
                "use_embeddings": bool(args.use_embeddings),
                "cyclical": bool(args.cyclical),
                "target_log1p": bool(args.target_log1p),
                "preprocessing": args.preprocessing,
                "seed": int(args.seed),
                "loss": args.loss,
                "lambda_early": args.lambda_early,
                "best_val_rmse": best_val_rmse,
                "best_epoch": best_epoch,
                "splits": {
                    "train": train_metrics,
                    "val": val_metrics,
                    "test": test_metrics,
                },
                "test": test_metrics,
                "model": args.model,
                "num_rows_caltech": int(len(df)),
                "num_numerical_features": num_numerical_features,
                "categorical_cardinalities": categorical_cardinalities,
            },
            f,
            indent=2,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
