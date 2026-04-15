from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import xgboost as xgb

from federated_acn.fl.config import CFG
from federated_acn.metrics import compute_regression_metrics, summarize_target
from federated_acn.ml.dataset import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    EarlySessionDataset,
    TabularPreprocessor,
)
from federated_acn.ml.cnn1d import TabularCNN1D
from federated_acn.ml.dcn import DeepCrossRegressor
from federated_acn.ml.features import (
    CYCLICAL_FEATURE_COLS,
    apply_cyclical_time_features,
)
from federated_acn.ml.gru import TabularGRU
from federated_acn.ml.losses import physics_constrained_loss
from federated_acn.ml.mlp import TabularMLP, TabularResMLP
from federated_acn.ml.target_transform import (
    inverse_transform_target_numpy,
)
from federated_acn.ml.transformer import TabularTransformer

def get_loss_criterion(loss_name: str) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "mae":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def resolve_data_path(raw_data_path: str) -> Path:
    candidate = Path(raw_data_path).expanduser()
    if candidate.is_absolute():
        return candidate

    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.exists():
        return cwd_path

    repo_root_path = (Path(__file__).resolve().parents[3] / candidate).resolve()
    if repo_root_path.exists():
        return repo_root_path

    return candidate


def load_dataframe(
    data_path: str, site_name: str = "caltech", cyclical: bool = True
) -> pd.DataFrame:
    resolved_data_path = resolve_data_path(data_path)

    if not resolved_data_path.exists():
        raise FileNotFoundError(
            "Dataset parquet not found. Checked path derived from:\n"
            f"  raw: {data_path}\n"
            f"  resolved: {resolved_data_path}\n"
            "Generate it first with federated_acn.data_dl.build_early_session_dataset."
        )

    df = pd.read_parquet(resolved_data_path)
    df = df[df["site_name"].astype(str).str.lower() == site_name.lower()].copy()
    df = apply_cyclical_time_features(df, cyclical=cyclical)
    df = df[df["kwh_delivered"].notna()].copy()
    df = df[df["n_current_points"] >= 5].copy()
    df = df.drop_duplicates(subset=["session_id"]).reset_index(drop=True)
    return df


def build_client_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["client_station"] = (
            out["site_name"].astype(str) + "::" + out["station_id"].astype(str)
    )
    out["client_cluster"] = (
            out["site_name"].astype(str)
            + "::"
            + out["cluster_id"].fillna("unassigned").astype(str)
    )
    return out


def split_df(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train: n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    return train_df, val_df, test_df


def load_global_splits(
    data_path: str,
    seed: int = 42,
    cyclical: bool = True,
    preprocessing: str = "standardization",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TabularPreprocessor]:
    df = load_dataframe(data_path, site_name="caltech", cyclical=cyclical)
    train_df, val_df, test_df = split_df(df, seed=seed)

    preprocessor = TabularPreprocessor(
        numerical_cols=[c for c in NUMERICAL_COLS if c in df.columns],
        categorical_cols=[c for c in CATEGORICAL_COLS if c in df.columns],
        passthrough_numerical_cols=[
            c for c in CYCLICAL_FEATURE_COLS if c in df.columns
        ],
        preprocessing=preprocessing,
    )
    preprocessor.fit(train_df)

    return train_df, val_df, test_df, preprocessor


def get_client_ids(
    df: pd.DataFrame,
    partition_by: str,
    min_total_sessions: int = 3,
) -> list[str]:
    if partition_by != "station":
        raise ValueError(f"Unsupported partition_by={partition_by}; only 'station' is supported.")
    client_col = "client_station"

    counts = df[client_col].dropna().astype(str).value_counts()

    eligible = counts[counts >= min_total_sessions].index.tolist()
    return sorted(eligible)


def load_partition(
    data_path: str,
    partition_id: int,
    num_partitions: int,
    partition_by: str,
    batch_size: int,
    seed: int = 42,
    min_total_sessions: int = 3,
):
    global_train_df, _, _, preprocessor = load_global_splits(
        data_path=data_path,
        seed=seed,
        cyclical=CFG.cyclical,
        preprocessing=CFG.preprocessing,
    )
    global_train_df = build_client_id_columns(global_train_df)

    client_ids = get_client_ids(
        global_train_df,
        partition_by=partition_by,
        min_total_sessions=min_total_sessions,
    )

    if num_partitions != len(client_ids):
        raise ValueError(
            f"num_partitions={num_partitions}, but discovered {len(client_ids)} eligible clients "
            f"for partition_by={partition_by}"
        )

    if partition_by != "station":
        raise ValueError(f"Unsupported partition_by={partition_by}; only 'station' is supported.")
    client_col = "client_station"
    client_id = client_ids[partition_id]

    client_df = global_train_df[
        global_train_df[client_col].astype(str) == client_id
        ].copy()
    train_df, val_df, _ = split_df(client_df, seed=seed)

    if len(train_df) == 0:
        raise ValueError(f"Client {client_id} has empty training split")
    if len(val_df) == 0:
        val_df = train_df.copy()

    train_ds = EarlySessionDataset(
        train_df, preprocessor, target_log1p=CFG.target_log1p
    )
    val_ds = EarlySessionDataset(val_df, preprocessor, target_log1p=CFG.target_log1p)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, preprocessor, client_id

def load_partition_xgb(
    data_path: str,
    partition_id: int,
    num_partitions: int,
    partition_by: str,
    seed: int = 42,
    min_total_sessions: int = 3,
):
    global_train_df, _, _, preprocessor = load_global_splits(
        data_path=data_path,
        seed=seed,
        cyclical=CFG.cyclical,
        preprocessing=CFG.preprocessing,
    )
    global_train_df = build_client_id_columns(global_train_df)

    client_ids = get_client_ids(
        global_train_df,
        partition_by=partition_by,
        min_total_sessions=min_total_sessions,
    )

    if num_partitions != len(client_ids):
        raise ValueError(
            f"num_partitions={num_partitions}, but discovered {len(client_ids)} eligible clients "
            f"for partition_by={partition_by}"
        )

    client_col = "client_station" if partition_by == "station" else "client_cluster"
    client_id = client_ids[partition_id]

    client_df = global_train_df[
        global_train_df[client_col].astype(str) == client_id
    ].copy()

    train_df, val_df, _ = split_df(client_df, seed=seed)

    if len(train_df) == 0:
        raise ValueError(f"Client {client_id} has empty training split")
    if len(val_df) == 0:
        val_df = train_df.copy()

    train_dmatrix = dataframe_to_dmatrix(train_df, preprocessor)
    val_dmatrix = dataframe_to_dmatrix(val_df, preprocessor)

    return (
        train_dmatrix,
        val_dmatrix,
        preprocessor,
        client_id,
        len(train_df),
        len(val_df),
    )

def load_global_eval_loaders(
    data_path: str,
    batch_size: int,
    seed: int = 42,
):
    train_df, val_df, test_df, preprocessor = load_global_splits(
        data_path=data_path,
        seed=seed,
        cyclical=CFG.cyclical,
        preprocessing=CFG.preprocessing,
    )

    train_ds = EarlySessionDataset(
        train_df, preprocessor, target_log1p=CFG.target_log1p
    )
    val_ds = EarlySessionDataset(val_df, preprocessor, target_log1p=CFG.target_log1p)
    test_ds = EarlySessionDataset(test_df, preprocessor, target_log1p=CFG.target_log1p)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, testloader, preprocessor, train_df, val_df, test_df

def load_global_eval_loaders_xgb(
    data_path: str,
    seed: int = 42,
):
    train_df, val_df, test_df, preprocessor = load_global_splits(
        data_path=data_path,
        seed=seed,
        cyclical=CFG.cyclical,
        preprocessing=CFG.preprocessing,
    )

    train_dmatrix = dataframe_to_dmatrix(train_df, preprocessor)
    val_dmatrix = dataframe_to_dmatrix(val_df, preprocessor)
    test_dmatrix = dataframe_to_dmatrix(test_df, preprocessor)

    return (
        train_dmatrix,
        val_dmatrix,
        test_dmatrix,
        preprocessor,
        train_df,
        val_df,
        test_df,
    )

def get_model(
    model_name: str,
    preprocessor: TabularPreprocessor,
    use_embeddings: bool,
) -> nn.Module:
    categorical_cardinalities = (
        preprocessor.categorical_cardinalities() if use_embeddings else {}
    )

    if model_name == "mlp":
        return TabularMLP(
            num_numerical_features=len(preprocessor.numerical_cols),
            categorical_cardinalities=categorical_cardinalities,
            hidden_dim=128,
            dropout=0.2,
        )
    if model_name == "dcn":
        return DeepCrossRegressor(
            num_numerical_features=len(preprocessor.numerical_cols),
            categorical_cardinalities=categorical_cardinalities,
            cross_layers=3,
            deep_hidden_dims=[128, 64],
            dropout=0.1,
            positive_output=True,
        )
    if model_name == "res-mlp":
        return TabularResMLP(
            num_numerical_features=len(preprocessor.numerical_cols),
            categorical_cardinalities=categorical_cardinalities,
            hidden_dim=128,
            dropout=0.2,
            num_blocks=2,
        )
    if model_name == "transformer":
        return TabularTransformer(
            num_numerical_features=len(preprocessor.numerical_cols),
            categorical_cardinalities=categorical_cardinalities,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.1,
        )
    if model_name == "cnn1d":
        return TabularCNN1D(
            num_numerical_features=len(preprocessor.numerical_cols),
            categorical_cardinalities=categorical_cardinalities,
            token_dim=16,
            hidden_channels=32,
            dropout=0.1,
        )
    if model_name == "gru":
        return TabularGRU(
            num_numerical_features=len(preprocessor.numerical_cols),
            categorical_cardinalities=categorical_cardinalities,
            token_dim=16,
            hidden_dim=32,
            dropout=0.1,
        )

    raise ValueError(f"Unknown model_name={model_name}")

def get_xgb_params() -> Dict:
    objective = "reg:absoluteerror" if CFG.loss == "mae" else "reg:squarederror"
    eval_metric = ["mae"] if CFG.loss == "mae" else ["rmse", "mae"]
    return {
        "objective": objective,
        "eval_metric": eval_metric,
        "eta": CFG.learning_rate,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "tree_method": "hist",
        "seed": CFG.seed,
    }


def train_one_round(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    loss_name: str,
    lambda_early: float,
) -> float:
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = get_loss_criterion(loss_name)

    running_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            pred = model(batch)
            loss, _, _ = physics_constrained_loss(
                pred=pred,
                target=batch["y"],
                base_loss_fn=criterion,
                early_energy_kwh=batch["early_energy_kwh"],
                target_log1p=CFG.target_log1p,
                lambda_early=lambda_early,
            )
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            num_batches += 1

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_name: str = "mae",
    lambda_early: float = 1.0,
) -> Dict[str, float]:
    model.to(device)
    model.eval()
    base_loss_fn = get_loss_criterion(loss_name)

    y_true_batches = []
    y_pred_batches = []
    total_loss = 0.0
    total_base_loss = 0.0
    total_loss_early = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch)
        y = batch["y"]
        loss, base_loss, loss_early = physics_constrained_loss(
            pred=pred,
            target=y,
            base_loss_fn=base_loss_fn,
            early_energy_kwh=batch["early_energy_kwh"],
            target_log1p=CFG.target_log1p,
            lambda_early=lambda_early,
        )
        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_base_loss += float(base_loss.item()) * bs
        total_loss_early += float(loss_early.item()) * bs

        y_true_batches.append(y.detach().cpu().numpy())
        y_pred_batches.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)
    y_true_eval = inverse_transform_target_numpy(y_true, use_log1p=CFG.target_log1p)
    y_pred_eval = inverse_transform_target_numpy(y_pred, use_log1p=CFG.target_log1p)
    metrics = compute_regression_metrics(y_true=y_true_eval, y_pred=y_pred_eval)
    denom = max(y_true_eval.shape[0], 1)
    metrics["loss"] = total_loss / denom
    metrics["base_loss"] = total_base_loss / denom
    metrics["loss_early"] = total_loss_early / denom
    metrics["num_samples"] = int(y_true_eval.shape[0])
    metrics["target_min"] = float(np.min(y_true_eval))
    metrics["target_max"] = float(np.max(y_true_eval))
    metrics["target_mean"] = float(np.mean(y_true_eval))
    metrics["target_std"] = float(np.std(y_true_eval))
    return metrics

def evaluate_xgb(
    bst: xgb.Booster,
    dmatrix: xgb.DMatrix,
) -> Dict[str, float]:
    y_true = dmatrix.get_label()
    y_pred = bst.predict(dmatrix)

    y_true_eval = inverse_transform_target_numpy(y_true, use_log1p=CFG.target_log1p)
    y_pred_eval = inverse_transform_target_numpy(y_pred, use_log1p=CFG.target_log1p)

    metrics = compute_regression_metrics(y_true=y_true_eval, y_pred=y_pred_eval)
    metrics["num_samples"] = int(y_true_eval.shape[0])
    metrics["target_min"] = float(np.min(y_true_eval))
    metrics["target_max"] = float(np.max(y_true_eval))
    metrics["target_mean"] = float(np.mean(y_true_eval))
    metrics["target_std"] = float(np.std(y_true_eval))
    return metrics

def create_linear_regression_model() -> SGDRegressor:
    linear_loss = "epsilon_insensitive" if CFG.loss == "mae" else "squared_error"
    return SGDRegressor(
        loss=linear_loss,
        penalty="l2",
        alpha=1e-4,
        max_iter=1,
        tol=None,
        warm_start=False,
        shuffle=True,
        learning_rate="invscaling",
        eta0=1e-5,
        power_t=0.25,
        epsilon=0.0,
        average=False,
        random_state=CFG.seed,
        fit_intercept=True,
    )


def get_linear_model_params(model: SGDRegressor):
    return [
        model.coef_.astype(np.float64),
        model.intercept_.astype(np.float64),
    ]


def fit_linear_model_with_params(
    model: SGDRegressor,
    x: np.ndarray,
    y: np.ndarray,
    params=None,
    local_epochs: int = 1,
) -> SGDRegressor:
    coef_init = None
    intercept_init = None

    if params is not None:
        coef_init = np.asarray(params[0], dtype=np.float64)
        intercept_init = np.asarray(params[1], dtype=np.float64)

    # Repeated one-epoch fits, carrying params forward explicitly
    for epoch in range(local_epochs):
        if epoch == 0:
            model.fit(x, y, coef_init=coef_init, intercept_init=intercept_init)
        else:
            model.fit(x, y, coef_init=model.coef_, intercept_init=model.intercept_)

    return model


def load_partition_linear(
    data_path: str,
    partition_id: int,
    num_partitions: int,
    partition_by: str,
    seed: int = 42,
    min_total_sessions: int = 3,
):
    global_train_df, _, _, preprocessor = load_global_splits(
        data_path=data_path,
        seed=seed,
        cyclical=CFG.cyclical,
        preprocessing=CFG.preprocessing,
    )
    global_train_df = build_client_id_columns(global_train_df)

    client_ids = get_client_ids(
        global_train_df,
        partition_by=partition_by,
        min_total_sessions=min_total_sessions,
    )

    if num_partitions != len(client_ids):
        raise ValueError(
            f"num_partitions={num_partitions}, but discovered {len(client_ids)} eligible clients "
            f"for partition_by={partition_by}"
        )

    client_col = "client_station" if partition_by == "station" else "client_cluster"
    client_id = client_ids[partition_id]

    client_df = global_train_df[
        global_train_df[client_col].astype(str) == client_id
    ].copy()

    train_df, val_df, _ = split_df(client_df, seed=seed)

    if len(train_df) == 0:
        raise ValueError(f"Client {client_id} has empty training split")
    if len(val_df) == 0:
        val_df = train_df.copy()

    x_train = build_sklearn_features(train_df, preprocessor).astype(np.float64)
    y_train = train_df["kwh_delivered"].to_numpy(dtype=np.float64)

    x_val = build_sklearn_features(val_df, preprocessor).astype(np.float64)
    y_val = val_df["kwh_delivered"].to_numpy(dtype=np.float64)

    if CFG.target_log1p:
        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)

    return x_train, y_train, x_val, y_val, preprocessor, client_id


def load_global_eval_linear(
    data_path: str,
    seed: int = 42,
):
    train_df, val_df, test_df, preprocessor = load_global_splits(
        data_path=data_path,
        seed=seed,
        cyclical=CFG.cyclical,
        preprocessing=CFG.preprocessing,
    )

    x_train = build_sklearn_features(train_df, preprocessor).astype(np.float64)
    y_train = train_df["kwh_delivered"].to_numpy(dtype=np.float64)

    x_val = build_sklearn_features(val_df, preprocessor).astype(np.float64)
    y_val = val_df["kwh_delivered"].to_numpy(dtype=np.float64)

    x_test = build_sklearn_features(test_df, preprocessor).astype(np.float64)
    y_test = test_df["kwh_delivered"].to_numpy(dtype=np.float64)

    if CFG.target_log1p:
        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)
        y_test = np.log1p(y_test)

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        preprocessor,
        train_df,
        val_df,
        test_df,
    )


def evaluate_linear_model(
    model: SGDRegressor,
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    y_pred = model.predict(x)

    if CFG.target_log1p:
        y_eval = inverse_transform_target_numpy(y, use_log1p=True)
        y_pred_eval = inverse_transform_target_numpy(y_pred, use_log1p=True)
    else:
        y_eval = y
        y_pred_eval = y_pred

    y_pred_eval = np.clip(y_pred_eval, a_min=0.0, a_max=None)

    metrics = compute_regression_metrics(y_true=y_eval, y_pred=y_pred_eval)
    metrics["num_samples"] = int(y_eval.shape[0])
    metrics["target_min"] = float(np.min(y_eval))
    metrics["target_max"] = float(np.max(y_eval))
    metrics["target_mean"] = float(np.mean(y_eval))
    metrics["target_std"] = float(np.std(y_eval))
    return metrics

def split_target_summaries(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    return {
        "train": summarize_target(train_df),
        "val": summarize_target(val_df),
        "test": summarize_target(test_df),
    }


def client_split_target_summaries(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    partition_by: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if partition_by != "station":
        raise ValueError(f"Unsupported partition_by={partition_by}; only 'station' is supported.")
    client_col = "client_station"

    train_df_with_client = build_client_id_columns(train_df)
    val_df_with_client = build_client_id_columns(val_df)
    test_df_with_client = build_client_id_columns(test_df)

    train_clients = train_df_with_client[client_col].dropna().astype(str)
    val_clients = val_df_with_client[client_col].dropna().astype(str)
    test_clients = test_df_with_client[client_col].dropna().astype(str)
    all_clients = sorted(set(train_clients) | set(val_clients) | set(test_clients))

    summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    for client_id in all_clients:
        client_train = train_df_with_client[
            train_df_with_client[client_col].astype(str) == client_id
        ]
        client_val = val_df_with_client[
            val_df_with_client[client_col].astype(str) == client_id
        ]
        client_test = test_df_with_client[
            test_df_with_client[client_col].astype(str) == client_id
        ]
        summaries[client_id] = {
            "train": summarize_target(client_train),
            "val": summarize_target(client_val),
            "test": summarize_target(client_test),
        }

    return summaries


def build_sklearn_features(
    df: pd.DataFrame,
    preprocessor: TabularPreprocessor,
) -> np.ndarray:
    x_num = preprocessor.transform_numerical(df)
    x_cat = preprocessor.transform_categorical(df)
    cat_arrays = [x_cat[col].astype(np.float32) for col in preprocessor.categorical_cols]
    if cat_arrays:
        return np.concatenate([x_num, np.stack(cat_arrays, axis=1)], axis=1)
    return x_num

def get_target_array(df: pd.DataFrame) -> np.ndarray:
    y = df["kwh_delivered"].to_numpy(dtype=np.float32)
    if CFG.target_log1p:
        y = np.log1p(y)
    return y


def dataframe_to_dmatrix(
    df: pd.DataFrame,
    preprocessor: TabularPreprocessor,
) -> xgb.DMatrix:
    x = build_sklearn_features(df, preprocessor).astype(np.float32)
    y = get_target_array(df)
    return xgb.DMatrix(x, label=y)
