import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from flwr.simulation import run_simulation
from sklearn.ensemble import RandomForestRegressor

from federated_acn.fl.client_app import app as client_app
from federated_acn.fl.config import CFG, write_env_overrides
from federated_acn.fl.server_app import app as server_app
from federated_acn.fl.task import (
    build_sklearn_features,
    build_client_id_columns,
    get_client_ids,
    load_global_eval_loaders,
    load_global_splits,
    split_target_summaries,
)
from federated_acn.metrics import compute_regression_metrics


def resolve_data_path(raw_data_path: str) -> Path:
    candidate = Path(raw_data_path).expanduser()
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Dataset parquet not found at: {candidate}")
        return candidate

    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.exists():
        return cwd_path

    repo_root_path = (Path(__file__).resolve().parents[3] / candidate).resolve()
    if repo_root_path.exists():
        return repo_root_path

    raise FileNotFoundError(
        "Dataset parquet not found. Checked both:\n"
        f"  - {cwd_path}\n"
        f"  - {repo_root_path}\n"
        "Pass --data-path with an absolute path, or run from the project root."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=CFG.data_path)
    parser.add_argument(
        "--model-name",
        type=str,
        default="mlp",
        choices=[
            "mlp",
            "res-mlp",
            "dcn",
            "transformer",
            "cnn1d",
            "gru",
            "linear_regression",
            "random_forest",
            "xgboost",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--num-server-rounds", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--loss", type=str, default="mae", choices=["mse", "mae"])
    parser.add_argument("--lambda-early", type=float, default=1.0)
    parser.add_argument("--fraction-train", type=float, default=0.1)
    parser.add_argument("--fraction-evaluate", type=float, default=1.0)
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
    parser.add_argument("--min-total-sessions-per-client", type=int, default=3)
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
    parser.add_argument(
        "--preprocessing",
        type=str,
        default="standardization",
        choices=["standardization", "minmax", "none"],
        help="Numerical preprocessing for non-cyclical features.",
    )
    parser.add_argument("--backend-num-cpus", type=float, default=1.0)
    parser.add_argument("--backend-num-gpus", type=float, default=0.0)
    parser.add_argument("--rf-local-trees", type=int, default=30)
    parser.add_argument("--xgb-local-rounds", type=int, default=1)
    parser.add_argument("--linear-local-epochs", type=int, default=3)
    parser.add_argument(
        "--xgb-train-method",
        type=str,
        default="bagging",
        choices=["bagging", "cyclic"],
    )
    args = parser.parse_args()

    CFG.data_path = resolve_data_path(args.data_path)
    CFG.partition_by = "station"
    CFG.model_name = args.model_name
    CFG.batch_size = args.batch_size
    CFG.local_epochs = args.local_epochs
    CFG.num_server_rounds = args.num_server_rounds
    CFG.learning_rate = args.learning_rate
    CFG.loss = args.loss
    CFG.lambda_early = args.lambda_early
    CFG.fraction_train = args.fraction_train
    CFG.fraction_evaluate = args.fraction_evaluate
    CFG.seed = args.seed
    CFG.use_embeddings = args.use_embeddings
    CFG.min_total_sessions_per_client = args.min_total_sessions_per_client
    CFG.cyclical = args.cyclical
    CFG.preprocessing = args.preprocessing
    CFG.target_log1p = args.target_log1p
    CFG.xgb_local_rounds = args.xgb_local_rounds
    CFG.xgb_train_method = args.xgb_train_method
    CFG.linear_local_epochs = args.linear_local_epochs
    write_env_overrides()

    global_train_df, val_df, test_df, _ = load_global_splits(
        CFG.data_path,
        seed=CFG.seed,
        cyclical=CFG.cyclical,
        preprocessing=CFG.preprocessing,
    )
    print("First preprocessed row (federated):")
    print(global_train_df.head(1).to_string(index=False))

    global_train_df = build_client_id_columns(global_train_df)

    client_ids = get_client_ids(
        global_train_df,
        CFG.partition_by,
        min_total_sessions=CFG.min_total_sessions_per_client,
    )
    num_supernodes = len(client_ids)

    print("Running Flower simulation")
    print(f"  partition_by      : {CFG.partition_by}")
    print(f"  model_name        : {CFG.model_name}")
    print(f"  eligible_clients  : {num_supernodes}")
    print(f"  use_embeddings    : {CFG.use_embeddings}")
    print(f"  loss              : {CFG.loss}")
    print(f"  seed              : {CFG.seed}")
    print(f"  lambda_early      : {CFG.lambda_early}")
    print(f"  cyclical          : {CFG.cyclical}")
    print(f"  preprocessing     : {CFG.preprocessing}")
    print(f"  target_log1p      : {CFG.target_log1p}")
    print(f"  train_rows        : {len(global_train_df)}")
    print(f"  val_rows          : {len(val_df)}")
    print(f"  test_rows         : {len(test_df)}")
    print(f"  data_path         : {Path(CFG.data_path).resolve()}")

    if CFG.model_name in {"mlp", "res-mlp", "dcn", "transformer", "cnn1d", "gru", "linear_regression", "xgboost"}:
        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=num_supernodes,
            backend_config={
                "client_resources": {
                    "num_cpus": args.backend_num_cpus,
                    "num_gpus": args.backend_num_gpus,
                }
            },
        )
        return 0

    trainloader, valloader, testloader, preprocessor, train_df, val_df, test_df = (
        load_global_eval_loaders(
            data_path=CFG.data_path,
            batch_size=CFG.batch_size,
            seed=CFG.seed,
        )
    )
    del trainloader, valloader, testloader
    train_df = build_client_id_columns(train_df)
    client_col = "client_station"
    outdir = Path("outputs") / "flower"
    outdir.mkdir(parents=True, exist_ok=True)
    run_tag = (
        f"{CFG.model_name}_no_embedding_"
        f"{'cyclical' if CFG.cyclical else 'no_cyclical'}_"
        f"{'target_log1p' if CFG.target_log1p else 'target_raw'}_"
        f"preproc_{CFG.preprocessing}_"
        f"{CFG.loss}_{CFG.seed}_{CFG.partition_by}_lambdaearly_{CFG.lambda_early:g}"
    )

    x_val = build_sklearn_features(val_df, preprocessor)
    y_val = val_df["kwh_delivered"].to_numpy(dtype=np.float32)
    x_test = build_sklearn_features(test_df, preprocessor)
    y_test = test_df["kwh_delivered"].to_numpy(dtype=np.float32)

    history: list[dict] = []
    rng = np.random.default_rng(CFG.seed)
    sample_count = max(1, int(np.ceil(CFG.fraction_train * len(client_ids))))

    def sample_clients() -> list[str]:
        sampled = rng.choice(client_ids, size=min(sample_count, len(client_ids)), replace=False)
        return [str(cid) for cid in sampled]

    if CFG.model_name == "random_forest":
        all_trees = []
        for rnd in range(1, CFG.num_server_rounds + 1):
            sampled_client_ids = sample_clients()
            round_trees = []
            for cid in sampled_client_ids:
                client_df = train_df[train_df[client_col].astype(str) == cid].copy()
                if client_df.empty:
                    continue
                x_client = build_sklearn_features(client_df, preprocessor)
                y_client = client_df["kwh_delivered"].to_numpy(dtype=np.float32)
                rf = RandomForestRegressor(
                    n_estimators=args.rf_local_trees,
                    criterion="absolute_error" if CFG.loss == "mae" else "squared_error",
                    random_state=CFG.seed + rnd,
                    n_jobs=-1,
                )
                rf.fit(x_client, y_client)
                round_trees.extend(rf.estimators_)
            all_trees.extend(round_trees)
            if not all_trees:
                print(f"[Round {rnd}] random_forest: no client data, skipping")
                continue
            val_pred = np.mean(np.stack([tree.predict(x_val) for tree in all_trees]), axis=0)
            test_pred = np.mean(np.stack([tree.predict(x_test) for tree in all_trees]), axis=0)
            val_pred = np.clip(val_pred, a_min=0.0, a_max=None)
            test_pred = np.clip(test_pred, a_min=0.0, a_max=None)
            val_rmse = compute_regression_metrics(y_val, val_pred)["rmse"]
            test_rmse = compute_regression_metrics(y_test, test_pred)["rmse"]
            print(
                f"[Round {rnd}] random_forest | "
                f"sampled_clients={len(sampled_client_ids)}/{len(client_ids)} | "
                f"new_trees={len(round_trees)} total_trees={len(all_trees)} | "
                f"val_rmse={val_rmse:.4f} | test_rmse={test_rmse:.4f}"
            )
            history.append(
                {
                    "round": rnd,
                    "sampled_clients": len(sampled_client_ids),
                    "global_val_rmse": val_rmse,
                    "global_test_rmse": test_rmse,
                    "preprocessing": CFG.preprocessing,
                }
            )

    if not history:
        raise RuntimeError(
            "No training history was produced. Check client filtering/sampling settings."
        )

    pd.DataFrame(history).to_csv(outdir / f"history_{run_tag}.csv", index=False)
    best_row = min(history, key=lambda row: row["global_val_rmse"])
    metrics_path = outdir / f"metrics_{run_tag}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": CFG.model_name,
                "partition_by": CFG.partition_by,
                "num_server_rounds": CFG.num_server_rounds,
                "local_epochs": CFG.local_epochs,
                "batch_size": CFG.batch_size,
                "loss": CFG.loss,
                "lambda_early": CFG.lambda_early,
                "preprocessing": CFG.preprocessing,
                "seed": CFG.seed,
                "best_round": int(best_row["round"]),
                "best_val_rmse": float(best_row["global_val_rmse"]),
                "final_test_rmse": float(history[-1]["global_test_rmse"]),
                "target_summaries": split_target_summaries(train_df, val_df, test_df),
            },
            f,
            indent=2,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
