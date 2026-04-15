import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedXgbBagging, FedXgbCyclic

from federated_acn.fl.config import CFG, apply_env_overrides
from federated_acn.fl.task import (
    client_split_target_summaries,
    create_linear_regression_model,
    evaluate,
    evaluate_linear_model,
    evaluate_xgb,
    get_model,
    get_xgb_params,
    load_global_eval_linear,
    load_global_eval_loaders,
    load_global_eval_loaders_xgb,
    split_target_summaries, fit_linear_model_with_params,
)

app = ServerApp()

_HISTORY: list[dict] = []
_BEST_VAL_RMSE = float("inf")
_BEST_ROUND = 0


def make_server_paths() -> tuple[Path, Path, Path, Path]:
    outdir = Path("outputs") / "flower"
    outdir.mkdir(parents=True, exist_ok=True)

    embedding_tag = "embedding" if CFG.use_embeddings else "no_embedding"
    cyclical_tag = "cyclical" if CFG.cyclical else "no_cyclical"
    target_tag = "target_log1p" if CFG.target_log1p else "target_raw"
    preprocessing_tag = f"preproc_{CFG.preprocessing}"
    run_tag = f"{CFG.model_name}_{embedding_tag}_{cyclical_tag}_{target_tag}_{preprocessing_tag}_{CFG.loss}_{CFG.seed}_{CFG.partition_by}"
    run_tag = f"{run_tag}_lambdaearly_{CFG.lambda_early:g}"

    if CFG.model_name == "xgboost":
        model_ext = "json"
    elif CFG.model_name == "linear_regression":
        model_ext = "pkl"
    else:
        model_ext = "pt"

    best_model_path = outdir / f"best_model_{run_tag}.{model_ext}"
    final_model_path = outdir / f"final_model_{run_tag}.{model_ext}"
    history_path = outdir / f"history_{run_tag}.csv"
    metrics_path = outdir / f"metrics_{run_tag}.json"
    return best_model_path, final_model_path, history_path, metrics_path


def make_centralized_evaluate_fn_torch():
    global _BEST_VAL_RMSE, _BEST_ROUND

    _, valloader, testloader, preprocessor, _, _, _ = load_global_eval_loaders(
        data_path=CFG.data_path,
        batch_size=CFG.batch_size,
        seed=CFG.seed,
    )
    best_model_path, _, _, _ = make_server_paths()

    def _evaluate(server_round: int, arrays: ArrayRecord):
        global _BEST_VAL_RMSE, _BEST_ROUND

        model = get_model(
            CFG.model_name, preprocessor, use_embeddings=CFG.use_embeddings
        )
        model.load_state_dict(arrays.to_torch_state_dict())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        val_metrics = evaluate(
            model,
            valloader,
            device=device,
            loss_name=CFG.loss,
            lambda_early=CFG.lambda_early,
        )
        test_metrics = evaluate(
            model,
            testloader,
            device=device,
            loss_name=CFG.loss,
            lambda_early=CFG.lambda_early,
        )

        if val_metrics["rmse"] < _BEST_VAL_RMSE:
            _BEST_VAL_RMSE = val_metrics["rmse"]
            _BEST_ROUND = server_round
            torch.save(model.state_dict(), best_model_path)

        _HISTORY.append(
            {
                "round": server_round,
                "preprocessing": CFG.preprocessing,
                **{f"global_val_{k}": float(v) for k, v in val_metrics.items()},
                **{f"global_test_{k}": float(v) for k, v in test_metrics.items()},
            }
        )

        print(
            f"[Server] Round {server_round} | "
            f"val MAE={val_metrics['mae']:.4f} | val RMSE={val_metrics['rmse']:.4f} | "
            f"val MAPE={val_metrics['mape']:.4f} | val R2={val_metrics['r2']:.4f} | "
            f"test MAE={test_metrics['mae']:.4f} | test RMSE={test_metrics['rmse']:.4f} | "
            f"test MAPE={test_metrics['mape']:.4f} | test R2={test_metrics['r2']:.4f}"
        )

        return RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "global_val_mae": float(val_metrics["mae"]),
                        "global_val_rmse": float(val_metrics["rmse"]),
                        "global_val_mape": float(val_metrics["mape"]),
                        "global_val_mase": float(val_metrics["mase"]),
                        "global_val_r2": float(val_metrics["r2"]),
                        "global_test_mae": float(test_metrics["mae"]),
                        "global_test_rmse": float(test_metrics["rmse"]),
                        "global_test_mape": float(test_metrics["mape"]),
                        "global_test_mase": float(test_metrics["mase"]),
                        "global_test_r2": float(test_metrics["r2"]),
                    }
                )
            }
        )

    return _evaluate

def make_centralized_evaluate_fn_xgb():
    global _BEST_VAL_RMSE, _BEST_ROUND

    _, val_dmatrix, test_dmatrix, _, _, _, _ = load_global_eval_loaders_xgb(
        data_path=CFG.data_path,
        seed=CFG.seed,
    )
    best_model_path, _, _, _ = make_server_paths()
    params = get_xgb_params()

    def _evaluate(server_round: int, arrays: ArrayRecord):
        global _BEST_VAL_RMSE, _BEST_ROUND

        if server_round == 0:
            return None

        bst = xgb.Booster(params=params)
        global_model = bytearray(arrays["0"].numpy().tobytes())
        bst.load_model(global_model)

        val_metrics = evaluate_xgb(bst, val_dmatrix)
        test_metrics = evaluate_xgb(bst, test_dmatrix)

        if val_metrics["rmse"] < _BEST_VAL_RMSE:
            _BEST_VAL_RMSE = val_metrics["rmse"]
            _BEST_ROUND = server_round
            bst.save_model(str(best_model_path))

        _HISTORY.append(
            {
                "round": server_round,
                "preprocessing": CFG.preprocessing,
                **{f"global_val_{k}": float(v) for k, v in val_metrics.items()},
                **{f"global_test_{k}": float(v) for k, v in test_metrics.items()},
            }
        )

        print(
            f"[Server] Round {server_round} | "
            f"val MAE={val_metrics['mae']:.4f} | val RMSE={val_metrics['rmse']:.4f} | "
            f"val MAPE={val_metrics['mape']:.4f} | val R2={val_metrics['r2']:.4f} | "
            f"test MAE={test_metrics['mae']:.4f} | test RMSE={test_metrics['rmse']:.4f} | "
            f"test MAPE={test_metrics['mape']:.4f} | test R2={test_metrics['r2']:.4f}"
        )

        return RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "global_val_mae": float(val_metrics["mae"]),
                        "global_val_rmse": float(val_metrics["rmse"]),
                        "global_val_mape": float(val_metrics["mape"]),
                        "global_val_mase": float(val_metrics["mase"]),
                        "global_val_r2": float(val_metrics["r2"]),
                        "global_test_mae": float(test_metrics["mae"]),
                        "global_test_rmse": float(test_metrics["rmse"]),
                        "global_test_mape": float(test_metrics["mape"]),
                        "global_test_mase": float(test_metrics["mase"]),
                        "global_test_r2": float(test_metrics["r2"]),
                    }
                )
            }
        )

    return _evaluate


def make_centralized_evaluate_fn_linear():
    global _BEST_VAL_RMSE, _BEST_ROUND

    x_train, y_train, x_val, y_val, x_test, y_test, _, _, _, _ = (
        load_global_eval_linear(
            data_path=CFG.data_path,
            seed=CFG.seed,
        )
    )
    best_model_path, _, _, _ = make_server_paths()

    def _evaluate(server_round: int, arrays: ArrayRecord):
        global _BEST_VAL_RMSE, _BEST_ROUND

        ndarrays = arrays.to_numpy_ndarrays()
        model = create_linear_regression_model()
        model.coef_ = np.asarray(ndarrays[0], dtype=np.float64)
        model.intercept_ = np.asarray(ndarrays[1], dtype=np.float64)
        model.n_features_in_ = model.coef_.shape[0]

        val_metrics = evaluate_linear_model(model, x_val, y_val)
        test_metrics = evaluate_linear_model(model, x_test, y_test)

        if val_metrics["rmse"] < _BEST_VAL_RMSE:
            _BEST_VAL_RMSE = val_metrics["rmse"]
            _BEST_ROUND = server_round
            joblib.dump(model, best_model_path)

        _HISTORY.append(
            {
                "round": server_round,
                "preprocessing": CFG.preprocessing,
                **{f"global_val_{k}": float(v) for k, v in val_metrics.items()},
                **{f"global_test_{k}": float(v) for k, v in test_metrics.items()},
            }
        )

        return RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "global_val_mae": float(val_metrics["mae"]),
                        "global_val_rmse": float(val_metrics["rmse"]),
                        "global_val_mape": float(val_metrics["mape"]),
                        "global_val_mase": float(val_metrics["mase"]),
                        "global_val_r2": float(val_metrics["r2"]),
                        "global_test_mae": float(test_metrics["mae"]),
                        "global_test_rmse": float(test_metrics["rmse"]),
                        "global_test_mape": float(test_metrics["mape"]),
                        "global_test_mase": float(test_metrics["mase"]),
                        "global_test_r2": float(test_metrics["r2"]),
                    }
                )
            }
        )

    return _evaluate


@app.main()
def main(grid: Grid, context: Context) -> None:
    global _BEST_VAL_RMSE, _BEST_ROUND

    apply_env_overrides()
    _BEST_VAL_RMSE = float("inf")
    _BEST_ROUND = 0
    _HISTORY.clear()

    if CFG.model_name == "linear_regression":
        x_train, y_train, x_val, y_val, x_test, y_test, _, train_df, val_df, test_df = (
            load_global_eval_linear(
                data_path=CFG.data_path,
                seed=CFG.seed,
            )
        )

        split_summaries = split_target_summaries(train_df, val_df, test_df)
        client_split_summaries = client_split_target_summaries(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            partition_by=CFG.partition_by,
        )

        print("Global split and target summary:")
        for split_name in ["train", "val", "test"]:
            summary = split_summaries[split_name]
            print(
                f"  {split_name:>5}: n={summary['num_samples']:,}, "
                f"min={summary['target_min']:.4f}, max={summary['target_max']:.4f}, "
                f"mean={summary['target_mean']:.4f}, std={summary['target_std']:.4f}"
            )

        print(f"Per-client split summary (partition_by={CFG.partition_by}):")
        for client_id in sorted(client_split_summaries):
            client_summary = client_split_summaries[client_id]
            print(f"  client={client_id}")
            for split_name in ["train", "val", "test"]:
                summary = client_summary[split_name]
                print(
                    f"    {split_name:>5}: n={summary['num_samples']:,}, "
                    f"min={summary['target_min']:.4f}, max={summary['target_max']:.4f}, "
                    f"mean={summary['target_mean']:.4f}, std={summary['target_std']:.4f}"
                )

        model = create_linear_regression_model()
        # Initialize once with zeros by fitting from explicit zero params
        zero_params = [
            np.zeros((x_train.shape[1],), dtype=np.float64),
            np.zeros((1,), dtype=np.float64),
        ]
        model = fit_linear_model_with_params(
            model=model,
            x=x_train[:1],
            y=y_train[:1],
            params=zero_params,
            local_epochs=1,
        )

        strategy = FedAvg(
            fraction_train=CFG.fraction_train,
            fraction_evaluate=CFG.fraction_evaluate,
        )

        initial_arrays = ArrayRecord(
            [
                np.zeros((x_train.shape[1],), dtype=np.float64),
                np.zeros((1,), dtype=np.float64),
            ]
        )

        result = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=CFG.num_server_rounds,
            evaluate_fn=make_centralized_evaluate_fn_linear(),
        )

        best_model_path, final_model_path, history_path, metrics_path = make_server_paths()

        final_model = create_linear_regression_model()
        final_params = result.arrays.to_numpy_ndarrays()

        final_model.coef_ = np.asarray(final_params[0], dtype=np.float64)
        final_model.intercept_ = np.asarray(final_params[1], dtype=np.float64)
        final_model.n_features_in_ = final_model.coef_.shape[0]
        joblib.dump(final_model, final_model_path)

        pd.DataFrame(_HISTORY).to_csv(history_path, index=False)

        if not best_model_path.exists():
            joblib.dump(final_model, best_model_path)

        best_model = joblib.load(best_model_path)

        best_train_metrics = evaluate_linear_model(best_model, x_train, y_train)
        best_val_metrics = evaluate_linear_model(best_model, x_val, y_val)
        best_test_metrics = evaluate_linear_model(best_model, x_test, y_test)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": CFG.model_name,
                    "cyclical": CFG.cyclical,
                    "preprocessing": CFG.preprocessing,
                    "seed": CFG.seed,
                    "partition_by": CFG.partition_by,
                    "num_server_rounds": CFG.num_server_rounds,
                    "linear_local_epochs": CFG.linear_local_epochs,
                    "fraction_train": CFG.fraction_train,
                    "fraction_evaluate": CFG.fraction_evaluate,
                    "learning_rate": CFG.learning_rate,
                    "best_val_mae": float(best_val_metrics["mae"]),
                    "best_val_rmse": float(best_val_metrics["rmse"]),
                    "best_model_round": int(_BEST_ROUND),
                    "splits": {
                        "train": best_train_metrics,
                        "val": best_val_metrics,
                        "test": best_test_metrics,
                    },
                    "target_summaries": split_summaries,
                    "client_target_summaries": client_split_summaries,
                },
                f,
                indent=2,
            )
        return

    if CFG.model_name == "xgboost":
        train_dmatrix, val_dmatrix, test_dmatrix, _, train_df, val_df, test_df = (
            load_global_eval_loaders_xgb(
                data_path=CFG.data_path,
                seed=CFG.seed,
            )
        )

        split_summaries = split_target_summaries(train_df, val_df, test_df)
        client_split_summaries = client_split_target_summaries(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            partition_by=CFG.partition_by,
        )

        print("Global split and target summary:")
        for split_name in ["train", "val", "test"]:
            summary = split_summaries[split_name]
            print(
                f"  {split_name:>5}: n={summary['num_samples']:,}, "
                f"min={summary['target_min']:.4f}, max={summary['target_max']:.4f}, "
                f"mean={summary['target_mean']:.4f}, std={summary['target_std']:.4f}"
            )

        print(f"Per-client split summary (partition_by={CFG.partition_by}):")
        for client_id in sorted(client_split_summaries):
            client_summary = client_split_summaries[client_id]
            print(f"  client={client_id}")
            for split_name in ["train", "val", "test"]:
                summary = client_summary[split_name]
                print(
                    f"    {split_name:>5}: n={summary['num_samples']:,}, "
                    f"min={summary['target_min']:.4f}, max={summary['target_max']:.4f}, "
                    f"mean={summary['target_mean']:.4f}, std={summary['target_std']:.4f}"
                )

        train_method = getattr(CFG, "xgb_train_method", "bagging")
        if train_method == "cyclic":
            strategy = FedXgbCyclic()
        else:
            strategy = FedXgbBagging(
                fraction_train=CFG.fraction_train,
                fraction_evaluate=CFG.fraction_evaluate,
            )

        initial_arrays = ArrayRecord([np.frombuffer(b"", dtype=np.uint8)])

        result = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=CFG.num_server_rounds,
            evaluate_fn=make_centralized_evaluate_fn_xgb(),
        )

        best_model_path, final_model_path, history_path, metrics_path = make_server_paths()

        final_bst = xgb.Booster(params=get_xgb_params())
        final_model = bytearray(result.arrays["0"].numpy().tobytes())
        final_bst.load_model(final_model)
        final_bst.save_model(str(final_model_path))

        pd.DataFrame(_HISTORY).to_csv(history_path, index=False)

        if not best_model_path.exists():
            final_bst.save_model(best_model_path)

        best_bst = xgb.Booster(params=get_xgb_params())
        best_bst.load_model(str(best_model_path))

        best_train_metrics = evaluate_xgb(best_bst, train_dmatrix)
        best_val_metrics = evaluate_xgb(best_bst, val_dmatrix)
        best_test_metrics = evaluate_xgb(best_bst, test_dmatrix)

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": CFG.model_name,
                    "use_embeddings": CFG.use_embeddings,
                    "cyclical": CFG.cyclical,
                    "preprocessing": CFG.preprocessing,
                    "seed": CFG.seed,
                    "target_log1p": CFG.target_log1p,
                    "partition_by": CFG.partition_by,
                    "num_server_rounds": CFG.num_server_rounds,
                    "xgb_local_rounds": CFG.xgb_local_rounds,
                    "fraction_train": CFG.fraction_train,
                    "fraction_evaluate": CFG.fraction_evaluate,
                    "learning_rate": CFG.learning_rate,
                    "best_val_mae": float(best_val_metrics["mae"]),
                    "best_val_rmse": float(best_val_metrics["rmse"]),
                    "best_model_round": int(_BEST_ROUND),
                    "splits": {
                        "train": best_train_metrics,
                        "val": best_val_metrics,
                        "test": best_test_metrics,
                    },
                    "target_summaries": split_summaries,
                    "client_target_summaries": client_split_summaries,
                },
                f,
                indent=2,
            )
        return

    trainloader, valloader, testloader, preprocessor, train_df, val_df, test_df = (
        load_global_eval_loaders(
            data_path=CFG.data_path,
            batch_size=CFG.batch_size,
            seed=CFG.seed,
        )
    )
    split_summaries = split_target_summaries(train_df, val_df, test_df)
    client_split_summaries = client_split_target_summaries(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        partition_by=CFG.partition_by,
    )
    print("Global split and target summary:")
    for split_name in ["train", "val", "test"]:
        summary = split_summaries[split_name]
        print(
            f"  {split_name:>5}: n={summary['num_samples']:,}, "
            f"min={summary['target_min']:.4f}, max={summary['target_max']:.4f}, "
            f"mean={summary['target_mean']:.4f}, std={summary['target_std']:.4f}"
        )
    print(f"Per-client split summary (partition_by={CFG.partition_by}):")
    for client_id in sorted(client_split_summaries):
        client_summary = client_split_summaries[client_id]
        print(f"  client={client_id}")
        for split_name in ["train", "val", "test"]:
            summary = client_summary[split_name]
            print(
                f"    {split_name:>5}: n={summary['num_samples']:,}, "
                f"min={summary['target_min']:.4f}, max={summary['target_max']:.4f}, "
                f"mean={summary['target_mean']:.4f}, std={summary['target_std']:.4f}"
            )

    model = get_model(CFG.model_name, preprocessor, use_embeddings=CFG.use_embeddings)

    strategy = FedAvg(
        fraction_train=CFG.fraction_train,
        fraction_evaluate=CFG.fraction_evaluate,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.state_dict()),
        train_config=ConfigRecord({"lr": CFG.learning_rate}),
        num_rounds=CFG.num_server_rounds,
        evaluate_fn=make_centralized_evaluate_fn_torch(),
    )

    best_model_path, final_model_path, history_path, metrics_path = make_server_paths()

    torch.save(result.arrays.to_torch_state_dict(), final_model_path)
    pd.DataFrame(_HISTORY).to_csv(history_path, index=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model = get_model(
        CFG.model_name, preprocessor, use_embeddings=CFG.use_embeddings
    )
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))

    best_train_metrics = evaluate(
        best_model,
        trainloader,
        device=device,
        loss_name=CFG.loss,
        lambda_early=CFG.lambda_early,
    )
    best_val_metrics = evaluate(
        best_model,
        valloader,
        device=device,
        loss_name=CFG.loss,
        lambda_early=CFG.lambda_early,
    )
    best_test_metrics = evaluate(
        best_model,
        testloader,
        device=device,
        loss_name=CFG.loss,
        lambda_early=CFG.lambda_early,
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": CFG.model_name,
                "use_embeddings": CFG.use_embeddings,
                "cyclical": CFG.cyclical,
                "preprocessing": CFG.preprocessing,
                "seed": CFG.seed,
                "target_log1p": CFG.target_log1p,
                "partition_by": CFG.partition_by,
                "num_server_rounds": CFG.num_server_rounds,
                "local_epochs": CFG.local_epochs,
                "batch_size": CFG.batch_size,
                "fraction_train": CFG.fraction_train,
                "fraction_evaluate": CFG.fraction_evaluate,
                "learning_rate": CFG.learning_rate,
                "loss": CFG.loss,
                "lambda_early": CFG.lambda_early,
                "best_val_mae": float(best_val_metrics["mae"]),
                "best_val_rmse": float(best_val_metrics["rmse"]),
                "best_model_round": int(_BEST_ROUND),
                "splits": {
                    "train": best_train_metrics,
                    "val": best_val_metrics,
                    "test": best_test_metrics,
                },
                "target_summaries": split_summaries,
                "client_target_summaries": client_split_summaries,
            },
            f,
            indent=2,
        )
