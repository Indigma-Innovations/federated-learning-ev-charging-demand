import numpy as np
import torch
import xgboost as xgb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from federated_acn.fl.config import CFG, apply_env_overrides
from federated_acn.fl.task import (
    create_linear_regression_model,
    evaluate,
    evaluate_linear_model,
    evaluate_xgb,
    fit_linear_model_with_params,
    get_linear_model_params,
    get_model,
    get_xgb_params,
    load_partition,
    load_partition_linear,
    load_partition_xgb,
    train_one_round,
)

app = ClientApp()

def _local_boost_xgb(
    bst_input: xgb.Booster,
    num_local_round: int,
    train_dmatrix: xgb.DMatrix,
    train_method: str,
) -> tuple[xgb.Booster, xgb.Booster]:
    for _ in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())

    bst_full = bst_input

    if train_method == "bagging":
        bst_to_send = bst_input[
            bst_input.num_boosted_rounds() - num_local_round : bst_input.num_boosted_rounds()
        ]
    else:
        bst_to_send = bst_input

    return bst_full, bst_to_send


@app.train()
def train(msg: Message, context: Context) -> Message:
    apply_env_overrides()
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    if CFG.model_name == "linear_regression":
        x_train, y_train, x_val, y_val, _, _ = load_partition_linear(
            data_path=CFG.data_path,
            partition_id=partition_id,
            num_partitions=num_partitions,
            partition_by=CFG.partition_by,
            seed=CFG.seed,
            min_total_sessions=CFG.min_total_sessions_per_client,
        )

        model = create_linear_regression_model()

        global_params = msg.content["arrays"].to_numpy_ndarrays()

        model = fit_linear_model_with_params(
            model=model,
            x=x_train,
            y=y_train,
            params=global_params,
            local_epochs=CFG.linear_local_epochs,
        )

        train_metrics = evaluate_linear_model(model, x_train, y_train)
        val_metrics = evaluate_linear_model(model, x_val, y_val)

        content = RecordDict(
            {
                "arrays": ArrayRecord(get_linear_model_params(model)),
                "metrics": MetricRecord(
                    {
                        "train_mae": float(train_metrics["mae"]),
                        "train_rmse": float(train_metrics["rmse"]),
                        "val_mae": float(val_metrics["mae"]),
                        "val_rmse": float(val_metrics["rmse"]),
                        "val_mape": float(val_metrics["mape"]),
                        "val_mase": float(val_metrics["mase"]),
                        "val_r2": float(val_metrics["r2"]),
                        "num-examples": int(len(x_train)),
                    }
                ),
            }
        )
        return Message(content=content, reply_to=msg)

    if CFG.model_name == "xgboost":
        train_dmatrix, val_dmatrix, _, _, num_train, _ = load_partition_xgb(
            data_path=CFG.data_path,
            partition_id=partition_id,
            num_partitions=num_partitions,
            partition_by=CFG.partition_by,
            seed=CFG.seed,
            min_total_sessions=CFG.min_total_sessions_per_client,
        )

        params = get_xgb_params()
        train_method = getattr(CFG, "xgb_train_method", "bagging")
        num_local_round = int(CFG.xgb_local_rounds)

        server_round = int(msg.content["config"]["server-round"])

        if server_round == 1:
            bst_full = xgb.train(
                params=params,
                dtrain=train_dmatrix,
                num_boost_round=num_local_round,
            )
            bst_to_send = bst_full
        else:
            bst = xgb.Booster(params=params)
            global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
            bst.load_model(global_model)
            bst_full, bst_to_send = _local_boost_xgb(
                bst_input=bst,
                num_local_round=num_local_round,
                train_dmatrix=train_dmatrix,
                train_method=train_method,
            )

        train_metrics = evaluate_xgb(bst_full, train_dmatrix)
        val_metrics = evaluate_xgb(bst_full, val_dmatrix)

        local_model = bst_to_send.save_raw("json")
        model_np = np.frombuffer(local_model, dtype=np.uint8)

        content = RecordDict(
            {
                "arrays": ArrayRecord([model_np]),
                "metrics": MetricRecord(
                    {
                        "train_mae": float(train_metrics["mae"]),
                        "train_rmse": float(train_metrics["rmse"]),
                        "val_mae": float(val_metrics["mae"]),
                        "val_rmse": float(val_metrics["rmse"]),
                        "val_mape": float(val_metrics["mape"]),
                        "val_mase": float(val_metrics["mase"]),
                        "val_r2": float(val_metrics["r2"]),
                        "num-examples": int(num_train),
                    }
                ),
            }
        )
        return Message(content=content, reply_to=msg)

    trainloader, valloader, preprocessor, _ = load_partition(
        data_path=CFG.data_path,
        partition_id=partition_id,
        num_partitions=num_partitions,
        partition_by=CFG.partition_by,
        batch_size=CFG.batch_size,
        seed=CFG.seed,
        min_total_sessions=CFG.min_total_sessions_per_client,
    )

    model = get_model(CFG.model_name, preprocessor, use_embeddings=CFG.use_embeddings)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    lr = float(msg.content["config"]["lr"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loss = train_one_round(
        model=model,
        trainloader=trainloader,
        epochs=CFG.local_epochs,
        lr=lr,
        device=device,
        loss_name=CFG.loss,
        lambda_early=CFG.lambda_early,
    )
    val_metrics = evaluate(
        model,
        valloader,
        device=device,
        loss_name=CFG.loss,
        lambda_early=CFG.lambda_early,
    )

    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(
                {
                    "train_loss": float(train_loss),
                    "val_mae": float(val_metrics["mae"]),
                    "val_rmse": float(val_metrics["rmse"]),
                    "val_mape": float(val_metrics["mape"]),
                    "val_mase": float(val_metrics["mase"]),
                    "val_r2": float(val_metrics["r2"]),
                    "num-examples": len(trainloader.dataset),
                }
            ),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_fn(msg: Message, context: Context) -> Message:
    apply_env_overrides()
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    if CFG.model_name == "linear_regression":
        _, _, x_val, y_val, _, _ = load_partition_linear(
            data_path=CFG.data_path,
            partition_id=partition_id,
            num_partitions=num_partitions,
            partition_by=CFG.partition_by,
            seed=CFG.seed,
            min_total_sessions=CFG.min_total_sessions_per_client,
        )

        model = create_linear_regression_model()
        global_params = msg.content["arrays"].to_numpy_ndarrays()

        # Build a fitted model from received params by doing a zero-update style fit init
        # using coef_init/intercept_init
        model = fit_linear_model_with_params(
            model=model,
            x=x_val[:1],
            y=y_val[:1],
            params=global_params,
            local_epochs=0,
        )

        model.coef_ = np.asarray(global_params[0], dtype=np.float64)
        model.intercept_ = np.asarray(global_params[1], dtype=np.float64)
        model.n_features_in_ = model.coef_.shape[0]

        val_metrics = evaluate_linear_model(model, x_val, y_val)

        content = RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "eval_mae": float(val_metrics["mae"]),
                        "eval_rmse": float(val_metrics["rmse"]),
                        "eval_mape": float(val_metrics["mape"]),
                        "eval_mase": float(val_metrics["mase"]),
                        "eval_r2": float(val_metrics["r2"]),
                        "num-examples": int(len(x_val)),
                    }
                )
            }
        )
        return Message(content=content, reply_to=msg)

    if CFG.model_name == "xgboost":
        _, val_dmatrix, _, _, _, num_val = load_partition_xgb(
            data_path=CFG.data_path,
            partition_id=partition_id,
            num_partitions=num_partitions,
            partition_by=CFG.partition_by,
            seed=CFG.seed,
            min_total_sessions=CFG.min_total_sessions_per_client,
        )

        params = get_xgb_params()
        bst = xgb.Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
        bst.load_model(global_model)

        val_metrics = evaluate_xgb(bst, val_dmatrix)

        content = RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "eval_mae": float(val_metrics["mae"]),
                        "eval_rmse": float(val_metrics["rmse"]),
                        "eval_mape": float(val_metrics["mape"]),
                        "eval_mase": float(val_metrics["mase"]),
                        "eval_r2": float(val_metrics["r2"]),
                        "num-examples": int(num_val),
                    }
                )
            }
        )
        return Message(content=content, reply_to=msg)

    _, valloader, preprocessor, _ = load_partition(
        data_path=CFG.data_path,
        partition_id=partition_id,
        num_partitions=num_partitions,
        partition_by=CFG.partition_by,
        batch_size=CFG.batch_size,
        seed=CFG.seed,
        min_total_sessions=CFG.min_total_sessions_per_client,
    )

    model = get_model(CFG.model_name, preprocessor, use_embeddings=CFG.use_embeddings)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_metrics = evaluate(
        model,
        valloader,
        device=device,
        loss_name=CFG.loss,
        lambda_early=CFG.lambda_early,
    )

    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "eval_mae": float(val_metrics["mae"]),
                    "eval_rmse": float(val_metrics["rmse"]),
                    "eval_mape": float(val_metrics["mape"]),
                    "eval_mase": float(val_metrics["mase"]),
                    "eval_r2": float(val_metrics["r2"]),
                    "num-examples": len(valloader.dataset),
                }
            )
        }
    )
    return Message(content=content, reply_to=msg)
