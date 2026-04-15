from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from federated_acn.ml.dataset import TARGET_COL


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MASE denominator uses one-step naive in-sample forecast error.
    # If y_true has only one sample or no variation in step-to-step deltas,
    # avoid divide-by-zero by returning NaN.
    if y_true.shape[0] < 2:
        mase = float("nan")
    else:
        naive_mae = mean_absolute_error(y_true[1:], y_true[:-1])
        mase = float(mae / naive_mae) if naive_mae > 0 else float("nan")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "mase": float(mase),
    }


def summarize_target(df: pd.DataFrame) -> Dict[str, float]:
    target = df[TARGET_COL].astype(float)
    return {
        "num_samples": int(len(target)),
        "target_min": float(target.min()),
        "target_max": float(target.max()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std(ddof=0)),
    }
