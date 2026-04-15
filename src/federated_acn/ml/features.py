import numpy as np
import pandas as pd

CYCLICAL_SOURCE_COLS = [
    "connect_hour",
    "connect_weekday",
    "connect_month",
    "connect_dayofyear",
]

CYCLICAL_FEATURE_COLS = [
    "connect_hour_sin",
    "connect_hour_cos",
    "connect_weekday_sin",
    "connect_weekday_cos",
    "connect_month_sin",
    "connect_month_cos",
    "connect_dayofyear_sin",
    "connect_dayofyear_cos",
]


def _add_sin_cos(df: pd.DataFrame, col: str, period: int) -> None:
    values = pd.to_numeric(df[col], errors="coerce")
    radians = 2.0 * np.pi * values / float(period)
    df[f"{col}_sin"] = np.sin(radians)
    df[f"{col}_cos"] = np.cos(radians)


def apply_cyclical_time_features(df: pd.DataFrame, cyclical: bool = True) -> pd.DataFrame:
    out = df.copy()
    if not cyclical:
        return out

    if "connect_hour" in out.columns:
        _add_sin_cos(out, "connect_hour", period=24)
    if "connect_weekday" in out.columns:
        _add_sin_cos(out, "connect_weekday", period=7)
    if "connect_month" in out.columns:
        _add_sin_cos(out, "connect_month", period=12)
    if "connect_dayofyear" in out.columns:
        _add_sin_cos(out, "connect_dayofyear", period=366)

    cols_to_drop = [c for c in CYCLICAL_SOURCE_COLS if c in out.columns]
    if cols_to_drop:
        out = out.drop(columns=cols_to_drop)

    return out
