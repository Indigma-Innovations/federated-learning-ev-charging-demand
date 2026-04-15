from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from federated_acn.ml.target_transform import transform_target_numpy

TARGET_COL = "kwh_delivered"
EARLY_ENERGY_COL = "approx_energy_first_window_kwh"

CATEGORICAL_COLS = [
    "station_id",
    "cluster_id",
]

NUMERICAL_COLS = [
    "connect_hour",
    "connect_weekday",
    "connect_month",
    "connect_dayofyear",
    "connect_hour_sin",
    "connect_hour_cos",
    "connect_weekday_sin",
    "connect_weekday_cos",
    "connect_month_sin",
    "connect_month_cos",
    "connect_dayofyear_sin",
    "connect_dayofyear_cos",
    "connect_is_weekend",
    "kwh_requested",
    "minutes_available",
    "requested_departure_minutes_from_connect",
    "early_window_min",
    "n_points_window",
    "n_current_points",
    "n_pilot_points",
    "window_observed_minutes",
    "mean_current",
    "max_current",
    "std_current",
    "min_current",
    "last_current",
    "mean_pilot",
    "max_pilot",
    "std_pilot",
    "min_pilot",
    "last_pilot",
    "mean_utilization",
    "max_utilization",
    "current_slope_per_sec",
    "pilot_slope_per_sec",
    "approx_energy_first_window_kwh",
    "has_enough_early_data",
]


@dataclass
class TabularPreprocessor:
    numerical_cols: List[str]
    categorical_cols: List[str]
    num_means: Optional[Dict[str, float]] = None
    num_stds: Optional[Dict[str, float]] = None
    cat_maps: Optional[Dict[str, Dict[str, int]]] = None
    passthrough_numerical_cols: Optional[List[str]] = None
    preprocessing: Literal["standardization", "minmax", "none"] = "standardization"
    num_mins: Optional[Dict[str, float]] = None
    num_maxs: Optional[Dict[str, float]] = None

    def fit(self, df: pd.DataFrame) -> None:
        self.num_means = {}
        self.num_stds = {}
        self.cat_maps = {}
        self.num_mins = {}
        self.num_maxs = {}
        passthrough = set(self.passthrough_numerical_cols or [])

        for col in self.numerical_cols:
            vals = pd.to_numeric(df[col], errors="coerce")
            mean = float(vals.mean()) if not vals.isna().all() else 0.0
            std = float(vals.std()) if not vals.isna().all() else 1.0
            if std == 0.0 or np.isnan(std):
                std = 1.0
            col_min = float(vals.min()) if not vals.isna().all() else 0.0
            col_max = float(vals.max()) if not vals.isna().all() else 1.0
            if np.isnan(col_min):
                col_min = 0.0
            if np.isnan(col_max):
                col_max = col_min + 1.0
            if col_max == col_min:
                col_max = col_min + 1.0

            self.num_means[col] = mean
            self.num_stds[col] = 1.0 if col in passthrough else std
            self.num_mins[col] = col_min
            self.num_maxs[col] = col_max

        for col in self.categorical_cols:
            values = df[col].fillna("UNK").astype(str).unique().tolist()
            # 0 reserved for unknown
            mapping = {v: i + 1 for i, v in enumerate(sorted(values))}
            self.cat_maps[col] = mapping

    def transform_numerical(self, df: pd.DataFrame) -> np.ndarray:
        assert self.num_means is not None and self.num_stds is not None
        assert self.num_mins is not None and self.num_maxs is not None
        arrays = []
        passthrough = set(self.passthrough_numerical_cols or [])
        for col in self.numerical_cols:
            vals = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(self.num_means[col])
                .astype(float)
            )
            if col in passthrough or self.preprocessing == "none":
                vals = vals.fillna(0.0)
            elif self.preprocessing == "standardization":
                vals = (vals - self.num_means[col]) / self.num_stds[col]
            elif self.preprocessing == "minmax":
                vals = (vals - self.num_mins[col]) / (self.num_maxs[col] - self.num_mins[col])
            else:
                raise ValueError(f"Unsupported preprocessing mode: {self.preprocessing}")
            arrays.append(vals.to_numpy(dtype=np.float32))
        if not arrays:
            return np.zeros((len(df), 0), dtype=np.float32)
        return np.stack(arrays, axis=1)

    def transform_categorical(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        assert self.cat_maps is not None
        out: Dict[str, np.ndarray] = {}
        for col in self.categorical_cols:
            mapping = self.cat_maps[col]
            vals = df[col].fillna("UNK").astype(str).map(lambda x: mapping.get(x, 0))
            out[col] = vals.to_numpy(dtype=np.int64)
        return out

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        self.fit(df)
        return self.transform_numerical(df), self.transform_categorical(df)

    def categorical_cardinalities(self) -> Dict[str, int]:
        assert self.cat_maps is not None
        return {
            col: max(mapping.values(), default=0) + 1
            for col, mapping in self.cat_maps.items()
        }


class EarlySessionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        preprocessor: TabularPreprocessor,
        target_log1p: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.target_log1p = target_log1p

        self.x_num = preprocessor.transform_numerical(self.df)
        self.x_cat = preprocessor.transform_categorical(self.df)
        y_values = (
            pd.to_numeric(self.df[TARGET_COL], errors="coerce")
            .astype(float)
            .to_numpy(dtype=np.float32)
        )
        self.y = transform_target_numpy(y_values, use_log1p=self.target_log1p)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        early_energy = float(
            pd.to_numeric(
                self.df.iloc[idx].get(EARLY_ENERGY_COL, 0.0), errors="coerce"
            )
        )
        if np.isnan(early_energy):
            early_energy = 0.0
        item = {
            "x_num": torch.tensor(self.x_num[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "early_energy_kwh": torch.tensor(early_energy, dtype=torch.float32),
        }
        for col in self.preprocessor.categorical_cols:
            item[col] = torch.tensor(self.x_cat[col][idx], dtype=torch.long)
        return item
