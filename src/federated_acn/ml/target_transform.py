import numpy as np
import torch


def transform_target_numpy(y: np.ndarray, use_log1p: bool) -> np.ndarray:
    if not use_log1p:
        return y.astype(np.float32, copy=False)
    return np.log1p(np.clip(y, a_min=0.0, a_max=None)).astype(np.float32, copy=False)


def inverse_transform_target_numpy(y: np.ndarray, use_log1p: bool) -> np.ndarray:
    if not use_log1p:
        return y.astype(np.float32, copy=False)
    return np.clip(np.expm1(y), a_min=0.0, a_max=None).astype(np.float32, copy=False)


def inverse_transform_target_tensor(y: torch.Tensor, use_log1p: bool) -> torch.Tensor:
    if not use_log1p:
        return y
    return torch.clamp(torch.expm1(y), min=0.0)
