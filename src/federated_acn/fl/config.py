from dataclasses import dataclass
import os


@dataclass
class FLConfig:
    data_path: str = "./acn_fl_data/processed/dataset_early_session_caltech.parquet"
    partition_by: str = "station"
    model_name: str = "mlp"  # mlp | transformer | cnn1d | gru | linear_regression | random_forest | xgboost
    batch_size: int = 64
    local_epochs: int = 3
    num_server_rounds: int = 50
    learning_rate: float = 1e-3
    loss: str = "mae"
    fraction_train: float = 0.1
    fraction_evaluate: float = 1.0
    seed: int = 42
    use_embeddings: bool = True
    min_total_sessions_per_client: int = 3
    cyclical: bool = True
    preprocessing: str = "standardization"
    target_log1p: bool = False
    lambda_early: float = 1.0
    xgb_local_rounds: int = 1
    xgb_train_method: str = "bagging"
    linear_local_epochs: int = 1


CFG = FLConfig()

_ENV_PREFIX = "FEDERATED_ACN_FL_"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def apply_env_overrides() -> None:
    """Override CFG values from process environment (used by Ray workers)."""
    for field_name, field_def in CFG.__dataclass_fields__.items():
        env_key = f"{_ENV_PREFIX}{field_name.upper()}"
        raw = os.getenv(env_key)
        if raw is None:
            continue

        current = getattr(CFG, field_name)
        if isinstance(current, bool):
            value = _parse_bool(raw)
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        else:
            value = raw

        setattr(CFG, field_name, value)


def write_env_overrides() -> None:
    """Write current CFG into process environment for child worker processes."""
    for field_name in CFG.__dataclass_fields__:
        value = getattr(CFG, field_name)
        os.environ[f"{_ENV_PREFIX}{field_name.upper()}"] = str(value)
