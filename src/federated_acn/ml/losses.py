import torch

from federated_acn.ml.target_transform import inverse_transform_target_tensor


def physics_constrained_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    base_loss_fn,
    early_energy_kwh: torch.Tensor,
    target_log1p: bool,
    lambda_early: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base_loss = base_loss_fn(pred, target)

    pred_kwh = inverse_transform_target_tensor(pred, use_log1p=target_log1p)
    pen_early = torch.relu(early_energy_kwh - pred_kwh) ** 2
    loss_early = pen_early.mean()

    total_loss = base_loss + (lambda_early * loss_early)
    return total_loss, base_loss, loss_early
