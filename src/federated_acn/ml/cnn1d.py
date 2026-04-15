from typing import Dict, List

import torch
import torch.nn as nn


class TabularCNN1D(nn.Module):
    """1D CNN over feature tokens (numerical and categorical embeddings)."""

    def __init__(
        self,
        num_numerical_features: int,
        categorical_cardinalities: Dict[str, int],
        token_dim: int = 16,
        hidden_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_numerical_features = num_numerical_features
        self.cat_cols = list(categorical_cardinalities.keys())

        self.num_linears = nn.ModuleList(
            [nn.Linear(1, token_dim) for _ in range(num_numerical_features)]
        )
        self.cat_embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(cardinality, token_dim)
                for col, cardinality in categorical_cardinalities.items()
            }
        )

        self.conv = nn.Sequential(
            nn.Conv1d(token_dim, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.output_activation = nn.Softplus()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_num = batch["x_num"]  # [B, F]
        tokens: List[torch.Tensor] = []

        for i in range(self.num_numerical_features):
            feat = x_num[:, i : i + 1]
            tok = self.num_linears[i](feat).unsqueeze(1)  # [B,1,D]
            tokens.append(tok)

        for col in self.cat_cols:
            tok = self.cat_embeddings[col](batch[col]).unsqueeze(1)  # [B,1,D]
            tokens.append(tok)

        x = torch.cat(tokens, dim=1)  # [B,T,D]
        x = x.transpose(1, 2)  # [B,D,T]
        x = self.conv(x)
        x = torch.mean(x, dim=2)  # Global average pool, [B,C]
        x = self.dropout(x)
        y = self.output_activation(self.head(x)).squeeze(1)
        return y
