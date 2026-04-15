from typing import Dict, List

import torch
import torch.nn as nn


class TabularGRU(nn.Module):
    """Lightweight GRU over tabular feature tokens."""

    def __init__(
        self,
        num_numerical_features: int,
        categorical_cardinalities: Dict[str, int],
        token_dim: int = 16,
        hidden_dim: int = 32,
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

        self.gru = nn.GRU(
            input_size=token_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.output_activation = nn.Softplus()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_num = batch["x_num"]  # [B,F]
        tokens: List[torch.Tensor] = []

        for i in range(self.num_numerical_features):
            feat = x_num[:, i : i + 1]
            tok = self.num_linears[i](feat).unsqueeze(1)  # [B,1,D]
            tokens.append(tok)

        for col in self.cat_cols:
            tok = self.cat_embeddings[col](batch[col]).unsqueeze(1)  # [B,1,D]
            tokens.append(tok)

        x = torch.cat(tokens, dim=1)  # [B,T,D]
        _, h = self.gru(x)
        y = self.output_activation(self.head(h[-1])).squeeze(1)
        return y
