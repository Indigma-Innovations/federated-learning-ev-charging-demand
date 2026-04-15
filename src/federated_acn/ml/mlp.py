from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def embedding_dim(cardinality: int) -> int:
    return min(32, max(4, (cardinality // 2) + 1))


class TabularMLP(nn.Module):
    def __init__(
        self,
        num_numerical_features: int,
        categorical_cardinalities: Dict[str, int],
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.cat_cols = list(categorical_cardinalities.keys())
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0

        for col, card in categorical_cardinalities.items():
            emb_dim = embedding_dim(card)
            self.embeddings[col] = nn.Embedding(
                num_embeddings=card, embedding_dim=emb_dim
            )
            total_emb_dim += emb_dim

        input_dim = num_numerical_features + total_emb_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.output_activation = nn.Softplus()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_num = batch["x_num"]
        parts = [x_num]

        for col in self.cat_cols:
            parts.append(self.embeddings[col](batch[col]))

        x = torch.cat(parts, dim=1)
        y = self.output_activation(self.net(x)).squeeze(1)
        return y

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h1, h2 = self.fc1(h).chunk(2, dim=-1)
        h = h1 * F.gelu(h2)   # lightweight gated activation
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


class TabularResMLP(nn.Module):
    def __init__(
        self,
        num_numerical_features: int,
        categorical_cardinalities: Dict[str, int],
        hidden_dim: int = 128,
        num_blocks: int = 3,
        emb_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cat_cols = list(categorical_cardinalities.keys())

        self.num_proj = nn.Sequential(
            nn.Linear(num_numerical_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.cat_embeddings = nn.ModuleDict({
            col: nn.Embedding(cardinality, emb_dim)
            for col, cardinality in categorical_cardinalities.items()
        })

        cat_total_dim = emb_dim * len(self.cat_cols)
        self.input_proj = nn.Linear(hidden_dim + cat_total_dim, hidden_dim)

        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.output_activation = nn.Softplus()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_num = batch["x_num"]
        x_num = self.num_proj(x_num)

        cat_toks: List[torch.Tensor] = []
        for col in self.cat_cols:
            cat_toks.append(self.cat_embeddings[col](batch[col]))

        if cat_toks:
            x_cat = torch.cat(cat_toks, dim=1)
            x = torch.cat([x_num, x_cat], dim=1)
        else:
            x = x_num

        x = self.input_proj(x)
        x = self.blocks(x)
        y = self.head(x)
        y = self.output_activation(y).squeeze(1)
        return y
