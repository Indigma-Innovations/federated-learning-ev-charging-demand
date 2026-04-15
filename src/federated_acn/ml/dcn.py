from typing import Dict, List

import torch
import torch.nn as nn

class CrossLayer(nn.Module):
    """
    Standard DCN-style cross layer:
        x_{l+1} = x_0 * (w^T x_l + b) + x_l

    where:
      - x_0 is the original input
      - x_l is the current layer representation
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        # x0: [B, D]
        # xl: [B, D]
        scale = self.weight(xl)  # [B, 1]
        return x0 * scale + self.bias + xl  # [B, D]


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepCrossRegressor(nn.Module):
    """
    Deep & Cross Network for tabular regression.

    Input:
        batch["x_num"]               -> [B, F_num]
        batch[col] for categorical   -> [B]

    Output:
        positive scalar regression prediction [B]
    """

    def __init__(
            self,
            num_numerical_features: int,
            categorical_cardinalities: Dict[str, int],
            embedding_dim_rule: str = "auto",
            cross_layers: int = 3,
            deep_hidden_dims: List[int] | None = None,
            dropout: float = 0.1,
            positive_output: bool = True,
    ) -> None:
        super().__init__()

        if deep_hidden_dims is None:
            deep_hidden_dims = [128, 64]

        self.num_numerical_features = num_numerical_features
        self.cat_cols = list(categorical_cardinalities.keys())
        self.positive_output = positive_output

        # Small embeddings for categorical columns
        self.cat_embeddings = nn.ModuleDict()
        self.cat_embedding_dims: Dict[str, int] = {}

        total_cat_dim = 0
        for col, cardinality in categorical_cardinalities.items():
            if embedding_dim_rule == "auto":
                emb_dim = min(16, max(4, (cardinality + 1) // 8))
            else:
                raise ValueError(
                    f"Unsupported embedding_dim_rule={embedding_dim_rule}"
                )

            self.cat_embedding_dims[col] = emb_dim
            self.cat_embeddings[col] = nn.Embedding(cardinality, emb_dim)
            total_cat_dim += emb_dim

        self.input_dim = num_numerical_features + total_cat_dim

        # Cross branch
        self.cross_net = nn.ModuleList(
            [CrossLayer(self.input_dim) for _ in range(cross_layers)]
        )

        # Deep branch
        deep_layers: List[nn.Module] = []
        prev_dim = self.input_dim
        for hidden_dim in deep_hidden_dims:
            deep_layers.append(MLPBlock(prev_dim, hidden_dim, dropout=dropout))
            prev_dim = hidden_dim
        self.deep_net = nn.Sequential(*deep_layers)

        # Final head uses concatenated [cross_out, deep_out]
        final_in_dim = self.input_dim + prev_dim
        self.head = nn.Sequential(
            nn.Linear(final_in_dim, max(32, final_in_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, final_in_dim // 2), 1),
        )

        self.output_activation = nn.Softplus() if positive_output else nn.Identity()

    def _build_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_num = batch["x_num"]  # [B, F_num]
        features = [x_num]

        for col in self.cat_cols:
            emb = self.cat_embeddings[col](batch[col])  # [B, emb_dim]
            features.append(emb)

        x = torch.cat(features, dim=1)  # [B, D]
        return x

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x0 = self._build_input(batch)  # [B, D]

        # Cross branch
        xc = x0
        for layer in self.cross_net:
            xc = layer(x0, xc)  # [B, D]

        # Deep branch
        xd = self.deep_net(x0)  # [B, H]

        # Combine
        x = torch.cat([xc, xd], dim=1)  # [B, D + H]
        y = self.head(x).squeeze(1)  # [B]
        y = self.output_activation(y)
        return y
