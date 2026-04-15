from typing import Dict, List

import torch
import torch.nn as nn


def embedding_dim(cardinality: int) -> int:
    return min(32, max(4, (cardinality // 2) + 1))


class TabularTransformer(nn.Module):
    def __init__(
        self,
        num_numerical_features: int,
        categorical_cardinalities: Dict[str, int],
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_numerical_features = num_numerical_features
        self.cat_cols = list(categorical_cardinalities.keys())

        self.num_linears = nn.ModuleList(
            [nn.Linear(1, d_model) for _ in range(num_numerical_features)]
        )

        self.cat_embeddings = nn.ModuleDict(
            {
                col: nn.Embedding(cardinality, d_model)
                for col, cardinality in categorical_cardinalities.items()
            }
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = None  # created on first forward if needed

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.output_activation = nn.Softplus()

    def _build_pos_embedding(
        self, seq_len: int, d_model: int, device: torch.device
    ) -> torch.Tensor:
        pos = torch.zeros(1, seq_len, d_model, device=device)
        nn.init.normal_(pos, mean=0.0, std=0.02)
        return pos

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_num = batch["x_num"]  # [B, F]
        bsz = x_num.shape[0]
        device = x_num.device

        tokens: List[torch.Tensor] = []

        for i in range(self.num_numerical_features):
            feat = x_num[:, i : i + 1]
            tok = self.num_linears[i](feat).unsqueeze(1)  # [B,1,D]
            tokens.append(tok)

        for col in self.cat_cols:
            tok = self.cat_embeddings[col](batch[col]).unsqueeze(1)  # [B,1,D]
            tokens.append(tok)

        x = torch.cat(tokens, dim=1)  # [B,T,D]

        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B,T+1,D]

        if self.pos_embedding is None or self.pos_embedding.shape[1] != x.shape[1]:
            self.pos_embedding = self._build_pos_embedding(
                x.shape[1], x.shape[2], device
            )

        x = x + self.pos_embedding
        x = self.encoder(x)
        cls_out = x[:, 0, :]
        y = self.output_activation(self.head(cls_out)).squeeze(1)
        return y
