"""PyTorch architectures shared by training and inference."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


FINGERPRINT_SIZE = 2048
ODOR_LABEL_COUNT = 113
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
LSTM_DROPOUT = 0.2


class OdorPredictor(nn.Module):
    """Multi-label odor descriptor model."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FINGERPRINT_SIZE, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, ODOR_LABEL_COUNT),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class SMILES_LSTM(nn.Module):
    """Character-level SMILES sequence model."""

    def __init__(self, vocab_size: int, pad_idx: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=LSTM_DROPOUT,
        )
        self.output = nn.Linear(HIDDEN_SIZE, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(token_ids)
        recurrent_output, hidden = self.lstm(embedded, hidden)
        return self.output(recurrent_output), hidden


def select_device() -> torch.device:
    """Prefer Apple Metal and use CPU as the stable fallback."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
