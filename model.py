"""Shared DeBERTa encoder with leaf and family classification heads."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from . import label_maps as LM


class AlaphiaMultitaskModel(nn.Module):
    """Shared encoder, dropout on pooled [CLS], leaf heads plus family heads."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        dropout: float = 0.2,
        from_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if from_pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:
            # Config-only init — encoder weights are populated by load_state_dict
            # immediately after instantiation (used for local inference to avoid
            # redundant weight download when the full checkpoint is already on disk).
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(config)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.emotion_head = nn.Linear(hidden, LM.NUM_EMOTION)
        self.emotion_family_head = nn.Linear(hidden, LM.NUM_EMOTION_FAMILY)
        self.need_head = nn.Linear(hidden, LM.NUM_NEED)
        self.need_family_head = nn.Linear(hidden, LM.NUM_NEED_FAMILY)
        self.money_theme_head = nn.Linear(hidden, LM.NUM_MONEY_THEME)
        self.life_context_head = nn.Linear(hidden, LM.NUM_LIFE_CONTEXT)
        self.life_context_family_head = nn.Linear(hidden, LM.NUM_LIFE_CONTEXT_FAMILY)
        self.status_head = nn.Linear(hidden, LM.NUM_STATUS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DeBERTa: first token is effective CLS
        pooled = out.last_hidden_state[:, 0, :]
        x = self.dropout(pooled)
        return (
            self.emotion_head(x),
            self.emotion_family_head(x),
            self.need_head(x),
            self.need_family_head(x),
            self.money_theme_head(x),
            self.life_context_head(x),
            self.life_context_family_head(x),
            self.status_head(x),
        )

    def logits_tuple_to_concat(self, logits: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Concatenate head logits for TorchScript export (fixed layout)."""
        return torch.cat(logits, dim=-1)


def concat_logit_layout() -> List[Tuple[str, int]]:
    """Order and sizes of segments in concatenated export tensor."""
    return [
        ("emotion", LM.NUM_EMOTION),
        ("emotion_family", LM.NUM_EMOTION_FAMILY),
        ("need", LM.NUM_NEED),
        ("need_family", LM.NUM_NEED_FAMILY),
        ("money_theme", LM.NUM_MONEY_THEME),
        ("life_context", LM.NUM_LIFE_CONTEXT),
        ("life_context_family", LM.NUM_LIFE_CONTEXT_FAMILY),
        ("status", LM.NUM_STATUS),
    ]


def split_concat_logits(tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Split exported concatenated logits back into leaf and family heads."""
    o = 0
    chunks = []
    for _, n in concat_logit_layout():
        chunks.append(tensor[:, o : o + n])
        o += n
    return tuple(chunks)  # type: ignore[return-value]
