"""PyTorch Dataset for JSON training files with leaf and family targets.

life_context is multi-label: the field may be a comma-separated string or a list of
strings. It is encoded as a float multi-hot vector of length NUM_LIFE_CONTEXT so the
life_context leaf head can be trained with BCEWithLogitsLoss. A second multi-hot
vector over NUM_LIFE_CONTEXT_FAMILY is derived automatically from the active leaf
labels. Emotion and need also emit derived single-label family targets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from . import label_maps as LM


def _life_context_multihot(raw: Any) -> torch.Tensor:
    """Convert a comma-separated string or list of label strings to a multi-hot float tensor."""
    if isinstance(raw, list):
        labels = [str(l).strip() for l in raw]
    else:
        labels = [l.strip() for l in str(raw).split(",")]
    vec = torch.zeros(LM.NUM_LIFE_CONTEXT, dtype=torch.float)
    for lbl in labels:
        if lbl and lbl in LM.life_context_to_id:
            vec[LM.life_context_to_id[lbl]] = 1.0
    return vec


def _life_context_family_multihot(raw: Any) -> torch.Tensor:
    """Derive a multi-hot family vector from the raw life-context labels."""
    if isinstance(raw, list):
        labels = [str(l).strip() for l in raw]
    else:
        labels = [l.strip() for l in str(raw).split(",")]
    vec = torch.zeros(LM.NUM_LIFE_CONTEXT_FAMILY, dtype=torch.float)
    for lbl in labels:
        if lbl and lbl in LM.life_context_to_id:
            vec[LM.life_context_family_id(lbl)] = 1.0
    return vec


class MultitaskDataset(Dataset):
    """
    Each record must contain: text, emotion, need, money_theme, life_context, status.
    - life_context may be a comma-separated string or list; encoded as multi-hot float vector.
    - emotion_family / need_family / life_context_family are derived from leaf labels.
    - Rows with no recognised life_context labels OR unknown single-label values are skipped.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        path = Path(data_path)
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            data = json.loads(raw)
            records = data if isinstance(data, list) else data.get("records", data.get("data", []))

        self.samples: List[Dict[str, Any]] = []
        skipped = 0
        for rec in records:
            if not all(
                k in rec and rec[k] is not None and str(rec[k]).strip() != ""
                for k in ("text", "emotion", "need", "money_theme", "life_context")
            ):
                skipped += 1
                continue
            try:
                emo = str(rec["emotion"]).strip()
                need = str(rec["need"]).strip()
                money = str(rec["money_theme"]).strip()
                lc_vec = _life_context_multihot(rec["life_context"])
                lc_family_vec = _life_context_family_multihot(rec["life_context"])
                if lc_vec.sum().item() == 0:
                    skipped += 1
                    continue

                # status is optional — null/missing → sentinel -1 (masked in loss)
                raw_stat = rec.get("status")
                if raw_stat is None or str(raw_stat).strip() == "":
                    status_id = -1
                else:
                    stat = str(raw_stat).strip().lower()
                    if stat not in LM.status_to_id:
                        stat = str(raw_stat).strip()
                    status_id = LM.status_to_id.get(stat, -1)

                self.samples.append(
                    {
                        "text": str(rec["text"]).strip(),
                        "emotion_id": LM.emotion_to_id[emo],
                        "emotion_family_id": LM.emotion_family_id(emo),
                        "need_id": LM.need_to_id[need],
                        "need_family_id": LM.need_family_id(need),
                        "money_theme_id": LM.money_theme_to_id[money],
                        "life_context_vec": lc_vec,
                        "life_context_family_vec": lc_family_vec,
                        "status_id": status_id,
                    }
                )
            except KeyError:
                skipped += 1
                continue

        if skipped:
            print(f"MultitaskDataset: skipped {skipped} incomplete/unknown-label rows")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        enc = self.tokenizer(
            s["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "emotion": torch.tensor(s["emotion_id"], dtype=torch.long),
            "emotion_family": torch.tensor(s["emotion_family_id"], dtype=torch.long),
            "need": torch.tensor(s["need_id"], dtype=torch.long),
            "need_family": torch.tensor(s["need_family_id"], dtype=torch.long),
            "money_theme": torch.tensor(s["money_theme_id"], dtype=torch.long),
            "life_context": s["life_context_vec"],           # float multi-hot (NUM_LIFE_CONTEXT,)
            "life_context_family": s["life_context_family_vec"],  # float multi-hot (NUM_LIFE_CONTEXT_FAMILY,)
            "status": torch.tensor(s["status_id"], dtype=torch.long),  # -1 = no label (masked)
        }
