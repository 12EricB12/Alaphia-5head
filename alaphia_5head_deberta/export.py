"""Export trained checkpoint to TorchScript (concatenated logits) for mobile/edge."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .model import AlaphiaMultitaskModel, concat_logit_layout


class TorchScriptWrapper(nn.Module):
    """Forward: input_ids, attention_mask -> concatenated logits (all five heads)."""

    def __init__(self, model: AlaphiaMultitaskModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_ids, attention_mask)
        return self.model.logits_tuple_to_concat(logits)


def export(
    checkpoint_path: Union[str, Path],
    output_path: Union[str, Path],
    device: str = "cpu",
) -> None:
    try:
        ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(checkpoint_path, map_location=device)

    model_name = ck.get("model_name", "microsoft/deberta-v3-small")
    base = AlaphiaMultitaskModel(model_name=model_name)
    base.load_state_dict(ck["model_state_dict"])
    base.eval()
    wrapped = TorchScriptWrapper(base)
    wrapped.eval()

    dev = torch.device(device)
    wrapped.to(dev)

    tok = AutoTokenizer.from_pretrained(model_name)
    enc = tok(
        "Journal entry export trace.",
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    dummy_ids = enc["input_ids"].to(dev)
    dummy_mask = enc["attention_mask"].to(dev)

    with torch.no_grad():
        traced = torch.jit.trace(wrapped, (dummy_ids, dummy_mask), strict=False)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))
    meta = {
        "logit_layout": [{"name": n, "dim": d} for n, d in concat_logit_layout()],
        "model_name": model_name,
    }
    import json

    (out.parent / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved TorchScript model to {out}")
    print(f"Wrote layout to {out.parent / 'export_meta.json'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="best_model.pt from train.py")
    p.add_argument("--output", type=str, default="alaphia_5head.pt")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    export(args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()
