# Alaphia 5-head Multitask Model (DeBERTa v3 Small)

Five single-label heads on `microsoft/deberta-v3-small`: **emotion**, **need**, **money theme**, **life context**, **status**. Tokenizer and weights are loaded from Hugging Face only (no files copied from the legacy DistilBERT project).

## Setup

```bash
cd /path/to/alaphia_3head_run_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r alaphia_5head_deberta/requirements.txt
```

## Training data format

JSON array or JSONL; each record:

```json
{
  "text": "…",
  "emotion": "Resentment",
  "need": "Trust",
  "money_theme": "Thinking about money",
  "life_context": "Partner",
  "status": "unmet"
}
```

All five labels must match strings in `label_maps.py` exactly. Rows with missing or unknown labels are skipped.

## Train

From the repo root (`alaphia_3head_run_1`):

```bash
python -m alaphia_5head_deberta.train \
  --data_path data/training_data.json \
  --output_dir models/alaphia_5head_run \
  --batch_size 16 \
  --epochs 5 \
  --lr 2e-5
```

Best checkpoint is saved when **need** macro-F1 on the validation split improves (`best_model.pt`).

## Inference

```python
from alaphia_5head_deberta.inference import load_model_for_inference, predict

model, tokenizer, device = load_model_for_inference("models/alaphia_5head_run/best_model.pt")
out = predict("I feel drained and unheard about money.", model, tokenizer, device)
# out["emotion"], out["need"], out["confidence"], etc.
```

## Export (TorchScript)

Concatenates logits from all five heads in a fixed order (see `export_meta.json` next to the `.pt`).

```bash
python -m alaphia_5head_deberta.export \
  --checkpoint models/alaphia_5head_run/best_model.pt \
  --output alaphia_5head.pt
```

Use `split_concat_logits` in `model.py` and `predict_from_concat_logits` in `inference.py` if you run the traced graph and receive a single tensor.

## Layout

| File | Purpose |
|------|---------|
| `label_maps.py` | Taxonomies, `*_to_id`, family/valence helpers |
| `model.py` | `AlaphiaMultitaskModel` |
| `dataset.py` | `MultitaskDataset` |
| `train.py` | Training loop |
| `inference.py` | Structured dict predictions |
| `export.py` | TorchScript export |
