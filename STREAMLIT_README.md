# Alaphia Inference Tester — Streamlit App

A tester-facing web UI for the Alaphia 5-head DeBERTa multitask model. Non-technical users can paste real journal entries or transcripts and immediately see structured predictions.

---

## Local Setup

```bash
# From the Sereniful-push/ directory:
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

**Python version:** 3.9+  
**First load:** model warm-up takes ~30 s; subsequent runs are fast.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ALAPHIA_MODEL_PATH` | `alaphia_5head_deberta/best_model.pt` | Path to the `.pt` checkpoint. Absolute or relative to `streamlit_app.py`. |
| `ALAPHIA_DEVICE` | *(auto)* | Force device: `cpu`, `mps`, or `cuda`. Omit to let the app pick the best available. |

Set them before launching:

```bash
export ALAPHIA_MODEL_PATH=/path/to/best_model.pt
export ALAPHIA_DEVICE=cpu
streamlit run streamlit_app.py
```

Or inline:

```bash
ALAPHIA_MODEL_PATH=/path/to/best_model.pt streamlit run streamlit_app.py
```

---

## Outputs

| Field | Type | Notes |
|---|---|---|
| Emotion | single label | Leaf label + family + valence |
| Emotion confidence | family + leaf | Both shown as percentages |
| Need | single label | Leaf label + family |
| Need confidence | family + leaf | Both shown as percentages |
| Status | `unmet` / `met` / `engaged` | Colour-coded |
| Money Theme | single label | 5-class |
| Life Context | list of labels | Filtered by leaf threshold and max labels |
| Life Context Families | list | Shown in expander when available |
| Raw JSON | full dict | Debug expander on each result |

---

## Batch Mode

Enable **Batch mode** in the UI, then separate multiple entries with a line containing only `---`:

```
I had a terrible day at work and couldn't stop worrying about money.
---
Finally paid off my credit card today — feeling relieved.
---
My partner and I argued again about the budget.
```

Each entry renders in its own collapsible panel.

---

## Sidebar Controls

- **Leaf threshold** (default 0.85): minimum sigmoid confidence for a life-context tag to appear. Raise to see fewer, more confident tags.
- **Family threshold** (default 0.75): minimum confidence for an active life-context family. Effective on hierarchical checkpoints.
- **Max labels** (default 6): cap on the number of life-context tags shown.

---

## Troubleshooting

**`FileNotFoundError: Checkpoint not found`**  
Set `ALAPHIA_MODEL_PATH` to point to `best_model.pt`.

**`ModuleNotFoundError: alaphia_5head_deberta`**  
Run the app from inside `Sereniful-push/`, or confirm that `alaphia_5head_deberta/` exists as a sibling directory.

**Slow first prediction**  
Expected — DeBERTa warm-up takes a few seconds. The model is cached for the server lifetime; subsequent predictions are fast.

**`sentence-transformers` import error**  
Required by the safety checker. Install with: `pip install sentence-transformers`.

**MPS errors on Apple Silicon**  
If you see Metal errors, force CPU: `ALAPHIA_DEVICE=cpu streamlit run streamlit_app.py`.

---

## Deployment — Streamlit Community Cloud

1. Push `Sereniful-push/` as the root of a public (or private) GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Set:
   - **Repository**: your repo
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. Add secrets under **Advanced settings → Secrets**:
   ```toml
   ALAPHIA_MODEL_PATH = "alaphia_5head_deberta/best_model.pt"
   ALAPHIA_DEVICE = "cpu"
   ```
5. Add `requirements-streamlit.txt` to the repo (already included).

**Note on the checkpoint:** `best_model.pt` must be committed to the repo or fetched at startup. For large files (>100 MB), use [Git LFS](https://git-lfs.github.com) or download from a remote store (S3, GCS) at startup via a custom `startup.sh`.

---

## Moving to an API-Backed Architecture

The current setup loads the model directly in the Streamlit process. This is fine for a single tester but does not scale for concurrent users or larger deployments.

For production or team-scale use, consider:

- **FastAPI inference service**: wrap `load_model_for_inference` + `predict` in a FastAPI endpoint, deploy on a GPU instance (or serverless GPU), and have Streamlit call the API over HTTP. The model stays in the API process; Streamlit becomes a thin frontend.
- **Batching**: the inference service can queue requests and call the model once per batch, improving GPU utilisation.
- **Authentication**: add an API key or OAuth layer in front of the Streamlit app for non-public access.
- **Caching**: for repeated identical inputs, cache predictions in Redis or a simple dict to avoid redundant forward passes.
