"""Streamlit inference tester for the Alaphia 5-head DeBERTa multitask model."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st
import torch

# Ensure the package directory is on sys.path so relative imports resolve
# whether the app is launched from Sereniful-push/ or from the parent directory.
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from alaphia_5head_deberta.inference import load_model_for_inference, predict  # noqa: E402

# ── Configuration from environment ──────────────────────────────────────────
def _cfg(name: str, default: str = "") -> Tuple[str, str]:
    """Resolve config from Streamlit secrets first, then env.

    Returns (value, source) where source is one of:
    - secrets:<key>
    - secrets:<section>.<key>
    - env
    - default
    """
    try:
        sec_val = st.secrets.get(name, None)
        if sec_val is not None and str(sec_val).strip():
            return str(sec_val).strip(), f"secrets:{name}"
    except Exception:
        pass
    # Also support grouped secret styles, e.g.:
    # [hf]
    # token = "..."
    # repo_id = "..."
    section_key_map = {
        "HF_TOKEN": ("hf", "token"),
        "HF_REPO_ID": ("hf", "repo_id"),
        "ALAPHIA_MODEL_PATH": ("alaphia", "model_path"),
        "ALAPHIA_DEVICE": ("alaphia", "device"),
    }
    if name in section_key_map:
        section, key = section_key_map[name]
        try:
            sec_section = st.secrets.get(section, None)
            if sec_section is not None:
                sec_val = sec_section.get(key, None)
                if sec_val is not None and str(sec_val).strip():
                    return str(sec_val).strip(), f"secrets:{section}.{key}"
        except Exception:
            pass
    env_val = os.environ.get(name, "").strip()
    if env_val:
        return env_val, "env"
    return default, "default"


_MODEL_PATH_ENV, _MODEL_PATH_SRC = _cfg("ALAPHIA_MODEL_PATH", "alaphia_5head_deberta/best_model.pt")
_DEVICE_ENV, _DEVICE_SRC = _cfg("ALAPHIA_DEVICE", "")
_HF_REPO_ID, _HF_REPO_SRC = _cfg("HF_REPO_ID", "")
_HF_TOKEN, _HF_TOKEN_SRC = _cfg("HF_TOKEN", "")

def _resolve_model_path() -> Path:
    p = Path(_MODEL_PATH_ENV)
    return p if p.is_absolute() else _HERE / p

def _resolve_device() -> torch.device | None:
    if not _DEVICE_ENV:
        return None
    try:
        return torch.device(_DEVICE_ENV)
    except Exception:
        return None

_MODEL_PATH = _resolve_model_path()
_DEVICE_OBJ = _resolve_device()

DELIMITER = "\n---\n"
MIN_CHARS = 20


def _ensure_checkpoint() -> None:
    """Download best_model.pt from HF Hub if it isn't already present locally."""
    if _MODEL_PATH.exists():
        return
    if not _HF_REPO_ID or not _HF_TOKEN:
        raise FileNotFoundError(
            f"Checkpoint not found at {_MODEL_PATH} and HF_REPO_ID / HF_TOKEN are not set.\n"
            "Either place best_model.pt in the expected path or set both env vars."
        )
    from huggingface_hub import hf_hub_download
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename="best_model.pt",
        local_dir=str(_MODEL_PATH.parent),
        token=_HF_TOKEN,
    )


# ── Model loading (cached for the lifetime of the server process) ────────────
@st.cache_resource(show_spinner="Loading model — this takes ~30 s the first time…")
def _load_model():
    _ensure_checkpoint()
    return load_model_for_inference(str(_MODEL_PATH), device=_DEVICE_OBJ)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Alaphia Inference Tester", layout="wide")
st.title("Alaphia Inference Tester")
st.caption("Paste a journal entry or transcript to see what the 5-head model predicts.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model")
    st.code(str(_MODEL_PATH), language=None)
    with st.expander("Config diagnostics"):
        st.write(f"`HF_REPO_ID` source: `{_HF_REPO_SRC}`")
        st.write(f"`HF_TOKEN` source: `{_HF_TOKEN_SRC}`")
        st.write(f"`ALAPHIA_MODEL_PATH` source: `{_MODEL_PATH_SRC}`")
        st.write(f"`ALAPHIA_DEVICE` source: `{_DEVICE_SRC}`")
        st.write(f"`HF_REPO_ID` present: `{bool(_HF_REPO_ID)}`")
        st.write(f"`HF_TOKEN` present: `{bool(_HF_TOKEN)}`")

    try:
        model, tokenizer, device = _load_model()
        active_device = str(device)
        if _DEVICE_ENV:
            active_device += f" (override: {_DEVICE_ENV})"
        st.success(f"Loaded · device: `{active_device}`")
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Model load failed:\n\n`{exc}`\n\nCheck logs for details.")
        st.stop()

    st.divider()
    st.header("Life Context Controls")
    lc_leaf_thresh = st.slider(
        "Leaf threshold",
        min_value=0.50, max_value=1.00, value=0.85, step=0.05,
        help="Minimum sigmoid confidence for a leaf life-context tag to appear.",
    )
    lc_family_thresh = st.slider(
        "Family threshold",
        min_value=0.50, max_value=1.00, value=0.75, step=0.05,
        help=(
            "Minimum confidence for an active life-context family. "
            "Used by hierarchical checkpoints; passed through to inference."
        ),
    )
    lc_max = st.slider(
        "Max labels",
        min_value=1, max_value=12, value=6, step=1,
        help="Maximum number of life-context leaf tags to show.",
    )

# ── Helpers ──────────────────────────────────────────────────────────────────
_VALENCE_COLOR = {"positive": "green", "negative": "red", "neutral": "gray"}
_STATUS_COLOR = {"unmet": "red", "met": "green", "engaged": "blue"}


def _colored(label: str, color: str) -> str:
    return f":{color}[**{label}**]"


def _render_result(result: Dict[str, Any]) -> None:
    if result.get("flagged"):
        safety = result.get("safety") or {}
        st.warning(
            "**Safety flag** — this entry was not processed by the model.  \n"
            f"Category: `{safety.get('category', 'unknown')}` · "
            f"Method: `{safety.get('method', 'unknown')}`\n\n"
            "If you believe this is a false positive, try rephrasing the input."
        )
        return

    conf = result.get("confidence") or {}

    col1, col2 = st.columns(2)

    with col1:
        valence = result.get("valence", "neutral")
        v_color = _VALENCE_COLOR.get(valence, "gray")
        emo_family = result.get("emotion_family") or ""
        emo_fam_conf = conf.get("emotion_family", 0.0)
        emo_leaf_conf = conf.get("emotion", 0.0)

        st.markdown(f"**Emotion** — {_colored(result['emotion'], v_color)}")
        st.caption(
            f"Family: {emo_family} ({emo_fam_conf:.0%}) · "
            f"Valence: {valence} · Leaf confidence: {emo_leaf_conf:.0%}"
        )

        st.write("")

        need_family = result.get("need_family") or ""
        need_fam_conf = conf.get("need_family", 0.0)
        need_leaf_conf = conf.get("need", 0.0)
        st.markdown(f"**Need** — **{result['need']}**")
        st.caption(
            f"Family: {need_family} ({need_fam_conf:.0%}) · "
            f"Leaf confidence: {need_leaf_conf:.0%}"
        )

    with col2:
        status = result.get("status") or ""
        s_color = _STATUS_COLOR.get(status, "gray")
        st.markdown(f"**Status** — {_colored(status, s_color)}")
        st.caption(f"Confidence: {conf.get('status', 0.0):.0%}")

        st.write("")

        money = result.get("money_theme") or ""
        st.markdown(f"**Money Theme** — **{money}**")
        st.caption(f"Confidence: {conf.get('money_theme', 0.0):.0%}")

    st.divider()

    lc_leaves = result.get("life_context") or []
    lc_leaf_conf = conf.get("life_context") or {}
    lc_families = result.get("life_context_family") or []
    lc_family_conf = conf.get("life_context_family") or {}

    st.markdown("**Life Context**")
    if lc_leaves:
        n_cols = min(len(lc_leaves), 3)
        cols = st.columns(n_cols)
        for i, tag in enumerate(lc_leaves):
            score = lc_leaf_conf.get(tag, 0.0) if isinstance(lc_leaf_conf, dict) else 0.0
            with cols[i % n_cols]:
                st.metric(label=tag, value=f"{score:.0%}")
    else:
        st.caption("No life-context tags met the leaf threshold.")

    if lc_families:
        with st.expander("Life Context Families"):
            for fam in lc_families:
                score = lc_family_conf.get(fam, 0.0) if isinstance(lc_family_conf, dict) else 0.0
                st.write(f"**{fam}** — {score:.0%}")

    with st.expander("Raw JSON"):
        st.json(result)


def _validate(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return "Input is empty — paste some text before running inference."
    if len(stripped) < MIN_CHARS:
        return (
            f"Input is too short ({len(stripped)} chars). "
            f"Please provide at least {MIN_CHARS} characters for meaningful predictions."
        )
    return None


# ── Input UI ─────────────────────────────────────────────────────────────────
st.subheader("Input")
batch_mode = st.checkbox(
    "Batch mode — split entries by `---` on its own line",
    value=False,
    help="When checked, separate multiple entries with a line containing only `---`.",
)
placeholder = (
    "Paste your text here…\n\nFor batch mode, separate entries with:\n---"
    if batch_mode
    else "Paste a journal entry or transcript here…"
)
raw_text = st.text_area("Text", height=220, placeholder=placeholder, label_visibility="collapsed")
run = st.button("Run Inference", type="primary")

# ── Inference ─────────────────────────────────────────────────────────────────
if run:
    err = _validate(raw_text)
    if err:
        st.error(err)
        st.stop()

    entries = (
        [e.strip() for e in raw_text.split("\n---\n") if e.strip()]
        if batch_mode
        else [raw_text.strip()]
    )

    if batch_mode and len(entries) == 0:
        st.error("No valid entries found after splitting. Check that entries are separated by `---`.")
        st.stop()

    for i, entry in enumerate(entries):
        preview = entry[:80].replace("\n", " ") + ("…" if len(entry) > 80 else "")

        if batch_mode:
            container = st.expander(f"Entry {i + 1} — {preview}", expanded=(i == 0))
        else:
            container = st.container()

        with container:
            with st.spinner(f"Running inference{f' on entry {i + 1}' if batch_mode else ''}…"):
                try:
                    result = predict(
                        entry,
                        model,
                        tokenizer,
                        device,
                        lc_threshold=lc_leaf_thresh,
                        lc_family_threshold=lc_family_thresh,
                        lc_max_labels=lc_max,
                    )
                except Exception as exc:
                    st.error(
                        f"Inference failed for entry {i + 1}:\n\n`{exc}`\n\n"
                        "Check that the checkpoint is compatible with this package version."
                    )
                    continue
            _render_result(result)
