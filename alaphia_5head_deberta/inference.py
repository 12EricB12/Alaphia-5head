"""Load checkpoint and return structured prediction dict with confidences and family lookups.

Safety check runs before model inference. If the input triggers a safety flag the
function returns immediately with {'flagged': True, ...} and skips the 5-head model.
Pass safety_check=False to disable (e.g. for batch offline processing).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from . import label_maps as LM
from .model import AlaphiaMultitaskModel, split_concat_logits
from .safety import SafetyChecker

# Single shared instance — loaded once, reused for all calls
_safety_checker: Optional[SafetyChecker] = None

def _get_safety_checker() -> SafetyChecker:
    global _safety_checker
    if _safety_checker is None:
        _safety_checker = SafetyChecker()
    return _safety_checker


_TOKENIZER_CACHE = Path(__file__).parent.parent / "deberta_tokenizer_cache"
LC_FAMILY_THRESHOLD = 0.75
LC_LEAF_THRESHOLD = 0.85
LC_FALLBACK_THRESHOLD = 0.70
LC_FALLBACK_MAX_LABELS = 3
STATUS_ENGAGED_THRESHOLD = 0.70
CHUNK_STRIDE = 128


def _family_mass_from_leaf_probs(
    probs: torch.Tensor,
    family_labels: list[str],
    index_lookup,
) -> Dict[str, float]:
    """Aggregate family confidence from leaf probabilities for backward compatibility."""
    out: Dict[str, float] = {}
    for family in family_labels:
        indices = index_lookup(family)
        out[family] = float(probs[indices].sum().item())
    return out


def _family_multilabel_scores_from_leaf_probs(
    probs: torch.Tensor,
    family_labels: list[str],
    index_lookup,
) -> Dict[str, float]:
    """Approximate family activation from leaf sigmoid scores for older checkpoints."""
    out: Dict[str, float] = {}
    for family in family_labels:
        indices = index_lookup(family)
        if not indices:
            out[family] = 0.0
            continue
        no_hit = 1.0
        for idx in indices:
            no_hit *= 1.0 - float(probs[idx].item())
        out[family] = 1.0 - no_hit
    return out


def _decode_single_label_with_family(
    leaf_logits: torch.Tensor,
    family_logits: Optional[torch.Tensor],
    id_to_leaf: Dict[int, str],
    id_to_family: Dict[int, str],
    family_labels: list[str],
    index_lookup,
    family_lookup,
) -> Tuple[str, str, float, float]:
    """Return leaf, family, family confidence, and within-family leaf confidence."""
    if family_logits is not None:
        fam_probs = F.softmax(family_logits, dim=-1)[0]
        fam_idx = int(fam_probs.argmax().item())
        family = id_to_family[fam_idx]
        family_conf = float(fam_probs[fam_idx].item())
    else:
        leaf_probs = F.softmax(leaf_logits, dim=-1)[0]
        fam_scores = _family_mass_from_leaf_probs(leaf_probs, family_labels, index_lookup)
        family, family_conf = max(fam_scores.items(), key=lambda item: item[1])

    family_indices = index_lookup(family)
    family_subset = leaf_logits[:, family_indices]
    subset_probs = F.softmax(family_subset, dim=-1)[0]
    local_idx = int(subset_probs.argmax().item())
    leaf_idx = family_indices[local_idx]
    leaf = id_to_leaf[leaf_idx]
    leaf_conf = float(subset_probs[local_idx].item())

    if family_lookup(leaf) != family:
        leaf = id_to_leaf[family_indices[0]]
        leaf_conf = float(subset_probs[0].item())
    return leaf, family, family_conf, leaf_conf


def _decode_life_context_with_family(
    leaf_logits: torch.Tensor,
    family_logits: Optional[torch.Tensor],
    lc_family_threshold: float,
    lc_leaf_threshold: float,
    lc_max_labels: int,
) -> Tuple[list[str], Dict[str, float], list[str], Dict[str, float]]:
    """Decode life-context families first, then leaf tags inside active families."""
    lc_probs = torch.sigmoid(leaf_logits)[0]
    if family_logits is not None:
        fam_probs = torch.sigmoid(family_logits)[0]
        active_families = (fam_probs >= lc_family_threshold).nonzero(as_tuple=True)[0].tolist()
        if not active_families:
            active_families = [int(fam_probs.argmax().item())]
        active_families = sorted(active_families, key=lambda i: fam_probs[i].item(), reverse=True)
        family_labels = [LM.id_to_life_context_family[i] for i in active_families]
        family_conf = {LM.id_to_life_context_family[i]: float(fam_probs[i].item()) for i in active_families}
    else:
        fam_scores = _family_multilabel_scores_from_leaf_probs(
            lc_probs,
            LM.LIFE_CONTEXT_FAMILY_LABELS,
            LM.life_context_indices_for_family,
        )
        active = [family for family, score in fam_scores.items() if score >= lc_family_threshold]
        if not active:
            active = [max(fam_scores.items(), key=lambda item: item[1])[0]]
        family_labels = sorted(active, key=lambda family: fam_scores[family], reverse=True)
        family_conf = {family: fam_scores[family] for family in family_labels}

    allowed_indices = sorted(
        {
            idx
            for family in family_labels
            for idx in LM.life_context_indices_for_family(family)
        }
    )
    active = [idx for idx in allowed_indices if float(lc_probs[idx].item()) >= lc_leaf_threshold]
    if not active and allowed_indices:
        fallback = [
            idx for idx in allowed_indices
            if float(lc_probs[idx].item()) >= LC_FALLBACK_THRESHOLD
        ]
        if fallback:
            active = sorted(
                fallback,
                key=lambda idx: float(lc_probs[idx].item()),
                reverse=True,
            )[:LC_FALLBACK_MAX_LABELS]
        else:
            active = [max(allowed_indices, key=lambda idx: float(lc_probs[idx].item()))]
    active = sorted(active, key=lambda idx: float(lc_probs[idx].item()), reverse=True)[:lc_max_labels]
    leaves = [LM.id_to_life_context[idx] for idx in active]
    leaf_conf = {LM.id_to_life_context[idx]: float(lc_probs[idx].item()) for idx in active}
    return family_labels, family_conf, leaves, leaf_conf


def load_model_for_inference(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[AlaphiaMultitaskModel, AutoTokenizer, torch.device]:
    try:
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(checkpoint_path, map_location="cpu")
    model_name = ck.get("model_name", "microsoft/deberta-v3-small")

    # Build model architecture from config (no weight download) to avoid the
    # macOS Apple Silicon mutex deadlock caused by safetensors' parallel loader
    # conflicting with the tokenizers Rust/abseil runtime.  Our checkpoint's
    # state dict is loaded immediately after via load_state_dict().
    model = AlaphiaMultitaskModel(model_name=model_name, from_pretrained=False)
    state_dict = ck["model_state_dict"]
    has_hierarchical_heads = all(
        key in state_dict
        for key in (
            "emotion_family_head.weight",
            "need_family_head.weight",
            "life_context_family_head.weight",
        )
    )
    model.load_state_dict(state_dict, strict=has_hierarchical_heads)
    model.has_hierarchical_heads = has_hierarchical_heads  # type: ignore[attr-defined]

    # Load tokenizer from the pre-converted local cache (has tokenizer.json),
    # which allows the Rust fast-tokenizer to be used without going through the
    # sentencepiece → protobuf conversion that triggers the abseil double-init.
    tok_path = _TOKENIZER_CACHE if _TOKENIZER_CACHE.exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), use_fast=False)

    if device is not None:
        dev = device
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    model.to(dev)
    model.eval()

    # Warm-up: one dummy forward pass so the first real request isn't slow
    _dummy = tokenizer("warmup", return_tensors="pt", max_length=16, truncation=True, padding="max_length")
    with torch.no_grad():
        model(_dummy["input_ids"].to(dev), _dummy["attention_mask"].to(dev))

    return model, tokenizer, dev


def predict(
    text: str,
    model: AlaphiaMultitaskModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 512,
    chunk_stride: int = CHUNK_STRIDE,
    lc_threshold: float = LC_LEAF_THRESHOLD,
    lc_family_threshold: float = LC_FAMILY_THRESHOLD,
    lc_max_labels: int = 6,
    safety_check: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    # ── Safety check (runs before model, returns immediately if flagged) ──────
    if safety_check:
        safety_result = _get_safety_checker().check(text)
        if safety_result["flagged"]:
            return {
                "flagged":       True,
                "safety":        safety_result,
                "emotion":       None,
                "emotion_family": None,
                "valence":       None,
                "need":          None,
                "need_family":   None,
                "money_theme":   None,
                "life_context_family": None,
                "life_context":  None,
                "status":        None,
                "confidence":    None,
            }

    # Use overflow windows so long posts are processed across multiple chunks.
    # Short posts still produce a single chunk.
    stride = max(0, min(int(chunk_stride), max(0, max_length - 2)))
    enc = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_overflowing_tokens=True,
        stride=stride,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        chunk_logits = model(input_ids, attention_mask)

    # Aggregate chunk logits into one "full post" decision surface.
    # - Single-label heads: mean over chunks (stable global decision)
    # - Multi-label life-context heads: max over chunks (capture salient labels anywhere in text)
    hierarchical = bool(getattr(model, "has_hierarchical_heads", True))
    if input_ids.shape[0] > 1:
        if hierarchical:
            logits = (
                chunk_logits[0].mean(dim=0, keepdim=True),  # emotion
                chunk_logits[1].mean(dim=0, keepdim=True),  # emotion_family
                chunk_logits[2].mean(dim=0, keepdim=True),  # need
                chunk_logits[3].mean(dim=0, keepdim=True),  # need_family
                chunk_logits[4].mean(dim=0, keepdim=True),  # money_theme
                chunk_logits[5].max(dim=0, keepdim=True).values,  # life_context
                chunk_logits[6].max(dim=0, keepdim=True).values,  # life_context_family
                chunk_logits[7].mean(dim=0, keepdim=True),  # status
            )
        else:
            logits = (
                chunk_logits[0].mean(dim=0, keepdim=True),  # emotion
                chunk_logits[1].mean(dim=0, keepdim=True),  # need
                chunk_logits[2].mean(dim=0, keepdim=True),  # money_theme
                chunk_logits[3].max(dim=0, keepdim=True).values,  # life_context
                chunk_logits[4].mean(dim=0, keepdim=True),  # status
            )
    else:
        logits = chunk_logits

    out: Dict[str, Any] = {}
    conf: Dict[str, Any] = {}

    # emotion
    emo, emo_family, emo_family_conf, emo_leaf_conf = _decode_single_label_with_family(
        logits[0],
        logits[1] if hierarchical else None,
        LM.id_to_emotion,
        LM.id_to_emotion_family,
        LM.EMOTION_FAMILY_LABELS,
        LM.emotion_indices_for_family,
        LM.emotion_family,
    )
    out["emotion"] = emo
    out["emotion_family"] = emo_family
    out["valence"] = LM.emotion_valence(emo)
    conf["emotion"] = emo_leaf_conf
    conf["emotion_family"] = emo_family_conf

    # need
    need, need_family, need_family_conf, need_leaf_conf = _decode_single_label_with_family(
        logits[2] if hierarchical else logits[1],
        logits[3] if hierarchical else None,
        LM.id_to_need,
        LM.id_to_need_family,
        LM.NEED_FAMILY_LABELS,
        LM.need_indices_for_family,
        LM.need_family,
    )
    out["need"] = need
    out["need_family"] = need_family
    conf["need"] = need_leaf_conf
    conf["need_family"] = need_family_conf

    # money theme
    money_idx = 4 if hierarchical else 2
    p = F.softmax(logits[money_idx], dim=-1)
    mi = int(p.argmax(dim=-1).item())
    out["money_theme"] = LM.id_to_money_theme[mi]
    conf["money_theme"] = float(p[0, mi].item())

    # life context
    lc_idx = 5 if hierarchical else 3
    lc_family_idx = 6 if hierarchical else None
    lc_families, lc_family_conf, lc_leaves, lc_leaf_conf = _decode_life_context_with_family(
        logits[lc_idx],
        logits[lc_family_idx] if lc_family_idx is not None else None,
        lc_family_threshold=lc_family_threshold,
        lc_leaf_threshold=lc_threshold,
        lc_max_labels=lc_max_labels,
    )
    out["life_context_family"] = lc_families
    out["life_context"] = lc_leaves
    conf["life_context_family"] = lc_family_conf
    conf["life_context"] = lc_leaf_conf

    # status
    status_idx = 7 if hierarchical else 4
    p = F.softmax(logits[status_idx], dim=-1)
    engaged_idx = LM.status_to_id["engaged"]
    engaged_prob = float(p[0, engaged_idx].item())
    if engaged_prob >= STATUS_ENGAGED_THRESHOLD:
        si = engaged_idx
    else:
        fallback_indices = [LM.status_to_id["unmet"], LM.status_to_id["met"]]
        si = max(fallback_indices, key=lambda idx: float(p[0, idx].item()))
    out["status"] = LM.id_to_status[si]
    conf["status"] = float(p[0, si].item())
    conf["status_raw"] = {LM.id_to_status[i]: float(p[0, i].item()) for i in range(p.shape[-1])}
    conf["meta"] = {"num_chunks": int(input_ids.shape[0]), "chunk_stride": int(stride), "max_length": int(max_length)}

    out["confidence"] = conf
    return out


def predict_from_concat_logits(
    concat_logits: torch.Tensor,
) -> Dict[str, Any]:
    """If using exported single-tensor logits, split then softmax (batch size 1)."""
    chunks = split_concat_logits(concat_logits)
    logits = chunks  # type: ignore
    # Reuse predict path by building a tiny wrapper — duplicate softmax logic once
    out: Dict[str, Any] = {}
    conf: Dict[str, Any] = {}

    hierarchical = len(logits) == 8

    emo, emo_family, emo_family_conf, emo_leaf_conf = _decode_single_label_with_family(
        logits[0],
        logits[1] if hierarchical else None,
        LM.id_to_emotion,
        LM.id_to_emotion_family,
        LM.EMOTION_FAMILY_LABELS,
        LM.emotion_indices_for_family,
        LM.emotion_family,
    )
    out["emotion"] = emo
    out["emotion_family"] = emo_family
    out["valence"] = LM.emotion_valence(emo)
    conf["emotion"] = emo_leaf_conf
    conf["emotion_family"] = emo_family_conf

    need, need_family, need_family_conf, need_leaf_conf = _decode_single_label_with_family(
        logits[2] if hierarchical else logits[1],
        logits[3] if hierarchical else None,
        LM.id_to_need,
        LM.id_to_need_family,
        LM.NEED_FAMILY_LABELS,
        LM.need_indices_for_family,
        LM.need_family,
    )
    out["need"] = need
    out["need_family"] = need_family
    conf["need"] = need_leaf_conf
    conf["need_family"] = need_family_conf

    money_idx = 4 if hierarchical else 2
    p = F.softmax(logits[money_idx], dim=-1)
    mi = int(p.argmax(dim=-1).item())
    out["money_theme"] = LM.id_to_money_theme[mi]
    conf["money_theme"] = float(p[0, mi].item())

    lc_idx = 5 if hierarchical else 3
    lc_family_idx = 6 if hierarchical else None
    lc_families, lc_family_conf, lc_leaves, lc_leaf_conf = _decode_life_context_with_family(
        logits[lc_idx],
        logits[lc_family_idx] if lc_family_idx is not None else None,
        lc_family_threshold=LC_FAMILY_THRESHOLD,
        lc_leaf_threshold=LC_LEAF_THRESHOLD,
        lc_max_labels=6,
    )
    out["life_context_family"] = lc_families
    out["life_context"] = lc_leaves
    conf["life_context_family"] = lc_family_conf
    conf["life_context"] = lc_leaf_conf

    status_idx = 7 if hierarchical else 4
    p = F.softmax(logits[status_idx], dim=-1)
    engaged_idx = LM.status_to_id["engaged"]
    engaged_prob = float(p[0, engaged_idx].item())
    if engaged_prob >= STATUS_ENGAGED_THRESHOLD:
        si = engaged_idx
    else:
        fallback_indices = [LM.status_to_id["unmet"], LM.status_to_id["met"]]
        si = max(fallback_indices, key=lambda idx: float(p[0, idx].item()))
    out["status"] = LM.id_to_status[si]
    conf["status"] = float(p[0, si].item())
    conf["status_raw"] = {LM.id_to_status[i]: float(p[0, i].item()) for i in range(p.shape[-1])}

    out["confidence"] = conf
    return out
