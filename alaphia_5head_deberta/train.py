"""Multitask training: weighted losses, AdamW, warmup + linear decay, best checkpoint by need macro-F1.

Head loss strategies
--------------------
emotion / need / money_theme / status  — CrossEntropyLoss with inverse-frequency class weights,
                                          capped per head and only applied when a class has enough
                                          training examples (otherwise weight = 1.0 to avoid rare-class
                                          noise dominating shared representations). Status uses a
                                          much lower cap so the rare engaged label is not overweighted.
                                          label_smoothing=0.1 on emotion and need heads.
life_context                           — BCEWithLogitsLoss with per-label pos_weight, capped and
                                          with a minimum positive count before boosting.
status                                 — loss is masked for rows where status_id == -1
                                          (no label collected); those rows still train all other heads.

Class weights are computed once from the training split and injected into the loss functions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .dataset import MultitaskDataset
from . import label_maps as LM
from .model import AlaphiaMultitaskModel


TASK_WEIGHTS = {
    "emotion": 3.0,
    "emotion_family": 1.0,
    "need": 3.0,
    "need_family": 1.0,
    "status": 0.5,
    "money_theme": 0.7,
    "life_context": 0.8,
    "life_context_family": 0.6,
}

LC_THRESHOLD = 0.5          # sigmoid threshold for life_context label activation
LABEL_SMOOTHING = 0.1       # applied to emotion and need heads only

# Per-head inverse-frequency caps and floors. Below min_count, class weight stays 1.0 (no boost).
CE_WEIGHT_EMOTION = {"max": 8.0, "min_count": 20}
CE_WEIGHT_EMOTION_FAMILY = {"max": 6.0, "min_count": 15}
CE_WEIGHT_NEED = {"max": 8.0, "min_count": 20}
CE_WEIGHT_NEED_FAMILY = {"max": 6.0, "min_count": 15}
CE_WEIGHT_MONEY = {"max": 5.0, "min_count": 30}
# Status: keep ``engaged`` from dominating — low cap; very rare classes get no boost.
CE_WEIGHT_STATUS = {"max": 3.0, "min_count": 80}

LC_POS_WEIGHT_MAX = 12.0
LC_POS_WEIGHT_MIN_POS = 25  # positives below this → pos_weight 1.0


SINGLE_LABEL_TASKS = ["emotion", "emotion_family", "need", "need_family", "money_theme", "status"]


# ── Class weight helpers ───────────────────────────────────────────────────────

def _ce_class_weights(
    taxonomy_size: int,
    counts: Counter,
    total_n: int,
    device: torch.device,
    *,
    max_weight: float,
    min_count_for_boost: int,
) -> torch.Tensor:
    """Inverse-frequency weights capped at max_weight; classes with count < min get weight 1.0."""
    t = torch.ones(taxonomy_size, dtype=torch.float, device=device)
    if total_n <= 0:
        return t
    for idx in range(taxonomy_size):
        c = counts.get(idx, 0)
        if c <= 0 or c < min_count_for_boost:
            continue
        w = total_n / (taxonomy_size * c)
        t[idx] = min(w, max_weight)
    return t


def compute_class_weights(dataset: MultitaskDataset, device: torch.device):
    """
    Walk the dataset once and compute per-class weights for every head.
    Returns a dict of tensors ready to pass to loss functions.
    """
    emo_cnt:   Counter = Counter()
    emo_fam_cnt: Counter = Counter()
    need_cnt:  Counter = Counter()
    need_fam_cnt: Counter = Counter()
    money_cnt: Counter = Counter()
    stat_cnt:  Counter = Counter()
    lc_pos:    Counter = Counter()   # positive label counts per life_context index
    lc_family_pos: Counter = Counter()
    total = len(dataset)

    for s in dataset.samples:
        emo_cnt[s["emotion_id"]]    += 1
        emo_fam_cnt[s["emotion_family_id"]] += 1
        need_cnt[s["need_id"]]      += 1
        need_fam_cnt[s["need_family_id"]] += 1
        money_cnt[s["money_theme_id"]] += 1
        if s["status_id"] >= 0:
            stat_cnt[s["status_id"]] += 1
        for idx in s["life_context_vec"].nonzero(as_tuple=True)[0].tolist():
            lc_pos[idx] += 1
        for idx in s["life_context_family_vec"].nonzero(as_tuple=True)[0].tolist():
            lc_family_pos[idx] += 1

    stat_total = sum(stat_cnt.values())
    stat_den = stat_total if stat_total > 0 else 1

    def _lc_pos_weight_tensor(pos_counter: Counter, n_labels: int) -> torch.Tensor:
        t = torch.ones(n_labels, dtype=torch.float, device=device)
        for idx in range(n_labels):
            pos = pos_counter.get(idx, 0)
            if pos <= 0 or pos < LC_POS_WEIGHT_MIN_POS:
                continue
            neg = total - pos
            pw = neg / pos
            t[idx] = min(pw, LC_POS_WEIGHT_MAX)
        return t

    return {
        "emotion": _ce_class_weights(
            LM.NUM_EMOTION, emo_cnt, total, device,
            max_weight=CE_WEIGHT_EMOTION["max"],
            min_count_for_boost=int(CE_WEIGHT_EMOTION["min_count"]),
        ),
        "emotion_family": _ce_class_weights(
            LM.NUM_EMOTION_FAMILY, emo_fam_cnt, total, device,
            max_weight=CE_WEIGHT_EMOTION_FAMILY["max"],
            min_count_for_boost=int(CE_WEIGHT_EMOTION_FAMILY["min_count"]),
        ),
        "need": _ce_class_weights(
            LM.NUM_NEED, need_cnt, total, device,
            max_weight=CE_WEIGHT_NEED["max"],
            min_count_for_boost=int(CE_WEIGHT_NEED["min_count"]),
        ),
        "need_family": _ce_class_weights(
            LM.NUM_NEED_FAMILY, need_fam_cnt, total, device,
            max_weight=CE_WEIGHT_NEED_FAMILY["max"],
            min_count_for_boost=int(CE_WEIGHT_NEED_FAMILY["min_count"]),
        ),
        "money_theme": _ce_class_weights(
            LM.NUM_MONEY_THEME, money_cnt, total, device,
            max_weight=CE_WEIGHT_MONEY["max"],
            min_count_for_boost=int(CE_WEIGHT_MONEY["min_count"]),
        ),
        "status": _ce_class_weights(
            LM.NUM_STATUS, stat_cnt, stat_den, device,
            max_weight=CE_WEIGHT_STATUS["max"],
            min_count_for_boost=int(CE_WEIGHT_STATUS["min_count"]),
        ),
        "lc_pos_weight": _lc_pos_weight_tensor(lc_pos, LM.NUM_LIFE_CONTEXT),
        "lc_family_pos_weight": _lc_pos_weight_tensor(lc_family_pos, LM.NUM_LIFE_CONTEXT_FAMILY),
    }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "emotion": torch.stack([b["emotion"] for b in batch]),
        "emotion_family": torch.stack([b["emotion_family"] for b in batch]),
        "need": torch.stack([b["need"] for b in batch]),
        "need_family": torch.stack([b["need_family"] for b in batch]),
        "money_theme": torch.stack([b["money_theme"] for b in batch]),
        # life_context: float multi-hot (B, NUM_LIFE_CONTEXT)
        "life_context": torch.stack([b["life_context"] for b in batch]),
        "life_context_family": torch.stack([b["life_context_family"] for b in batch]),
        "status": torch.stack([b["status"] for b in batch]),
    }


def _logit_map(logits):
    return {
        "emotion": logits[0],
        "emotion_family": logits[1],
        "need": logits[2],
        "need_family": logits[3],
        "money_theme": logits[4],
        "life_context": logits[5],
        "life_context_family": logits[6],
        "status": logits[7],
    }


def evaluate(model, loader, device, cw: dict):
    model.eval()
    ce_emo   = nn.CrossEntropyLoss(weight=cw["emotion"],     label_smoothing=LABEL_SMOOTHING)
    ce_emo_f = nn.CrossEntropyLoss(weight=cw["emotion_family"], label_smoothing=LABEL_SMOOTHING)
    ce_need  = nn.CrossEntropyLoss(weight=cw["need"],        label_smoothing=LABEL_SMOOTHING)
    ce_need_f = nn.CrossEntropyLoss(weight=cw["need_family"], label_smoothing=LABEL_SMOOTHING)
    ce_money = nn.CrossEntropyLoss(weight=cw["money_theme"])
    ce_stat  = nn.CrossEntropyLoss(weight=cw["status"])
    bce_lc   = nn.BCEWithLogitsLoss(pos_weight=cw["lc_pos_weight"])
    bce_lc_f = nn.BCEWithLogitsLoss(pos_weight=cw["lc_family_pos_weight"])

    ce_map = {
        "emotion": ce_emo,
        "emotion_family": ce_emo_f,
        "need": ce_need,
        "need_family": ce_need_f,
        "money_theme": ce_money,
        "status": ce_stat,
    }

    losses = {k: [] for k in TASK_WEIGHTS}
    all_pred = {k: [] for k in ("emotion", "emotion_family", "need", "need_family", "money_theme")}
    all_true = {k: [] for k in ("emotion", "emotion_family", "need", "need_family", "money_theme")}
    stat_pred, stat_true = [], []
    lc_pred_all: list = []
    lc_true_all: list = []
    lc_family_pred_all: list = []
    lc_family_true_all: list = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            log_map = _logit_map(logits)

            for name in ("emotion", "emotion_family", "need", "need_family", "money_theme"):
                tgt = batch[name].to(device)
                losses[name].append(ce_map[name](log_map[name], tgt).item())
                all_pred[name].append(log_map[name].argmax(dim=-1).cpu().numpy())
                all_true[name].append(tgt.cpu().numpy())

            # status — only rows with a real label (status_id >= 0)
            st_ids = batch["status"].to(device)
            mask   = st_ids >= 0
            if mask.any():
                st_loss = ce_stat(log_map["status"][mask], st_ids[mask])
                losses["status"].append(st_loss.item())
                stat_pred.append(log_map["status"][mask].argmax(dim=-1).cpu().numpy())
                stat_true.append(st_ids[mask].cpu().numpy())

            # life_context — multi-label
            lc_tgt = batch["life_context"].to(device)
            losses["life_context"].append(bce_lc(log_map["life_context"], lc_tgt).item())
            lc_pred_all.append((torch.sigmoid(log_map["life_context"]) >= LC_THRESHOLD).cpu().numpy())
            lc_true_all.append(lc_tgt.cpu().numpy())

            lc_family_tgt = batch["life_context_family"].to(device)
            losses["life_context_family"].append(bce_lc_f(log_map["life_context_family"], lc_family_tgt).item())
            lc_family_pred_all.append((torch.sigmoid(log_map["life_context_family"]) >= LC_THRESHOLD).cpu().numpy())
            lc_family_true_all.append(lc_family_tgt.cpu().numpy())

    metrics = {}
    for name in ("emotion", "emotion_family", "need", "need_family", "money_theme"):
        y_p = np.concatenate(all_pred[name])
        y_t = np.concatenate(all_true[name])
        metrics[f"{name}_loss"] = float(np.mean(losses[name]))
        metrics[f"{name}_acc"]  = float(accuracy_score(y_t, y_p))
        metrics[f"{name}_f1"]   = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    if stat_pred:
        y_p = np.concatenate(stat_pred)
        y_t = np.concatenate(stat_true)
        metrics["status_loss"] = float(np.mean(losses["status"]))
        metrics["status_acc"]  = float(accuracy_score(y_t, y_p))
        metrics["status_f1"]   = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    else:
        metrics.update({"status_loss": 0.0, "status_acc": 0.0, "status_f1": 0.0})

    lc_p = np.concatenate(lc_pred_all, axis=0)
    lc_t = np.concatenate(lc_true_all, axis=0)
    metrics["life_context_loss"]     = float(np.mean(losses["life_context"]))
    metrics["life_context_f1"]       = float(f1_score(lc_t, lc_p, average="micro", zero_division=0))
    metrics["life_context_f1_macro"] = float(f1_score(lc_t, lc_p, average="macro", zero_division=0))
    lc_family_p = np.concatenate(lc_family_pred_all, axis=0)
    lc_family_t = np.concatenate(lc_family_true_all, axis=0)
    metrics["life_context_family_loss"] = float(np.mean(losses["life_context_family"]))
    metrics["life_context_family_f1"] = float(f1_score(lc_family_t, lc_family_p, average="micro", zero_division=0))
    metrics["life_context_family_f1_macro"] = float(f1_score(lc_family_t, lc_family_p, average="macro", zero_division=0))

    return metrics


def train_epoch(model, loader, optimizer, scheduler, device, cw: dict, grad_clip: float = 1.0):
    model.train()
    ce_emo   = nn.CrossEntropyLoss(weight=cw["emotion"],     label_smoothing=LABEL_SMOOTHING)
    ce_emo_f = nn.CrossEntropyLoss(weight=cw["emotion_family"], label_smoothing=LABEL_SMOOTHING)
    ce_need  = nn.CrossEntropyLoss(weight=cw["need"],        label_smoothing=LABEL_SMOOTHING)
    ce_need_f = nn.CrossEntropyLoss(weight=cw["need_family"], label_smoothing=LABEL_SMOOTHING)
    ce_money = nn.CrossEntropyLoss(weight=cw["money_theme"])
    ce_stat  = nn.CrossEntropyLoss(weight=cw["status"])
    bce_lc   = nn.BCEWithLogitsLoss(pos_weight=cw["lc_pos_weight"])
    bce_lc_f = nn.BCEWithLogitsLoss(pos_weight=cw["lc_family_pos_weight"])

    totals = {k: 0.0 for k in TASK_WEIGHTS}
    n = 0

    for batch in tqdm(loader, desc="train"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention_mask)
        log_map = _logit_map(logits)

        loss_total = torch.tensor(0.0, device=device)

        for name, ce_fn in (
            ("emotion", ce_emo),
            ("emotion_family", ce_emo_f),
            ("need", ce_need),
            ("need_family", ce_need_f),
            ("money_theme", ce_money),
        ):
            tgt = batch[name].to(device)
            li  = ce_fn(log_map[name], tgt)
            loss_total = loss_total + TASK_WEIGHTS[name] * li
            totals[name] += li.item()

        # status — mask rows without a label
        st_ids = batch["status"].to(device)
        mask   = st_ids >= 0
        if mask.any():
            st_li = ce_stat(log_map["status"][mask], st_ids[mask])
            loss_total = loss_total + TASK_WEIGHTS["status"] * st_li
            totals["status"] += st_li.item()

        lc_tgt = batch["life_context"].to(device)
        lc_li  = bce_lc(log_map["life_context"], lc_tgt)
        loss_total = loss_total + TASK_WEIGHTS["life_context"] * lc_li
        totals["life_context"] += lc_li.item()

        lc_family_tgt = batch["life_context_family"].to(device)
        lc_family_li  = bce_lc_f(log_map["life_context_family"], lc_family_tgt)
        loss_total = loss_total + TASK_WEIGHTS["life_context_family"] * lc_family_li
        totals["life_context_family"] += lc_family_li.item()

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        n += 1

    return {k: totals[k] / max(n, 1) for k in totals}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="models/alaphia_5head_run")
    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every_epoch", action="store_true",
                   help="Save a checkpoint after every epoch (epoch_1.pt, epoch_2.pt, …).")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to an epoch checkpoint to resume training from.")
    p.add_argument("--drive_backup_dir", type=str, default=None,
                   help="If set, copy every epoch checkpoint and best_model.pt here immediately "
                        "after saving (use a Google Drive path so nothing is lost on runtime crash).")
    p.add_argument("--warm_start_from", type=str, default=None,
                   help="Load encoder and compatible heads from an older checkpoint with strict=False "
                        "and start a fresh training run for the current architecture.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    full_ds = MultitaskDataset(args.data_path, tokenizer, max_length=args.max_length)
    n = len(full_ds)
    if n == 0:
        raise SystemExit("No training samples after filtering. Check JSON and label strings.")

    indices = np.random.permutation(n)
    n_val = max(1, int(n * args.val_ratio))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Compute class weights from training split only ────────────────────────
    print("Computing class weights from training split …")
    train_subset_ds = Subset(full_ds, train_idx)
    # Build a lightweight counter dataset (reuse full_ds.samples via indices)
    class _CounterDS:
        def __init__(self, samples): self.samples = samples
        def __len__(self): return len(self.samples)
    cw_ds = _CounterDS([full_ds.samples[i] for i in train_idx])
    cw = compute_class_weights(cw_ds, device)  # type: ignore[arg-type]
    print(f"  Emotion  weight range : {cw['emotion'].min():.3f} – {cw['emotion'].max():.3f}")
    print(f"  Emotion family weight range : {cw['emotion_family'].min():.3f} – {cw['emotion_family'].max():.3f}")
    print(f"  Need     weight range : {cw['need'].min():.3f} – {cw['need'].max():.3f}")
    print(f"  Need family weight range : {cw['need_family'].min():.3f} – {cw['need_family'].max():.3f}")
    print(f"  Money    weight range : {cw['money_theme'].min():.3f} – {cw['money_theme'].max():.3f}")
    print(f"  Status   weight range : {cw['status'].min():.3f} – {cw['status'].max():.3f}")
    print(f"  LC pos_weight range   : {cw['lc_pos_weight'].min():.3f} – {cw['lc_pos_weight'].max():.3f}")
    print(f"  LC family pos_weight range : {cw['lc_family_pos_weight'].min():.3f} – {cw['lc_family_pos_weight'].max():.3f}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    model = AlaphiaMultitaskModel(model_name=args.model_name).to(device)

    num_steps = len(train_loader) * args.epochs
    warmup = int(0.1 * num_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, num_steps)

    best_need_f1 = -1.0
    best_path = out_dir / "best_model.pt"
    start_epoch = 1
    backup_dir = Path(args.drive_backup_dir) if args.drive_backup_dir else None
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)

    if args.resume_from:
        ck = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_need_f1 = ck.get("val_metrics", {}).get("need_f1", -1.0)
        print(f"Resumed from {args.resume_from} (epoch {ck['epoch']}, best need F1: {best_need_f1:.4f})")
    elif args.warm_start_from:
        ck = torch.load(args.warm_start_from, map_location=device)
        missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
        print(f"Warm-started from {args.warm_start_from}")
        if missing:
            print(f"  Missing keys (new heads expected): {missing}")
        if unexpected:
            print(f"  Unexpected keys ignored: {unexpected}")

    for epoch in range(start_epoch, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, scheduler, device, cw)
        va = evaluate(model, val_loader, device, cw)
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("  train losses:", {k: round(tr[k], 4) for k in tr})
        print("  val:", {k: round(va[k], 4) for k in va if k.endswith("_f1") or k.endswith("_acc")})

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_name": args.model_name,
            "epoch": epoch,
            "val_metrics": va,
        }

        if args.save_every_epoch:
            epoch_path = out_dir / f"epoch_{epoch}.pt"
            torch.save(checkpoint, epoch_path)
            print(f"  → Epoch checkpoint saved: {epoch_path}")
            if backup_dir:
                shutil.copy(epoch_path, backup_dir / epoch_path.name)
                print(f"  → Backed up to Drive: {backup_dir / epoch_path.name}")

        nf1 = va["need_f1"]
        if nf1 > best_need_f1:
            best_need_f1 = nf1
            torch.save(checkpoint, best_path)
            print(f"  ✓ New best need macro-F1: {best_need_f1:.4f} → saved {best_path}")
            if backup_dir:
                shutil.copy(best_path, backup_dir / "best_model.pt")
                print(f"  → Best model backed up to Drive")

    with open(out_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Done. Best need macro-F1: {best_need_f1:.4f}")


if __name__ == "__main__":
    main()
