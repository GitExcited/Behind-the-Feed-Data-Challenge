"""
Shared Milestone 2 code: data, metrics, scoring and training for the same-story models.

    from src.m2_common import *

Bi-encoder = MiniLM that turns each headline into a vector (cosine similarity between vectors).
Cross-encoder = mMiniLM that reads both headlines together and outputs a same-story probability.
"""

import random
import time
from pathlib import Path

import sentence_transformers  # noqa: F401  (import before torch: the other order crashes on some Windows setups)
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, losses
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.embeddings import clean_headline

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "pairs"
MODELS = ROOT / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 6450
BASE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Same recipe as v1 (milestone2_training.ipynb)
CFG = dict(epochs=3, batch_size=32, learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1, max_grad_norm=1.0,
           max_seq_length=128, margin=0.5, freeze_word_embeddings=True, patience=1)


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_pairs():
    return {s: pd.read_csv(PAIRS / f"{s}.csv", dtype={"timestamp": str}) for s in ["train", "val", "test"]}


def best_threshold(labels, scores):
    """Threshold with the best F1 (chosen on val, then applied unchanged elsewhere)."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(labels, scores, threshold):
    pred = (scores >= threshold).astype(int)
    return {"precision": precision_score(labels, pred, zero_division=0), "recall": recall_score(labels, pred, zero_division=0),
            "f1": f1_score(labels, pred, zero_division=0), "accuracy": accuracy_score(labels, pred),
            "roc_auc": roc_auc_score(labels, scores), "pr_auc": average_precision_score(labels, scores)}


def to_device(features):
    return {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in features.items()}


# ---------------------------------------------------------------- bi-encoder
def load_biencoder(path_or_name=BASE_MODEL):
    return SentenceTransformer(str(path_or_name), device=DEVICE, cache_folder=str(MODELS))


def pair_scores(model, df, prep=clean_headline, batch_size=64):
    """Cosine similarity per pair; every distinct headline is embedded once."""
    a, b = [prep(h) for h in df.headline_a], [prep(h) for h in df.headline_b]
    unique = list(dict.fromkeys(a + b))
    index = {h: i for i, h in enumerate(unique)}
    E = model.encode(unique, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return (E[[index[h] for h in a]] * E[[index[h] for h in b]]).sum(axis=1)


def train_biencoder(train_df, tag, val_df, prep=clean_headline, cfg=CFG):
    """Fine-tune MiniLM with the v1 recipe; keep the epoch with the best val PR-AUC. Saved to models/exp/<tag>."""
    set_seed()
    model = load_biencoder()
    model.max_seq_length = cfg["max_seq_length"]
    if cfg["freeze_word_embeddings"]:
        next(m for n, m in model.named_modules() if n.endswith("word_embeddings")).requires_grad_(False)
    loss_fn = losses.OnlineContrastiveLoss(model, margin=cfg["margin"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    bs = cfg["batch_size"]
    steps = int(np.ceil(len(train_df) / bs))
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps * cfg["epochs"]), steps * cfg["epochs"])
    A, B = [prep(h) for h in train_df.headline_a], [prep(h) for h in train_df.headline_b]
    y = train_df.label.values.astype(np.float32)
    rng, best, bad, t0, history = np.random.default_rng(SEED), -1.0, 0, time.perf_counter(), []
    path = MODELS / "exp" / tag
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = rng.permutation(len(train_df))
        for s in range(steps):
            idx = order[s * bs:(s + 1) * bs]
            loss = loss_fn([to_device(model.tokenize([A[i] for i in idx])), to_device(model.tokenize([B[i] for i in idx]))],
                           torch.tensor(y[idx], device=DEVICE))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]); opt.step(); sched.step()
        model.eval()
        scores = pair_scores(model, val_df, prep)
        m = evaluate(val_df.label.values, scores, best_threshold(val_df.label.values, scores))
        history.append({"epoch": epoch, **m})
        print(f"  {tag} epoch {epoch}: val F1 {m['f1']:.4f}, PR-AUC {m['pr_auc']:.4f}", flush=True)
        if m["pr_auc"] > best:
            best, bad, best_epoch = m["pr_auc"], 0, epoch
            model.save(str(path))
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"kind": "bi", "path": path, "prep": prep, "best_epoch": best_epoch,
            "minutes": round((time.perf_counter() - t0) / 60, 2), "history": pd.DataFrame(history)}


# ---------------------------------------------------------------- cross-encoder
def load_crossencoder(path):
    return AutoModelForSequenceClassification.from_pretrained(str(path)).to(DEVICE).eval(), AutoTokenizer.from_pretrained(str(path))


def ce_scores(model, tok, a, b, batch_size=128):
    """Same-story probability, averaged over both orders of the two headlines."""
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(a), batch_size):
            s = 0
            for x, z in [(a[i:i + batch_size], b[i:i + batch_size]), (b[i:i + batch_size], a[i:i + batch_size])]:
                f = tok(list(x), list(z), padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
                s = s + torch.sigmoid(model(**f).logits[:, 0].float())
            out.append((s / 2).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_crossencoder(train_df, tag, val_df, prep=clean_headline, cfg=CFG):
    set_seed()
    tok = AutoTokenizer.from_pretrained(CE_MODEL, cache_dir=str(MODELS))
    model = AutoModelForSequenceClassification.from_pretrained(CE_MODEL, cache_dir=str(MODELS), num_labels=1).to(DEVICE)
    bs = cfg["batch_size"]
    steps = int(np.ceil(len(train_df) / bs))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps * cfg["epochs"]), steps * cfg["epochs"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    A, B = [prep(h) for h in train_df.headline_a], [prep(h) for h in train_df.headline_b]
    va, vb = [prep(h) for h in val_df.headline_a], [prep(h) for h in val_df.headline_b]
    y = train_df.label.values.astype(np.float32)
    rng, best, bad, t0, history = np.random.default_rng(SEED), -1.0, 0, time.perf_counter(), []
    path = MODELS / "exp" / tag
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = rng.permutation(len(train_df))
        for s in range(steps):
            idx = order[s * bs:(s + 1) * bs]
            swap = rng.random(len(idx)) < 0.5            # random order, so the model learns a symmetric score
            x = [B[i] if w else A[i] for i, w in zip(idx, swap)]
            z = [A[i] if w else B[i] for i, w in zip(idx, swap)]
            f = tok(x, z, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
            loss = loss_fn(model(**f).logits[:, 0], torch.tensor(y[idx], device=DEVICE))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"]); opt.step(); sched.step()
        scores = ce_scores(model, tok, va, vb)
        m = evaluate(val_df.label.values, scores, best_threshold(val_df.label.values, scores))
        history.append({"epoch": epoch, **m})
        print(f"  {tag} epoch {epoch}: val F1 {m['f1']:.4f}, PR-AUC {m['pr_auc']:.4f}", flush=True)
        if m["pr_auc"] > best:
            best, bad, best_epoch = m["pr_auc"], 0, epoch
            model.save_pretrained(path); tok.save_pretrained(path)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"kind": "cross", "path": path, "prep": prep, "best_epoch": best_epoch,
            "minutes": round((time.perf_counter() - t0) / 60, 2), "history": pd.DataFrame(history)}


def score_model(run, df):
    """Scores for any trained run dict (bi or cross)."""
    if run["kind"] == "bi":
        return pair_scores(load_biencoder(run["path"]), df, run["prep"])
    model, tok = load_crossencoder(run["path"])
    return ce_scores(model, tok, [run["prep"](h) for h in df.headline_a], [run["prep"](h) for h in df.headline_b])
