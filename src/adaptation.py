"""Milestone 3, Section 4.3: adaptation experiment helpers (used by milestone3_adaptation.ipynb).

Builds on src/robustness.py (same metrics, threshold rule, Platt calibration, bi-encoder wrapper and noise
augmentation) so every number here is computed the same way as in Section 2 and in Milestone 2.

  drift statistics   psi(), drift_row(), estimate_prior()       -> is the stream drifting, and do triggers fire?
  simulated drift    resample_prior(), language split           -> label shift and a new text domain
  update             finetune()                                  -> incremental fine-tune with replay (or full retrain)
"""
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, losses
from transformers import get_linear_schedule_with_warmup

import src.robustness as R
from src.embeddings import clean_headline
from src.robustness import MODELS, SEED, augment, best_threshold, evaluate

OUT = R.OUT / "adaptation"
ADAPT_MODELS = Path(os.environ.get("M3_ADAPT_MODELS", MODELS / "exp" / "adaptation"))

# Robustness.py picks cuda or cpu; on the team's Macs use the Apple GPU for training (latency is measured on CPU).
if R.DEVICE == "cpu" and torch.backends.mps.is_available():
    R.DEVICE = "mps"


def month(df):
    return df.timestamp.str[:6]


# drift statistics
def psi(ref, cur, bins=10, eps=1e-4):
    """Population stability index of `cur` against `ref`, with bins at the deciles of `ref`.
    Rule of thumb used in the monitoring plan: < 0.1 stable, 0.1 to 0.2 watch, > 0.2 drift."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1) / len(ref)
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref, cur, eps=1e-4):
    cats = sorted(set(ref) | set(cur))
    r = np.clip(np.array([np.mean(np.asarray(ref) == k) for k in cats]), eps, None)
    c = np.clip(np.array([np.mean(np.asarray(cur) == k) for k in cats]), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


TOKEN = re.compile(r"\w+", re.UNICODE)


def tokens(text):
    return TOKEN.findall(clean_headline(text).lower())


def vocabulary(df):
    v = set()
    for h in pd.unique(pd.concat([df.headline_a, df.headline_b])):
        v.update(tokens(h))
    return v


def headline_features(df, vocab):
    """Label-free input features per pair: words per headline and share of words never seen in training."""
    heads = pd.concat([df.headline_a, df.headline_b])
    toks = [tokens(h) for h in heads]
    n_words = np.array([len(t) for t in toks])
    oov = np.array([np.mean([w not in vocab for w in t]) if t else 0.0 for t in toks])
    return n_words, oov


def estimate_prior(pred_rate, tpr, fpr):
    """Black-box shift estimate of the same-story rate from unlabeled predictions (Lipton et al., 2018, binary case):
    observed positive-prediction rate q = tpr * pi + fpr * (1 - pi)  ->  pi = (q - fpr) / (tpr - fpr)."""
    return float(np.clip((pred_rate - fpr) / max(tpr - fpr, 1e-6), 0, 1))


def drift_row(pipe, df, ref, vocab, name):
    """One monitoring window: label-free statistics first (available live), then delayed-label metrics."""
    s = pipe.raw(df.headline_a, df.headline_b)
    n_words, oov = headline_features(df, vocab)
    pred = (s >= pipe.threshold).astype(int)
    row = {"window": name, "pairs": len(df),
           "french_share": float((df.edition == "fr-CA").mean()),
           "psi_edition": psi_categorical(ref["edition"], df.edition.values),
           "words_mean": float(n_words.mean()), "psi_words": psi(ref["words"], n_words),
           "oov_rate": float(oov.mean()), "psi_oov": psi(ref["oov"], oov),
           "score_mean": float(s.mean()), "psi_score": psi(ref["score"], s),
           "pred_same_rate": float(pred.mean()),
           "est_same_rate": estimate_prior(pred.mean(), ref["tpr"], ref["fpr"]),
           "true_same_rate": float(df.label.mean())}
    row.update({k: v for k, v in evaluate(df.label.values, s, pipe.threshold).items()})
    return row


def reference_stats(pipe, ref_df, vocab):
    s = pipe.raw(ref_df.headline_a, ref_df.headline_b)
    n_words, oov = headline_features(ref_df, vocab)
    y, pred = ref_df.label.values, (s >= pipe.threshold).astype(int)
    return {"score": s, "words": n_words, "oov": oov, "edition": ref_df.edition.values,
            "tpr": float(pred[y == 1].mean()), "fpr": float(pred[y == 0].mean()),
            "f1": float(evaluate(y, s, pipe.threshold)["f1"])}


def prior_shift_probs(platt, scores, pi_old, pi_new):
    """Label-shift correction of a Platt model (Saerens et al., 2002): add the change in prior log-odds to its logit."""
    z = platt.decision_function(np.asarray(scores).reshape(-1, 1))
    pi_new = float(np.clip(pi_new, 1e-3, 1 - 1e-3))
    z = z + np.log(pi_new / (1 - pi_new)) - np.log(pi_old / (1 - pi_old))
    return 1 / (1 + np.exp(-z))


def expected_f1_threshold(scores, probs):
    """Score threshold that maximises the F1 expected under calibrated probabilities (no labels needed):
    E[TP] = sum p over predicted positives, E[FP] = sum (1 - p) over them, E[FN] = sum p over the rest."""
    order = np.argsort(-np.asarray(scores))
    s, p = np.asarray(scores)[order], np.asarray(probs)[order]
    tp = np.cumsum(p); fp = np.cumsum(1 - p); fn = p.sum() - tp
    f1 = 2 * tp / np.clip(2 * tp + fp + fn, 1e-12, None)
    return float(s[int(np.argmax(f1))])


# simulated label shift
def resample_prior(df, pi, n, seed):
    """n pairs with a share `pi` of same-story pairs (sampling with replacement only if a class runs out)."""
    rng = np.random.default_rng(seed)
    pos, neg = df[df.label == 1], df[df.label == 0]
    k = int(round(pi * n))
    take = lambda d, m: d.iloc[rng.choice(len(d), m, replace=m > len(d))]
    return pd.concat([take(pos, k), take(neg, n - k)]).sample(frac=1, random_state=seed).reset_index(drop=True)


# update: incremental fine-tune with replay, or full retrain
def finetune(start, train_df, val_df, path, seed=SEED, epochs=2, aug_prob=0.5, lr=None, log=print):
    """The Milestone 2 / Section 2 recipe (OnlineContrastiveLoss, frozen word embeddings, AdamW with linear warmup,
    early stopping on val PR-AUC, augmentation as in v2), started from any checkpoint (`start`) with its own seed.
    Started from a fine-tuned model with a small labeled batch + replay it is an incremental update; started from
    the base model with all data it is a full retrain."""
    cfg = dict(R.CFG, epochs=epochs)
    if lr:
        cfg["learning_rate"] = lr
    np.random.seed(seed); torch.manual_seed(seed)
    model = SentenceTransformer(str(start), device=R.DEVICE, cache_folder=str(MODELS))
    model.max_seq_length = cfg["max_seq_length"]
    if cfg["freeze_word_embeddings"]:
        next(m for n, m in model.named_modules() if n.endswith("word_embeddings")).requires_grad_(False)
    loss_fn = losses.OnlineContrastiveLoss(model, margin=cfg["margin"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    bs = cfg["batch_size"]
    steps = int(np.ceil(len(train_df) / bs))
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps * epochs), steps * epochs)
    A = [clean_headline(h) for h in train_df.headline_a]
    B = [clean_headline(h) for h in train_df.headline_b]
    y = train_df.label.values.astype(np.float32)
    rng, aug_rng = np.random.default_rng(seed), np.random.default_rng(seed + 100)
    va = [clean_headline(h) for h in val_df.headline_a]
    vb = [clean_headline(h) for h in val_df.headline_b]
    vy = val_df.label.values
    uniq = list(dict.fromkeys(va + vb))
    index = {h: i for i, h in enumerate(uniq)}
    best, bad, best_epoch, done, t0 = -1.0, 0, 0, 0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_df))
        for s in range(steps):
            idx = order[s * bs:(s + 1) * bs]
            xa = [augment(A[i], aug_rng, aug_prob) if aug_prob else A[i] for i in idx]
            xb = [augment(B[i], aug_rng, aug_prob) if aug_prob else B[i] for i in idx]
            loss = loss_fn([R.to_device(model.tokenize(xa)), R.to_device(model.tokenize(xb))],
                           torch.tensor(y[idx], device=R.DEVICE))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]); opt.step(); sched.step()
            done += 1
        model.eval()
        E = model.encode(uniq, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        sc = (E[[index[h] for h in va]] * E[[index[h] for h in vb]]).sum(axis=1)
        m = evaluate(vy, sc, best_threshold(vy, sc))
        log(f"    epoch {epoch}: val F1 {m['f1']:.4f}, PR-AUC {m['pr_auc']:.4f}")
        if m["pr_auc"] > best:
            best, bad, best_epoch = m["pr_auc"], 0, epoch
            model.save(str(path))
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    minutes = (time.perf_counter() - t0) / 60
    del model
    if R.DEVICE == "mps":
        torch.mps.empty_cache()
    elif R.DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"path": Path(path), "best_epoch": best_epoch, "steps": done, "minutes": minutes, "train_pairs": len(train_df)}


def grouping(pipe, pages):
    """Card-level grouping vs Google's own cards (same procedure as the Section 2 notebook, clean input):
    average-linkage clustering of the page's headlines cut at the pipeline's threshold. Returns (mean ARI, exact cards)."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score
    rows = []
    for heads, truth, _ in pages:
        if len(heads) < 2:
            pred = np.zeros(len(heads), int)
        else:
            E = pipe.scorer.embed(pipe.prep(heads))
            S = E @ E.T
            pred = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                           distance_threshold=1 - pipe.threshold).fit_predict(np.clip(1 - S, 0, None))
        google = {frozenset(np.where(truth == k)[0]) for k in set(truth)}
        ours = {frozenset(np.where(pred == k)[0]) for k in set(pred)}
        rows.append((adjusted_rand_score(truth, pred), len(google & ours), len(google)))
    r = np.array(rows)
    return float(r[:, 0].mean()), float(r[:, 1].sum() / r[:, 2].sum())


def interleaved_latency(pipes, pages, rounds=10, per_round=20, warmup=20):
    """Per-page latency (Milestone 2 Section 3 definition: embed the page's headlines and score all pairs) for several
    models, measured round-robin so that every model sees the same machine state (thermals, caches, memory)."""
    def one(pipe, h):
        i, j = np.triu_indices(len(h), 1)
        t0 = time.perf_counter()
        E = pipe.scorer.embed(pipe.prep(h))
        _ = (E @ E.T)[i, j]
        return (time.perf_counter() - t0) * 1000
    for pipe in pipes.values():
        for r in range(warmup):
            one(pipe, pages[r % len(pages)])
    times = {k: [] for k in pipes}
    for r in range(rounds):
        for k, pipe in pipes.items():
            for q in range(per_round):
                times[k].append(one(pipe, pages[(r * per_round + q) % len(pages)]))
    return {k: {"p50_ms": float(np.percentile(t, 50)), "p90_ms": float(np.percentile(t, 90)), "n_pages": len(t)}
            for k, t in times.items()}


def drop_model(path):
    shutil.rmtree(path, ignore_errors=True)


def dir_mb(p):
    p = Path(p)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6 if p.exists() else np.nan
