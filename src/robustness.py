import json
import os
import random
import re
import time
import unicodedata
from pathlib import Path

import sentence_transformers  # noqa: F401
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, losses
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from transformers import get_linear_schedule_with_warmup

from src.embeddings import MAX_SEQ_LENGTH, clean_headline

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "data" / "pairs"
MODELS = ROOT / "models"
OUT = ROOT / "outputs" / "milestone3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 6450
BASE_MODEL = os.environ.get("M3_BASE", "paraphrase-multilingual-MiniLM-L12-v2")
V1_PATH = Path(os.environ.get("M3_V1", MODELS / "minilm_finetuned"))
V2_PATH = MODELS / "minilm_robust"

CFG = dict(epochs=3, batch_size=32, learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.1, max_grad_norm=1.0,
           max_seq_length=MAX_SEQ_LENGTH, margin=0.5, freeze_word_embeddings=True, patience=1)


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_pairs():
    return {s: pd.read_csv(PAIRS / f"{s}.csv", dtype={"timestamp": str}) for s in ["train", "val", "test"]}


# metrics
# best_threshold and evaluate are copied from src/baselines.py
def best_threshold(labels, scores):
    """Threshold with the best F1 (chosen on val, then applied unchanged elsewhere)."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(labels, scores, threshold):
    pred = (scores >= threshold).astype(int)
    out = {"precision": precision_score(labels, pred, zero_division=0), "recall": recall_score(labels, pred, zero_division=0),
           "f1": f1_score(labels, pred, zero_division=0), "accuracy": accuracy_score(labels, pred)}
    if len(set(labels)) == 2:
        out.update(roc_auc=roc_auc_score(labels, scores), pr_auc=average_precision_score(labels, scores))
    return out


def bootstrap_ci(labels, scores, threshold, metric="f1", n=1000, seed=SEED):
    """95% percentile bootstrap interval over pairs."""
    rng = np.random.default_rng(seed)
    labels, scores = np.asarray(labels), np.asarray(scores)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(labels), len(labels))
        if metric == "f1":
            vals.append(f1_score(labels[i], (scores[i] >= threshold).astype(int), zero_division=0))
        elif metric == "pr_auc":
            vals.append(average_precision_score(labels[i], scores[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ece(labels, probs, bins=15):
    """Expected calibration error with equal-width bins; also returns the reliability-diagram table."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    rows, total = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf, acc = probs[m].mean(), labels[m].mean()
        total += m.mean() * abs(conf - acc)
        rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()), "confidence": conf, "accuracy": acc})
    return float(total), pd.DataFrame(rows)


# scorers
class Tfidf:
    """Milestone 2 TF-IDF baseline, same settings as src/baselines.py, fitted on train headlines only. The fitted
    vectorizer was not saved in Milestone 2, so it is refitted here (under a second, same result)."""
    name = "TF-IDF"

    def __init__(self, train_df):
        heads = pd.unique(pd.concat([train_df.headline_a, train_df.headline_b]))
        self.vec = TfidfVectorizer(preprocessor=lambda t: clean_headline(t).lower(), strip_accents="unicode",
                                   ngram_range=(1, 2), sublinear_tf=True).fit(heads)

    def embed(self, texts):
        return self.vec.transform(texts)

    def scores(self, a, b):
        return np.asarray(self.vec.transform(a).multiply(self.vec.transform(b)).sum(axis=1)).ravel()


class BiEncoder:
    """A fine-tuned MiniLM bi-encoder: cosine similarity between mean-pooled, normalized headline vectors."""

    def __init__(self, path, name):
        self.name = name
        self.model = SentenceTransformer(str(path), device=DEVICE, cache_folder=str(MODELS))
        self.model.max_seq_length = CFG["max_seq_length"]
        self.model.eval()

    def embed(self, texts, batch_size=64):
        return self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

    def scores(self, a, b):
        unique = list(dict.fromkeys(list(a) + list(b)))
        index = {h: i for i, h in enumerate(unique)}
        E = self.embed(unique)
        return (E[[index[h] for h in a]] * E[[index[h] for h in b]]).sum(axis=1)


class Pipeline:
    """Scorer + optional input sanitizer + Platt calibration (fitted on clean val) + val threshold."""

    def __init__(self, scorer, sanitize_inputs=False, name=None):
        self.scorer, self.sanitize_inputs = scorer, sanitize_inputs
        self.name = name or scorer.name + (" + sanitizer" if sanitize_inputs else "")

    def prep(self, texts):
        return [sanitize(t) if self.sanitize_inputs else clean_headline(t) for t in texts]

    def raw(self, a, b):
        return self.scorer.scores(self.prep(a), self.prep(b))

    def fit(self, val_df, threshold=None):
        """Threshold = best val F1 (the Milestone 2 rule), unless one is given (a sanitized pipeline keeps the
        threshold of its model: the sanitizer is an input filter, it should not move the decision rule)."""
        s = self.raw(val_df.headline_a, val_df.headline_b)
        y = val_df.label.values
        self.threshold = best_threshold(y, s) if threshold is None else float(threshold)
        self.platt = LogisticRegression(C=1e4).fit(s.reshape(-1, 1), y)
        return self

    def prob(self, scores):
        return self.platt.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]


# Milestone 2 outputs (reused)
M2_OUT = ROOT / "outputs" / "milestone2"
M2_CONFIG_KEYS = ["epochs", "batch_size", "learning_rate", "weight_decay", "warmup_ratio", "max_grad_norm",
                  "max_seq_length", "margin", "freeze_word_embeddings", "patience"]


def load_m2(m2_out=M2_OUT):
    """Reads what Milestone 2 saved, so Milestone 3 reuses it instead of re-deciding it:
      config      the v1 training recipe (train_config.json) -> v2 is trained with exactly the same recipe
      reference   v1 / TF-IDF / pretrained val threshold and test metrics (pair_metrics_all_models.csv),
                  card grouping (grouping_metrics.csv), training time (training_efficiency.csv) and
                  per-page latency (efficiency/inference_efficiency.csv)
    Missing files are reported and skipped. Use the committed versions (RTX 3080 run):
        git checkout origin/ege/milestone2/efficiency -- outputs/milestone2
    """
    m2_out = Path(m2_out)
    out, missing = {"config": None, "pairs": {}, "grouping": {}, "train_minutes": None, "train_hardware": None,
                    "page_latency": None}, []
    f = m2_out / "train_config.json"
    if f.exists():
        j = json.load(open(f))
        out["config"] = {k: j["config"][k] for k in M2_CONFIG_KEYS if k in j["config"]}
        out["seed"] = j["config"].get("seed")
        out["train_hardware"] = (j.get("hardware") or {}).get("gpu") or (j.get("hardware") or {}).get("cpu")
        out["smoke"] = j.get("smoke")
    else:
        missing.append(f.name)
    f = m2_out / "pair_metrics_all_models.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[d.edition == "all"]
        name = {"tfidf": "TF-IDF", "minilm_pretrained": "pretrained", "minilm_finetuned": "v1"}
        for _, r in d.iterrows():
            out["pairs"].setdefault(name.get(r.model, r.model), {})[r.split] = r.to_dict()
    else:
        missing.append(f.name)
    f = m2_out / "grouping_metrics.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[d.edition == "all"]
        name = {"tfidf": "TF-IDF", "minilm_pretrained": "pretrained", "minilm_finetuned": "v1"}
        out["grouping"] = {name.get(r.model, r.model): {"ARI": r["ARI (mean per page)"], "exact_cards": r["exact cards"]}
                           for _, r in d.iterrows()}
    else:
        missing.append(f.name)
    f = m2_out / "training_efficiency.csv"
    if f.exists():
        d = pd.read_csv(f, index_col=0)["value"]
        out["train_minutes"] = float(d.get("total training time (min)", np.nan))
    else:
        missing.append(f.name)
    f = m2_out / "efficiency" / "inference_efficiency.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[(d.model == "bi-encoder") & d.variant.str.contains("CPU")]
        if len(d):
            out["page_latency"] = {"p50_ms": float(d["latency p50, one front page (ms)"].iloc[0]),
                                   "p90_ms": float(d["latency p90, one front page (ms)"].iloc[0]),
                                   "hardware": "Apple M4 Max CPU (Milestone 2, Section 3)"}
    else:
        missing.append("efficiency/" + f.name)
    out["missing"] = missing
    return out


def same_model_as_m2(pipe, val_df, test_df, m2_pairs, tol=0.001):
    """True when this model reproduces Milestone 2's recorded val threshold and test F1, i.e. it IS the M2 model."""
    if not m2_pairs or "val" not in m2_pairs or "test" not in m2_pairs:
        return None, {}
    s_test = pipe.raw(test_df.headline_a, test_df.headline_b)
    m = evaluate(test_df.label.values, s_test, pipe.threshold)
    check = {"threshold_m2": m2_pairs["val"]["threshold"], "threshold_m3": round(pipe.threshold, 4),
             "test_f1_m2": m2_pairs["test"]["f1"], "test_f1_m3": m["f1"],
             "test_pr_auc_m2": m2_pairs["test"]["pr_auc"], "test_pr_auc_m3": m["pr_auc"]}
    ok = abs(check["threshold_m2"] - check["threshold_m3"]) <= 1e-4 and abs(check["test_f1_m2"] - m["f1"]) <= tol
    return ok, check


# text corruptions
QWERTY = "qwertyuiop asdfghjkl zxcvbnm".split()
NEIGHBOURS = {}
for row in QWERTY:
    for i, ch in enumerate(row):
        NEIGHBOURS[ch] = row[max(0, i - 1):i] + row[i + 1:i + 2]


def _typo(word, rng):
    if len(word) < 2:
        return word
    i = int(rng.integers(0, len(word) - 1))
    op = rng.integers(0, 4)
    if op == 0:                                   # swap two neighbouring letters
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == 1:                                   # drop a letter
        return word[:i] + word[i + 1:]
    if op == 2:                                   # double a letter
        return word[:i] + word[i] + word[i:]
    near = NEIGHBOURS.get(word[i].lower(), "")    # hit the key next to it
    return word[:i] + (near[int(rng.integers(0, len(near)))] if near else word[i]) + word[i + 1:]


def char_noise(text, rng, p):
    """Typos: each word gets one character edit with probability p."""
    return " ".join(_typo(w, rng) if rng.random() < p else w for w in text.split())


def word_dropout(text, rng, p):
    """Partial loss: each word is dropped with probability p (at least one word is kept)."""
    words = text.split()
    kept = [w for w in words if rng.random() >= p]
    return " ".join(kept or words[:1])


def strip_accents(text, rng=None, p=None):
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def change_case(text, rng, p=None):
    return text.upper() if rng.random() < 0.5 else text.lower()


PREFIXES = ["LIVE: ", "Live updates: ", "BREAKING: ", "Opinion: ", "Analysis | ", "CP NewsAlert: ",
            "EN DIRECT | ", "[Couverture en direct] ", "Opinion | ", "Chronique : ", "Dossier | "]


def add_prefix(text, rng, p=None):
    return PREFIXES[int(rng.integers(0, len(PREFIXES)))] + text


# Latin letter -> visually identical Cyrillic / Greek letter
HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і", "j": "ј", "s": "ѕ",
              "A": "А", "E": "Е", "O": "О", "P": "Р", "C": "С", "X": "Х", "B": "В", "H": "Н", "K": "К", "M": "М",
              "T": "Т", "N": "Ν", "I": "І"}
INVISIBLE = ["​", "‌", "‍", "⁠", "﻿", "­"]


def homoglyph(text, rng, p):
    """Looks identical to a reader: each word, with probability p, gets one letter swapped for a look-alike
    and an invisible character inserted."""
    out = []
    for w in text.split():
        if rng.random() < p:
            spots = [i for i, ch in enumerate(w) if ch in HOMOGLYPHS]
            if spots:
                i = spots[int(rng.integers(0, len(spots)))]
                w = w[:i] + HOMOGLYPHS[w[i]] + w[i + 1:]
            j = int(rng.integers(0, len(w) + 1))
            w = w[:j] + INVISIBLE[int(rng.integers(0, len(INVISIBLE)))] + w[j:]
        out.append(w)
    return " ".join(out)


CORRUPTIONS = {   # name -> (function, strengths); strength None = single setting
    "typos": (char_noise, [0.1, 0.2, 0.3, 0.5]),
    "word dropout": (word_dropout, [0.1, 0.2, 0.3, 0.5]),
    "homoglyphs": (homoglyph, [0.1, 0.2, 0.3, 0.5]),
    "accents stripped": (strip_accents, [None]),
    "case changed": (change_case, [None]),
    "prefix added": (add_prefix, [None]),
}
NATURAL = ["typos", "word dropout", "accents stripped", "case changed", "prefix added"]


def corrupt_df(df, fn, strength, seed, sides="both"):
    """Copy of the pairs with one corruption applied (to both headlines, or only headline_a)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["headline_a"] = [fn(clean_headline(h), rng, strength) for h in df.headline_a]
    if sides == "both":
        out["headline_b"] = [fn(clean_headline(h), rng, strength) for h in df.headline_b]
    return out


# inference-level defence: sanitizer
CONFUSABLES = {v: k for k, v in HOMOGLYPHS.items()}
CONFUSABLES.update({"ο": "o", "α": "a", "ε": "e", "ρ": "p", "τ": "t", "ν": "v", "κ": "k", "Ι": "I", "Α": "A", "Β": "B",
                    "Ε": "E", "Η": "H", "Κ": "K", "Μ": "M", "Ο": "O", "Ρ": "P", "Τ": "T", "Χ": "X", "Ζ": "Z"})
BOILERPLATE = [re.compile(p, re.I) for p in [
    r"^\s*\[\s*(couverture en direct|en direct|live|direct)\s*\]\s*",
    r"^\s*(en direct|live updates?|live|direct|breaking( news)?|alerte)\s*[:|,\-–]\s*",
    r"^\s*(dossier|opinion|analyse|analysis|chronique|éditorial|editorial|cp newsalert|breaking|exclusive|exclusif)\s*[:|]\s*"]]
TRAILING = re.compile(r"\s*\(\d+ articles?\)\s*$", re.I)


def _fold(ch):
    if ch in CONFUSABLES:
        return CONFUSABLES[ch]
    return ch


def sanitize(text):
    """NFKC, drop invisible format characters, map Cyrillic/Greek look-alikes inside Latin words back to Latin,
    strip live/opinion prefixes (extends the Milestone 2 boilerplate rule), collapse whitespace."""
    t = unicodedata.normalize("NFKC", str(text))
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Cf")
    words = []
    for w in t.split():
        latin = sum(ch.isascii() and ch.isalpha() for ch in w)
        other = sum(ch.isalpha() and not ch.isascii() and ch in CONFUSABLES for ch in w)
        # only fold mixed-script words, so real Greek/Cyrillic text is left alone
        words.append("".join(_fold(ch) for ch in w) if latin and other else w)
    t = " ".join(words)
    changed = True
    while changed:
        changed = False
        for p in BOILERPLATE:
            new = p.sub("", t, count=1)
            if new != t and new.strip():
                t, changed = new, True
    t = TRAILING.sub("", t).strip()
    return " ".join(unicodedata.normalize("NFC", t).split()) or clean_headline(text)


def abstain_band(pipe, val_df, coverage=0.9):
    """Probability band around the decision threshold, sized on val so that `coverage` of pairs are still decided.
    Pairs inside the band are not merged automatically: the extension keeps Google's own card for them."""
    s = pipe.raw(val_df.headline_a, val_df.headline_b)
    p, p_t = pipe.prob(s), float(pipe.prob([pipe.threshold])[0])
    width = float(np.quantile(np.abs(p - p_t), 1 - coverage))
    return p_t - width, p_t + width


def selective(labels, probs, scores, threshold, band):
    keep = (probs < band[0]) | (probs > band[1])
    m = evaluate(labels[keep], scores[keep], threshold) if keep.sum() and len(set(labels[keep])) == 2 else {"f1": np.nan}
    return {"coverage": keep.mean(), "selective_f1": m["f1"]}


# data-level defence: noise augmentation
AUG_OPS = [lambda t, r: char_noise(t, r, 0.15), lambda t, r: word_dropout(t, r, 0.15), strip_accents,
           lambda t, r: change_case(t, r), lambda t, r: add_prefix(t, r)]


def augment(text, rng, prob=0.5):
    """With probability `prob`, apply one natural corruption. Homoglyphs, entity stuffing and the gradient
    attacks are NOT used in training, so they test whether robustness carries over to unseen attacks."""
    if rng.random() >= prob:
        return text
    return AUG_OPS[int(rng.integers(0, len(AUG_OPS)))](text, rng)


def to_device(features):
    return {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in features.items()}


def train_biencoder(train_df, val_df, path, aug_prob=0.0, base=BASE_MODEL, cfg=CFG, log=print):
    """Milestone 2 recipe (OnlineContrastiveLoss, frozen word embeddings, AdamW, early stopping on clean val PR-AUC),
    plus on-the-fly augmentation of each headline with probability aug_prob (fresh draw every epoch)."""
    set_seed()
    model = SentenceTransformer(str(base), device=DEVICE, cache_folder=str(MODELS))
    model.max_seq_length = cfg["max_seq_length"]
    if cfg["freeze_word_embeddings"]:
        next(m for n, m in model.named_modules() if n.endswith("word_embeddings")).requires_grad_(False)
    loss_fn = losses.OnlineContrastiveLoss(model, margin=cfg["margin"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    bs = cfg["batch_size"]
    steps = int(np.ceil(len(train_df) / bs))
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps * cfg["epochs"]), steps * cfg["epochs"])
    A, B = [clean_headline(h) for h in train_df.headline_a], [clean_headline(h) for h in train_df.headline_b]
    y = train_df.label.values.astype(np.float32)
    rng, aug_rng = np.random.default_rng(SEED), np.random.default_rng(SEED + 100)
    va, vb, vy = [clean_headline(h) for h in val_df.headline_a], [clean_headline(h) for h in val_df.headline_b], val_df.label.values
    best, bad, best_epoch, t0, history = -1.0, 0, 0, time.perf_counter(), []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = rng.permutation(len(train_df))
        for s in range(steps):
            idx = order[s * bs:(s + 1) * bs]
            xa = [augment(A[i], aug_rng, aug_prob) if aug_prob else A[i] for i in idx]
            xb = [augment(B[i], aug_rng, aug_prob) if aug_prob else B[i] for i in idx]
            loss = loss_fn([to_device(model.tokenize(xa)), to_device(model.tokenize(xb))], torch.tensor(y[idx], device=DEVICE))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"]); opt.step(); sched.step()
        model.eval()
        E = model.encode(list(dict.fromkeys(va + vb)), batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False)
        index = {h: i for i, h in enumerate(dict.fromkeys(va + vb))}
        sc = (E[[index[h] for h in va]] * E[[index[h] for h in vb]]).sum(axis=1)
        m = evaluate(vy, sc, best_threshold(vy, sc))
        history.append({"epoch": epoch, "minutes": (time.perf_counter() - t0) / 60, **m})
        log(f"  epoch {epoch}: val F1 {m['f1']:.4f}, PR-AUC {m['pr_auc']:.4f}")
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
    return {"path": path, "best_epoch": best_epoch, "minutes": round((time.perf_counter() - t0) / 60, 2),
            "history": pd.DataFrame(history)}


# attacks
STOP = set("""a an the of to in on for and or but with at by from as is are was were be been has have had it its this that
these those after over into about than says said new le la les un une des de du et à au aux en pour sur par dans avec
qui que est sont a été plus ses son sa leur leurs ce cette""".split())


def salient_words(text):
    """Words a reader would call the story's key terms: capitalized names, numbers, long words; no stop words."""
    words = [w.strip(".,:;!?«»\"'()[]’‘“”") for w in text.split()]
    scored = []
    for i, w in enumerate(words):
        if not w or w.lower() in STOP:
            continue
        s = (2 if (w[0].isupper() and i > 0) else 0) + (1 if any(c.isdigit() for c in w) else 0) + (1 if len(w) >= 7 else 0)
        scored.append((s, -i, w))
    return [w for _, _, w in sorted(scored, reverse=True)]


def entity_stuff(df, k):
    """False-merge attack (black-box, no queries): on different-story pairs, the attacker appends the k key terms
    of the other (trending) story to its own headline, hoping to be grouped with it. Same-story pairs untouched."""
    out = df.copy()
    new_a = []
    for a, b, y in zip(df.headline_a, df.headline_b, df.label):
        a = clean_headline(a)
        if y == 0 and k > 0:
            have = set(w.lower() for w in a.split())
            extra = [w for w in salient_words(clean_headline(b)) if w.lower() not in have][:k]
            a = a + (" | " + " ".join(extra) if extra else "")
        new_a.append(a)
    out["headline_a"] = new_a
    return out


def greedy_attack(pipe, df, max_edits=3, log_every=0):
    """Black-box attack with score access (the attacker queries the public extension/API and sees the grouping).
    Same-story pairs: each step tries deleting or misspelling every word of headline_a and keeps the edit that lowers
    the similarity most. Different-story pairs: each step tries appending one key term of headline_b and keeps the one
    that raises it most. Stops when the decision flips. Returns one row per pair."""
    rows, rng = [], np.random.default_rng(SEED)
    for n, (a, b, y) in enumerate(zip(df.headline_a, df.headline_b, df.label)):
        a, b = clean_headline(a), clean_headline(b)
        cur = float(pipe.raw([a], [b])[0])
        start, queries, edits = cur, 1, 0
        correct0 = (cur >= pipe.threshold) == bool(y)
        while correct0 and edits < max_edits and ((cur >= pipe.threshold) == bool(y)):
            words = a.split()
            if y == 1:
                cands = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words)) if len(words) > 1]
                cands += [" ".join(words[:i] + [_typo(words[i], rng)] + words[i + 1:]) for i in range(len(words))]
            else:
                have = set(w.lower() for w in words)
                cands = [a + " " + w for w in salient_words(b)[:8] if w.lower() not in have]
            if not cands:
                break
            s = pipe.raw(cands, [b] * len(cands))
            queries += len(cands)
            j = int(np.argmin(s) if y == 1 else np.argmax(s))
            a, cur, edits = cands[j], float(s[j]), edits + 1
        rows.append({"label": int(y), "start_score": start, "end_score": cur, "edits": edits, "queries": queries,
                     "correct_before": bool(correct0), "correct_after": (cur >= pipe.threshold) == bool(y),
                     "headline_a_adv": a, "headline_b": b})
        if log_every and (n + 1) % log_every == 0:
            print(f"    {n + 1}/{len(df)} pairs")
    return pd.DataFrame(rows)


def _encode_from_embeds(model, embeds, mask):
    """Mean-pooled, normalized sentence vector from token embeddings (same pooling as the saved model)."""
    out = model[0].auto_model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
    m = mask.unsqueeze(-1).float()
    return torch.nn.functional.normalize((out * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=1)


def gradient_attack(biencoder, df, eps, steps=1, batch_size=64):
    """White-box attack on the token embeddings of headline_a (the int8 fallback model ships inside the extension,
    so its weights are effectively public). eps is the L-inf budget per coordinate, in units of the RMS of the
    word-embedding table. steps=1 is FGSM; steps>1 is PGD with step size 2.5*eps/steps. Special tokens are not
    perturbed. Same-story pairs are pushed apart, different-story pairs pulled together. Returns cosine scores."""
    model = biencoder.model
    model.eval()
    emb = model[0].auto_model.get_input_embeddings()
    scale = float(emb.weight.detach().pow(2).mean().sqrt())
    a = [clean_headline(h) for h in df.headline_a]
    b = [clean_headline(h) for h in df.headline_b]
    sign_y = torch.tensor(np.where(df.label.values == 1, 1.0, -1.0), dtype=torch.float32, device=DEVICE)
    out = []
    for i in range(0, len(a), batch_size):
        fa = to_device(model.tokenize(a[i:i + batch_size]))
        with torch.no_grad():
            vb = torch.tensor(biencoder.embed(b[i:i + batch_size]), device=DEVICE)
            base = emb(fa["input_ids"])
        special = torch.isin(fa["input_ids"], torch.tensor(model.tokenizer.all_special_ids, device=DEVICE))
        free = ((~special) & fa["attention_mask"].bool()).unsqueeze(-1).float()
        delta = torch.zeros_like(base)
        budget, step = eps * scale, (2.5 * eps * scale / steps if steps > 1 else eps * scale)
        for _ in range(steps if eps > 0 else 0):
            delta.requires_grad_(True)
            va = _encode_from_embeds(model, base + delta * free, fa["attention_mask"])
            loss = (sign_y[i:i + batch_size] * (va * vb).sum(1)).sum()   # attacker lowers this
            grad, = torch.autograd.grad(loss, delta)
            delta = (delta.detach() - step * grad.sign()).clamp(-budget, budget)
        with torch.no_grad():
            va = _encode_from_embeds(model, base + delta * free, fa["attention_mask"])
            out.append((va * vb).sum(1).float().cpu().numpy())
    return np.concatenate(out)


# latency
def page_latency(pipe, pages, repeats=200, warmup=20):
    """Per-page latency (embed the page's headlines and score all pairs), as in Milestone 2 Section 3."""
    times = []
    for r in range(warmup + repeats):
        h = pages[r % len(pages)]
        i, j = np.triu_indices(len(h), 1)
        t0 = time.perf_counter()
        hh = pipe.prep(h)
        E = pipe.scorer.embed(hh)
        S = E @ E.T
        _ = np.asarray(S.todense() if hasattr(S, "todense") else S)[i, j]
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        if r >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p90_ms": float(np.percentile(times, 90))}


def front_pages(min_date="20260501000000"):
    """Test-period front pages (Top stories / À la une), same filter as Milestone 2 grouping. Returns
    [(headlines, google_card_ids, edition)]."""
    fp = ROOT / "data" / "google_news_front_page"
    snaps = pd.read_csv(fp / "snapshots.csv", dtype={"timestamp": str})
    keep = snaps[(snaps.purpose != "repeat") & (snaps.exclude.astype(str).str.lower() != "true")][["timestamp", "edition"]]
    arts = pd.read_csv(fp / "front_page_articles.csv", dtype={"timestamp": str}).merge(keep, on=["timestamp", "edition"])
    arts = arts[arts.section.isin(["Top stories", "À la une"]) & arts.headline.notna() & (arts.timestamp >= min_date)]
    return [([clean_headline(h) for h in p.headline], p.card.values, ed) for (_, ed), p in arts.groupby(["timestamp", "edition"])]
