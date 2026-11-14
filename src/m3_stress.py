"""
Milestone 3 stress tests: ways to damage headlines, and helpers to score and measure the damaged pairs.

    from src.m3_stress import *

Every corruption takes (headline, rng) and returns the damaged headline. It is applied once per distinct
headline, so a headline that appears in several pairs gets the same damage everywhere (like a real typo would).
"""

import math
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.embeddings import clean_headline
from src.m2_common import SEED, best_threshold, evaluate

LETTERS = list("abcdefghijklmnopqrstuvwxyz")


# ---------------------------------------------------------------- corruptions
def char_noise(p):
    """Typos: each letter is deleted, replaced, doubled with a random letter, or swapped with the next one, with probability p."""
    def f(text, rng):
        chars, out, i = list(text), [], 0
        while i < len(chars):
            c = chars[i]
            if c.isalpha() and rng.random() < p:
                op = rng.integers(4)
                if op == 1:
                    out.append(rng.choice(LETTERS))
                elif op == 2:
                    out += [c, rng.choice(LETTERS)]
                elif op == 3 and i + 1 < len(chars):
                    out += [chars[i + 1], c]
                    i += 1
                elif op == 3:
                    out.append(c)
                # op == 0: letter deleted
            else:
                out.append(c)
            i += 1
        return "".join(out)
    return f


def word_dropout(p):
    """Each word is removed with probability p (at least one word is kept)."""
    def f(text, rng):
        words = text.split()
        kept = [w for w in words if rng.random() >= p]
        return " ".join(kept) if kept else words[rng.integers(len(words))]
    return f


def word_mask(p):
    """Each word is replaced by a blank '_' with probability p."""
    def f(text, rng):
        return " ".join("_" if rng.random() < p else w for w in text.split())
    return f


def keep_start(frac):
    """Partial feature loss: only the first part of the headline is kept (e.g. a cut-off preview)."""
    def f(text, rng):
        words = text.split()
        return " ".join(words[:max(1, math.ceil(len(words) * frac))])
    return f


def all_caps(text, rng):
    return text.upper()


def strip_accents(text, rng):
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


# name -> list of (level, function); level is what goes on the x-axis of the robustness curves
CORRUPTIONS = {
    "typos (share of letters)": [(p, char_noise(p)) for p in (0.05, 0.10, 0.20)],
    "word dropout (share of words)": [(p, word_dropout(p)) for p in (0.10, 0.20, 0.30)],
    "word masking (share of words)": [(p, word_mask(p)) for p in (0.10, 0.20, 0.30)],
    "only start kept (share removed)": [(1 - k, keep_start(k)) for k in (0.75, 0.50, 0.25)],
    "all caps": [(1, all_caps)],
    "accents removed": [(1, strip_accents)],
}


def corrupt_pairs(df, fn, seed=SEED):
    """Apply fn once to every distinct headline of df (fixed seed) and return the damaged copy."""
    rng = np.random.default_rng(seed)
    unique = pd.unique(pd.concat([df.headline_a, df.headline_b]))
    damaged = {h: fn(clean_headline(h), rng) for h in unique}
    return df.assign(headline_a=df.headline_a.map(damaged), headline_b=df.headline_b.map(damaged))


# ---------------------------------------------------------------- scoring and metrics
def tfidf_scorer(train_df):
    """Same TF-IDF baseline as Milestone 2 (src/baselines.py), fitted on train headlines only."""
    tfidf = TfidfVectorizer(preprocessor=lambda t: clean_headline(t).lower(), strip_accents="unicode",
                            ngram_range=(1, 2), sublinear_tf=True)
    tfidf.fit(pd.unique(pd.concat([train_df.headline_a, train_df.headline_b])))

    def score(df):
        A, B = tfidf.transform(df.headline_a), tfidf.transform(df.headline_b)
        return np.asarray(A.multiply(B).sum(axis=1)).ravel()
    return score


def bootstrap_ci(labels, scores, threshold, n=1000, seed=SEED):
    """95% interval for F1 and PR-AUC by resampling the pairs."""
    rng, f1s, aps = np.random.default_rng(seed), [], []
    for _ in range(n):
        i = rng.integers(0, len(labels), len(labels))
        m = evaluate(labels[i], scores[i], threshold)
        f1s.append(m["f1"]); aps.append(m["pr_auc"])
    return np.percentile(f1s, [2.5, 97.5]), np.percentile(aps, [2.5, 97.5])


def resample_prevalence(df, share, rng):
    """Keep every different-story pair and draw same-story pairs so they make up `share` of the result."""
    neg, pos = df[df.label == 0], df[df.label == 1]
    n_pos = round(share / (1 - share) * len(neg))
    return pd.concat([neg, pos.sample(n_pos, random_state=int(rng.integers(1 << 31)))])


def page_pairs(local_df, max_pairs=3000, seed=SEED):
    """All headline pairs within each local news page (no labels: used only to look at score distributions)."""
    rows = []
    for (run, town), g in local_df.groupby(["run", "town_id"]):
        h = g.headline.dropna().tolist()
        rows += [(h[i], h[j]) for i in range(len(h)) for j in range(i + 1, len(h))]
    out = pd.DataFrame(rows, columns=["headline_a", "headline_b"]).drop_duplicates()
    return out.sample(min(max_pairs, len(out)), random_state=seed)
