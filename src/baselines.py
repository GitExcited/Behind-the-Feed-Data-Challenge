"""
Baselines for the same-story task: TF-IDF and pretrained MiniLM.

For each pair of headlines, each baseline gives a similarity score.
A threshold turns the score into a decision (>= threshold -> "same story").
The threshold is chosen on VAL (the one with the best F1), then applied
unchanged to TEST, so the test numbers are never tuned on test data.

  TF-IDF : word counts weighted by rarity, cosine similarity.
           Fitted on TRAIN headlines only (no peeking at val/test).

  MiniLM : pretrained paraphrase-multilingual-MiniLM-L12-v2, no fine-tuning,
           cosine similarity from src/embeddings.py.

Outputs (outputs/milestone2/):
  baseline_metrics.csv  precision / recall / F1 / accuracy / ROC-AUC / PR-AUC
                        per baseline and split, plus the threshold and timing

  baseline_scores.csv   every val/test pair with both scores (for plots)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.embeddings import clean_headline, similarity  # import functions from embeddings.py

PAIRS = ROOT / "data" / "pairs"
OUT = ROOT / "outputs" / "milestone2"


"""
Step 1. Load the pairs built by build_pairs.py
"""
splits = {} # load the split of train before 2026, val Jan–Apr, test May onward
for s in ["train", "val", "test"]:
    splits[s] = pd.read_csv(PAIRS / f"{s}.csv", dtype={"timestamp": str})
    
for s, df in splits.items():
    print(f"{s:5s}: {len(df):,} pairs ({df['label'].mean():.1%} same story)")


"""
Step 2. Baseline scorers: each takes two lists of headlines, returns one score per pair
"""
# TF-IDF: learn the vocabulary and word weights from TRAIN headlines only
train_headlines = pd.unique(pd.concat([splits["train"]["headline_a"], splits["train"]["headline_b"]]))

tfidf = TfidfVectorizer(
    preprocessor=lambda t: clean_headline(t).lower(),
    strip_accents="unicode", # "é" and "e" count as the same letter
    ngram_range=(1, 2), # single words and two-word phrases
    sublinear_tf=True,
)

tfidf.fit(train_headlines)


def tfidf_similarity(headlines_a, headlines_b):
    # rows are L2-normalized by default, so row-wise dot product = cosine similarity
    A = tfidf.transform(headlines_a)
    B = tfidf.transform(headlines_b)
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


BASELINES = {
    "tfidf": tfidf_similarity, # the TF-IDF scoring function
    "minilm_pretrained": similarity, # the MiniLM scoring function
}


"""
Step 3. Score val and test, pick the threshold on val, evaluate both
"""
def best_threshold(labels, scores):
    """Threshold with the highest F1 on the given (validation) data."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = np.argmax(f1[:-1]) # last point has no threshold
    return float(thresholds[best])


def evaluate(labels, scores, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        "precision": precision_score(labels, pred, zero_division=0),
        "recall": recall_score(labels, pred, zero_division=0),
        "f1": f1_score(labels, pred, zero_division=0),
        "accuracy": accuracy_score(labels, pred),
        "roc_auc": roc_auc_score(labels, scores), # threshold-free
        "pr_auc": average_precision_score(labels, scores), # threshold-free
    }


metrics, score_tables = [], []
for name, score_fn in BASELINES.items(): # results for TF-IDF and MiniLM
    # A. score val and test
    # B. pick the threshold on val
    # C. evaluate val and test with that threshold, save results
    scores = {}
    for split in ["val", "test"]:
        df = splits[split]
        start = time.perf_counter()
        scores[split] = score_fn(df["headline_a"].tolist(), df["headline_b"].tolist())
        seconds = time.perf_counter() - start # how long scoring took -> might be useful for section 3 as an efficiency metric
        print(f"{name:18s} scored {split:4s} in {seconds:6.2f} s")
        scores[split + "_seconds"] = seconds

    # Find a threshold to evaluate the score and if they belong to "same story” or “different story”
    # to pick a threshold on val it tries every possible cut-off on the val scores and returns the one with the highest F1
    # only val is used, so test stays untouched
    threshold = best_threshold(splits["val"]["label"], scores["val"])

    # evaluate val and test with the threshold
    for split in ["val", "test"]:
        df = splits[split]
        row = {"baseline": name, "split": split, "threshold": round(threshold, 4)}
        # compares the decisions (score >= threshold) with the true label column 
        # and returns precision, recall, F1, accuracy, ROC-AUC and PR-AUC
        row.update(evaluate(df["label"].values, scores[split], threshold))
        row["n_pairs"] = len(df)
        row["seconds"] = round(scores[split + "_seconds"], 3)
        metrics.append(row)

        # we have four rows: TF-IDF val, TF-IDF test, MiniLM val and MiniLM test.
        table = df[["timestamp", "edition", "headline_a", "headline_b", "label"]].copy()
        table["split"], table["baseline"], table["score"] = split, name, scores[split]
        score_tables.append(table)


"""
Step 4. Save and print
"""
OUT.mkdir(parents=True, exist_ok=True)
metrics = pd.DataFrame(metrics)
metrics.to_csv(OUT / "baseline_metrics.csv", index=False)
pd.concat(score_tables).to_csv(OUT / "baseline_scores.csv", index=False)

pd.set_option("display.width", 120)
print("\n" + metrics.round(3).to_string(index=False))
print(f"\nSaved to {OUT}")