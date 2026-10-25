from pathlib import Path
import pandas as pd
import itertools

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "google_news_front_page"
TOP_STORIES_SECTIONS = {"À la une", "Top stories"} # Section names for Top Stories in each edition.

# Time split boundaries (timestamps are YYYYMMDDhhmmss strings).
VAL_START = "20260101000000"
TEST_START = "20260501000000"


"""
Step 1. Take top stories articles from front_page_articles.csv 
(only the main snapshots, cards with 2+ articles, and group by timestamp + card, not story_id since it's half empty)
"""
# Import main snapshots and cards with 2+ articles
snapshots = pd.read_csv(DATA/"snapshots.csv", dtype={"timestamp": str})
articles = pd.read_csv(DATA/"front_page_articles.csv", dtype={"timestamp": str})

# Find repeated snapshots
is_repeat = snapshots["purpose"] == "repeat"
# Drop the 5 snapshots that have Exclude=True (were more than 3h off and are marked, we won't take them)
is_excluded = snapshots["exclude"].astype(str).str.strip().str.lower() == "true"

# Remove repeats & excluded from main snapshots (drop same-day repeats)
main = snapshots[~is_repeat & ~is_excluded][["timestamp", "edition"]]
articles = articles.merge(main, on=["timestamp", "edition"], how="inner")

# Keep only Top Stories (English + French)
top_stories = articles[articles["section"].isin(TOP_STORIES_SECTIONS)].copy()
top_stories = top_stories.dropna(subset=["headline"])

# keep only cards with 2+ articles
card_cols = ["timestamp", "edition", "card"]
top_stories["n_in_card"] = top_stories.groupby(card_cols)["headline"].transform("size") # create column to count the number of articles n in card
top_stories = top_stories[top_stories["n_in_card"] >= 2].copy()

# Checks
print("Snapshots:", top_stories.groupby(["timestamp", "edition"]).ngroups)
print("Cards with 2+ articles:", top_stories.groupby(card_cols).ngroups)
print("Articles:", len(top_stories))
print("\nCard sizes (number of cards of each size):")
print(top_stories.groupby(card_cols).size().value_counts().sort_index())
print("\nRows where our count differs from Google's cluster_size:",
      (top_stories["n_in_card"] != top_stories["cluster_size"]).sum())


"""
Step 2. Build pairs:
  same card                         -> label 1 (same story)
  different cards, same snapshot    -> label 0 (different story)
"""
rows = []
for (ts, ed), snap in top_stories.groupby(["timestamp", "edition"]):
    cards = [card[["headline", "outlet", "card"]].to_dict("records")
             for _, card in snap.groupby("card")]

    # Positives: every pair of headlines inside the same card
    for card in cards:
        for a, b in itertools.combinations(card, 2):
            rows.append({"timestamp": ts, "edition": ed,
                         "headline_a": a["headline"], "headline_b": b["headline"],
                         "outlet_a": a["outlet"], "outlet_b": b["outlet"], "label": 1})

    # Negatives: every pair of headlines from two different cards
    for card_a, card_b in itertools.combinations(cards, 2):
        for a in card_a:
            for b in card_b:
                rows.append({"timestamp": ts, "edition": ed,
                             "headline_a": a["headline"], "headline_b": b["headline"],
                             "outlet_a": a["outlet"], "outlet_b": b["outlet"], "label": 0})

pairs = pd.DataFrame(rows)
print("\nPairs:", len(pairs))
print(pairs["label"].value_counts())


"""
Step 3. Clean, split by time, save.
  train: before 2026 | val: Jan-Apr 2026 | test: May 2026 onward
"""
# Drop pairs where both headlines are identical (same syndicated text)
pairs = pairs[pairs["headline_a"] != pairs["headline_b"]].copy()

# Order-independent key, so (A, B) and (B, A) count as the same pair
a, b = pairs["headline_a"], pairs["headline_b"]
pairs["pair_key"] = a.where(a < b, b) + " ||| " + b.where(a < b, a)

# Time split (timestamps are YYYYMMDDhhmmss strings)
pairs["split"] = "train"
pairs.loc[pairs["timestamp"] >= VAL_START, "split"] = "val"
pairs.loc[pairs["timestamp"] >= TEST_START, "split"] = "test"

# Drop pairs labelled both 1 and 0 in the same split (ambiguous)
n_labels = pairs.groupby(["split", "pair_key"])["label"].transform("nunique")
pairs = pairs[n_labels == 1]

# Same pair often repeats across snapshots: keep it once per split
pairs = pairs.sort_values("timestamp").drop_duplicates(["split", "pair_key"])

# No leakage: drop val/test pairs that already appear in an earlier split
train_keys = set(pairs.loc[pairs["split"] == "train", "pair_key"])
val_keys = set(pairs.loc[pairs["split"] == "val", "pair_key"])
leak = ((pairs["split"] == "val") & pairs["pair_key"].isin(train_keys)) | \
       ((pairs["split"] == "test") & pairs["pair_key"].isin(train_keys | val_keys))
pairs = pairs[~leak]

# Save
out = ROOT/"data"/"pairs"
out.mkdir(parents=True, exist_ok=True)
cols = ["timestamp", "edition", "headline_a", "headline_b", "outlet_a", "outlet_b", "label"]
for split in ["train", "val", "test"]:
    pairs.loc[pairs["split"] == split, cols].to_csv(out / f"{split}.csv", index=False)

print("\nAfter cleaning:")
print(pairs.groupby(["split", "label"]).size().unstack())