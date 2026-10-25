"""
Translate every train headline into the other language (English <-> French) for data augmentation.

Reads data/pairs/train.csv and writes data/augment/translations_train.csv
(columns: edition, headline, translation). Headlines already in that file are skipped,
so the script can be re-run safely.

  English (en-CA) headlines -> French with Helsinki-NLP/opus-mt-en-fr
  French (fr-CA) headlines  -> English with Helsinki-NLP/opus-mt-fr-en

Run from the repo root:  python src/translate_headlines.py
Uses the GPU when one is available; on a CPU it is slow (about an hour).
"""

import sys
import time
from pathlib import Path

import sentence_transformers  # noqa: F401  (import before torch: the other order crashes on some Windows setups)
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.embeddings import clean_headline  # noqa: E402

TRAIN = ROOT / "data" / "pairs" / "train.csv"
CACHE = ROOT / "data" / "augment" / "translations_train.csv"
MODELS_DIR = ROOT / "models"
MT_MODELS = {"en-CA": "Helsinki-NLP/opus-mt-en-fr", "fr-CA": "Helsinki-NLP/opus-mt-fr-en"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


def translate(texts, model_name):
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODELS_DIR))
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=str(MODELS_DIR)).to(DEVICE).eval()
    out = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = tok(texts[i:i + BATCH_SIZE], return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            generated = model.generate(**batch, max_new_tokens=96, num_beams=2)
        out += tok.batch_decode(generated, skip_special_tokens=True)
        if (i // BATCH_SIZE) % 20 == 0:
            print(f"  {i + len(batch['input_ids']):,}/{len(texts):,}", flush=True)
    return out


def main():
    train = pd.read_csv(TRAIN, dtype={"timestamp": str})
    need = pd.concat([train[["edition", "headline_a"]].rename(columns={"headline_a": "headline"}),
                      train[["edition", "headline_b"]].rename(columns={"headline_b": "headline"})]).drop_duplicates()
    cache = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["edition", "headline", "translation"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    todo_all = need.merge(cache, how="left", on=["edition", "headline"])
    for edition, model_name in MT_MODELS.items():
        todo = todo_all[(todo_all.edition == edition) & todo_all.translation.isna()].headline.tolist()
        print(f"{edition}: {len(todo):,} headlines to translate with {model_name} on {DEVICE}", flush=True)
        if not todo:
            continue
        start = time.perf_counter()
        new = pd.DataFrame({"edition": edition, "headline": todo,
                            "translation": translate([clean_headline(h) for h in todo], model_name)})
        cache = pd.concat([cache, new]).drop_duplicates(["edition", "headline"], keep="last")
        cache.to_csv(CACHE, index=False)
        print(f"  done in {(time.perf_counter() - start) / 60:.1f} min", flush=True)
    print(f"{len(cache):,} translations in {CACHE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
