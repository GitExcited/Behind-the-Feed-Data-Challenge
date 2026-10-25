"""
Shared headline embedding functions.

Everyone imports from here so we all use the same model and settings:
    from src.embeddings import embed_headlines, similarity

    E = embed_headlines(["headline 1", "headline 2"])         # (n, 384) array
    s = similarity(pairs["headline_a"], pairs["headline_b"])  # one score per pair

To use the fine-tuned model later, pass its folder as model_name,
or change DEFAULT_MODEL below once it is ready.
"""

from functools import lru_cache
import unicodedata

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1] # repo root
MODELS_DIR = ROOT / "models" # stored models (gitignored)

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_SEQ_LENGTH = 128  # headlines are short so length of text being 128 is enough
BATCH_SIZE = 64


def clean_headline(text) -> str:
    """
    normalize(): an accented letter like “é” can be stored in two different ways in Unicode. NFC makes them identical
    collapses double spaces, tabs and line breaks into single spaces 
    """
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


@lru_cache(maxsize=2) # maxsize=2 to keep pretrained and fine-tuned versions when comparing them later
def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """Load the model once, later calls reuse the same copy in memory."""
    MODELS_DIR.mkdir(exist_ok=True)
    model = SentenceTransformer(model_name, cache_folder=str(MODELS_DIR))
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def embed_headlines(headlines, model_name: str = DEFAULT_MODEL, 
                    batch_size: int = BATCH_SIZE, show_progress: bool = False) -> np.ndarray:
    """
    Embedding function to turn the stories into vectors.
    Turn a list of headlines into vectors.

    Returns an (n, 384) float32 array, one row per headline, each row
    L2-normalized (length 1), so a dot product = cosine similarity.
    """
    texts = [clean_headline(h) for h in headlines] # clean every headline
    return load_model(model_name).encode( # run MiniLM on all of them
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )


def similarity(headlines_a, headlines_b, model_name: str = DEFAULT_MODEL, 
               batch_size: int = BATCH_SIZE, show_progress: bool = False) -> np.ndarray:
    """
    Cosine similarity between paired headlines: a[0] vs b[0], a[1] vs b[1], ...

    Returns an (n,) array. Higher = more likely the same story.
    Each unique headline is embedded only once, which matters because the
    same headline appears in many pairs.
    """
    a = [clean_headline(h) for h in headlines_a]
    b = [clean_headline(h) for h in headlines_b]
    if len(a) != len(b):
        raise ValueError(f"headlines_a has {len(a)} items but headlines_b has {len(b)}")

    unique = list(dict.fromkeys(a + b))          # keeps order, drops repeats
    index = {h: i for i, h in enumerate(unique)}
    E = embed_headlines(unique, model_name, batch_size, show_progress)

    ia = np.fromiter((index[h] for h in a), dtype=np.int64, count=len(a))
    ib = np.fromiter((index[h] for h in b), dtype=np.int64, count=len(b))
    return (E[ia] * E[ib]).sum(axis=1)


if __name__ == "__main__":
    # Quick check: python src/embeddings.py
    a = ["Carney to accept Trump's offer for role on Gaza board",
         "Carney to accept Trump's offer for role on Gaza board"]
    b = ["Carney acceptera un rôle au conseil de Trump sur Gaza",   # same story, French
         "Wildfire forces evacuations near Kamloops"]               # different story
    print("Embedding shape:", embed_headlines(a + b).shape)
    print("Same story:", round(float(similarity(a, b)[0]), 3))
    print("Different story:", round(float(similarity(a, b)[1]), 3))