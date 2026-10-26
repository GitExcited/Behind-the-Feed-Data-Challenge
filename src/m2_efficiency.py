"""
Milestone 2, Section 3: inference and training efficiency of the two same-story models.

Each measurement runs in its own Python process, so memory numbers are not mixed between models:

    python -m src.m2_efficiency infer --model bi --backend onnx_int8 --device cpu
    python -m src.m2_efficiency train --model cross --device mps

The trained models are loaded with plain `transformers` (mean pooling done here for the bi-encoder), so the code runs with
sentence-transformers 5 or 6 and transformers 4 or 5. Scores match `m2_common.pair_scores` / `ce_scores`.

Backends: torch (fp32, CPU or Apple GPU), torch_int8 (PyTorch dynamic int8 quantization, CPU), onnx (ONNX Runtime fp32,
CPU) and onnx_int8 (ONNX Runtime dynamic int8 quantization, CPU).
"""

import argparse
import json
import platform
import resource
import time
import warnings

import sentence_transformers  # noqa: F401  (import before torch: the other order crashes on some Windows setups)
import numpy as np
import pandas as pd
import psutil
import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from src.embeddings import clean_headline
from src.m2_common import MODELS, ROOT, SEED, best_threshold, evaluate, load_pairs

warnings.filterwarnings("ignore")
PATHS = {"bi": MODELS / "minilm_finetuned", "cross": MODELS / "exp" / "cross_encoder"}
ONNX = MODELS / "onnx"
MAX_LEN = 128
BATCH = 64            # pairs per batch for the batch-latency and throughput numbers
N_SINGLE = 300        # pairs timed one at a time for the per-sample latency
WARMUP = 20
GPU_MB = []           # live tensor memory on the Apple GPU, sampled while a batch's tensors are still allocated


def sample_gpu(device):
    if device == "mps":
        GPU_MB.append(torch.mps.current_allocated_memory() / 1e6)


# ---------------------------------------------------------------- data
def front_pages():
    """Test front pages (Top stories / À la une), same filter as the grouping in milestone2_training.ipynb."""
    fp = ROOT / "data" / "google_news_front_page"
    snaps = pd.read_csv(fp / "snapshots.csv", dtype={"timestamp": str})
    keep = snaps[(snaps.purpose != "repeat") & (snaps.exclude.astype(str).str.lower() != "true")][["timestamp", "edition"]]
    arts = pd.read_csv(fp / "front_page_articles.csv", dtype={"timestamp": str}).merge(keep, on=["timestamp", "edition"])
    arts = arts[arts.section.isin(["Top stories", "À la une"]) & arts.headline.notna()]
    arts = arts[arts.timestamp >= "20260501000000"]
    return [[clean_headline(h) for h in page.headline] for _, page in arts.groupby(["timestamp", "edition"])]


# ---------------------------------------------------------------- models
def tokenizer(kind):
    try:
        return AutoTokenizer.from_pretrained(str(PATHS[kind]))
    except ValueError:   # tokenizer_config.json written by transformers 5 ("TokenizersBackend"), read here by transformers 4
        from transformers import XLMRobertaTokenizerFast
        return XLMRobertaTokenizerFast(tokenizer_file=str(PATHS[kind] / "tokenizer.json"))


class Head(torch.nn.Module):
    """Plain tensor in, plain tensor out: token vectors (bi) or the same-story logit (cross). Used for torch and ONNX."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        cls = AutoModel if kind == "bi" else AutoModelForSequenceClassification
        self.net = cls.from_pretrained(str(PATHS[kind])).eval()

    def forward(self, input_ids, attention_mask):
        out = self.net(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state if self.kind == "bi" else out.logits[:, 0]


def export_onnx(kind):
    """fp32 ONNX export, then ONNX Runtime dynamic int8 quantization (weights int8, activations quantized on the fly)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    ONNX.mkdir(parents=True, exist_ok=True)
    fp32, int8 = ONNX / f"{kind}_fp32.onnx", ONNX / f"{kind}_int8.onnx"
    tok = tokenizer(kind)
    f = tok(["Exemple de titre", "Example headline for the export"], padding=True, return_tensors="pt")
    axes = {"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"}, "out": {0: "batch"}}
    with torch.no_grad():
        torch.onnx.export(Head(kind), (f["input_ids"], f["attention_mask"]), str(fp32), input_names=["input_ids", "attention_mask"],
                          output_names=["out"], dynamic_axes=axes, opset_version=17, dynamo=False)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    return {"onnx fp32 (MB)": round(fp32.stat().st_size / 1e6, 1), "onnx int8 (MB)": round(int8.stat().st_size / 1e6, 1)}


class Scorer:
    """Same-story score for headline pairs, for one model kind on one backend."""

    def __init__(self, kind, backend="torch", device="cpu"):
        self.kind, self.backend, self.device = kind, backend, device
        self.tok = tokenizer(kind)
        if backend.startswith("onnx"):
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = torch.get_num_threads()
            name = f"{kind}_{'int8' if backend == 'onnx_int8' else 'fp32'}.onnx"
            self.session = ort.InferenceSession(str(ONNX / name), opts, providers=["CPUExecutionProvider"])
        else:
            self.net = Head(kind)
            if backend == "torch_int8":
                torch.backends.quantized.engine = "qnnpack" if platform.machine() == "arm64" else "fbgemm"
                self.net = torch.ao.quantization.quantize_dynamic(self.net, {torch.nn.Linear}, dtype=torch.qint8)
            self.net = self.net.to(device)

    def _run(self, x, z=None):
        """Model output for a batch of texts (bi) or text pairs (cross)."""
        if self.backend.startswith("onnx"):
            f = self.tok(x, z, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="np")
            out = self.session.run(None, {"input_ids": f["input_ids"].astype(np.int64),
                                          "attention_mask": f["attention_mask"].astype(np.int64)})[0]
            return torch.from_numpy(out), torch.from_numpy(f["attention_mask"])
        f = self.tok(x, z, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.net(f["input_ids"], f["attention_mask"])
            sample_gpu(self.device)
            return out.float().cpu(), f["attention_mask"].cpu()

    def embed(self, texts):
        """Mean-pooled, normalized headline vectors (bi-encoder only)."""
        h, mask = self._run(texts)
        m = mask.unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1).numpy()

    def pairs(self, a, b):
        """Scores for one batch of pairs. Bi: cosine of the two vectors. Cross: both orders in one batch, averaged."""
        if self.kind == "bi":
            E = self.embed(list(a) + list(b))
            return (E[:len(a)] * E[len(a):]).sum(1)
        logits, _ = self._run(list(a) + list(b), list(b) + list(a))
        p = torch.sigmoid(logits).numpy()
        return (p[:len(a)] + p[len(a):]) / 2

    def score(self, a, b, batch_size=BATCH):
        return np.concatenate([self.pairs(a[i:i + batch_size], b[i:i + batch_size]) for i in range(0, len(a), batch_size)])

    def page(self, headlines):
        """All within-page pair scores, as the extension needs them. Bi: each headline embedded once."""
        if self.kind == "bi":
            E = self.embed(headlines)
            return E @ E.T
        i, j = np.triu_indices(len(headlines), k=1)
        return self.score([headlines[k] for k in i], [headlines[k] for k in j])


# ---------------------------------------------------------------- measurements
def rss_mb():
    return psutil.Process().memory_info().rss / 1e6


def peak_rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if platform.system() == "Darwin" else peak / 1e3   # bytes on macOS, KB on Linux


def timed(fn, *args):
    t = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t


def pct(seconds, q):
    return round(float(np.percentile(seconds, q)) * 1000, 2)


def hardware(device):
    import subprocess
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else f"{cpu} GPU (MPS)" if device == "mps" else "none"
    return {"cpu": cpu, "torch threads": torch.get_num_threads(), "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "gpu": gpu, "torch": torch.__version__}


def flops_per_pair(kind, n=200):
    """Forward FLOPs per pair at batch size 1 (no padding), counted by PyTorch on the real test headlines."""
    test = load_pairs()["test"].sample(n, random_state=SEED)
    s = Scorer(kind)
    with FlopCounterMode(display=False) as counter:
        for a, b in zip(test.headline_a, test.headline_b):
            s.pairs([clean_headline(a)], [clean_headline(b)])
    return counter.get_total_flops() / n


def infer(kind, backend, device):
    rng = np.random.default_rng(SEED)
    base = rss_mb()
    t = time.perf_counter()
    s = Scorer(kind, backend, device)
    load_seconds = time.perf_counter() - t
    after_load = rss_mb()
    data = load_pairs()
    prep = {k: ([clean_headline(h) for h in d.headline_a], [clean_headline(h) for h in d.headline_b]) for k, d in data.items()
            if k != "train"}
    a, b = prep["test"]
    s.score(a[:WARMUP * 4], b[:WARMUP * 4])

    # Per-sample latency: one pair at a time
    pick = rng.choice(len(a), N_SINGLE, replace=False)
    for k in pick[:WARMUP]:
        s.pairs([a[k]], [b[k]])
    single = [timed(s.pairs, [a[k]], [b[k]]) for k in pick]

    # Batch latency and throughput: the whole test set in batches of BATCH pairs
    batches = []
    for i in range(0, len(a), BATCH):
        batches.append(timed(s.pairs, a[i:i + BATCH], b[i:i + BATCH]))
    full = [x for x in batches[:-1]]   # last batch is smaller

    # One front page: every pair of headlines on it
    pages = front_pages()
    for p in pages[:5]:
        s.page(p)
    page = [timed(s.page, p) for p in pages]

    # Accuracy: threshold picked on val, applied unchanged to test (same rule as baselines.py)
    sv, st = s.score(*prep["val"]), s.score(*prep["test"])
    thr = best_threshold(data["val"].label.values, sv)
    m = evaluate(data["test"].label.values, st, thr)

    return {
        "model": kind, "backend": backend, "device": device,
        "load time (s)": round(load_seconds, 2),
        "latency p50, 1 pair (ms)": pct(single, 50), "latency p90, 1 pair (ms)": pct(single, 90),
        f"latency p50, batch of {BATCH} pairs (ms)": pct(full, 50), f"latency p90, batch of {BATCH} pairs (ms)": pct(full, 90),
        f"throughput, batch {BATCH} (pairs/s)": round(len(a) / sum(batches), 1),
        "latency p50, one front page (ms)": pct(page, 50), "latency p90, one front page (ms)": pct(page, 90),
        "RSS before loading (MB)": round(base), "RSS after loading (MB)": round(after_load),
        "peak RSS (MB)": round(peak_rss_mb()),
        "peak GPU tensor memory, MPS (MB)": round(max(GPU_MB)) if GPU_MB else None,
        "threshold (val)": round(thr, 4), "test F1": round(m["f1"], 4), "test PR-AUC": round(m["pr_auc"], 4),
        "test precision": round(m["precision"], 4), "test recall": round(m["recall"], 4),
        "test pairs": len(a), "front pages": len(pages), "mean headlines per page": round(float(np.mean([len(p) for p in pages])), 1),
        "hardware": hardware(device),
    }


def online_contrastive(model, fa, fb, labels, margin=0.5):
    """sentence-transformers' OnlineContrastiveLoss written with masks instead of boolean indexing: same value and the
    same FLOPs, but no empty tensors when a batch has no hard pairs (those crash on MPS in torch 2.8)."""
    ea, eb = model(fa)["sentence_embedding"], model(fb)["sentence_embedding"]
    d = 1 - torch.nn.functional.cosine_similarity(ea, eb)
    pos, neg = labels == 1, labels == 0
    inf = torch.tensor(float("inf"), device=d.device)
    max_pos = torch.where(pos, d, -inf).max() if pos.sum() > 1 else d[neg].mean()
    min_neg = torch.where(neg, d, inf).min() if neg.sum() > 1 else d[pos].mean()
    hard_pos, hard_neg = pos & (d > min_neg), neg & (d < max_pos)
    return (hard_pos * d.pow(2)).sum() + (hard_neg * torch.relu(margin - d).pow(2)).sum()


def train(kind, device, steps=60, batch_size=32):
    """Short training run with the real recipe (batch 32, AdamW, same loss) to time steps and read peak memory on this
    machine. Starts from the trained weights: same architecture, so the cost per step is the same as the real run."""
    from sentence_transformers import SentenceTransformer
    from transformers import get_linear_schedule_with_warmup
    torch.manual_seed(SEED)
    df = load_pairs()["train"].sample(steps * batch_size, random_state=SEED, replace=True)
    A, B = [clean_headline(h) for h in df.headline_a], [clean_headline(h) for h in df.headline_b]
    y = torch.tensor(df.label.values, dtype=torch.float)
    to_dev = lambda f: {k: v.to(device) for k, v in f.items()}
    if kind == "bi":
        # Base model from the HF cache (the fine-tuned copy was saved by a newer sentence-transformers); same architecture
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", cache_folder=str(MODELS), device=device)
        model.max_seq_length = MAX_LEN
        next(m for n, m in model.named_modules() if n.endswith("word_embeddings")).requires_grad_(False)
        step_loss = lambda i: online_contrastive(model, to_dev(model.tokenize(A[i])), to_dev(model.tokenize(B[i])), y[i].to(device))
    else:
        model, tok = Head("cross").net.to(device).train(), tokenizer("cross")
        bce = torch.nn.BCEWithLogitsLoss()
        step_loss = lambda i: bce(model(**to_dev(tok(A[i], B[i], padding=True, truncation=True, max_length=MAX_LEN,
                                                      return_tensors="pt"))).logits[:, 0], y[i].to(device))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, 0, steps)
    batches = [slice(i * batch_size, (i + 1) * batch_size) for i in range(steps)]

    def step(i):
        loss = step_loss(i)
        sample_gpu(device)   # activations are still alive before backward, so this is close to the peak
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if device == "mps":
            torch.mps.synchronize()

    with FlopCounterMode(display=False) as counter:   # forward + backward FLOPs of the first 10 batches
        for i in batches[:10]:
            step(i)
    times = []
    GPU_MB.clear()
    for i in batches[10:]:
        times.append(timed(step, i))
        sample_gpu(device)
    return {"model": kind, "device": device, "steps timed": len(times), "batch size": batch_size,
            "steps per second": round(1 / np.mean(times), 2),
            "train FLOPs per pair (GFLOPs)": round(counter.get_total_flops() / (10 * batch_size) / 1e9, 3),
            "total parameters": sum(p.numel() for p in model.parameters()),
            "trainable parameters": sum(p.numel() for p in params),
            "peak RSS (MB)": round(peak_rss_mb()), "peak GPU tensor memory, MPS (MB)": round(max(GPU_MB)) if GPU_MB else None,
            "hardware": hardware(device)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["infer", "train", "flops", "export"])
    p.add_argument("--model", choices=["bi", "cross"], required=True)
    p.add_argument("--backend", default="torch", choices=["torch", "torch_int8", "onnx", "onnx_int8"])
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = p.parse_args()
    if args.task == "infer":
        result = infer(args.model, args.backend, args.device)
    elif args.task == "train":
        result = train(args.model, args.device)
    elif args.task == "flops":
        result = {"model": args.model, "forward FLOPs per pair (GFLOPs)": round(flops_per_pair(args.model) / 1e9, 3)}
    else:
        result = {"model": args.model, **export_onnx(args.model)}
    print("RESULT " + json.dumps(result))
