"""Milestone 4, Part 1: continual learning module.

Chosen strategy: drift-triggered replay fine-tuning with an accept/reject gate.

  ReplayBuffer    stratified reservoir of old labelled pairs (edition x label), mixed 1:1 with new pairs
  Registry        versioned model store (models/cl/registry.json): weights folder, parent version, threshold,
                  Platt calibration, data hashes, gate result, status (active / archived / rejected / rolled_back)
  check_triggers  T1 data drift, T2 performance drop, T3 class-balance change (Milestone 3, Section 4.1)
  update()        one cycle: fine-tune the active model on new pairs + replay -> gate -> register / promote
  recalibrate()   T3 path: label-free threshold and calibration update, no training (Milestone 3, Section 4.3)
  run_stream()    test-then-train over a sequence of batches (months); logs before/after metrics and efficiency
  plot_stream()   metric trajectories across the stream

Everything reuses src/robustness.py and src/adaptation.py (same metrics, threshold rule, calibration, training
recipe), so numbers are comparable with Milestones 2 and 3.

Command line (outputs go to outputs/milestone4/continual/):
    python -m src.continual --policy triggered          # real May-Oct 2026 stream, update only when a trigger fires
    python -m src.continual --policy always             # update every month
    python -m src.continual --drift typos --policy always  # simulated drift: typo rate grows month by month
    python -m src.continual --smoke                     # tiny run to check the code works (CPU, a few minutes)
The starting model is the deployed Milestone 3 model: v2 (noise-augmented) + input sanitizer, read from
models/minilm_robust (M4_START env var or --start-model). Use --start-model <v1 folder> --start-name v1
--no-sanitizer to start from Milestone 2's v1 instead.
"""
import argparse
import hashlib
import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.linear_model import LogisticRegression

import src.adaptation as A
import src.robustness as R
from src.robustness import SEED, evaluate

CL_MODELS = Path(os.environ.get("CL_MODELS", R.MODELS / "cl"))
START_MODEL = Path(os.environ.get("M4_START", R.V2_PATH))   # deployed model at the end of Milestone 3
OUT = R.ROOT / "outputs" / "milestone4" / "continual"

CL_CFG = dict(
    # update
    replay_ratio=1.0,      # old pairs per new pair; 1:1 matched a full retrain in half the time in Milestone 3
    buffer_size=4000,      # replay buffer, split evenly over (edition, label) -> 1,000 pairs per stratum
    holdout_frac=0.2,      # share of each batch's front-page snapshots kept out of training, used by the gate
    min_new_pairs=200,     # below this many new training pairs we do not fine-tune (too little signal)
    epochs=2,              # Milestone 3 incremental-update setting; batch size, lr etc. come from R.CFG (Milestone 2)
    lr=None,               # None = Milestone 2 learning rate (2e-5)
    aug_prob=0.5,          # noise augmentation as in v2 (Milestone 3, Section 2)
    # gate
    gate_tol_new=0.0,      # candidate F1 on the new holdout must be >= current F1 - this
    gate_tol_old=0.01,     # candidate F1 on old reference data must be >= current F1 - this (forgetting guard)
    # triggers (Milestone 3, Section 4.1)
    psi_limit=0.2, psi_features=2,       # T1: PSI > 0.2 on at least 2 of 4 features
    f1_drop=0.03, f1_windows=2,          # T2: F1 more than 0.03 below reference for 2 batches in a row
    prior_change=0.10,                   # T3: estimated same-story share moves by more than 0.10
    keep_weights=3,        # keep weights of the last N archived versions (plus the root); older ones are deleted
)

PAIR_COLS = ["timestamp", "edition", "headline_a", "headline_b", "outlet_a", "outlet_b", "label"]


# helpers
def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def data_hash(df):
    cols = [c for c in PAIR_COLS if c in df.columns]
    return hashlib.sha256(df[cols].to_csv(index=False).encode("utf-8")).hexdigest()[:16]


def rel(path):
    """Paths are stored relative to the repo root so the registry works on every machine."""
    for p in (Path(path).absolute(), Path(path).resolve()):   # unresolved first: models/ may hold symlinks
        try:
            return p.relative_to(R.ROOT).as_posix()
        except ValueError:
            pass
    return Path(path).as_posix()


def absolute(path):
    p = Path(path)
    return p if p.is_absolute() else R.ROOT / p


def dir_mb(path):
    return A.dir_mb(absolute(path))


class PeakMemory:
    """Peak process RAM (sampled every 0.2 s) and peak GPU memory while the block runs. On Apple silicon (MPS)
    there is no peak counter, so the memory the MPS driver holds is sampled the same way as RAM."""

    def __enter__(self):
        self.proc, self.peak, self.stop = psutil.Process(), 0, threading.Event()
        self.mps_peak = 0
        if R.DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()

        def watch():
            while not self.stop.is_set():
                self.peak = max(self.peak, self.proc.memory_info().rss)
                if R.DEVICE == "mps":
                    self.mps_peak = max(self.mps_peak, torch.mps.driver_allocated_memory())
                time.sleep(0.2)
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)
        self.ram_mb = self.peak / 1e6
        self.vram_mb = (torch.cuda.max_memory_allocated() / 1e6 if R.DEVICE == "cuda"
                        else self.mps_peak / 1e6 if R.DEVICE == "mps" else np.nan)
        return False


def split_batch(df, frac, seed):
    """Hold out whole front-page snapshots (not single pairs), so headlines from one page never sit on both sides."""
    snaps = np.array(sorted(df.timestamp.unique()))
    rng = np.random.default_rng(seed)
    k = max(1, int(round(frac * len(snaps)))) if len(snaps) > 1 else 0
    hold = set(rng.choice(snaps, k, replace=False)) if k else set()
    mask = df.timestamp.isin(hold)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


# replay buffer
class ReplayBuffer:
    """Reservoir sampling inside each (edition, label) stratum: every pair seen so far has the same chance of
    being kept, so the buffer covers all past months instead of only the latest one, and both languages and both
    classes stay represented even when a month has no French (June 2026) or few same-story pairs."""

    def __init__(self, size=CL_CFG["buffer_size"], strata=("edition", "label"), seed=SEED):
        self.size, self.strata = size, list(strata)
        self.rng = np.random.default_rng(seed)
        self.data = pd.DataFrame(columns=PAIR_COLS)
        self.seen = {}

    def add(self, df):
        old = {k: g for k, g in self.data.groupby(self.strata)} if len(self.data) else {}
        new = {k: g for k, g in df.groupby(self.strata)}
        keys = sorted(set(old) | set(new), key=str)
        cap = self.size // max(len(keys), 1)
        out = []
        for k in keys:
            rows = old[k].to_dict("records") if k in old else []
            seen = self.seen.get(k, len(rows))
            for r in (new[k].to_dict("records") if k in new else []):
                seen += 1
                if len(rows) < cap:
                    rows.append(r)
                else:
                    j = int(self.rng.integers(0, seen))
                    if j < cap:
                        rows[j] = r
            if len(rows) > cap:   # a new stratum appeared, so every stratum's share shrank
                rows = [rows[i] for i in sorted(self.rng.choice(len(rows), cap, replace=False))]
            self.seen[k] = seen
            out.extend(rows)
        self.data = pd.DataFrame(out, columns=self.data.columns if len(self.data.columns) else None)
        return self

    def sample(self, n, seed):
        if n <= 0 or not len(self.data):
            return self.data.iloc[:0]
        return self.data.sample(n=n, replace=n > len(self.data), random_state=seed).reset_index(drop=True)

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(folder / "buffer.csv", index=False)
        json.dump({"size": self.size, "strata": self.strata,
                   "seen": [[*map(lambda v: v.item() if hasattr(v, "item") else v, k), int(c)]
                            for k, c in self.seen.items()]}, open(folder / "buffer.json", "w"), indent=1)

    @classmethod
    def load(cls, folder, seed=SEED):
        folder = Path(folder)
        meta = json.load(open(folder / "buffer.json"))
        b = cls(meta["size"], meta["strata"], seed)
        b.data = pd.read_csv(folder / "buffer.csv", dtype={"timestamp": str})
        b.seen = {tuple(x[:-1]): x[-1] for x in meta["seen"]}
        return b

    def summary(self):
        return self.data.groupby(self.strata).size().to_dict()


# model registry (versioning)
class Registry:
    """One record per model version. Weights live in one folder per version (models/cl/<version>/); a
    recalibration-only version points at its parent's weights. Only one version is 'active' (the one serving)."""

    def __init__(self, root=CL_MODELS):
        self.root = Path(root)
        self.file = self.root / "registry.json"
        self.versions = json.load(open(self.file))["versions"] if self.file.exists() else []

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        json.dump({"versions": self.versions}, open(self.file, "w"), indent=1)

    def get(self, name):
        return next(v for v in self.versions if v["version"] == name)

    def active(self):
        act = [v for v in self.versions if v["status"] == "active"]
        return act[-1] if act else None

    def next_name(self, parent):
        root = parent.split(".")[0]
        n = sum(v["version"].startswith(root + ".") for v in self.versions)
        return f"{root}.{n + 1}"

    def add(self, rec):
        rec = {"created": now(), **rec}
        self.versions.append(rec)
        self.save()
        return rec

    def promote(self, name):
        for v in self.versions:
            if v["status"] == "active":
                v["status"] = "archived"
        self.get(name)["status"] = "active"
        self.save()

    def rollback(self):
        """Serving model goes back to the most recent archived version."""
        cur = self.active()
        prev = [v for v in self.versions if v["status"] == "archived" and Path(absolute(v["weights"])).exists()]
        if not cur or not prev:
            raise RuntimeError("nothing to roll back to")
        cur["status"] = "rolled_back"
        prev[-1]["status"] = "active"
        self.save()
        return prev[-1]["version"]

    def init(self, weights, name, ref_df, sanitize=True, note=""):
        """Register the starting model (the deployed v2 + sanitizer by default) with a threshold and calibration
        fitted on ref_df."""
        if not Path(weights).exists():
            raise FileNotFoundError(f"starting model not found: {weights} (copy it there or pass --start-model)")
        self.versions = []
        pipe = fit_pipeline(weights, name, ref_df, sanitize)
        self.add({"version": name, "parent": None, "kind": "initial", "weights": rel(weights), "status": "active",
                  "sanitize": sanitize, "threshold": pipe.threshold, "platt": platt_params(pipe.platt),
                  "ref_hash": data_hash(ref_df), "ref_pairs": len(ref_df), "note": note})
        return pipe

    def pipeline(self, name=None):
        rec = self.get(name) if name else self.active()
        pipe = R.Pipeline(R.BiEncoder(absolute(rec["weights"]), rec["version"]),
                          sanitize_inputs=rec.get("sanitize", False), name=rec["version"])
        pipe.threshold = rec["threshold"]
        pipe.platt = make_platt(*rec["platt"])
        return pipe

    def prune(self, keep=CL_CFG["keep_weights"]):
        """Delete weights of old archived and rejected versions (records stay, so the history is kept)."""
        in_use = {v["weights"] for v in self.versions if v["status"] == "active" or v["kind"] == "initial"}
        archived = [v for v in self.versions if v["status"] == "archived" and v["kind"] != "recalibration"]
        old = archived[:-keep] if keep else archived
        for v in old + [v for v in self.versions if v["status"] == "rejected"]:
            if v["weights"] not in in_use and absolute(v["weights"]).exists() and \
                    absolute(v["weights"]).is_relative_to(self.root):
                A.drop_model(absolute(v["weights"]))
                v["weights_deleted"] = True
        self.save()


def fit_pipeline(weights, name, ref_df, sanitize):
    """Threshold = best F1 on ref_df, as in Milestone 2. With the sanitizer on, the threshold is the one of the
    model without it (Milestone 3: the sanitizer is an input filter and should not move the decision rule);
    Platt calibration is then fitted on the sanitized scores."""
    pipe = R.Pipeline(R.BiEncoder(weights, name)).fit(ref_df)
    return R.Pipeline(pipe.scorer, sanitize_inputs=True, name=name).fit(ref_df, threshold=pipe.threshold) if sanitize else pipe


def platt_params(platt):
    return [float(platt.coef_[0, 0]), float(platt.intercept_[0])]


def make_platt(coef, intercept):
    p = LogisticRegression()
    p.coef_, p.intercept_, p.classes_ = np.array([[coef]]), np.array([intercept]), np.array([0, 1])
    return p


# triggers
def reference(pipe, ref_df, vocab):
    st = A.reference_stats(pipe, ref_df, vocab)
    st["prior"] = float(ref_df.label.mean())
    return st


def check_triggers(row, history, ref, cfg=CL_CFG):
    """row = A.drift_row(...) for this batch, history = earlier rows since the last model change."""
    psis = {k: row[k] for k in ["psi_score", "psi_words", "psi_oov", "psi_edition"]}
    t1 = sum(v > cfg["psi_limit"] for v in psis.values()) >= cfg["psi_features"]
    recent = [h["f1"] for h in history[-(cfg["f1_windows"] - 1):]] + [row["f1"]] if cfg["f1_windows"] > 1 else [row["f1"]]
    t2 = len(recent) >= cfg["f1_windows"] and all(f < ref["f1"] - cfg["f1_drop"] for f in recent)
    t3 = abs(row["est_same_rate"] - ref["prior"]) > cfg["prior_change"]
    return {"T1_data_drift": bool(t1), "T2_perf_drop": bool(t2), "T3_balance_change": bool(t3)}


# gate
def gate_check(cand, cur, hold, ref_df, cfg=CL_CFG):
    """Accept the candidate only if (a) it is at least as good as the serving model on the new holdout (both with
    their own thresholds, neither model trained on it) and (b) it does not forget: F1 on the old reference data
    (both thresholds refitted there, so neither gets an in-sample advantage) drops by at most gate_tol_old."""
    def f1(p, df, refit=False):
        s = p.raw(df.headline_a, df.headline_b)
        thr = A.best_threshold(df.label.values, s) if refit else p.threshold
        return float(evaluate(df.label.values, s, thr)["f1"])
    r = {"new_f1_current": f1(cur, hold), "new_f1_candidate": f1(cand, hold),
         "old_f1_current": f1(cur, ref_df, True), "old_f1_candidate": f1(cand, ref_df, True)}
    r["accept"] = bool(r["new_f1_candidate"] >= r["new_f1_current"] - cfg["gate_tol_new"]
                       and r["old_f1_candidate"] >= r["old_f1_current"] - cfg["gate_tol_old"])
    return r


# one continual-learning cycle
def update(reg, buffer, batch, ref_df, cfg=CL_CFG, mode="replay", full_data=None, seed=SEED, log=print):
    """Fine-tune the active model on the new batch (+ replay), gate it, register it, promote it if accepted.

    mode: 'replay'   (chosen) new pairs + replay_ratio x as many buffer pairs, start from the active model
          'finetune' new pairs only (no replay), start from the active model
          'full'     retrain from the base model on full_data + new pairs (the expensive reference point)
    The buffer is updated with the new batch after training, whatever the gate decides (the labels are still good
    data). Returns a dict with the gate result and efficiency numbers."""
    cur_rec = reg.active()
    cur = reg.pipeline()
    train_new, hold = split_batch(batch, cfg["holdout_frac"], seed)
    out = {"mode": mode, "parent": cur_rec["version"], "new_train_pairs": len(train_new), "holdout_pairs": len(hold)}
    if len(train_new) < cfg["min_new_pairs"] or hold.label.nunique() < 2:
        buffer.add(batch)
        return {**out, "decision": "skipped (too few new pairs)"}
    if mode == "replay":
        train, start = pd.concat([train_new, buffer.sample(int(round(cfg["replay_ratio"] * len(train_new))), seed)]), \
            absolute(cur_rec["weights"])
    elif mode == "finetune":
        train, start = train_new, absolute(cur_rec["weights"])
    elif mode == "full":
        train, start = pd.concat([full_data, train_new]), R.BASE_MODEL
    else:
        raise ValueError(mode)
    train = train.sample(frac=1, random_state=seed).reset_index(drop=True)
    name = reg.next_name(cur_rec["version"])
    path = reg.root / name
    log(f"  training {name} ({mode}) from {cur_rec['version'] if mode != 'full' else 'base model'} on {len(train)} pairs")
    with PeakMemory() as mem:
        info = A.finetune(start, train, ref_df, path, seed=seed, epochs=cfg["epochs"], aug_prob=cfg["aug_prob"],
                          lr=cfg["lr"], log=log)
    cand = fit_pipeline(path, name, ref_df, cur_rec.get("sanitize", False))
    gate = gate_check(cand, cur, hold, ref_df, cfg)
    rec = reg.add({"version": name, "parent": cur_rec["version"], "kind": mode, "weights": rel(path),
                   "status": "candidate", "sanitize": cur_rec.get("sanitize", False),
                   "threshold": cand.threshold, "platt": platt_params(cand.platt),
                   "train_hash": data_hash(train), "train_pairs": len(train), "new_pairs": len(train_new),
                   "replay_pairs": len(train) - len(train_new) if mode == "replay" else 0,
                   "ref_hash": data_hash(ref_df), "holdout_hash": data_hash(hold), "gate": gate,
                   "config": {k: cfg[k] for k in ["replay_ratio", "buffer_size", "epochs", "lr", "aug_prob",
                                                  "holdout_frac"]} | {"seed": seed, "train_cfg": R.CFG},
                   "update_minutes": info["minutes"], "best_epoch": info["best_epoch"],
                   "device": R.DEVICE})
    if gate["accept"]:
        reg.promote(name)
    else:
        rec["status"] = "rejected"
        reg.save()
    reg.prune(cfg["keep_weights"])
    buffer.add(batch)
    del cand
    log(f"  gate: new F1 {gate['new_f1_current']:.3f} -> {gate['new_f1_candidate']:.3f}, "
        f"old F1 {gate['old_f1_current']:.3f} -> {gate['old_f1_candidate']:.3f} -> "
        f"{'ACCEPT' if gate['accept'] else 'REJECT'}")
    return {**out, "version": name, "decision": "accepted" if gate["accept"] else "rejected", **gate,
            "update_minutes": info["minutes"], "peak_ram_mb": mem.ram_mb, "peak_vram_mb": mem.vram_mb,
            "model_mb": dir_mb(path) if absolute(path).exists() else np.nan, "hold": hold}


def recalibrate(reg, batch, ref, log=print):
    """T3 path (class balance changed): no training. Estimate the new same-story share from the model's own
    predictions, shift the calibration by the change in prior log-odds and pick the threshold with the best
    expected F1. Uses no labels (Milestone 3, Section 4.3). Registered as a new version on the same weights."""
    cur_rec = reg.active()
    pipe = reg.pipeline()
    s = pipe.raw(batch.headline_a, batch.headline_b)
    pi_new = A.estimate_prior(float((s >= pipe.threshold).mean()), ref["tpr"], ref["fpr"])
    probs = A.prior_shift_probs(pipe.platt, s, ref["prior"], pi_new)
    thr = A.expected_f1_threshold(s, probs)
    pi_old = float(np.clip(ref["prior"], 1e-3, 1 - 1e-3))
    pi_c = float(np.clip(pi_new, 1e-3, 1 - 1e-3))
    coef, icpt = cur_rec["platt"]
    icpt = icpt + np.log(pi_c / (1 - pi_c)) - np.log(pi_old / (1 - pi_old))
    name = reg.next_name(cur_rec["version"])
    reg.add({"version": name, "parent": cur_rec["version"], "kind": "recalibration", "weights": cur_rec["weights"],
             "status": "candidate", "sanitize": cur_rec.get("sanitize", False),
             "threshold": float(thr), "platt": [coef, float(icpt)],
             "estimated_prior": pi_new, "batch_hash": data_hash(batch)})
    reg.promote(name)
    log(f"  recalibrated -> {name}: estimated same-story share {pi_new:.2f}, threshold {pipe.threshold:.3f} -> {thr:.3f}")
    return {"version": name, "decision": "recalibrated", "estimated_prior": pi_new, "threshold": thr}


# test-then-train stream
def run_stream(batches, reg, buffer, ref_df, vocab, policy="triggered", mode="replay", full_data=None,
               eval_sets=None, cfg=CL_CFG, log=print, seed=SEED):
    """For each (name, df) batch, in time order:
      1. before: score the batch with the serving model (it has never seen it) -> drift stats + metrics + latency
      2. triggers T1-T3 (policy 'triggered'), or always / never update
      3. update (replay fine-tune + gate) or recalibrate
      4. after: metrics of the serving model on the batch holdout and on fixed eval_sets (forgetting check)
    seed: base training seed (update at step k uses seed + k), so runs can be repeated with different seeds.
    Returns one row per batch. ref_df grows with every holdout, so thresholds are refitted on recent data too."""
    eval_sets = eval_sets or {}
    pipe = reg.pipeline()
    ref = reference(pipe, ref_df, vocab)
    history, rows = [], []
    start = {"step": 0, "batch": "start", "serving_after": reg.active()["version"]}
    for ename, edf in eval_sets.items():   # baseline for the forgetting check
        s = pipe.raw(edf.headline_a, edf.headline_b)
        start[f"eval_{ename}_f1"] = float(evaluate(edf.label.values, s, pipe.threshold)["f1"])
    rows.append(start)
    for step, (name, df) in enumerate(batches, 1):
        log(f"[{step}/{len(batches)}] {name}: {len(df)} pairs, serving {reg.active()['version']}")
        t0 = time.perf_counter()
        drow = A.drift_row(pipe, df, ref, vocab, name)
        ms_per_pair = (time.perf_counter() - t0) * 1000 / max(len(df), 1)
        trig = check_triggers(drow, history, ref, cfg)
        row = {"step": step, "batch": name, "pairs": len(df), "serving_before": reg.active()["version"],
               **{f"before_{k}": drow[k] for k in ["precision", "recall", "f1", "accuracy", "pr_auc"] if k in drow},
               **{k: drow[k] for k in ["french_share", "psi_edition", "psi_words", "psi_oov", "psi_score",
                                       "est_same_rate", "true_same_rate"]},
               **trig, "score_ms_per_pair": ms_per_pair}
        do_update = policy == "always" or (policy == "triggered" and (trig["T1_data_drift"] or trig["T2_perf_drop"]))
        do_recal = policy == "triggered" and trig["T3_balance_change"] and not do_update
        if do_update:
            res = update(reg, buffer, df, ref_df, cfg, mode, full_data, seed=seed + step, log=log)
        elif do_recal:
            res = recalibrate(reg, df, ref, log)
            buffer.add(df)
        else:
            res = {"decision": "no update"}
            buffer.add(df)
        hold = res.pop("hold", None)
        row.update({k: v for k, v in res.items() if not isinstance(v, (pd.DataFrame, dict))})
        changed = res.get("decision") in ("accepted", "recalibrated")
        if changed:
            del pipe
            pipe = reg.pipeline()
        if hold is not None and len(hold):
            s = pipe.raw(hold.headline_a, hold.headline_b)
            row["after_holdout_f1"] = float(evaluate(hold.label.values, s, pipe.threshold)["f1"])
            row["before_holdout_f1"] = res.get("new_f1_current")
            ref_df = pd.concat([ref_df, hold], ignore_index=True)
        for ename, edf in eval_sets.items():
            s = pipe.raw(edf.headline_a, edf.headline_b)
            row[f"eval_{ename}_f1"] = float(evaluate(edf.label.values, s, pipe.threshold)["f1"])
        row["serving_after"] = reg.active()["version"]
        if changed:   # new model -> new score distribution and threshold, so rebuild the drift reference
            ref = reference(pipe, ref_df, vocab)
            history = []
        else:
            history.append(drow)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_stream(log_df, path, title="Continual learning stream"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(log_df))
    ax[0].plot(x, log_df.before_f1, "o-", label="F1 on the new batch (before update)")
    if "after_holdout_f1" in log_df:
        ax[0].plot(x, log_df.after_holdout_f1, "s--", label="F1 on batch holdout (after update)")
    for c in [c for c in log_df.columns if c.startswith("eval_")]:
        ax[0].plot(x, log_df[c], "^:", label=c[5:-3] + " F1 (forgetting check)")
    for i, d in enumerate(log_df.get("decision", [])):
        if d in ("accepted", "rejected", "recalibrated"):
            ax[0].axvline(i, color={"accepted": "tab:green", "rejected": "tab:red", "recalibrated": "tab:orange"}[d],
                          alpha=0.25, lw=6)
    ax[0].set_xticks(x, log_df.batch, rotation=30)
    ax[0].set_ylabel("F1"); ax[0].set_title("Accuracy over time (green = accepted update, red = rejected)")
    ax[0].legend(fontsize=8)
    if "update_minutes" in log_df:
        ax[1].bar(x, log_df.update_minutes.fillna(0))
        ax[1].set_ylabel("update time (min)")
    ax[1].set_xticks(x, log_df.batch, rotation=30); ax[1].set_title("Update cost per batch")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# command line
def monthly(df):
    return [(m, g.reset_index(drop=True)) for m, g in df.groupby(A.month(df))]


def noisy_stream(batches, kind="typos", start=0.0, stop=0.3, seed=SEED):
    """Simulated gradual drift: the corruption strength grows linearly from `start` to `stop` across the batches
    (same corruptions as the Milestone 3 stress tests)."""
    fn = R.CORRUPTIONS[kind][0]
    levels = np.linspace(start, stop, len(batches))
    return [(f"{n} ({kind} {p:.2f})", R.corrupt_df(df, fn, float(p), seed + i) if p > 0 else df)
            for i, ((n, df), p) in enumerate(zip(batches, levels))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="triggered", choices=["triggered", "always", "never"])
    ap.add_argument("--mode", default="replay", choices=["replay", "finetune", "full"])
    ap.add_argument("--drift", default=None, help="simulate drift with a Milestone 3 corruption, e.g. typos")
    ap.add_argument("--start-model", default=str(START_MODEL), help="default: v2 (models/minilm_robust)")
    ap.add_argument("--start-name", default="v2", help="version name of the starting model (v2 -> v2.1, v2.2, ...)")
    ap.add_argument("--no-sanitizer", action="store_true", help="score without the Milestone 3 input sanitizer")
    ap.add_argument("--tag", default=None, help="name of the output subfolder")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 2 months, 300 pairs each, 1 epoch")
    a = ap.parse_args()

    cfg = dict(CL_CFG)
    pairs = R.load_pairs()
    train, val, test = pairs["train"], pairs["val"], pairs["test"]
    batches = monthly(test)
    if a.smoke:
        cfg.update(epochs=1, min_new_pairs=50, buffer_size=400)
        batches = [(n, d.sample(n=min(300, len(d)), random_state=SEED).reset_index(drop=True)) for n, d in batches[:2]]
        val = val.sample(n=400, random_state=SEED).reset_index(drop=True)
        train = train.sample(n=2000, random_state=SEED).reset_index(drop=True)
    if a.drift:
        batches = noisy_stream(batches, a.drift)
    tag = a.tag or f"{a.policy}_{a.mode}" + (f"_{a.drift}" if a.drift else "") + ("_smoke" if a.smoke else "")
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)

    reg = Registry(CL_MODELS / tag)
    log = lambda m: print(m, flush=True)
    log(f"device {R.DEVICE}; start model {a.start_model} ({a.start_name}, sanitizer {'off' if a.no_sanitizer else 'on'}); "
        f"{len(batches)} batches")
    reg.init(a.start_model, a.start_name, val, sanitize=not a.no_sanitizer, note="starting model")
    buffer = ReplayBuffer(cfg["buffer_size"]).add(train)
    vocab = A.vocabulary(train)
    eval_sets = {"old_val": val, "french_val": val[val.edition == "fr-CA"]}
    t0 = time.perf_counter()
    log_df = run_stream(batches, reg, buffer, val, vocab, a.policy, a.mode,
                        full_data=train if a.mode == "full" else None, eval_sets=eval_sets, cfg=cfg, log=log)
    log_df.to_csv(out / "stream_log.csv", index=False)
    shutil.copy(reg.file, out / "registry.json")
    buffer.save(out)
    plot_stream(log_df, out / "stream.png", title=f"policy={a.policy}, mode={a.mode}" + (f", drift={a.drift}" if a.drift else ""))
    json.dump({"device": R.DEVICE, "torch": torch.__version__, "cfg": cfg, "train_cfg": R.CFG,
               "minutes_total": (time.perf_counter() - t0) / 60, "start_model": rel(a.start_model),
               "start_name": a.start_name, "sanitizer": not a.no_sanitizer},
              open(out / "run_info.json", "w"), indent=1, default=str)
    log(f"saved to {rel(out)}")
    cols = ["batch", "before_f1", "decision", "after_holdout_f1", "eval_old_val_f1", "update_minutes"]
    log(log_df[[c for c in cols if c in log_df]].to_string(index=False))


if __name__ == "__main__":
    main()
