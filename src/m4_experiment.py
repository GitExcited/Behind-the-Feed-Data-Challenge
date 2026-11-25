"""Milestone 4, Section 1.4: simulated drift + continual update experiments.

Built on the continual learning module (src/continual.py): every run starts from the deployed model, v2 + input
sanitizer, and replays the test period (May-Oct 2026) one month at a time (test-then-train: each month is scored by
the model serving at that time before any update uses it).

  A  real stream      the real months, no added drift
  B  typo drift       simulated gradual drift: typo rate grows from 0 to 0.3 of letters across the six months
                      (same corruption as the Milestone 3 stress tests), applied to the pairs and to the front pages
  C  bad labels       one update on May-Jul labels where a share of the labels is wrong (simulated annotator errors),
                      checked on clean Aug-Oct data: does the accept/reject gate stop a bad update?
  D  latency / size   CPU page latency of v2 vs the updated models (same procedure as Milestones 2-3)

Update strategies compared (A and B): no update, fine-tune on the new month only (no replay), fine-tune with replay +
gate (our method, every month or only when a Milestone 3 trigger fires) and full retrain from the base model.
Runs that train are repeated with three seeds (6450-6452) and reported as mean +- std; the no-update baseline and
the full retrain (the expensive reference point) use one seed.

Metrics are the Milestone 2-3 ones: pair F1 and PR-AUC on each month, card ARI against Google's cards on that
month's front pages, F1 on the old validation set (forgetting check), update time, peak RAM / GPU memory, model size.

    python -m src.m4_experiment                 # everything (about 1.5 h on an M4 Max laptop)
    python -m src.m4_experiment --parts A C     # some parts only
    python -m src.m4_experiment --smoke         # quick check (2 months, small data, 1 epoch)
    python -m src.m4_experiment --report        # only rebuild the summary tables and figures from saved runs

Finished runs are cached in outputs/milestone4/experiment/runs/, so an interrupted job continues where it stopped.
Model versions go to models/cl/experiment/ (not in git); only the final models used for latency are kept.
"""
import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch

import src.adaptation as A
import src.continual as C
import src.robustness as R
from src.robustness import SEED, evaluate

OUT = R.ROOT / "outputs" / "milestone4" / "experiment"
RUNS = C.CL_MODELS / "experiment"
SEEDS = [SEED, SEED + 1, SEED + 2]
DRIFT, DRIFT_STOP = "typos", 0.3
FLIP_RATES = [0.1, 0.2, 0.3, 0.4]
KEEP = {"A_always_replay_6450", "B_always_replay_6450"}   # final models kept for the latency check (part D)

STRATEGIES = {   # name -> (policy, mode, seeds)
    "A": [("never", "replay", [SEED]), ("triggered", "replay", SEEDS), ("always", "replay", SEEDS)],
    "B": [("never", "replay", [SEED]), ("always", "finetune", SEEDS), ("always", "replay", SEEDS),
          ("triggered", "replay", SEEDS), ("always", "full", [SEED])],
}
LABEL = {"never/replay": "no update", "always/finetune": "fine-tune, no replay", "always/replay": "replay + gate (monthly)",
         "triggered/replay": "replay + gate (when triggered)", "always/full": "full retrain (monthly)"}
COLORS = {"no update": "#6b6b6b", "fine-tune, no replay": "#E69F00", "replay + gate (monthly)": "#0072B2",
          "replay + gate (when triggered)": "#56B4E9", "full retrain (monthly)": "#009E73"}   # Okabe-Ito, colour-blind safe


def log(m):
    print(m, flush=True)


# data
def pages_by_month():
    """Test-period front pages (Top stories / A la une), same filter as R.front_pages, keyed by month (YYYYMM)."""
    fp = R.ROOT / "data" / "google_news_front_page"
    snaps = pd.read_csv(fp / "snapshots.csv", dtype={"timestamp": str})
    keep = snaps[(snaps.purpose != "repeat") & (snaps.exclude.astype(str).str.lower() != "true")][["timestamp", "edition"]]
    arts = pd.read_csv(fp / "front_page_articles.csv", dtype={"timestamp": str}).merge(keep, on=["timestamp", "edition"])
    arts = arts[arts.section.isin(["Top stories", "À la une"]) & arts.headline.notna() & (arts.timestamp >= "20260501000000")]
    out = {}
    for (ts, ed), p in arts.groupby(["timestamp", "edition"]):
        out.setdefault(ts[:6], []).append(([R.clean_headline(h) for h in p.headline], p.card.values, ed))
    return out


def corrupt_pages(pages, level, seed):
    fn = R.CORRUPTIONS[DRIFT][0]
    rng = np.random.default_rng(seed)
    return [([fn(h, rng, level) for h in heads], cards, ed) for heads, cards, ed in pages] if level > 0 else pages


def setup(smoke):
    pairs = R.load_pairs()
    train, val, test = pairs["train"], pairs["val"], pairs["test"]
    batches, pages = C.monthly(test), pages_by_month()
    cfg = dict(C.CL_CFG, keep_weights=99)   # keep every version until the card ARI is computed, then clean up
    if smoke:
        cfg.update(epochs=1, min_new_pairs=50, buffer_size=400)
        batches = [(n, d.sample(n=min(300, len(d)), random_state=SEED).reset_index(drop=True)) for n, d in batches[:2]]
        pages = {m: p[:10] for m, p in pages.items()}
        val = val.sample(n=400, random_state=SEED).reset_index(drop=True)
        train = train.sample(n=2000, random_state=SEED).reset_index(drop=True)
    return dict(train=train, val=val, test=test, batches=batches, pages=pages, cfg=cfg, vocab=A.vocabulary(train))


def stream(part, D):
    """Month batches and front pages for part A (real) or B (typo drift growing month by month)."""
    if part == "A":
        return D["batches"], {m[:6]: D["pages"].get(m[:6], []) for m, _ in D["batches"]}
    levels = np.linspace(0, DRIFT_STOP, len(D["batches"]))
    batches = C.noisy_stream(D["batches"], DRIFT, 0.0, DRIFT_STOP, seed=SEED)   # same drifted data for every seed
    pages = {n[:6]: corrupt_pages(D["pages"].get(n[:6], []), float(p), SEED + 1000 + i)
             for i, ((n, _), p) in enumerate(zip(D["batches"], levels))}
    return batches, pages


# parts A and B: one stream run
def stream_run(part, policy, mode, seed, D):
    name = f"{part}_{policy}_{mode}_{seed}"
    out = OUT / "runs" / name
    if (out / "stream_log.csv").exists():
        log(f"{name}: cached")
        return
    log(f"=== {name}")
    out.mkdir(parents=True, exist_ok=True)
    batches, pages = stream(part, D)
    root = RUNS / name
    if root.exists():   # leftovers of an interrupted run
        shutil.rmtree(root)
    reg = C.Registry(root)
    reg.init(R.V2_PATH, "v2", D["val"], sanitize=True, note="starting model (Milestone 3 deployed)")
    buffer = C.ReplayBuffer(D["cfg"]["buffer_size"], seed=seed).add(D["train"])
    val = D["val"]
    t0 = time.perf_counter()
    df = C.run_stream(batches, reg, buffer, val, D["vocab"], policy, mode,
                      full_data=D["train"] if mode == "full" else None,
                      eval_sets={"old_val": val, "french_val": val[val.edition == "fr-CA"]},
                      cfg=D["cfg"], log=log, seed=seed)
    minutes = (time.perf_counter() - t0) / 60
    # card ARI of the model that served each month (it had not seen that month yet)
    cache = {}
    for i, row in df.iterrows():
        if row.step == 0:
            continue
        v = row.serving_before
        if v not in cache:
            cache.clear()
            cache[v] = reg.pipeline(v)
        ari, exact = A.grouping(cache[v], pages[str(row.batch)[:6]])
        df.loc[i, "card_ari"], df.loc[i, "exact_cards"] = ari, exact
    cache.clear()
    df.insert(0, "run", name)
    df.insert(1, "part", part)
    df.insert(2, "strategy", LABEL[f"{policy}/{mode}"])
    df.insert(3, "seed", seed)
    df["month"] = df.batch.astype(str).str[:6]
    df.to_csv(out / "stream_log.csv", index=False)
    shutil.copy(reg.file, out / "registry.json")
    json.dump({"device": R.DEVICE, "torch": torch.__version__, "minutes_total": minutes, "cfg": D["cfg"],
               "train_cfg": R.CFG, "start_model": "v2 + sanitizer (models/minilm_robust)"},
              open(out / "run_info.json", "w"), indent=1, default=str)
    if name not in KEEP:   # free the disk (about 0.5 GB per version); the registry record stays
        for v in reg.versions:
            if v["kind"] != "initial":
                A.drop_model(C.absolute(v["weights"]))
    log(f"{name}: done in {minutes:.1f} min")


# part C: bad labels
def flip(df, rate, kind, seed):
    """Simulated annotation errors. random: any label flipped with probability `rate`. missed: an annotator
    misses the link, so same-story pairs are labelled 'different' with probability `rate`."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    hit = rng.random(len(df)) < rate
    if kind == "missed":
        hit &= df.label.values == 1
    out.loc[hit, "label"] = 1 - out.loc[hit, "label"]
    return out, int(hit.sum())


def bad_labels(D):
    path_csv = OUT / "bad_labels.csv"
    done = pd.read_csv(path_csv) if path_csv.exists() else pd.DataFrame(columns=["run"])
    m = A.month(D["test"])
    months = sorted(m.unique())
    new, future = D["test"][m.isin(months[:3])].reset_index(drop=True), D["test"][m.isin(months[3:])].reset_index(drop=True)
    fut_pages = [p for mm in months[3:] for p in D["pages"].get(mm, [])]
    cur = C.fit_pipeline(R.V2_PATH, "v2", D["val"], sanitize=True)
    fs = cur.raw(future.headline_a, future.headline_b)
    base = {"cur_future_f1": evaluate(future.label.values, fs, cur.threshold)["f1"],
            "cur_future_ari": A.grouping(cur, fut_pages)[0]}
    rows = done.to_dict("records")
    jobs = [("none", 0.0)] + [(k, r) for k in ["random", "missed"] for r in FLIP_RATES]
    for seed in SEEDS:
        train_clean, hold_clean = C.split_batch(new, D["cfg"]["holdout_frac"], seed)
        for kind, rate in jobs:
            run = f"{kind}-{rate}-{seed}"
            if run in set(done.run):
                continue
            log(f"=== bad labels {run}")
            noisy, n_flipped = flip(new, rate, kind, seed) if rate else (new, 0)
            train_new, hold = C.split_batch(noisy, D["cfg"]["holdout_frac"], seed)   # same snapshots as the clean split
            buffer = C.ReplayBuffer(D["cfg"]["buffer_size"], seed=seed).add(D["train"])
            train = pd.concat([train_new, buffer.sample(len(train_new), seed)]).sample(frac=1, random_state=seed)
            path = RUNS / "bad_labels" / run
            with C.PeakMemory() as mem:
                info = A.finetune(R.V2_PATH, train.reset_index(drop=True), D["val"], path, seed=seed,
                                  epochs=D["cfg"]["epochs"], aug_prob=D["cfg"]["aug_prob"], log=log)
            cand = C.fit_pipeline(path, run, D["val"], sanitize=True)
            gate = C.gate_check(cand, cur, hold, D["val"], D["cfg"])                # what the system sees (noisy labels)
            gate_clean = C.gate_check(cand, cur, hold_clean, D["val"], D["cfg"])    # same gate with a trusted, clean holdout
            cs = cand.raw(future.headline_a, future.headline_b)
            mf = evaluate(future.label.values, cs, cand.threshold)
            rows.append({"run": run, "kind": kind, "flip_rate": rate, "seed": seed, "flipped": n_flipped,
                         "train_pairs": len(train), **{f"gate_{k}": v for k, v in gate.items()},
                         "accept_clean_holdout": gate_clean["accept"],
                         "cand_clean_holdout_f1": gate_clean["new_f1_candidate"],
                         "cur_clean_holdout_f1": gate_clean["new_f1_current"],
                         "cand_future_f1": mf["f1"], "cand_future_pr_auc": mf["pr_auc"],
                         "cand_future_ari": A.grouping(cand, fut_pages)[0], **base,
                         "threshold": cand.threshold, "update_minutes": info["minutes"],
                         "peak_ram_mb": mem.ram_mb, "peak_vram_mb": mem.vram_mb})
            pd.DataFrame(rows).to_csv(path_csv, index=False)
            del cand
            A.drop_model(path)
            log(f"  accept={gate['accept']} (clean holdout: {gate_clean['accept']}), future F1 {mf['f1']:.3f} "
                f"vs v2 {base['cur_future_f1']:.3f}")


# part D: latency and size
def latency(D):
    finals = {}
    for name in sorted(KEEP):
        f = OUT / "runs" / name / "registry.json"
        if f.exists():
            rec = [v for v in json.load(open(f))["versions"] if v["status"] == "active"][-1]
            finals[name] = rec
    if not finals:
        log("no kept models yet, skipping latency")
        return
    dev = R.DEVICE
    R.DEVICE = "cpu"   # latency on CPU, as in Milestones 2-3
    pipes = {"v2 (start)": C.Registry(RUNS / sorted(finals)[0]).pipeline("v2")}
    for name, rec in finals.items():
        pipes[f"{rec['version']} ({name})"] = C.Registry(RUNS / name).pipeline(rec["version"])
    pages = [h for p in D["pages"].values() for h, _, _ in p]
    lat = A.interleaved_latency(pipes, pages, rounds=10, per_round=20)
    rows = []
    for k, v in lat.items():
        w = R.V2_PATH if k.startswith("v2 (") else C.absolute(finals[k.split("(")[1].rstrip(")")]["weights"])
        rows.append({"model": k, **v, "model_mb": A.dir_mb(w)})
    pd.DataFrame(rows).to_csv(OUT / "latency_size.csv", index=False)
    R.DEVICE = dev
    log(pd.DataFrame(rows).to_string(index=False))


# summary tables and figures
def load_runs():
    files = sorted((OUT / "runs").glob("*/stream_log.csv"))
    return pd.concat([pd.read_csv(f, dtype={"month": str, "batch": str}) for f in files], ignore_index=True) if files else None


def summarize():
    logs = load_runs()
    if logs is None:
        return
    logs.to_csv(OUT / "stream_logs_all.csv", index=False)
    steps = logs[logs.step > 0]
    metrics = ["before_f1", "before_pr_auc", "card_ari", "eval_old_val_f1", "eval_french_val_f1"]
    per_month = steps.groupby(["part", "strategy", "step", "month"])[metrics + ["update_minutes"]].agg(["mean", "std"])
    per_month.columns = [f"{a}_{b}" for a, b in per_month.columns]
    per_month.reset_index().to_csv(OUT / "per_month.csv", index=False)
    # one row per strategy: averages over the six months, first computed per seed, then mean +- std over seeds
    per_seed = steps.groupby(["part", "strategy", "seed"]).agg(
        f1=("before_f1", "mean"), pr_auc=("before_pr_auc", "mean"), card_ari=("card_ari", "mean"),
        last_f1=("before_f1", "last"), old_val_f1_end=("eval_old_val_f1", "last"),
        updates_accepted=("decision", lambda d: int((d == "accepted").sum())),
        updates_rejected=("decision", lambda d: int((d == "rejected").sum())),
        recalibrations=("decision", lambda d: int((d == "recalibrated").sum())),
        update_min_total=("update_minutes", "sum"), update_min_mean=("update_minutes", "mean"),
        peak_ram_mb=("peak_ram_mb", "max"), peak_gpu_mb=("peak_vram_mb", "max")).reset_index()
    summ = per_seed.groupby(["part", "strategy"]).agg(
        **{f"{c}_{s}": (c, s) for c in ["f1", "pr_auc", "card_ari", "last_f1", "old_val_f1_end", "update_min_total"]
           for s in ["mean", "std"]},
        updates_accepted=("updates_accepted", "mean"), updates_rejected=("updates_rejected", "mean"),
        recalibrations=("recalibrations", "mean"), update_min_mean=("update_min_mean", "mean"),
        peak_ram_mb=("peak_ram_mb", "max"), peak_gpu_mb=("peak_gpu_mb", "max"), seeds=("seed", "nunique")).reset_index()
    order = list(LABEL.values())
    summ["o"] = summ.strategy.map(order.index)
    summ = summ.sort_values(["part", "o"]).drop(columns="o")
    summ.to_csv(OUT / "summary.csv", index=False)
    log(summ.to_string(index=False))
    for part in summ.part.unique():
        plot_part(steps[steps.part == part], logs[(logs.part == part) & (logs.step == 0)], part)
        plot_updates(steps[steps.part == part], part)


def plot_part(steps, starts, part):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e3e3e3", "grid.linewidth": 0.6, "font.size": 9})
    months = sorted(steps.month.unique())
    x = np.arange(len(months))
    ticks = [f"{pd.Timestamp(m + '01'):%b}" for m in months]
    if part == "B":
        ticks = [f"{t}\n{lvl:.0%} typos" for t, lvl in zip(ticks, np.linspace(0, DRIFT_STOP, len(months)))]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    panels = [("before_f1", "Pair F1 on the new month\n(serving model, before it updates)"),
              ("card_ari", "Card ARI vs Google's cards\n(same month's front pages)"),
              ("eval_old_val_f1", "F1 on old validation data\n(forgetting check, after the month's update)")]
    strategies = [s for s in LABEL.values() if s in set(steps.strategy)]
    for a, (col, title) in zip(ax.flat, panels):
        for s in strategies:
            g = steps[steps.strategy == s].groupby("month")[col].agg(["mean", "std"]).reindex(months)
            base = s == "no update"   # dashed and on top: it can coincide with 'when triggered' if nothing fires
            a.plot(x, g["mean"], "o--" if base else "o-", color=COLORS[s], lw=2, ms=5, label=s, zorder=5 if base else 3)
            if steps[steps.strategy == s].seed.nunique() > 1:
                a.fill_between(x, g["mean"] - g["std"], g["mean"] + g["std"], color=COLORS[s], alpha=0.15, lw=0)
        if col == "eval_old_val_f1" and len(starts):
            a.axhline(starts.eval_old_val_f1.iloc[0], color="#999", ls="--", lw=1)
            a.annotate("v2 at start", (x[-1], starts.eval_old_val_f1.iloc[0]), textcoords="offset points",
                       xytext=(0, 4), ha="right", fontsize=8, color="#666")
        a.set_xticks(x, ticks)
        a.set_title(title, fontsize=10)
    a = ax[1, 1]
    trained = [s for s in strategies if steps[steps.strategy == s].update_minutes.notna().any()]
    w = 0.8 / max(len(trained), 1)
    for k, s in enumerate(trained):
        g = steps[steps.strategy == s].groupby("month").update_minutes.mean().reindex(months) * 60
        a.bar(x - 0.4 + w * (k + 0.5), g.where(g > 0), w * 0.9, color=COLORS[s])
    from matplotlib.ticker import FixedLocator, NullLocator
    a.set_yscale("log")
    top = steps.update_minutes.max() * 60
    a.set_ylim(1, max(10, top * 2) if top == top else 10)
    a.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 30, 60, 180, 600, 1800]))
    a.yaxis.set_minor_locator(NullLocator())
    a.yaxis.set_major_formatter(lambda v, _: f"{v:.0f} s" if v < 60 else f"{v / 60:.0f} min")
    a.set_ylabel("seconds (log scale)")
    a.set_xticks(x, ticks)
    a.set_title("Update time per month (training only, mean over seeds)", fontsize=10)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
    fig.suptitle({"A": "Real stream (May-Oct 2026), starting from v2 + sanitizer",
                  "B": f"Simulated drift: typo rate grows 0 to {DRIFT_STOP:.0%} of letters, starting from v2 + sanitizer"}[part]
                 + "\nshaded band = +-1 std over 3 seeds", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"{part}_trajectories.png", dpi=160)
    plt.close(fig)


def plot_updates(steps, part):
    """One figure per part about the updates themselves: F1 before vs after each update (our method), update time and
    peak memory per update for every strategy that trains."""
    import matplotlib.pyplot as plt
    u = steps[steps.decision.isin(["accepted", "rejected"])]
    if not len(u):
        return
    months = sorted(steps.month.unique())[:-1] if (steps.decision.astype(str).str.startswith("skipped")).any() else sorted(steps.month.unique())
    x = np.arange(len(months))
    ticks = [f"{pd.Timestamp(m + '01'):%b}" for m in months]
    trained = [s for s in LABEL.values() if s in set(u.strategy)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    # 1. before vs after, our method
    ours = "replay + gate (monthly)"
    r = u[u.strategy == ours].groupby("month").agg(
        b=("new_f1_current", "mean"), bs=("new_f1_current", "std"), a=("new_f1_candidate", "mean"),
        as_=("new_f1_candidate", "std"), acc=("decision", lambda d: f"{(d == 'accepted').sum()}/{len(d)}")).reindex(months)
    a = ax[0]
    a.errorbar(x - 0.08, r.b, yerr=r.bs, fmt="o", color="#6b6b6b", ms=7, capsize=3, label="before (serving model)")
    a.errorbar(x + 0.08, r.a, yerr=r.as_, fmt="o", color=COLORS[ours], ms=7, capsize=3, label="after (new candidate)")
    for i in range(len(months)):
        a.annotate("", (x[i] + 0.08, r.a.iloc[i]), (x[i] - 0.08, r.b.iloc[i]),
                   arrowprops=dict(arrowstyle="->", color="#999", lw=1))
    a.set_xticks(x, [f"{t}\naccepted {c}" for t, c in zip(ticks, r.acc)], fontsize=8)
    a.set_ylabel("F1 on the month's holdout")
    a.set_title(f"F1 before vs after each update\n({ours}, mean +- std over seeds)", fontsize=10)
    a.legend(frameon=False, fontsize=8)
    # 2. update time and 3. peak memory, every strategy that trains
    w = 0.8 / len(trained)
    for k, s in enumerate(trained):
        g = u[u.strategy == s].groupby("month").agg(t=("update_minutes", "mean"), ram=("peak_ram_mb", "max"),
                                                    gpu=("peak_vram_mb", "max")).reindex(months)
        off = x - 0.4 + w * (k + 0.5)
        ax[1].bar(off, g.t * 60, w * 0.9, color=COLORS[s], label=s)
        ax[2].bar(off, g.gpu / 1000, w * 0.9, color=COLORS[s], alpha=0.35)
        ax[2].bar(off, g.ram / 1000, w * 0.9, color=COLORS[s])
    from matplotlib.ticker import FixedLocator, NullLocator
    ax[1].set_yscale("log")
    ax[1].set_ylim(1, max(10, u.update_minutes.max() * 60 * 2))
    ax[1].yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 30, 60, 180, 600]))
    ax[1].yaxis.set_minor_locator(NullLocator())
    ax[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0f} s" if v < 60 else f"{v / 60:.0f} min")
    ax[1].set_title("Update time (training, mean over seeds)", fontsize=10)
    ax[2].set_ylabel("GB")
    ax[2].set_title("Peak memory per update (max over seeds)\nsolid = RAM, light = Apple GPU (MPS driver, incl. cache)",
                    fontsize=10)
    for a in ax[1:]:
        a.set_xticks(x, ticks)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(color="#e3e3e3", lw=0.6, axis="y")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
    fig.suptitle({"A": "Updates on the real stream", "B": "Updates under simulated typo drift"}[part], fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "figures" / f"{part}_updates.png", dpi=160)
    plt.close(fig)


def plot_bad_labels():
    f = OUT / "bad_labels.csv"
    if not f.exists():
        return
    import matplotlib.pyplot as plt
    d = pd.read_csv(f)
    base = d[d.kind == "none"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for a, kind, title in zip(ax, ["random", "missed"], ["Random label flips", "Missed same-story links (1 -> 0)"]):
        g = pd.concat([base.assign(kind=kind), d[d.kind == kind]]).groupby("flip_rate")
        m = g.agg(f1=("cand_future_f1", "mean"), sd=("cand_future_f1", "std"), acc=("gate_accept", "mean"),
                  acc_clean=("accept_clean_holdout", "mean"), cur=("cur_future_f1", "first")).reset_index()
        a.errorbar(m.flip_rate, m.f1, yerr=m.sd, fmt="o-", color="#0072B2", lw=2, capsize=3, label="updated model")
        a.axhline(m.cur.iloc[0], color="#6b6b6b", ls="--", lw=1.5, label="v2 (no update)")
        a.set_xticks(m.flip_rate, [f"{v:.0%} wrong\ngate accepts {r:.0%}" for v, r in zip(m.flip_rate, m.acc)],
                     fontsize=8)
        a.set_xlabel("share of labels wrong / share of the 3 seeds where the gate accepted the update")
        a.set_ylabel("F1 on clean Aug-Oct pairs")
        a.set_title(title, fontsize=10)
        a.spines[["top", "right"]].set_visible(False); a.grid(color="#e3e3e3", lw=0.6)
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("One replay update on May-Jul labels with simulated annotator errors (mean +- std, 3 seeds)", fontsize=11)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / "C_bad_labels.png", dpi=160)
    plt.close(fig)
    s = d.groupby(["kind", "flip_rate"]).agg(
        flipped=("flipped", "mean"), gate_accept=("gate_accept", "mean"),
        accept_with_clean_holdout=("accept_clean_holdout", "mean"),
        future_f1_mean=("cand_future_f1", "mean"), future_f1_std=("cand_future_f1", "std"),
        future_ari_mean=("cand_future_ari", "mean"), v2_future_f1=("cur_future_f1", "first"),
        v2_future_ari=("cur_future_ari", "first"), threshold=("threshold", "mean")).reset_index()
    s.to_csv(OUT / "bad_labels_summary.csv", index=False)
    log(s.to_string(index=False))


def main():
    global OUT, RUNS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report", action="store_true", help="only rebuild tables and figures")
    a = ap.parse_args()
    if a.smoke:
        OUT, RUNS = OUT.parent / "experiment_smoke", RUNS.parent / "experiment_smoke"
        STRATEGIES["A"] = [("always", "replay", [SEED])]
        STRATEGIES["B"] = [("never", "replay", [SEED]), ("always", "replay", [SEED])]
        FLIP_RATES[:] = [0.3]
        SEEDS[:] = [SEED]
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.report:
        D = setup(a.smoke)
        log(f"device {R.DEVICE}; start model {C.rel(R.V2_PATH)} (v2 + sanitizer); {len(D['batches'])} months")
        t0 = time.perf_counter()
        for part in [p for p in ["A", "B"] if p in a.parts]:
            for policy, mode, seeds in STRATEGIES[part]:
                for seed in seeds:
                    stream_run(part, policy, mode, seed, D)
        if "C" in a.parts:
            bad_labels(D)
        if "D" in a.parts:
            latency(D)
        log(f"total {(time.perf_counter() - t0) / 60:.1f} min")
    summarize()
    plot_bad_labels()


if __name__ == "__main__":
    main()
