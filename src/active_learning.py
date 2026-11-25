"""Active learning query strategy for the HITL loop (INSE 6450, Milestone 4, Section 2.3).

Selects the weekly batch of headline pairs sent to human annotators.

A pair enters the review pool when either:
  1. it is uncertain: its calibrated P(same) is close to the decision threshold, or
  2. it confidently disagrees with Google's card: |P(same) - y_google| >= 0.8.

Pairs are ranked by an informativeness score, then picked greedily under
diversity caps (max pairs per card, max pairs per outlet) and an EN/FR quota,
so one big story or one outlet cannot fill the whole batch.

The module only needs a DataFrame of already-scored pairs, so it plugs into the
existing pipeline: score with the active model + Platt calibration, call
select_batch(), send the batch to annotators, then pass the labels to
continual.update() as new labelled pairs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class QueryConfig:
    batch_size: int = 100            # pairs sent to annotators each week
    threshold_prob: float = 0.5      # decision threshold expressed as a calibrated probability
    band: float = 0.15               # |p - threshold| below this counts as uncertain
    disagree_min: float = 0.8        # |p - y_google| at or above this counts as disagreement
    disagree_share: float = 0.3      # target share of the batch reserved for disagreement pairs
    max_per_card: int = 2            # diversity cap per Google card
    max_per_outlet: int = 10         # diversity cap per outlet
    fr_share: float = 0.3            # minimum share of French pairs, if enough are available
    strategy: str = "uncertainty"    # "uncertainty" or "random" (the control in Section 2.4)
    seed: int = 6450


REQUIRED = ["pair_id", "p_same", "y_google", "card_a", "card_b", "outlet_a", "outlet_b", "lang"]


def score_pairs(pairs: pd.DataFrame, cfg: QueryConfig) -> pd.DataFrame:
    """Add uncertainty, disagreement and informativeness columns."""
    missing = [c for c in REQUIRED if c not in pairs.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = pairs.copy()
    p = df["p_same"].clip(0, 1)
    # 1 at the threshold, 0 at the edge of the band
    df["uncertainty"] = (1 - (p - cfg.threshold_prob).abs() / cfg.band).clip(lower=0)
    df["disagreement"] = (p - df["y_google"]).abs()
    df["is_uncertain"] = df["uncertainty"] > 0
    df["is_disagree"] = df["disagreement"] >= cfg.disagree_min
    df["reason"] = np.select(
        [df["is_disagree"], df["is_uncertain"]], ["disagree_google", "uncertain"], default="none"
    )
    # A disagreement pair can also be uncertain; take the stronger signal.
    df["info"] = np.maximum(df["uncertainty"], df["disagreement"].where(df["is_disagree"], 0))
    return df


def _greedy_pick(cands: pd.DataFrame, n: int, cfg: QueryConfig,
                 card_count: dict, outlet_count: dict) -> list:
    """Take candidates in order, skipping any that would break a diversity cap."""
    chosen = []
    for row in cands.itertuples(index=False):
        if len(chosen) >= n:
            break
        cards = {row.card_a, row.card_b}
        outlets = {row.outlet_a, row.outlet_b}
        if any(card_count.get(c, 0) >= cfg.max_per_card for c in cards):
            continue
        if any(outlet_count.get(o, 0) >= cfg.max_per_outlet for o in outlets):
            continue
        for c in cards:
            card_count[c] = card_count.get(c, 0) + 1
        for o in outlets:
            outlet_count[o] = outlet_count.get(o, 0) + 1
        chosen.append(row.pair_id)
    return chosen


def select_batch(pairs: pd.DataFrame, cfg: QueryConfig | None = None,
                 already_labelled: set | None = None) -> pd.DataFrame:
    """Return the batch of pairs to send to annotators, most informative first."""
    cfg = cfg or QueryConfig()
    rng = np.random.default_rng(cfg.seed)
    df = score_pairs(pairs, cfg)
    if already_labelled:
        df = df[~df["pair_id"].isin(already_labelled)]

    # Shuffle first so ties break randomly.
    df = df.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    if cfg.strategy == "random":
        # Control arm: same batch size and caps, no informativeness ranking.
        pool, n_dis = df, 0
    elif cfg.strategy == "uncertainty":
        pool = df[df["reason"] != "none"].sort_values("info", ascending=False, kind="stable")
        n_dis = int(round(cfg.batch_size * cfg.disagree_share))
    else:
        raise ValueError(f"unknown strategy: {cfg.strategy}")

    card_count, outlet_count, chosen = {}, {}, []

    def pick(cands: pd.DataFrame, n: int) -> None:
        cands = cands[~cands["pair_id"].isin(chosen)]
        n = min(n, cfg.batch_size - len(chosen))
        if n > 0:
            chosen.extend(_greedy_pick(cands, n, cfg, card_count, outlet_count))

    # Step 1: reserve part of the batch for confident disagreements with Google.
    pick(pool[pool["reason"] == "disagree_google"], n_dis)
    # Step 2: top up French pairs to the minimum share (French is the rarer language).
    n_fr_have = int(df[df["pair_id"].isin(chosen)]["lang"].eq("fr").sum())
    pick(pool[pool["lang"] == "fr"], int(round(cfg.batch_size * cfg.fr_share)) - n_fr_have)
    # Step 3: fill the rest with the most informative pairs of any kind.
    pick(pool, cfg.batch_size)

    out = df.set_index("pair_id").loc[chosen].reset_index()
    out = out.sort_values("info", ascending=False, kind="stable")
    return out[REQUIRED + ["reason", "uncertainty", "disagreement", "info"]]


def batch_report(batch: pd.DataFrame) -> dict:
    """Small summary to log with each weekly batch."""
    # Count pairs per card / outlet (a pair with the same outlet on both sides counts once).
    def per_pair(a: str, b: str) -> pd.Series:
        return pd.Series([v for x, y in zip(batch[a], batch[b]) for v in {x, y}])
    cards = per_pair("card_a", "card_b")
    outlets = per_pair("outlet_a", "outlet_b")
    return {
        "n": len(batch),
        "by_reason": batch["reason"].value_counts().to_dict(),
        "fr_share": round(float(batch["lang"].eq("fr").mean()), 3) if len(batch) else 0.0,
        "max_pairs_per_card": int(cards.value_counts().max()) if len(batch) else 0,
        "max_pairs_per_outlet": int(outlets.value_counts().max()) if len(batch) else 0,
        "mean_p_same": round(float(batch["p_same"].mean()), 3) if len(batch) else 0.0,
    }


def run_loop(pool: pd.DataFrame,
             score_fn: Callable[[pd.DataFrame], pd.Series],
             annotate_fn: Callable[[pd.DataFrame], pd.Series],
             update_fn: Callable[[pd.DataFrame], dict],
             cfg: QueryConfig | None = None,
             cycles: int = 5) -> list[dict]:
    """Weekly active-learning loop: score -> select -> annotate -> update.

    score_fn(pool)     -> calibrated P(same) for every pair, using the CURRENT model
    annotate_fn(batch) -> human labels (in the simulation: the held-out true labels)
    update_fn(labelled)-> dict from continual.update() / recalibration, e.g. {"accepted": True, "f1": ...}
    """
    cfg = cfg or QueryConfig()
    labelled_ids, log = set(), []
    for cycle in range(1, cycles + 1):
        scored = pool.assign(p_same=score_fn(pool).values)
        batch = select_batch(scored, QueryConfig(**{**asdict(cfg), "seed": cfg.seed + cycle}),
                             already_labelled=labelled_ids)
        batch = batch.assign(y_human=annotate_fn(batch).values)
        labelled_ids |= set(batch["pair_id"])
        result = update_fn(batch)
        log.append({"cycle": cycle, "labels_total": len(labelled_ids), **batch_report(batch), **result})
    return log
