import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from src.active_learning import QueryConfig, select_batch, batch_report, run_loop
rng = np.random.default_rng(0)
N = 3000
outlets = [f"o{i}" for i in range(60)]
w = np.r_[np.full(5, 8.0), np.ones(55)]; w /= w.sum() # 5 big outlets dominate
y = rng.integers(0, 2, N)
p = np.clip(np.where(y == 1, rng.beta(6, 2, N), rng.beta(2, 6, N)), 0, 1)
pairs = pd.DataFrame({
    "pair_id": np.arange(N), "p_same": p, "y_google": y,
    "card_a": rng.integers(0, 300, N), "card_b": rng.integers(0, 300, N),
    "outlet_a": rng.choice(outlets, N, p=w), "outlet_b": rng.choice(outlets, N, p=w),
    "lang": rng.choice(["en", "fr"], N, p=[0.75, 0.25]),
})
pairs["y_true"] = y

for strat in ["uncertainty", "random"]:
    b = select_batch(pairs, QueryConfig(strategy=strat))
    r = batch_report(b)
    print(strat, r)
    assert r["n"] == 100 and r["max_pairs_per_card"] <= 2 and r["max_pairs_per_outlet"] <= 10
    assert b["pair_id"].is_unique
    if strat == "uncertainty":
        assert r["fr_share"] >= 0.3 and (b["reason"] != "none").all()
        print("  mean |p - 0.5| uncertain picks:", round((b[b.reason=="uncertain"].p_same-0.5).abs().mean(), 3))

# Same seed -> same batch (reproducibility)
assert select_batch(pairs).pair_id.tolist() == select_batch(pairs).pair_id.tolist()

# Loop with stand-in model: each accepted update sharpens scores a bit
state = {"sharp": 1.0}
def score_fn(pool):
    s = state["sharp"]; q = pool["y_true"]
    return pool["p_same"] ** s * q + (1 - (1 - pool["p_same"]) ** s) * (1 - q)
def annotate_fn(batch): return pairs.set_index("pair_id").loc[batch.pair_id, "y_true"]
def update_fn(lab):
    state["sharp"] += 0.2; return {"accepted": True}
for row in run_loop(pairs, score_fn, annotate_fn, update_fn, cycles=5):
    print(row)
