"""
Shared monitoring/drift functions for Milestone 3, Section 3.

Everyone imports from here so the drift math is computed the same way everywhere:
    from src.m3_monitoring import load_stream, load_reference, psi, categorical_psi, \
        js_distance, wasserstein, rolling_window_metrics

The "stream" is data/pairs/test.csv (May 2026 onward), replayed in chronological order
to simulate a production query log. The "reference" is data/pairs/val.csv (Jan-Apr 2026)
-- the last distribution the deployed model was validated against before this simulated
go-live, which is what a real monitoring system would compare incoming traffic to.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTLETS_CSV = DATA / "google_news_front_page" / "outlets.csv"

RARE_GROUPS = {"local_en", "franco", "cdn_unpaid"}  # small/independent outlets, the project's fairness focus


def load_outlet_groups() -> pd.DataFrame:
    return pd.read_csv(OUTLETS_CSV)[["domain", "group"]]


def _attach_outlet_groups(df: pd.DataFrame) -> pd.DataFrame:
    groups = load_outlet_groups()
    df = df.merge(groups.rename(columns={"domain": "outlet_a", "group": "group_a"}), on="outlet_a", how="left")
    df = df.merge(groups.rename(columns={"domain": "outlet_b", "group": "group_b"}), on="outlet_b", how="left")
    df["group_a"] = df["group_a"].fillna("unknown")
    df["group_b"] = df["group_b"].fillna("unknown")
    return df


def _base_load(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df.timestamp.astype(str), format="%Y%m%d%H%M%S")
    df["headline_length_a"] = df.headline_a.astype(str).str.len()
    df["headline_length_b"] = df.headline_b.astype(str).str.len()
    df["headline_length"] = (df.headline_length_a + df.headline_length_b) / 2.0
    df = _attach_outlet_groups(df)
    df["rare_outlet_pair"] = (df.group_a.isin(RARE_GROUPS) | df.group_b.isin(RARE_GROUPS)).astype(int)
    return df.sort_values("ts").reset_index(drop=True)


def load_reference(path=DATA / "pairs" / "val.csv") -> pd.DataFrame:
    """The last known-good distribution: the validation pairs used to pick the deployment threshold."""
    return _base_load(path)


def load_stream(path=DATA / "pairs" / "test.csv", window="W") -> pd.DataFrame:
    """The simulated production stream: the test pairs, replayed in time order and bucketed into windows."""
    df = _base_load(path)
    df["window"] = df.ts.dt.to_period(window).dt.start_time
    return df


def psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index for a continuous variable.
    Bucket edges come from the reference distribution's quantiles, so a PSI of 0
    means the current data falls into those same buckets in the same proportions.
    Conventional reading: <0.1 no real shift, 0.1-0.25 moderate, >0.25 significant.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(reference: pd.Series, current: pd.Series, categories=None) -> float:
    """Same PSI formula, but bucketed by category label instead of quantile bin."""
    categories = categories if categories is not None else sorted(set(reference) | set(current))
    ref_pct = reference.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    cur_pct = current.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    ref_pct = np.clip(ref_pct.to_numpy(), 1e-6, None)
    cur_pct = np.clip(cur_pct.to_numpy(), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def js_distance(reference: np.ndarray, current: np.ndarray, buckets: int = 20) -> float:
    """Jensen-Shannon distance between two continuous distributions, via a shared histogram."""
    lo = min(reference.min(), current.min())
    hi = max(reference.max(), current.max())
    edges = np.linspace(lo, hi, buckets + 1)
    ref_hist = np.histogram(reference, bins=edges)[0].astype(float)
    cur_hist = np.histogram(current, bins=edges)[0].astype(float)
    ref_hist = ref_hist / ref_hist.sum()
    cur_hist = cur_hist / cur_hist.sum()
    return float(jensenshannon(ref_hist, cur_hist, base=2))


def wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    return float(wasserstein_distance(reference, current))


def score_pairs(df: pd.DataFrame, model_name: str = "models/minilm_finetuned") -> np.ndarray:
    """Run the deployed bi-encoder on headline_a/headline_b. Imports sentence-transformers lazily
    so the drift-stat functions above stay usable without it installed."""
    from src.embeddings import similarity
    return similarity(df.headline_a.tolist(), df.headline_b.tolist(), model_name=model_name)


def sanitizer_trigger_rate(headlines) -> float:
    """Share of headlines the Milestone 3 Section 2 sanitizer (src/robustness.py) actually
    changes, beyond plain NFC/whitespace cleanup -- homoglyph folding, invisible-character
    removal, or a stripped boilerplate prefix like "LIVE:". A rising rate is a signal in its
    own right: either a genuine shift in how outlets format headlines, or an attack attempt
    (Section 2.2's threat model)."""
    from src.robustness import sanitize
    from src.embeddings import clean_headline
    headlines = list(headlines)
    if not headlines:
        return 0.0
    changed = sum(1 for h in headlines if sanitize(h) != clean_headline(h))
    return changed / len(headlines)


def sanitizer_rate_by_window(stream: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window, g in stream.groupby("window"):
        texts = pd.concat([g.headline_a, g.headline_b])
        rows.append({
            "window": window,
            "n_pairs": len(g),
            "sanitizer_trigger_rate": sanitizer_trigger_rate(texts),
        })
    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)


def fit_v1_pipeline(reference: pd.DataFrame, model_path="models/minilm_finetuned"):
    """Fit the same Pipeline (val-F1 threshold + Platt calibration on clean val) that Section 2
    uses for v1 -- reused here rather than re-derived, so the threshold/calibration are exactly
    what Section 2 reports, not a second independent fit that might drift from it."""
    from src.robustness import BiEncoder, Pipeline
    scorer = BiEncoder(ROOT / model_path, "v1")
    return Pipeline(scorer, sanitize_inputs=False, name="v1").fit(reference)


def coverage_by_window(stream: pd.DataFrame, pipe, band: tuple) -> pd.DataFrame:
    """Per-window coverage (share of pairs outside the abstention band) and F1 on the pairs
    the pipeline actually decides on -- reuses Section 2's own `selective()` so "coverage" and
    "selective F1" mean exactly what Section 2.3.2/2.4.1 report, not a redefinition."""
    from src.robustness import selective

    scores = pipe.raw(stream.headline_a, stream.headline_b)
    probs = pipe.prob(scores)
    labels = stream.label.to_numpy()

    rows = []
    for window, idx in stream.groupby("window").indices.items():
        idx = np.asarray(idx)
        r = selective(labels[idx], probs[idx], scores[idx], pipe.threshold, band)
        r["window"] = window
        r["n_pairs"] = len(idx)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)


def rolling_window_metrics(df: pd.DataFrame, scores: np.ndarray, threshold: float = 0.724) -> pd.DataFrame:
    """
    Per-window model-quality metrics against the real labels -- a stand-in for "delayed truth
    metrics when labels arrive": in production these would lag behind the predictions by
    however long it takes Google's own cards to confirm a grouping; here we just compute them
    directly since test.csv's labels are already known.
    """
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

    out = df[["window"]].copy()
    out["score"] = scores
    out["pred"] = (scores >= threshold).astype(int)
    out["label"] = df.label.to_numpy()

    rows = []
    for window, g in out.groupby("window"):
        row = {
            "window": window,
            "n_pairs": len(g),
            "f1": f1_score(g.label, g.pred) if g.label.nunique() > 1 else np.nan,
            "roc_auc": roc_auc_score(g.label, g.score) if g.label.nunique() > 1 else np.nan,
            "pr_auc": average_precision_score(g.label, g.score) if g.label.nunique() > 1 else np.nan,
            "mean_score": g.score.mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)


def _resample_window(reference: pd.DataFrame, window_size: int, rng) -> pd.DataFrame:
    """A random window_size-sized draw (with replacement) from the reference data --
    i.e. what a real stream window looks like if nothing has actually drifted."""
    return reference.iloc[rng.integers(0, len(reference), size=window_size)]


def calibrate_numeric_psi(reference: pd.DataFrame, column: str, window_size: int, n_boot: int = 500,
                           percentile: float = 95, seed: int = 6450) -> tuple:
    """
    PSI threshold for `column`, calibrated the same way drift_by_window actually computes
    it: the full reference as the stable baseline, vs. a window_size-sized random draw
    from that same reference (so it's a null sample -- no real drift by construction,
    but sized exactly like the real stream windows will be). The threshold is the
    `percentile`-th value of that null distribution.
    """
    rng = np.random.default_rng(seed)
    ref_vals = reference[column].to_numpy()
    null_vals = np.array([
        psi(ref_vals, _resample_window(reference, window_size, rng)[column].to_numpy())
        for _ in range(n_boot)
    ])
    return float(np.percentile(null_vals, percentile)), null_vals


def calibrate_categorical_psi(reference: pd.DataFrame, column: str, window_size: int, n_boot: int = 500,
                               percentile: float = 95, seed: int = 6450) -> tuple:
    """Same idea as calibrate_numeric_psi, for a categorical column."""
    rng = np.random.default_rng(seed)
    categories = sorted(reference[column].unique())
    ref_col = reference[column]
    null_vals = np.array([
        categorical_psi(ref_col, _resample_window(reference, window_size, rng)[column], categories=categories)
        for _ in range(n_boot)
    ])
    return float(np.percentile(null_vals, percentile)), null_vals


def calibrate_window_band(reference: pd.DataFrame, metric_fn, window_size: int, n_boot: int = 500,
                           percentiles: tuple = (2.5, 97.5), seed: int = 6450) -> tuple:
    """
    General-purpose calibration for any metric computed on a single window of a given
    size (e.g. rolling F1, rare-outlet share): repeatedly draw random windows of that
    size from the reference data (with replacement, since real windows are smaller
    than the reference set) and see how much the metric swings on its own. Returns
    the requested percentile band plus the raw null samples for plotting.

    metric_fn(window_df) -> float, and may return np.nan (e.g. F1 needs both classes
    present); NaNs are dropped before taking percentiles.
    """
    rng = np.random.default_rng(seed)
    n = len(reference)
    vals = []
    for _ in range(n_boot):
        sample = reference.iloc[rng.integers(0, n, size=window_size)]
        v = metric_fn(sample)
        if v is not None and not np.isnan(v):
            vals.append(v)
    vals = np.array(vals)
    lo, hi = np.percentile(vals, percentiles)
    return (float(lo), float(hi)), vals


def calibrate_coverage_floor(reference: pd.DataFrame, pipe, band: tuple, window_size: int,
                              n_boot: int = 500, percentile: float = 5, seed: int = 6425) -> tuple:
    """Same null-resampling idea as calibrate_window_band, specialized for coverage: how low
    can coverage go on a window_size-sized draw from the reference alone, with no real drift?"""
    from src.robustness import selective

    scores = pipe.raw(reference.headline_a, reference.headline_b)
    probs = pipe.prob(scores)
    labels = reference.label.to_numpy()
    rng = np.random.default_rng(seed)
    n = len(reference)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=window_size)
        vals.append(selective(labels[idx], probs[idx], scores[idx], pipe.threshold, band)["coverage"])
    vals = np.array(vals)
    return float(np.percentile(vals, percentile)), vals


def calibrate_sanitizer_ceiling(reference: pd.DataFrame, window_size: int, n_boot: int = 500,
                                 percentile: float = 95, seed: int = 6426) -> tuple:
    """Same idea for the sanitizer trigger rate: its own baseline noise on reference data,
    with no attack and no real formatting shift."""
    rng = np.random.default_rng(seed)
    n = len(reference)
    ref_a, ref_b = reference.headline_a.to_numpy(), reference.headline_b.to_numpy()
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=window_size)
        vals.append(sanitizer_trigger_rate(list(ref_a[idx]) + list(ref_b[idx])))
    vals = np.array(vals)
    return float(np.percentile(vals, percentile)), vals


def sustained(flags: pd.Series, min_consecutive: int = 2) -> pd.Series:
    """True only where `flags` has been True for `min_consecutive` windows in a row --
    the rule that turns a noisy single-window alert into something worth paging someone over."""
    result = flags.copy()
    for shift in range(1, min_consecutive):
        result = result & flags.shift(shift, fill_value=False)
    return result


def calibrate_all_thresholds(reference: pd.DataFrame, reference_scores: np.ndarray,
                              window_size: int, n_boot: int = 500, threshold: float = 0.724,
                              pipe=None, band: tuple = None) -> dict:
    """
    Runs all four calibrations against data this project actually has, instead of
    borrowing generic industry PSI bands. window_size should match the real stream's
    typical window size, so the F1/rare-share noise bands reflect the sample size
    alerts will actually be computed on.
    """
    from sklearn.metrics import f1_score

    length_psi_threshold, length_psi_null = calibrate_numeric_psi(
        reference, "headline_length", window_size=window_size, n_boot=n_boot)
    group_psi_threshold, group_psi_null = calibrate_categorical_psi(
        reference, "group_a", window_size=window_size, n_boot=n_boot)

    rare_share_band, rare_share_null = calibrate_window_band(
        reference, lambda w: w.rare_outlet_pair.mean(), window_size, n_boot=n_boot
    )

    ref_with_scores = reference.assign(_score=reference_scores)

    def _f1(window_df):
        if window_df.label.nunique() < 2:
            return np.nan
        pred = (window_df._score >= threshold).astype(int)
        return f1_score(window_df.label, pred)

    f1_band, f1_null = calibrate_window_band(ref_with_scores, _f1, window_size, n_boot=n_boot,
                                              percentiles=(5, 95))

    if pipe is None or band is None:
        from src.robustness import abstain_band
        pipe = fit_v1_pipeline(reference)
        band = abstain_band(pipe, reference, coverage=0.9)
    coverage_floor, coverage_null = calibrate_coverage_floor(reference, pipe, band, window_size, n_boot=n_boot)
    sanitizer_ceiling, sanitizer_null = calibrate_sanitizer_ceiling(reference, window_size, n_boot=n_boot)

    return {
        "length_psi_threshold": length_psi_threshold,
        "group_psi_threshold": group_psi_threshold,
        "rare_share_band": rare_share_band,
        "f1_floor": f1_band[0],  # only the low side matters: a drop, not a rise, is the SLO concern
        "coverage_floor": coverage_floor,
        "sanitizer_ceiling": sanitizer_ceiling,
        "pipe": pipe,
        "band": band,
        "_null_samples": {
            "length_psi": length_psi_null, "group_psi": group_psi_null,
            "rare_share": rare_share_null, "f1": f1_null,
            "coverage": coverage_null, "sanitizer": sanitizer_null,
        },
    }


def apply_alerts(drift: pd.DataFrame, rolling: pd.DataFrame, thresholds: dict,
                  coverage: pd.DataFrame = None, sanitizer: pd.DataFrame = None,
                  min_consecutive: int = 2) -> pd.DataFrame:
    """Merge drift + rolling-metric (+ coverage + sanitizer, if given) tables, flag each window
    against the calibrated thresholds, and add the sustained (2+ consecutive window) version
    of each flag."""
    alerts = drift.merge(rolling[["window", "f1"]], on="window")

    alerts["length_drift_raw"] = alerts.length_psi > thresholds["length_psi_threshold"]
    alerts["group_drift_raw"] = alerts.group_psi > thresholds["group_psi_threshold"]
    alerts["rare_share_drop_raw"] = alerts.rare_outlet_share < thresholds["rare_share_band"][0]
    alerts["slo_violation_raw"] = alerts.f1 < thresholds["f1_floor"]

    flag_cols = ["length_drift", "group_drift", "rare_share_drop", "slo_violation"]

    if coverage is not None:
        alerts = alerts.merge(coverage[["window", "coverage"]], on="window")
        alerts["coverage_drop_raw"] = alerts.coverage < thresholds["coverage_floor"]
        flag_cols.append("coverage_drop")

    if sanitizer is not None:
        alerts = alerts.merge(sanitizer[["window", "sanitizer_trigger_rate"]], on="window")
        alerts["sanitizer_spike_raw"] = alerts.sanitizer_trigger_rate > thresholds["sanitizer_ceiling"]
        flag_cols.append("sanitizer_spike")

    for col in flag_cols:
        alerts[f"{col}_sustained"] = sustained(alerts[f"{col}_raw"], min_consecutive=min_consecutive)

    return alerts


def drift_by_window(stream: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """One row per window: PSI/JS/Wasserstein on headline length, PSI on outlet group, rare-outlet share."""
    ref_len = reference.headline_length.to_numpy()
    ref_group = reference.group_a
    categories = sorted(set(reference.group_a) | set(stream.group_a))

    rows = []
    for window, g in stream.groupby("window"):
        rows.append({
            "window": window,
            "n_pairs": len(g),
            "length_psi": psi(ref_len, g.headline_length.to_numpy()),
            "length_js": js_distance(ref_len, g.headline_length.to_numpy()),
            "length_wasserstein": wasserstein(ref_len, g.headline_length.to_numpy()),
            "group_psi": categorical_psi(ref_group, g.group_a, categories=categories),
            "rare_outlet_share": g.rare_outlet_pair.mean(),
        })
    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)


# Shared font sizes for every panel, both in the combined dashboard and the
# individually-saved versions -- bigger than matplotlib's defaults, since the
# combined dashboard shrinks each panel a lot.
TITLE_FS = 15
LABEL_FS = 13
LEGEND_FS = 11
TICK_FS = 11


def _style_panel(ax, title, ylabel=None, legend=True, rotate_x=True):
    ax.set_title(title, fontsize=TITLE_FS)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    if legend:
        ax.legend(fontsize=LEGEND_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    if rotate_x:
        ax.tick_params(axis="x", rotation=45)


def _panel_model_quality(ax, rolling, thresholds):
    ax.plot(rolling.window, rolling.f1, marker="o", label="rolling F1", color="#2a6f97")
    ax.plot(rolling.window, rolling.roc_auc, marker="o", label="rolling ROC-AUC", color="#a3a3a3", alpha=0.7)
    ax.axhline(0.90, color="gray", linestyle=":", linewidth=1, label="fixed SLO target: F1 >= 0.90")
    ax.axhline(thresholds["f1_floor"], color="crimson", linestyle="--", linewidth=1,
               label="calibrated noise floor (5th pct, null)")
    _style_panel(ax, "Rolling model quality (delayed truth)", ylabel="score")


def _panel_length_drift(ax, drift, thresholds):
    ax.bar(drift.window, drift.length_psi, width=5, color="#2a6f97", label="PSI")
    ax.axhline(thresholds["length_psi_threshold"], color="crimson", linestyle="--", linewidth=1,
               label="calibrated threshold (95th pct, null)")
    _style_panel(ax, "Headline-length drift (PSI vs. validation baseline)")


def _panel_group_drift(ax, drift, thresholds):
    ax.bar(drift.window, drift.group_psi, width=5, color="#52796f", label="PSI")
    ax.axhline(thresholds["group_psi_threshold"], color="crimson", linestyle="--", linewidth=1,
               label="calibrated threshold (95th pct, null)")
    _style_panel(ax, "Outlet-group mix drift (PSI vs. validation baseline)")


def _panel_fairness_kpi(ax, drift, reference, thresholds):
    ax.plot(drift.window, drift.rare_outlet_share, marker="o", color="#bb3e03",
            label="small/local-outlet pair share")
    ax.axhline(reference.rare_outlet_pair.mean(), color="gray", linestyle=":", linewidth=1,
               label="validation baseline")
    lo, hi = thresholds["rare_share_band"]
    ax.axhspan(lo, hi, color="gray", alpha=0.15, label="calibrated null band (2.5-97.5th pct)")
    _style_panel(ax, "Fairness KPI: small/local-outlet pair share", ylabel="share of pairs")


def _panel_score_distribution(ax, reference_scores, stream_scores):
    bins = np.linspace(min(reference_scores.min(), stream_scores.min()),
                        max(reference_scores.max(), stream_scores.max()), 30)
    ax.hist(reference_scores, bins=bins, alpha=0.5, density=True, label="validation (reference)",
            color="#2a6f97")
    ax.hist(stream_scores, bins=bins, alpha=0.5, density=True, label="test stream (all windows)",
            color="#bb3e03")
    _style_panel(ax, "Score-distribution drift", rotate_x=False)


def _panel_traffic_volume(ax, drift):
    ax.bar(drift.window, drift.n_pairs, width=5, color="#6c757d")
    _style_panel(ax, "Traffic volume per window", ylabel="pairs", legend=False)


def _panel_coverage(ax, coverage, thresholds):
    ax.plot(coverage.window, coverage.coverage, marker="o", color="#2a6f97", label="coverage")
    ax.axhline(thresholds["coverage_floor"], color="crimson", linestyle="--", linewidth=1,
               label="calibrated floor (5th pct, null)")
    ax.axhline(0.919, color="gray", linestyle=":", linewidth=1, label="clean-validation baseline (91.9%)")
    _style_panel(ax, "Abstention coverage (share of pairs decided)", ylabel="coverage")


def _panel_sanitizer_rate(ax, sanitizer, thresholds):
    ax.bar(sanitizer.window, sanitizer.sanitizer_trigger_rate, width=5, color="#f9a825", label="trigger rate")
    ax.axhline(thresholds["sanitizer_ceiling"], color="crimson", linestyle="--", linewidth=1,
               label="calibrated ceiling (95th pct, null)")
    _style_panel(ax, "Sanitizer trigger rate", ylabel="share of headlines")


def _panels(rolling, drift, reference, reference_scores, stream_scores, thresholds, coverage, sanitizer):
    """(name, plot_fn) for every panel that's available, in dashboard order."""
    out = [
        ("rolling_model_quality", lambda ax: _panel_model_quality(ax, rolling, thresholds)),
        ("headline_length_drift", lambda ax: _panel_length_drift(ax, drift, thresholds)),
        ("outlet_group_drift", lambda ax: _panel_group_drift(ax, drift, thresholds)),
        ("fairness_kpi", lambda ax: _panel_fairness_kpi(ax, drift, reference, thresholds)),
        ("score_distribution", lambda ax: _panel_score_distribution(ax, reference_scores, stream_scores)),
        ("traffic_volume", lambda ax: _panel_traffic_volume(ax, drift)),
    ]
    if coverage is not None:
        out.append(("abstention_coverage", lambda ax: _panel_coverage(ax, coverage, thresholds)))
    if sanitizer is not None:
        out.append(("sanitizer_trigger_rate", lambda ax: _panel_sanitizer_rate(ax, sanitizer, thresholds)))
    return out


def plot_dashboard(rolling: pd.DataFrame, drift: pd.DataFrame, reference: pd.DataFrame,
                    reference_scores: np.ndarray, stream_scores: np.ndarray, thresholds: dict, out_path,
                    coverage: pd.DataFrame = None, sanitizer: pd.DataFrame = None, show: bool = True):
    """Render the compact monitoring dashboard, using calibrated (not generic) thresholds,
    and save it to out_path (a .png). Adds the coverage and sanitizer-rate panels when those
    tables are given (both optional, since they need src/robustness.py's Pipeline). Displays
    the figure inline (show=True, the default) so it's visible when reading the notebook
    top to bottom, not just as a saved file someone has to open separately."""
    import matplotlib.pyplot as plt

    panels = _panels(rolling, drift, reference, reference_scores, stream_scores, thresholds, coverage, sanitizer)
    ncols = 4 if len(panels) > 6 else 3
    nrows = -(-len(panels) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for (name, plot_fn), ax in zip(panels, axes_flat):
        plot_fn(ax)
    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def save_individual_panels(rolling: pd.DataFrame, drift: pd.DataFrame, reference: pd.DataFrame,
                            reference_scores: np.ndarray, stream_scores: np.ndarray, thresholds: dict, out_dir,
                            coverage: pd.DataFrame = None, sanitizer: pd.DataFrame = None,
                            show: bool = True) -> list:
    """Save every panel from plot_dashboard as its own full-size PNG in out_dir
    (e.g. out_dir/rolling_model_quality.png), for whenever the compact grid is too
    small to read a specific panel closely. Displays each one inline (show=True, the
    default) so every panel is visible in the notebook itself, not just saved to disk.
    Returns the list of paths written."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = _panels(rolling, drift, reference, reference_scores, stream_scores, thresholds, coverage, sanitizer)

    written = []
    for name, plot_fn in panels:
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_fn(ax)
        fig.tight_layout()
        path = out_dir / f"panel_{name}.png"
        fig.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        written.append(path)
    return written
