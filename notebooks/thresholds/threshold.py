#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Threshold sweep (daily + hourly + 5-min): jump detection + D1/D2/D3 profile stability
#
# This notebook sweeps **jump detection thresholds** on **daily, hourly, and 5-min** data and shows (per frequency):
#
# - **Jump-score distribution** (histogram) with threshold lines
# - **Exceedance curve**: fraction of points with \(|x(t)| > \tau\) as a function of \(\tau\)
# - **Counts** per threshold: number of detected jumps and number of valid extracted windows
# - **Overlay plots** for **D1/D2/D3**:
#   - For each threshold, compute scores and plot the average x(t) profile for:
#     - Q0.1 (lowest 10% of the score)
#     - Q0.9 (highest 10% of the score)
#   - Superpose all thresholds on the same figure (Plotly)

# %%
from __future__ import annotations

# %% [markdown]
# ### Imports

# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data.curating_stooq import curate_stooq_dir_5min, curate_stooq_dir_daily, curate_stooq_dir_hourly
from utils.data.jump_detection import compute_jump_score, detect_jumps_many
from model.wavelet.wavelet import WaveletModel

# %% [markdown]
# ### Config

# %%
ScoreMode = Literal["kpca", "handcrafted"]
Freq = Literal["5min", "hourly", "daily"]


@dataclass(frozen=True)
class ThresholdSweepConfig:
    # data
    min_len: int = 500
    max_tickers: int = 200

    # jump detection thresholds to compare  # we allow per-frequency thresholds
    thresholds_common: Tuple[float, ...] = (2.5, 3.0, 3.5, 4.0, 4.5)  # we use these for 5min and daily by default
    thresholds_hourly: Tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5)  # we include 2.0 for hourly as requested

    # per-frequency windows (±window_steps around jump)
    window_steps_5min: int = 12
    window_steps_hourly: int = 12
    window_steps_daily: int = 20
    max_windows_per_threshold: int = 4000
    q: float = 0.1

    # intraday trimming (5min/hourly only)
    trim_intraday_minutes: int = 60

    # embedding/scoring
    J: int = 3
    n_components: int = 3
    include_scattering_spectra: bool = False
    score_mode: ScoreMode = "kpca"  # kpca uses D1,D2,D3 = embedding[:,0:3]; handcrafted uses D2/D3 filters

    random_seed: int = 0


CFG = ThresholdSweepConfig()


def thresholds_for_freq(freq: Freq) -> Tuple[float, ...]:
    if freq == "hourly":
        return CFG.thresholds_hourly
    return CFG.thresholds_common

# %% [markdown]
# ### Project root (notebook-safe)

# %%
def project_root() -> Path:
    """
    Return repo root in both script and notebook contexts.
    """
    try:
        here = Path(__file__).resolve()
        return here.parents[2]
    except NameError:
        cwd = Path.cwd().resolve()
        for p in [cwd, *cwd.parents]:
            if (p / "utils").is_dir() and (p / "data").is_dir():
                return p
        return cwd


def out_dir() -> Path:
    d = project_root() / "notebooks" / "thresholds" / "outputs" / "threshold_sweep"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir(freq: Freq) -> Path:
    root = project_root()
    if freq == "5min":
        return root / "data" / "stooq" / "poland" / "5_min" / "pl" / "wsestocks"
    if freq == "hourly":
        return root / "data" / "stooq" / "poland" / "hourly" / "ncstocks"
    if freq == "daily":
        return root / "data" / "stooq" / "poland" / "daily" / "ncstocks"
    raise ValueError(freq)


def out_dir_freq(freq: Freq) -> Path:
    d = out_dir() / freq
    d.mkdir(parents=True, exist_ok=True)
    return d


def window_steps_for_freq(freq: Freq) -> int:
    if freq == "5min":
        return CFG.window_steps_5min
    if freq == "hourly":
        return CFG.window_steps_hourly
    if freq == "daily":
        return CFG.window_steps_daily
    raise ValueError(freq)

# %% [markdown]
# ### Load data (per frequency)

# %%
def _infer_bar_timedelta(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    idx = index.sort_values()
    deltas = idx.to_series().diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return None
    td = deltas.median()
    if pd.isna(td) or td <= pd.Timedelta(0):
        return None
    return td


def trim_intraday_by_bars(df: pd.DataFrame, trim_minutes: int) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex) or trim_minutes <= 0:
        return df
    bar_td = _infer_bar_timedelta(df.index)
    if bar_td is None or bar_td >= pd.Timedelta(days=1):
        return df
    trim_bars = max(1, int(round(pd.Timedelta(minutes=trim_minutes) / bar_td)))

    days: List[pd.DataFrame] = []
    for _, day_df in df.groupby(df.index.date):
        day_df = day_df.sort_index()
        if len(day_df) <= (2 * trim_bars + 1):
            continue
        days.append(day_df.iloc[trim_bars:-trim_bars])
    if not days:
        return df.iloc[0:0]
    out = pd.concat(days).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def load_freq(freq: Freq, min_len: int, max_tickers: int) -> Dict[str, pd.DataFrame]:
    d = data_dir(freq)
    if not d.exists():
        raise FileNotFoundError(f"Data directory not found: {d}")

    if freq == "5min":
        dfs = curate_stooq_dir_5min(str(d), pattern="*.txt", recursive=True)
    elif freq == "hourly":
        dfs = curate_stooq_dir_hourly(str(d), pattern="*.txt", recursive=True)
    elif freq == "daily":
        dfs = curate_stooq_dir_daily(str(d), pattern="*.txt", recursive=True)
    else:
        raise ValueError(freq)

    tickers = [t for t, d in dfs.items() if d is not None and not d.empty and len(d) >= min_len]
    tickers.sort(key=lambda t: len(dfs[t]), reverse=True)
    tickers = tickers[:max_tickers]

    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = dfs[t].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if freq == "5min":
            df = df  # keep time-trim simple here; bars are dense
        elif freq == "hourly":
            df = trim_intraday_by_bars(df, trim_minutes=CFG.trim_intraday_minutes)
        # daily: no trimming
        out[t] = df
    return out


dfs_by_freq: Dict[Freq, Dict[str, pd.DataFrame]] = {}
for freq in ("5min", "hourly", "daily"):
    dfs_by_freq[freq] = load_freq(freq, min_len=CFG.min_len, max_tickers=CFG.max_tickers)
    print(f"Loaded {len(dfs_by_freq[freq])} {freq} tickers.")

# %% [markdown]
# ### Jump-score distribution (x(t)) on a representative ticker (per frequency)
#
# We compute the full score series once; thresholds only change how many events are selected.

# %%
def plot_score_diagnostics(freq: Freq, dfs: Dict[str, pd.DataFrame]) -> None:
    if not dfs:
        return
    sample_ticker, sample_df = max(dfs.items(), key=lambda kv: len(kv[1]))
    scores_df = compute_jump_score(sample_df, price_col="close")
    print(f"[{freq}] Score series for {sample_ticker}: n={len(scores_df)}")

    # histogram
    fig_hist = px.histogram(
        scores_df.reset_index(),
        x="score",
        nbins=200,
        title=f"{freq} jump score distribution x(t) - {sample_ticker}",
    )
    for thr in thresholds_for_freq(freq):
        fig_hist.add_vline(x=thr, line_dash="dash", line_color="red", opacity=0.6)
        fig_hist.add_vline(x=-thr, line_dash="dash", line_color="red", opacity=0.6)
    fig_hist.update_layout(template="plotly_white")
    fig_hist.write_html(out_dir_freq(freq) / f"score_hist_{freq}.html")
    fig_hist.show()

    # exceedance
    abs_x = scores_df["score"].abs().to_numpy(dtype=float)
    tau_grid = np.linspace(0.0, max(6.0, float(np.nanpercentile(abs_x, 99.9))), 200)
    tail = [(abs_x > t).mean() for t in tau_grid]
    fig_tail = go.Figure()
    fig_tail.add_trace(go.Scatter(x=tau_grid, y=tail, mode="lines", name="P(|x|>τ)"))
    for thr in thresholds_for_freq(freq):
        fig_tail.add_vline(x=thr, line_dash="dash", line_color="red", opacity=0.6)
    fig_tail.update_layout(
        title=f"{freq} exceedance curve - {sample_ticker}",
        xaxis_title="τ",
        yaxis_title="fraction(|x| > τ)",
        template="plotly_white",
        yaxis_type="log",
    )
    fig_tail.write_html(out_dir_freq(freq) / f"score_exceedance_{freq}.html")
    fig_tail.show()

# %% [markdown]
for freq in ("5min", "hourly", "daily"):
    plot_score_diagnostics(freq, dfs_by_freq[freq])

# %% [markdown]
# ### Helpers: window extraction + scoring

# %%
def extract_windows(
    dfs: Dict[str, pd.DataFrame],
    jumps_df: pd.DataFrame,
    window_steps: int,
    max_windows: int,
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    if jumps_df is None or jumps_df.empty:
        return np.empty((0, 2 * window_steps + 1)), jumps_df.iloc[0:0]

    rng = np.random.default_rng(seed)
    idxs = np.arange(len(jumps_df))
    if len(idxs) > max_windows:
        idxs = rng.choice(idxs, size=max_windows, replace=False)
        idxs = np.sort(idxs)
        jumps_df = jumps_df.iloc[idxs].reset_index(drop=True)

    windows: List[np.ndarray] = []
    valid_rows: List[int] = []
    center = window_steps

    for i, row in jumps_df.iterrows():
        ticker = row["ticker"]
        ts = row["timestamp"]
        if ticker not in dfs:
            continue
        df = dfs[ticker]
        if ts not in df.index:
            continue
        loc = df.index.get_loc(ts)
        if loc - window_steps < 0 or loc + window_steps + 1 > len(df):
            continue
        subset = df.iloc[loc - window_steps : loc + window_steps + 1]
        r_window = subset["close"].pct_change().fillna(0.0).to_numpy(dtype=float)

        norm = float(row.get("f", 1.0)) * float(row.get("sigma", 1.0))
        if not np.isfinite(norm) or norm == 0.0:
            norm = 1e-4
        x_profile = r_window / norm

        sgn = float(np.sign(x_profile[center]))
        if sgn == 0.0:
            sgn = 1.0
        windows.append(x_profile * sgn)
        valid_rows.append(i)

    if not windows:
        return np.empty((0, 2 * window_steps + 1)), jumps_df.iloc[0:0]

    X = np.asarray(windows, dtype=float)
    return X, jumps_df.iloc[valid_rows].reset_index(drop=True)


def compute_scores(
    X: np.ndarray,
    include_ss: bool,
    J: int,
    n_components: int,
    score_mode: ScoreMode,
    seed: int,
) -> Dict[str, np.ndarray]:
    if X.size == 0:
        return {"D1": np.array([]), "D2": np.array([]), "D3": np.array([])}
    center = X.shape[1] // 2

    wm = WaveletModel(
        n_layers=0,
        n_neurons=0,
        n_outputs=0,
        J=J,
        n_components=n_components,
        include_scattering_spectra=include_ss,
        random_state=seed,
    )
    emb = wm.fit_transform(X)
    d1 = emb[:, 0].copy() if emb.shape[1] >= 1 else np.zeros((X.shape[0],), dtype=float)  # we guard against degenerate embeddings

    # orient D1 to correlate with post>pre activity
    act_post = np.sum(np.abs(X[:, center + 1 :]), axis=1)
    act_pre = np.sum(np.abs(X[:, :center]), axis=1)
    asym = (act_post - act_pre) / (act_post + act_pre + 1e-9)
    corr = np.corrcoef(d1, asym)[0, 1] if len(d1) > 1 else 1.0
    if np.isfinite(corr) and corr < 0:
        d1 *= -1

    if score_mode == "kpca":
        if emb.shape[1] >= 3:
            d2 = emb[:, 1]  # we use kpca component 2 as mean-reversion score
            d3 = emb[:, 2]  # we use kpca component 3 as trend score
        else:
            x_pre = X[:, center - 1]  # we fall back to handcrafted scores when kpca is underdetermined
            x_post = X[:, center + 1]  # we fall back to handcrafted scores when kpca is underdetermined
            d2 = x_pre - x_post  # we define D2 as local mean-reversion proxy
            d3 = x_pre + x_post  # we define D3 as local trend proxy
    else:
        x_pre = X[:, center - 1]
        x_post = X[:, center + 1]
        d2 = x_pre - x_post
        d3 = x_pre + x_post

    return {"D1": d1, "D2": np.asarray(d2), "D3": np.asarray(d3)}


def q_low_high_profiles(X: np.ndarray, score: np.ndarray, q: float) -> Tuple[np.ndarray, np.ndarray]:
    if X.size == 0 or score.size == 0:
        return np.array([]), np.array([])
    n = len(score)
    k = max(1, int(round(q * n)))
    order = np.argsort(score)
    low = np.mean(X[order[:k]], axis=0)
    high = np.mean(X[order[-k:]], axis=0)
    return low, high

# %% [markdown]
# ### Sweep thresholds (per frequency): counts + profiles

# %%
def run_sweep_for_freq(freq: Freq, dfs: Dict[str, pd.DataFrame]) -> None:
    w = window_steps_for_freq(freq)
    results_rows: List[Dict] = []
    profiles: Dict[str, Dict[float, Tuple[np.ndarray, np.ndarray]]] = {"D1": {}, "D2": {}, "D3": {}}

    thr_list = thresholds_for_freq(freq)
    for thr in thr_list:
        jumps_df = detect_jumps_many(dfs, threshold=thr)
        X, _ = extract_windows(
            dfs,
            jumps_df,
            window_steps=w,
            max_windows=CFG.max_windows_per_threshold,
            seed=CFG.random_seed,
        )

        scores = compute_scores(
            X,
            include_ss=CFG.include_scattering_spectra,
            J=CFG.J,
            n_components=CFG.n_components,
            score_mode=CFG.score_mode,
            seed=CFG.random_seed,
        )

        for d in ("D1", "D2", "D3"):
            low, high = q_low_high_profiles(X, scores[d], q=CFG.q)
            profiles[d][thr] = (low, high)

        results_rows.append({"threshold": thr, "n_jumps": int(len(jumps_df)), "n_windows": int(len(X))})

    summary = pd.DataFrame(results_rows).sort_values("threshold")
    print(f"\n[{freq}] threshold summary")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir_freq(freq) / f"threshold_summary_{freq}.csv", index=False)

    # overlay plots per direction
    wl = 2 * w + 1
    center = wl // 2
    t_axis = np.arange(-center, center + 1)
    palette = px.colors.qualitative.Plotly

    for direction in ("D1", "D2", "D3"):
        fig = go.Figure()
        for i, thr in enumerate(thr_list):
            low, high = profiles[direction].get(thr, (np.array([]), np.array([])))
            color = palette[i % len(palette)]
            if low.size:
                fig.add_trace(go.Scatter(x=t_axis, y=low, mode="lines", name=f"τ={thr} | Q{CFG.q}", line=dict(color=color, dash="dot", width=2)))
            if high.size:
                fig.add_trace(go.Scatter(x=t_axis, y=high, mode="lines", name=f"τ={thr} | Q{1-CFG.q}", line=dict(color=color, dash="solid", width=2)))
        fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.6)
        fig.update_layout(
            title=f"{freq} threshold sweep: {direction} profiles | includeSS={CFG.include_scattering_spectra} | mode={CFG.score_mode}",
            xaxis_title=f"Time (steps) relative to jump",
            yaxis_title="Jump-aligned normalized return x(t)",
            template="plotly_white",
            hovermode="x unified",
        )
        fig.write_html(out_dir_freq(freq) / f"overlay_{direction}_thresholds_{freq}.html")
        fig.show()


for freq in ("5min", "hourly", "daily"):
    run_sweep_for_freq(freq, dfs_by_freq[freq])

# %% [markdown]
# (Overlay plots are produced inside the per-frequency sweep.)

# %%
def overlay_profiles(direction: Literal["D1", "D2", "D3"]) -> go.Figure:
    wl = 2 * CFG.window_steps + 1
    center = wl // 2
    t_axis = np.arange(-center, center + 1)

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, thr in enumerate(CFG.thresholds_common):
        low, high = profiles[direction].get(thr, (np.array([]), np.array([])))
        color = palette[i % len(palette)]
        if low.size:
            fig.add_trace(
                go.Scatter(
                    x=t_axis,
                    y=low,
                    mode="lines",
                    name=f"τ={thr} | Q0.1",
                    line=dict(color=color, dash="dot", width=2),
                )
            )
        if high.size:
            fig.add_trace(
                go.Scatter(
                    x=t_axis,
                    y=high,
                    mode="lines",
                    name=f"τ={thr} | Q0.9",
                    line=dict(color=color, dash="solid", width=2),
                )
            )

    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.6)
    fig.update_layout(
        title=f"Daily threshold sweep: {direction} profiles (Q{CFG.q} vs Q{1-CFG.q}) | includeSS={CFG.include_scattering_spectra} | mode={CFG.score_mode}",
        xaxis_title="Time (days) relative to jump",
        yaxis_title="Jump-aligned normalized return x(t)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


for d in ("D1", "D2", "D3"):
    fig = overlay_profiles(d)
    fig.write_html(out_dir() / f"overlay_{d}_thresholds_daily.html")
    fig.show()


