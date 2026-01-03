#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Distributions of D1 / D2 / D3 (reflexivity / mean-reversion / trend)
#
# This notebook computes the scores:
# - **D1 (reflexivity)**,
# - **D2 (mean-reversion)**,
# - **D3 (trend)**
#
# and plots **histogram distributions** across:
# - **Frequencies**: 5-min, hourly, daily
# - **With / without Scattering Spectra** in `WaveletModel`
# - **Score modes**:
#   - `kpca`: D1/D2/D3 from KPCA embedding
#   - `handcrafted`: D1 from KPCA embedding, but D2/D3 from filters:
#     - D2 = x_pre − x_post
#     - D3 = x_pre + x_post

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

from utils.data.curating_stooq import curate_stooq_dir_5min, curate_stooq_dir_hourly, curate_stooq_dir_daily
from utils.data.jump_detection import detect_jumps_many
from model.wavelet.wavelet import WaveletModel

Freq = Literal["5min", "hourly", "daily"]
ScoreMode = Literal["kpca", "handcrafted"]

# %% [markdown]
# ### Config

# %%
@dataclass(frozen=True)
class FeatureDistConfig:
    # dataset selection
    min_len: int = 500
    max_tickers: int = 120
    threshold_5min: float = 4.0
    threshold_hourly: float = 2.0
    threshold_daily: float = 2.5

    # windows
    window_steps_intraday: int = 12
    window_steps_daily: int = 20
    max_windows: int = 4000

    # wavelet embedding
    J: int = 3
    n_components: int = 3

    # plot
    nbins: int = 120
    show_plots: bool = False


CFG = FeatureDistConfig()


def threshold_for_freq(freq: Freq) -> float:
    if freq == "5min":
        return CFG.threshold_5min
    if freq == "hourly":
        return CFG.threshold_hourly
    if freq == "daily":
        return CFG.threshold_daily
    raise ValueError(freq)

# %% [markdown]
# ### Repo root + data dirs (notebook-safe)

# %%
def project_root() -> Path:
    try:
        here = Path(__file__).resolve()
        return here.parents[2]  # .../notebooks/features_distribution/feature_distribution.py -> repo root
    except NameError:
        cwd = Path.cwd().resolve()
        for p in [cwd, *cwd.parents]:
            if (p / "utils").is_dir() and (p / "data").is_dir():
                return p
        return cwd


def out_dir() -> Path:
    d = project_root() / "notebooks" / "features_distribution" / "outputs"
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
    raise ValueError(f"Unknown freq: {freq}")

# %% [markdown]
# ### Preprocessing helpers

# %%
def trim_intraday(df: pd.DataFrame, minutes: int = 60) -> pd.DataFrame:
    """
    Remove first/last `minutes` of each day (intraday only).
    """
    if df.empty or not isinstance(df.index, pd.DatetimeIndex) or minutes <= 0:
        return df
    out_days: List[pd.DataFrame] = []
    for _, day_df in df.groupby(df.index.date):
        day_df = day_df.sort_index()
        if day_df.empty:
            continue
        start = day_df.index[0] + pd.Timedelta(minutes=minutes)
        end = day_df.index[-1] - pd.Timedelta(minutes=minutes)
        mask = (day_df.index >= start) & (day_df.index <= end)
        if mask.any():
            out_days.append(day_df.loc[mask])
    if not out_days:
        return df.iloc[0:0]
    out = pd.concat(out_days).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


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


def trim_intraday_by_bars(df: pd.DataFrame, trim_minutes: int = 60) -> pd.DataFrame:
    """
    For irregular hourly-like data, trimming by absolute time can drop everything.
    Instead, trim by *number of bars* per day inferred from median bar size.
    """
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


def load_poland(freq: Freq, min_len: int, max_tickers: int) -> Dict[str, pd.DataFrame]:
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

    tickers = [t for t, df in dfs.items() if df is not None and not df.empty and len(df) >= min_len]
    tickers.sort(key=lambda t: len(dfs[t]), reverse=True)
    tickers = tickers[:max_tickers]

    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = dfs[t].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if freq == "5min":
            df = trim_intraday(df, minutes=60)
        elif freq == "hourly":
            df = trim_intraday_by_bars(df, trim_minutes=60)
        if not df.empty:
            out[t] = df
    return out

# %% [markdown]
# ### Window extraction
#
# We use the same definition as the other notebooks: for each detected jump at timestamp `t0`,
# we extract a centered window and normalize by `f(t0)*sigma(t0)` (constant over the window),
# then align sign so x(0) > 0.

# %%
def extract_windows(
    dfs: Dict[str, pd.DataFrame],
    jumps_df: pd.DataFrame,
    window_steps: int,
    max_windows: int,
    seed: int = 0,
) -> np.ndarray:
    if jumps_df is None or jumps_df.empty:
        return np.empty((0, 2 * window_steps + 1))

    rng = np.random.default_rng(seed)
    idxs = np.arange(len(jumps_df))
    if len(idxs) > max_windows:
        idxs = rng.choice(idxs, size=max_windows, replace=False)
        idxs = np.sort(idxs)
        jumps_df = jumps_df.iloc[idxs].reset_index(drop=True)

    windows: List[np.ndarray] = []
    center = window_steps

    for _, row in jumps_df.iterrows():
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

    if not windows:
        return np.empty((0, 2 * window_steps + 1))
    return np.asarray(windows, dtype=float)

# %% [markdown]
# ### Compute D1/D2/D3 scores

# %%
def compute_scores(
    X: np.ndarray,
    include_ss: bool,
    score_mode: ScoreMode,
    J: int,
    n_components: int,
    seed: int = 0,
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
    d1 = emb[:, 0].copy()

    # Orient D1: positive correlates with post-jump activity > pre-jump activity
    act_post = np.sum(np.abs(X[:, center + 1 :]), axis=1)
    act_pre = np.sum(np.abs(X[:, :center]), axis=1)
    asym = (act_post - act_pre) / (act_post + act_pre + 1e-9)
    corr = np.corrcoef(d1, asym)[0, 1] if len(d1) > 1 else 1.0
    if np.isfinite(corr) and corr < 0:
        d1 *= -1

    if score_mode == "kpca":
        if emb.shape[1] < 3:
            raise ValueError("Need n_components>=3 for kpca mode")
        d2 = emb[:, 1]
        d3 = emb[:, 2]
    else:
        x_pre = X[:, center - 1]
        x_post = X[:, center + 1]
        d2 = x_pre - x_post
        d3 = x_pre + x_post

    return {"D1": np.asarray(d1), "D2": np.asarray(d2), "D3": np.asarray(d3)}

# %% [markdown]
# ### Run + plot distributions

# %%
rows: List[Dict] = []

for freq in ("5min", "hourly", "daily"):
    dfs = load_poland(freq, min_len=CFG.min_len, max_tickers=CFG.max_tickers)
    print(f"{freq}: loaded {len(dfs)} tickers")

    thr = threshold_for_freq(freq)
    jumps_df = detect_jumps_many(dfs, threshold=thr)
    print(f"{freq}: detected {len(jumps_df)} jumps (threshold={thr})")

    w = CFG.window_steps_daily if freq == "daily" else CFG.window_steps_intraday
    X = extract_windows(dfs, jumps_df, window_steps=w, max_windows=CFG.max_windows, seed=0)
    print(f"{freq}: extracted {len(X)} windows (w={w})")

    for include_ss in (False, True):
        for mode in ("kpca", "handcrafted"):
            scores = compute_scores(X, include_ss=include_ss, score_mode=mode, J=CFG.J, n_components=CFG.n_components, seed=0)
            for dname in ("D1", "D2", "D3"):
                vals = scores[dname]
                if vals.size == 0:
                    continue
                for v in vals:
                    rows.append(
                        {
                            "freq": freq,
                            "include_ss": include_ss,
                            "mode": mode,
                            "direction": dname,
                            "value": float(v),
                            "config": f"{freq}|{'SS' if include_ss else 'noSS'}|{mode}",
                        }
                    )

df_scores = pd.DataFrame(rows)
print("Total score rows:", len(df_scores))

df_scores.to_csv(out_dir() / "scores_long.csv", index=False)
df_scores["abs_value"] = df_scores["value"].abs()

# %% [markdown]
# #### Histograms (overlay by config, faceted by frequency)

# %%
def plot_hist(direction: Literal["D1", "D2", "D3"]) -> None:
    sub = df_scores[df_scores["direction"] == direction].copy()
    if sub.empty:
        print(f"No data for {direction}")
        return

    fig = px.histogram(
        sub,
        x="value",
        color="config",
        facet_col="freq",
        barmode="overlay",
        opacity=0.45,
        nbins=CFG.nbins,
        title=f"Distribution of {direction} (reflexivity/mean-reversion/trend) across configs",
        category_orders={"freq": ["5min", "hourly", "daily"]},
    )
    fig.update_layout(template="plotly_white", legend_title_text="config")
    fig.update_xaxes(matches=None)
    fig.write_html(out_dir() / f"hist_{direction}.html")
    if CFG.show_plots:
        fig.show()


for direction in ("D1", "D2", "D3"):
    plot_hist(direction)


# %% [markdown]
# #### Log-log plots (tail diagnostics)
#
# We plot:
# - histogram of **|D|** with log-x and log-y
# - CCDF of **|D|** (log-log), which is usually the cleanest view of tail behavior

# %%
def _ccdf_points(x: np.ndarray, max_points: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    n = x.size
    k = min(max_points, n)
    idx = np.unique(np.linspace(0, n - 1, k).astype(int))
    xv = x[idx]
    ccdf = (n - idx) / n
    return xv, ccdf


def plot_abs_hist_loglog(direction: Literal["D1", "D2", "D3"]) -> None:
    sub = df_scores[(df_scores["direction"] == direction) & (df_scores["abs_value"] > 0)].copy()
    if sub.empty:
        print(f"No abs-value data for {direction}")
        return
    fig = px.histogram(
        sub,
        x="abs_value",
        color="config",
        facet_col="freq",
        barmode="overlay",
        opacity=0.45,
        nbins=CFG.nbins,
        title=f"Log-log histogram of |{direction}| across configs",
        category_orders={"freq": ["5min", "hourly", "daily"]},
        log_y=True,
    )
    fig.update_layout(template="plotly_white", legend_title_text="config")
    fig.update_xaxes(type="log", matches=None, title_text=f"|{direction}|")
    fig.update_yaxes(type="log", title_text="count")
    fig.write_html(out_dir() / f"hist_abs_loglog_{direction}.html")
    if CFG.show_plots:
        fig.show()


def plot_ccdf_loglog(direction: Literal["D1", "D2", "D3"]) -> None:
    sub = df_scores[(df_scores["direction"] == direction) & (df_scores["abs_value"] > 0)].copy()
    if sub.empty:
        print(f"No abs-value data for {direction}")
        return

    rows_ccdf: List[Dict] = []
    for (freq, config), g in sub.groupby(["freq", "config"], sort=False):
        xv, cc = _ccdf_points(g["abs_value"].to_numpy(dtype=float), max_points=500)
        for x_i, c_i in zip(xv, cc):
            rows_ccdf.append({"freq": freq, "config": config, "abs_value": float(x_i), "ccdf": float(c_i)})

    ccdf_df = pd.DataFrame(rows_ccdf)
    if ccdf_df.empty:
        print(f"No CCDF points for {direction}")
        return

    fig = px.line(
        ccdf_df,
        x="abs_value",
        y="ccdf",
        color="config",
        facet_col="freq",
        title=f"Log-log CCDF of |{direction}| across configs",
        category_orders={"freq": ["5min", "hourly", "daily"]},
        log_x=True,
        log_y=True,
    )
    fig.update_layout(template="plotly_white", legend_title_text="config")
    fig.update_xaxes(matches=None, title_text=f"|{direction}|")
    fig.update_yaxes(title_text="P(|D| > x)")
    fig.write_html(out_dir() / f"ccdf_loglog_{direction}.html")
    if CFG.show_plots:
        fig.show()


for direction in ("D1", "D2", "D3"):
    plot_abs_hist_loglog(direction)
    plot_ccdf_loglog(direction)


# %% [markdown]
# #### Stacked log-log CCDF across frequencies (3 colors)
#
# This gives a direct comparison **daily vs hourly vs 5-min** on the same axes.
# We plot the CCDF of **|D|** for the baseline config: **noSS + kpca**.

# %%
def plot_ccdf_loglog_stacked_by_freq(direction: Literal["D1", "D2", "D3"], include_ss: bool = False, mode: ScoreMode = "kpca") -> None:
    sub = df_scores[
        (df_scores["direction"] == direction)
        & (df_scores["include_ss"] == include_ss)
        & (df_scores["mode"] == mode)
        & (df_scores["abs_value"] > 0)
    ].copy()
    if sub.empty:
        print(f"No data for stacked CCDF {direction}")
        return

    palette = {"5min": "#636EFA", "hourly": "#EF553B", "daily": "#00CC96"}
    fig = px.line(
        title=f"Stacked log-log CCDF of |{direction}| (noSS, {mode})",
    )
    # manual build to guarantee exactly 3 colors by freq
    import plotly.graph_objects as go

    for freq in ("5min", "hourly", "daily"):
        g = sub[sub["freq"] == freq]
        xv, cc = _ccdf_points(g["abs_value"].to_numpy(dtype=float), max_points=800)
        if xv.size == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=xv,
                y=cc,
                mode="lines",
                name=f"{freq}",
                line=dict(width=2, color=palette[freq]),
            )
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title=f"|{direction}|",
        yaxis_title="P(|D| > x)",
    )
    fig.write_html(out_dir() / f"ccdf_loglog_STACKED_{direction}_noSS_{mode}.html")
    if CFG.show_plots:
        fig.show()


for direction in ("D1", "D2", "D3"):
    plot_ccdf_loglog_stacked_by_freq(direction, include_ss=False, mode="kpca")

