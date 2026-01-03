#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Dataset presentation: curves + distributions of r(t) and sigma(t)
#
# We reuse the project definition from `utils/data/jump_detection.py`:
#
# - r(t) = pct_change(close)
# - sigma(t) = EWM std of deseasonalized returns (with frequency-aware span)
#
# We produce:
# - time-series curves of r(t) and sigma(t) for a representative ticker per frequency
# - pooled distributions of r(t) and sigma(t) across tickers for each frequency (5-min / hourly / daily)

# %%
from __future__ import annotations

from dataclasses import dataclass  # we import dataclass for configuration
from pathlib import Path  # we import path handling
from typing import Dict, List, Literal, Optional, Tuple  # we import strict typing

import numpy as np  # we import numpy
import pandas as pd  # we import pandas
import plotly.express as px  # we import plotly express for histograms
import plotly.graph_objects as go  # we import plotly graph objects for custom plots
from plotly.subplots import make_subplots  # we import subplot helper

from utils.data.curating_stooq import curate_stooq_dir_5min, curate_stooq_dir_daily, curate_stooq_dir_hourly  # we import curated loaders
from utils.data.jump_detection import compute_jump_score  # we import r(t), sigma(t) definition

Freq = Literal["5min", "hourly", "daily"]  # we define supported frequencies


@dataclass(frozen=True)
class DatasetPresentationConfig:
    min_len: int = 800  # we require a minimum length per ticker for stable estimates
    max_tickers: int = 80  # we cap the number of tickers to keep runtime reasonable
    trim_intraday_minutes: int = 60  # we remove open/close bars for intraday data
    max_points_per_freq: int = 250_000  # we cap pooled points per frequency for histograms
    ts_points: int = 2_500  # we plot only the last ts_points for legibility
    use_sequential_index: bool = True  # we plot against a sequential index to avoid overnight gaps in intraday series
    random_seed: int = 0  # we keep sampling deterministic
    outputs_subdir: str = "notebooks/dataset_presentation/outputs"  # we store html outputs here


CFG = DatasetPresentationConfig()  # we instantiate configuration


def project_root() -> Path:
    try:
        here = Path(__file__).resolve()  # we locate file path when executed as a script
        return here.parents[2]  # we map notebooks/dataset_presentation/dataset.py -> repo root
    except NameError:
        cwd = Path.cwd().resolve()  # we fall back to notebook cwd
        for p in [cwd, *cwd.parents]:  # we search upward for repo markers
            if (p / "utils").is_dir() and (p / "data").is_dir():
                return p  # we return the first plausible repo root
        return cwd  # we fall back to cwd


def out_dir() -> Path:
    d = project_root() / CFG.outputs_subdir  # we build output directory
    d.mkdir(parents=True, exist_ok=True)  # we ensure directory exists
    return d  # we return output directory


def data_dir(freq: Freq) -> Path:
    root = project_root()  # we locate repo root
    if freq == "5min":
        return root / "data" / "stooq" / "poland" / "5_min" / "pl" / "wsestocks"  # we follow existing notebook conventions
    if freq == "hourly":
        return root / "data" / "stooq" / "poland" / "hourly" / "ncstocks"  # we follow existing notebook conventions
    if freq == "daily":
        return root / "data" / "stooq" / "poland" / "daily" / "ncstocks"  # we follow existing notebook conventions
    raise ValueError(f"unknown freq: {freq}")  # we validate frequency


def _infer_bar_timedelta(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:  # we guard for invalid index
        return None  # we cannot infer
    idx = index.sort_values()  # we sort for stable diffs
    deltas = idx.to_series().diff().dropna()  # we compute deltas
    deltas = deltas[deltas > pd.Timedelta(0)]  # we keep positive deltas
    if deltas.empty:  # we handle degenerate case
        return None  # we cannot infer
    td = deltas.median()  # we use median for robustness
    if pd.isna(td) or td <= pd.Timedelta(0):  # we validate timedelta
        return None  # we cannot infer
    return td  # we return representative bar size


def trim_intraday(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex) or minutes <= 0:  # we validate inputs
        return df  # we do nothing
    days: List[pd.DataFrame] = []  # we collect day slices
    for _, day_df in df.groupby(df.index.date):  # we split by day
        day_df = day_df.sort_index()  # we sort within day
        if day_df.empty:  # we skip empty days
            continue  # we continue
        start = day_df.index[0] + pd.Timedelta(minutes=minutes)  # we trim start by minutes
        end = day_df.index[-1] - pd.Timedelta(minutes=minutes)  # we trim end by minutes
        mask = (day_df.index >= start) & (day_df.index <= end)  # we build inclusion mask
        if mask.any():  # we keep only if something remains
            days.append(day_df.loc[mask])  # we append trimmed day
    if not days:  # we handle case where trimming removed all data
        return df.iloc[0:0]  # we return empty
    out = pd.concat(days).sort_index()  # we concatenate and sort
    out = out[~out.index.duplicated(keep="last")]  # we de-duplicate
    return out  # we return trimmed dataframe


def trim_intraday_by_bars(df: pd.DataFrame, trim_minutes: int) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex) or trim_minutes <= 0:  # we validate inputs
        return df  # we do nothing
    bar_td = _infer_bar_timedelta(df.index)  # we infer bar size
    if bar_td is None or bar_td >= pd.Timedelta(days=1):  # we skip daily-like data
        return df  # we return unchanged
    trim_bars = max(1, int(round(pd.Timedelta(minutes=trim_minutes) / bar_td)))  # we compute how many bars to trim
    days: List[pd.DataFrame] = []  # we collect day slices
    for _, day_df in df.groupby(df.index.date):  # we split by day
        day_df = day_df.sort_index()  # we sort within day
        if len(day_df) <= (2 * trim_bars + 1):  # we ensure enough bars remain
            continue  # we skip tiny days
        days.append(day_df.iloc[trim_bars:-trim_bars])  # we trim by bar count
    if not days:  # we handle empty result
        return df.iloc[0:0]  # we return empty
    out = pd.concat(days).sort_index()  # we concatenate and sort
    out = out[~out.index.duplicated(keep="last")]  # we de-duplicate
    return out  # we return trimmed dataframe


def load_freq(freq: Freq, min_len: int, max_tickers: int) -> Dict[str, pd.DataFrame]:
    d = data_dir(freq)  # we locate the data directory
    if not d.exists():  # we validate existence
        raise FileNotFoundError(f"data directory not found: {d}")  # we raise clear error
    if freq == "5min":
        dfs = curate_stooq_dir_5min(str(d), pattern="*.txt", recursive=True)  # we load 5min data
    elif freq == "hourly":
        dfs = curate_stooq_dir_hourly(str(d), pattern="*.txt", recursive=True)  # we load hourly data
    elif freq == "daily":
        dfs = curate_stooq_dir_daily(str(d), pattern="*.txt", recursive=True)  # we load daily data
    else:
        raise ValueError(f"unknown freq: {freq}")  # we validate frequency
    tickers = [t for t, df in dfs.items() if df is not None and not df.empty and len(df) >= min_len]  # we filter tickers
    tickers.sort(key=lambda t: len(dfs[t]), reverse=True)  # we sort by length
    tickers = tickers[:max_tickers]  # we cap count
    out: Dict[str, pd.DataFrame] = {}  # we store cleaned dfs
    for t in tickers:  # we iterate selected tickers
        df = dfs[t].sort_index()  # we sort by time
        df = df[~df.index.duplicated(keep="last")]  # we de-duplicate
        if freq == "5min":
            df = trim_intraday(df, minutes=CFG.trim_intraday_minutes)  # we trim open/close minutes
        elif freq == "hourly":
            df = trim_intraday_by_bars(df, trim_minutes=CFG.trim_intraday_minutes)  # we trim by bars for irregular hourly
        if not df.empty:  # we keep only non-empty
            out[t] = df  # we store
    return out  # we return mapping


def representative_ticker(dfs: Dict[str, pd.DataFrame]) -> str:
    if not dfs:  # we validate input
        raise ValueError("no tickers loaded")  # we raise clear error
    t = max(dfs.keys(), key=lambda k: len(dfs[k]))  # we pick longest series
    return t  # we return ticker


def _finite_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")  # we coerce numeric
    s = s.replace([np.inf, -np.inf], np.nan).dropna()  # we drop non-finite
    return s  # we return cleaned series


def plot_curves_r_sigma(freq: Freq, df: pd.DataFrame, ticker: str, ts_points: int) -> go.Figure:
    if df.empty:  # we validate input
        raise ValueError("empty dataframe")  # we raise clear error
    scores = compute_jump_score(df, price_col="close")  # we compute r and sigma
    if scores.empty:  # we validate output
        raise ValueError("could not compute r/sigma")  # we raise clear error
    r = _finite_series(scores["return"])  # we extract r(t)
    sigma = _finite_series(scores["sigma"])  # we extract sigma(t)
    if r.empty or sigma.empty:  # we validate series
        raise ValueError("r or sigma series is empty after cleaning")  # we raise clear error
    r = r.iloc[-ts_points:] if len(r) > ts_points else r  # we select tail window
    sigma = sigma.loc[r.index]  # we align sigma to r index
    x_axis = r.index  # we default to datetime x axis
    x_label = "time"  # we label x axis
    if CFG.use_sequential_index:  # we avoid discontinuities due to overnight gaps by using a dense index
        x_axis = np.arange(len(r), dtype=int)  # we build a sequential index
        x_label = "bar index"  # we label x axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])  # we build a dual-axis figure
    fig.add_trace(go.Scatter(x=x_axis, y=r.values, name="r(t) = pct_change(close)", line=dict(color="#1f77b4")), secondary_y=False)  # we plot returns
    fig.add_trace(go.Scatter(x=x_axis, y=sigma.values, name="sigma(t) = EWM std (deseasonalized)", line=dict(color="#d62728")), secondary_y=True)  # we plot sigma
    fig.update_layout(
        title=f"{freq}: curves of r(t) and sigma(t) for {ticker}",
        xaxis_title=x_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=60, t=70, b=60),
        template="plotly_white",
        height=520,
    )  # we set plot layout
    fig.update_yaxes(title_text="return r(t)", secondary_y=False)  # we label left axis
    fig.update_yaxes(title_text="sigma(t)", secondary_y=True)  # we label right axis
    return fig  # we return figure


def pooled_r_sigma(dfs: Dict[str, pd.DataFrame], max_points: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))  # we create rng
    r_all: List[np.ndarray] = []  # we collect returns arrays
    s_all: List[np.ndarray] = []  # we collect sigma arrays
    for _, df in dfs.items():  # we iterate tickers
        if df.empty or "close" not in df.columns:  # we validate required column
            continue  # we skip invalid
        scores = compute_jump_score(df, price_col="close")  # we compute r and sigma
        if scores.empty:  # we skip if failed
            continue  # we continue
        r = scores["return"].to_numpy(dtype=float, copy=False)  # we extract r
        s = scores["sigma"].to_numpy(dtype=float, copy=False)  # we extract sigma
        mask = np.isfinite(r) & np.isfinite(s)  # we keep finite pairs
        r = r[mask]  # we filter returns
        s = s[mask]  # we filter sigma
        if r.size == 0:  # we skip empty
            continue  # we continue
        r_all.append(r)  # we collect
        s_all.append(s)  # we collect
    if not r_all or not s_all:  # we validate pooled arrays
        return np.asarray([], dtype=float), np.asarray([], dtype=float)  # we return empty arrays
    r_cat = np.concatenate(r_all, axis=0)  # we concatenate
    s_cat = np.concatenate(s_all, axis=0)  # we concatenate
    n = int(min(max_points, r_cat.size))  # we cap points
    if n <= 0:  # we guard
        return np.asarray([], dtype=float), np.asarray([], dtype=float)  # we return empty arrays
    if r_cat.size > n:  # we subsample without replacement
        idx = rng.choice(r_cat.size, size=n, replace=False)  # we sample indices
        r_cat = r_cat[idx]  # we subsample returns
        s_cat = s_cat[idx]  # we subsample sigma
    return r_cat, s_cat  # we return pooled samples


def plot_distributions_r_sigma(pooled: Dict[Freq, Tuple[np.ndarray, np.ndarray]]) -> go.Figure:
    rows: List[Dict[str, object]] = []  # we build a long-form table for plotly
    for freq, (r, s) in pooled.items():  # we iterate pooled samples
        for v in r:  # we add return values
            rows.append({"freq": freq, "variable": "return", "value": float(v)})  # we add one row
        for v in s:  # we add sigma values
            rows.append({"freq": freq, "variable": "sigma", "value": float(v)})  # we add one row
    df_long = pd.DataFrame(rows)  # we create dataframe
    if df_long.empty:  # we validate
        raise ValueError("no pooled points available for distributions")  # we raise clear error
    fig = px.histogram(
        df_long,
        x="value",
        color="freq",
        facet_row="variable",
        nbins=200,
        opacity=0.75,
        barmode="overlay",
        histnorm="probability density",
        title="Distributions of r(t) and sigma(t) across 5min / hourly / daily (pooled across tickers)",
    )  # we create faceted histograms
    fig.update_layout(template="plotly_white", height=780, margin=dict(l=60, r=40, t=80, b=60))  # we set layout
    fig.update_xaxes(matches=None)  # we allow different x-scales per facet
    return fig  # we return figure


def run(show_plots: bool = False) -> None:
    np.random.seed(int(CFG.random_seed))  # we seed numpy for reproducibility
    figures_dir = out_dir()  # we ensure output directory exists
    pooled: Dict[Freq, Tuple[np.ndarray, np.ndarray]] = {}  # we store pooled samples per freq
    for freq in ("5min", "hourly", "daily"):  # we iterate frequencies
        dfs = load_freq(freq=freq, min_len=int(CFG.min_len), max_tickers=int(CFG.max_tickers))  # we load data
        if not dfs:  # we validate
            raise ValueError(f"no tickers loaded for freq={freq}")  # we raise clear error
        t = representative_ticker(dfs)  # we pick representative ticker
        fig_ts = plot_curves_r_sigma(freq=freq, df=dfs[t], ticker=t, ts_points=int(CFG.ts_points))  # we build curves plot
        fig_ts_path = figures_dir / f"curves_r_sigma_{freq}.html"  # we set output path
        fig_ts.write_html(str(fig_ts_path), include_plotlyjs="cdn")  # we save html
        if show_plots:  # we optionally display
            fig_ts.show()  # we show plot
        r_pool, s_pool = pooled_r_sigma(dfs, max_points=int(CFG.max_points_per_freq), seed=int(CFG.random_seed))  # we pool values
        pooled[freq] = (r_pool, s_pool)  # we store pooled samples
    fig_dist = plot_distributions_r_sigma(pooled)  # we build distribution plot
    fig_dist_path = figures_dir / "distributions_r_sigma_5min_hourly_daily.html"  # we set output path
    fig_dist.write_html(str(fig_dist_path), include_plotlyjs="cdn")  # we save html
    if show_plots:  # we optionally display
        fig_dist.show()  # we show plot
    print(f"saved figures to: {figures_dir}")  # we print output location


if __name__ == "__main__":
    run(show_plots=False)  # we run non-interactively by default

