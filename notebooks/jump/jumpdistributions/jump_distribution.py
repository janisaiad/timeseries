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
# ## QQ plots: jump score vs Gumbel (daily / hourly / 5-min)
#
# We compute the standardized jump score time series:
#
# \[
# x(t) = \frac{r(t)}{f(t)\,\sigma(t)}
# \]
#
# and then compare the empirical distribution of **\(|x(t)|\)** to a **Gumbel** distribution via a QQ plot.
#
# Implementation notes:
# - We use \(|x(t)|\) because Gumbel is supported on \(\mathbb{R}\) but is typically used for positive extremes;
#   the paper discusses Gumbel in the context of \(|x(t)|\) under the null.
# - We fit Gumbel parameters (loc, scale) by method-of-moments (no SciPy dependency).
# - We pool scores across multiple tickers but cap total points for speed/memory.

# %%
from __future__ import annotations

# %% [markdown]
# ### Imports

# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.data.curating_stooq import curate_stooq_dir_5min, curate_stooq_dir_hourly, curate_stooq_dir_daily
from utils.data.jump_detection import compute_jump_score

Freq = Literal["5min", "hourly", "daily"]

# %% [markdown]
# ### Config

# %%
@dataclass(frozen=True)
class QQConfig:
    min_len: int = 500
    max_tickers: int = 200
    max_points_total: int = 200_000  # pooled across tickers
    qq_points: int = 10_000          # points used in QQ plot (subsample after pooling)
    trim_intraday_minutes: int = 60  # remove open/close for intraday frequencies
    out_subdir: str = "notebooks/jump/jumpdistributions/outputs"


CFG = QQConfig()

# %% [markdown]
# ### Repo root + data dirs (notebook-safe)

# %%
def project_root() -> Path:
    try:
        here = Path(__file__).resolve()
        return here.parents[3]  # .../notebooks/jump/jumpdistributions/jump_distribution.py -> repo root
    except NameError:
        cwd = Path.cwd().resolve()
        for p in [cwd, *cwd.parents]:
            if (p / "utils").is_dir() and (p / "data").is_dir():
                return p
        return cwd


def out_dir() -> Path:
    d = project_root() / CFG.out_subdir
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
# ### Helpers: trim intraday open/close

# %%
def trim_intraday(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex) or minutes <= 0:
        return df
    days: List[pd.DataFrame] = []
    for _, day_df in df.groupby(df.index.date):
        day_df = day_df.sort_index()
        if day_df.empty:
            continue
        start = day_df.index[0] + pd.Timedelta(minutes=minutes)
        end = day_df.index[-1] - pd.Timedelta(minutes=minutes)
        mask = (day_df.index >= start) & (day_df.index <= end)
        if mask.any():
            days.append(day_df.loc[mask])
    if not days:
        return df.iloc[0:0]
    out = pd.concat(days).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out


def _infer_bar_timedelta(index: pd.DatetimeIndex) -> pd.Timedelta | None:
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
    """
    Hourly data can be irregular; trimming by absolute time can drop everything.
    Trim by number of bars per day inferred from median bar size.
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

# %% [markdown]
# ### Load data (top N tickers)

# %%
def load_poland(freq: Freq, min_len: int, max_tickers: int) -> Dict[str, pd.DataFrame]:
    d = data_dir(freq)
    if not d.exists():
        raise FileNotFoundError(f"Data dir not found: {d}")

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
            df = trim_intraday(df, minutes=CFG.trim_intraday_minutes)
        elif freq == "hourly":
            df = trim_intraday_by_bars(df, trim_minutes=CFG.trim_intraday_minutes)
        if not df.empty:
            out[t] = df
    return out

# %% [markdown]
# ### Gumbel fit (method of moments) + QQ plot

# %%
_EULER_GAMMA = 0.5772156649015329


def fit_gumbel_mom(x: np.ndarray) -> Tuple[float, float]:
    """
    Fit Gumbel(loc, scale) by method-of-moments:
      mean = loc + gamma*scale
      std  = (pi/sqrt(6))*scale
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("Not enough points to fit gumbel")
    m = float(np.mean(x))
    s = float(np.std(x, ddof=0))
    scale = s * np.sqrt(6.0) / np.pi if s > 0 else 1.0
    loc = m - _EULER_GAMMA * scale
    return loc, scale


def gumbel_ppf(p: np.ndarray, loc: float, scale: float) -> np.ndarray:
    """
    Quantile function for Gumbel (right) distribution.
    """
    p = np.asarray(p, dtype=float)
    # clamp to avoid inf
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return loc - scale * np.log(-np.log(p))


def gumbel_ppf_standard(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -np.log(-np.log(p))


def qq_data_abs_score_vs_gumbel(abs_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Return (q_std, empirical_sorted, loc, scale) where:
      q_std = standard gumbel quantiles -log(-log(p))
      empirical_sorted = sorted |x(t)| samples
    """
    x = np.asarray(abs_scores, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    if x.size == 0:
        raise ValueError("No finite abs scores")

    # subsample for QQ
    if x.size > CFG.qq_points:
        rng = np.random.default_rng(0)
        x = rng.choice(x, size=CFG.qq_points, replace=False)
    x = np.sort(x)
    n = x.size
    p = (np.arange(1, n + 1) - 0.5) / n

    loc, scale = fit_gumbel_mom(x)
    q_std = gumbel_ppf_standard(p)
    return q_std, x, loc, scale


def qq_plot_abs_score_vs_gumbel(abs_scores: np.ndarray, title: str) -> go.Figure:
    q_std, x, loc, scale = qq_data_abs_score_vs_gumbel(abs_scores)

    # fitted line in (q_std, x) space
    y_fit = loc + scale * q_std

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_std, y=x, mode="markers", name="QQ points", marker=dict(size=4, opacity=0.6)))
    fig.add_trace(go.Scatter(x=q_std, y=y_fit, mode="lines", name="fit: loc + scale*q", line=dict(color="black", dash="dash")))
    fig.update_layout(
        title=title + f"<br>Gumbel fit: loc={loc:.3f}, scale={scale:.3f}, n={n}",
        xaxis_title="Theoretical quantiles (standard Gumbel)",
        yaxis_title="Empirical quantiles (|x(t)|)",
        template="plotly_white",
        hovermode="closest",
    )
    return fig

# %% [markdown]
# ### Run for daily / hourly / 5-min

# %%
def pooled_abs_scores(freq: Freq) -> np.ndarray:
    dfs = load_poland(freq, min_len=CFG.min_len, max_tickers=CFG.max_tickers)
    print(f"{freq}: loaded {len(dfs)} tickers")
    pooled: List[np.ndarray] = []
    total = 0

    # deterministic order (longest first)
    tickers = sorted(dfs.keys(), key=lambda t: len(dfs[t]), reverse=True)
    for t in tickers:
        df = dfs[t]
        s = compute_jump_score(df, price_col="close")["score"].to_numpy(dtype=float)
        s = np.abs(s)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        # cap per ticker to keep pool balanced
        cap = min(s.size, max(1000, CFG.max_points_total // max(1, len(tickers))))
        if s.size > cap:
            rng = np.random.default_rng(0)
            s = rng.choice(s, size=cap, replace=False)
        pooled.append(s)
        total += s.size
        if total >= CFG.max_points_total:
            break

    if not pooled:
        return np.array([])

    x = np.concatenate(pooled)
    if x.size > CFG.max_points_total:
        rng = np.random.default_rng(0)
        x = rng.choice(x, size=CFG.max_points_total, replace=False)
    return x


for freq in ("daily", "hourly", "5min"):
    abs_scores = pooled_abs_scores(freq)  # pooled |x(t)|
    if abs_scores.size == 0:
        print(f"{freq}: no scores, skipping")
        continue

    fig = qq_plot_abs_score_vs_gumbel(
        abs_scores,
        title=f"QQ plot: |x(t)| vs Gumbel ({freq})",
    )
    out_path = out_dir() / f"qq_abs_score_vs_gumbel_{freq}.html"
    fig.write_html(out_path)
    fig.show()
    print(f"Saved: {out_path}")


# %% [markdown]
# ### Stacked overlays (3 colors): QQ + log-log tail across frequencies

# %%
def ccdf_points(x: np.ndarray, max_points: int = 600) -> tuple[np.ndarray, np.ndarray]:
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


abs_by_freq: dict[Freq, np.ndarray] = {}
for freq in ("5min", "hourly", "daily"):
    abs_by_freq[freq] = pooled_abs_scores(freq)

palette = {"5min": "#636EFA", "hourly": "#EF553B", "daily": "#00CC96"}

# QQ overlay: standard-gumbel quantiles on x-axis for all freqs
fig_qq = go.Figure()
for freq in ("5min", "hourly", "daily"):
    x = abs_by_freq[freq]
    if x.size == 0:
        continue
    q_std, emp, loc, scale = qq_data_abs_score_vs_gumbel(x)
    # subsample points for plotting density
    if emp.size > 4000:
        take = np.unique(np.linspace(0, emp.size - 1, 4000).astype(int))
        q_std_p = q_std[take]
        emp_p = emp[take]
    else:
        q_std_p, emp_p = q_std, emp
    fig_qq.add_trace(
        go.Scatter(
            x=q_std_p,
            y=emp_p,
            mode="markers",
            name=f"{freq}",
            marker=dict(size=4, opacity=0.5, color=palette[freq]),
        )
    )
fig_qq.update_layout(
    title="Stacked QQ: |x(t)| vs Gumbel (3 freqs)",
    xaxis_title="Theoretical quantiles (standard Gumbel)",
    yaxis_title="Empirical quantiles (|x(t)|)",
    template="plotly_white",
)
fig_qq.write_html(out_dir() / "qq_abs_score_vs_gumbel_STACKED.html")
fig_qq.show()

# Log-log CCDF overlay of |x(t)|
fig_ccdf = go.Figure()
for freq in ("5min", "hourly", "daily"):
    x = abs_by_freq[freq]
    xv, cc = ccdf_points(x, max_points=800)
    if xv.size == 0:
        continue
    fig_ccdf.add_trace(
        go.Scatter(
            x=xv,
            y=cc,
            mode="lines",
            name=f"{freq}",
            line=dict(width=2, color=palette[freq]),
        )
    )
fig_ccdf.update_layout(
    title="Stacked log-log tail: CCDF of |x(t)| (3 freqs)",
    xaxis_title="|x(t)|",
    yaxis_title="P(|x| > x)",
    template="plotly_white",
    xaxis_type="log",
    yaxis_type="log",
)
fig_ccdf.write_html(out_dir() / "score_ccdf_loglog_STACKED.html")
fig_ccdf.show()


