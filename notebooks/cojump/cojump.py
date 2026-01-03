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
# ## Co-jump analysis (paper-style metrics + plots)
#
# A **co-jump** (as in `refs/texsources/ridingwavelets/main.tex`, Section "Classification of co-jumps") means:
# multiple tickers have a detected jump whose timestamps fall in the **same time bin**.
# The **size** \(S\) is the number of assets in that bin.
#
# We reproduce the key diagnostics from the paper section:
# - size distribution (hist + CCDF in log-log)
# - sign alignment (average sign vs size)
# - co-jump indicators from the per-jump reflexivity score \(D_1\): mean/max/min across constituents
# - correlation metric \(\rho\) based on the per-jump trend score \(D_3\) (Appendix cojump-correlation)
#
# Notes:
# - If your underlying data is 5-minute bars, then grouping by `1min` vs `5min` will often be identical
#   because your bar timestamps are already multiples of 5 minutes.
# - This script is notebook-friendly (Jupytext). Run cell-by-cell.

# %% [markdown]
# ### Imports

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio  # we configure plotly rendering in notebooks

from utils.data.curating_stooq import curate_stooq_dir_5min, curate_stooq_dir_hourly, curate_stooq_dir_daily
from utils.data.jump_detection import detect_jumps_many
from model.wavelet.wavelet import WaveletModel

# %% [markdown]
# ### Config

# %%
BaseFreq = Literal["5min", "hourly", "daily"]
ScoreMode = Literal["kpca", "handcrafted"]


@dataclass(frozen=True)
class CojumpConfig:
    base_freq: BaseFreq = "5min"
    threshold_5min: float = 4.0
    threshold_hourly: float = 2.0
    threshold_daily: float = 2.5
    min_len: int = 500
    max_tickers: int = 200
    min_cojump_size: int = 2
    # intraday trimming (only used for 5min/hourly)
    trim_intraday_minutes: int = 60
    # which cojump bin widths to compute
    bins: Tuple[str, ...] = ("1min", "5min", "1h", "1D")
    # jump-profile window (in bars, per side)
    window_steps_intraday: int = 12
    window_steps_daily: int = 20
    max_jumps_total: int = 200_000
    max_windows_total: int = 30_000
    random_seed: int = 0
    # wavelet embedding / scoring (D1/D2/D3)
    J: int = 3
    n_components: int = 3
    include_scattering_spectra: bool = False
    score_mode: ScoreMode = "kpca"
    # plotting
    show_plots: bool = True
    show_only_profile_plots: bool = True  # we avoid spamming the notebook with all plots when showing inline
    # cojump profile plotting
    profile_bin: str = "5min"  # we plot profiles for this bin (use a bin that matches the base bar timestamps)
    top_k_profiles: int = 5  # we plot the top-k largest cojumps
    profile_window_steps: int = 12  # we plot ±window steps around the cojump timestamp
    max_tickers_per_profile: int = 35  # we cap plotted tickers per cojump to keep plots readable


CFG = CojumpConfig()


def threshold_for_freq(base_freq: BaseFreq) -> float:
    if base_freq == "5min":
        return float(CFG.threshold_5min)
    if base_freq == "hourly":
        return float(CFG.threshold_hourly)
    if base_freq == "daily":
        return float(CFG.threshold_daily)
    raise ValueError(base_freq)


def configure_plotly_renderer() -> None:
    """
    we ensure plotly renders inline in notebook contexts (vscode/cursor)  # we avoid browser-only rendering in wsl
    """
    candidates = ["vscode", "plotly_mimetype", "notebook_connected", "notebook"]  # we try notebook-friendly renderers first
    available = set(getattr(pio, "renderers", {}).keys())  # we list available renderers
    for r in candidates:
        if r in available:
            pio.renderers.default = r  # we set the renderer
            return
    if "plotly_mimetype" in available:  # we fall back to mimetype renderer if nothing else matched
        pio.renderers.default = "plotly_mimetype"

# %% [markdown]
# ### Paths

# %%

def project_root() -> Path:
    """
    Return repository root.

    - When run as a script, __file__ is defined and we can resolve relative to it.
    - In Jupyter notebooks, __file__ is often undefined; we fall back to Path.cwd()
      and walk upwards until we find a directory that looks like the repo root.
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


def poland_data_dir(base_freq: BaseFreq) -> Path:
    root = project_root()
    if base_freq == "5min":
        return root / "data" / "stooq" / "poland" / "5_min" / "pl" / "wsestocks"
    if base_freq == "hourly":
        return root / "data" / "stooq" / "poland" / "hourly" / "ncstocks"
    if base_freq == "daily":
        return root / "data" / "stooq" / "poland" / "daily" / "ncstocks"
    raise ValueError(f"Unknown base_freq: {base_freq}")


def out_dir() -> Path:
    d = project_root() / "notebooks" / "cojump" / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d

# %% [markdown]
# ### Load + preprocess

# %%

def _trim_intraday(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Remove first/last `minutes` of each day (intraday only)."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
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


def _trim_intraday_by_bars(df: pd.DataFrame, trim_minutes: int) -> pd.DataFrame:
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


def load_poland(base_freq: BaseFreq, min_len: int, max_tickers: int) -> Dict[str, pd.DataFrame]:
    data_dir = poland_data_dir(base_freq)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if base_freq == "5min":
        dfs = curate_stooq_dir_5min(str(data_dir), pattern="*.txt", recursive=True)
    elif base_freq == "hourly":
        dfs = curate_stooq_dir_hourly(str(data_dir), pattern="*.txt", recursive=True)
    elif base_freq == "daily":
        dfs = curate_stooq_dir_daily(str(data_dir), pattern="*.txt", recursive=True)
    else:
        raise ValueError(f"Unknown base_freq: {base_freq}")

    tickers = [t for t, d in dfs.items() if d is not None and not d.empty and len(d) >= min_len]
    tickers.sort(key=lambda t: len(dfs[t]), reverse=True)
    tickers = tickers[:max_tickers]
    return {t: dfs[t] for t in tickers}


def preprocess(base_freq: BaseFreq, dfs: Dict[str, pd.DataFrame], trim_minutes: int) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for t, df in dfs.items():
        if df is None or df.empty:
            continue
        d = df.sort_index()
        d = d[~d.index.duplicated(keep="last")]
        if base_freq in ("5min", "hourly") and trim_minutes > 0:
            if base_freq == "hourly":
                d = _trim_intraday_by_bars(d, trim_minutes=trim_minutes)
            else:
                d = _trim_intraday(d, minutes=trim_minutes)
        if not d.empty:
            out[t] = d
    return out

# %% [markdown]
# ### Co-jump grouping

# %%

def group_cojumps(
    jumps_df: pd.DataFrame,
    bin_freq: str,
    min_size: int = 2,
) -> pd.DataFrame:
    """
    Group jump events into co-jumps using a time bin (floor).

    Expected `jumps_df` columns: timestamp, ticker, score, return (others optional)
    Returns a dataframe with:
      - bin: start timestamp of the time bucket
      - size: number of tickers jumping in that bin
      - tickers: list[str]
      - scores: list[float]
      - returns: list[float]
    """
    if jumps_df is None or jumps_df.empty:
        return pd.DataFrame()

    if "timestamp" not in jumps_df.columns:
        raise ValueError("jumps_df must have a 'timestamp' column")

    j = jumps_df.copy()
    j["bin"] = pd.to_datetime(j["timestamp"]).dt.floor(bin_freq)

    cols: List[str] = ["ticker", "score", "return"]  # we aggregate these core columns
    for c in ("D1", "D2", "D3"):
        if c in j.columns:
            cols.append(c)
    agg_dict = {  # we build per-column aggregations
        "tickers": ("ticker", lambda x: list(x)),
        "scores": ("score", lambda x: list(x)),
        "returns": ("return", lambda x: list(x)),
    }
    if "D1" in cols:
        agg_dict["D1s"] = ("D1", lambda x: list(x))
    if "D2" in cols:
        agg_dict["D2s"] = ("D2", lambda x: list(x))
    if "D3" in cols:
        agg_dict["D3s"] = ("D3", lambda x: list(x))
    agg = j.groupby("bin").agg(**agg_dict).reset_index()
    agg["size"] = agg["tickers"].apply(len)
    agg = agg.loc[agg["size"] >= min_size].sort_values("bin").reset_index(drop=True)
    return agg


def window_steps_for_freq(base_freq: BaseFreq) -> int:
    if base_freq in ("5min", "hourly"):
        return int(CFG.window_steps_intraday)
    if base_freq == "daily":
        return int(CFG.window_steps_daily)
    raise ValueError(base_freq)


def extract_windows(
    dfs: Dict[str, pd.DataFrame],
    jumps_df: pd.DataFrame,
    window_steps: int,
    max_windows: int,
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    if jumps_df is None or jumps_df.empty:
        return np.empty((0, 2 * window_steps + 1)), pd.DataFrame()
    if "timestamp" not in jumps_df.columns or "ticker" not in jumps_df.columns:
        raise ValueError("jumps_df must have columns: timestamp, ticker")
    rng = np.random.default_rng(int(seed))
    j = jumps_df.copy()
    if len(j) > int(max_windows):
        idxs = rng.choice(len(j), size=int(max_windows), replace=False)
        idxs = np.sort(idxs)
        j = j.iloc[idxs].reset_index(drop=True)
    windows: List[np.ndarray] = []
    valid_rows: List[int] = []
    center = int(window_steps)
    for i, row in j.iterrows():
        ticker = str(row["ticker"])
        ts = row["timestamp"]
        if ticker not in dfs:
            continue
        df = dfs[ticker]
        if df is None or df.empty or "close" not in df.columns:
            continue
        if ts not in df.index:
            continue
        loc = df.index.get_loc(ts)
        if isinstance(loc, slice):
            continue
        if int(loc) - window_steps < 0 or int(loc) + window_steps + 1 > len(df):
            continue
        subset = df.iloc[int(loc) - window_steps : int(loc) + window_steps + 1]
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
        return np.empty((0, 2 * window_steps + 1)), j.iloc[0:0]
    X = np.asarray(windows, dtype=float)
    return X, j.iloc[valid_rows].reset_index(drop=True)


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
        J=int(J),
        n_components=int(n_components),
        include_scattering_spectra=bool(include_ss),
        random_state=int(seed),
    )
    emb = wm.fit_transform(X)
    d1 = emb[:, 0].copy() if emb.shape[1] >= 1 else np.zeros((X.shape[0],), dtype=float)
    act_post = np.sum(np.abs(X[:, center + 1 :]), axis=1)
    act_pre = np.sum(np.abs(X[:, :center]), axis=1)
    asym = (act_post - act_pre) / (act_post + act_pre + 1e-9)
    corr = np.corrcoef(d1, asym)[0, 1] if len(d1) > 1 else 1.0
    if np.isfinite(corr) and corr < 0:
        d1 *= -1
    if score_mode == "kpca":
        if emb.shape[1] >= 3:
            d2 = emb[:, 1]
            d3 = emb[:, 2]
        else:
            x_pre = X[:, center - 1]
            x_post = X[:, center + 1]
            d2 = x_pre - x_post
            d3 = x_pre + x_post
    else:
        x_pre = X[:, center - 1]
        x_post = X[:, center + 1]
        d2 = x_pre - x_post
        d3 = x_pre + x_post
    return {"D1": d1, "D2": np.asarray(d2), "D3": np.asarray(d3)}


def _rho_from_scores(d3: np.ndarray) -> float:
    d3 = np.asarray(d3, dtype=float)
    d3 = d3[np.isfinite(d3)]
    s = int(d3.size)
    if s < 2:
        return float("nan")
    ss = float(np.sum(d3**2))
    if not np.isfinite(ss) or ss <= 0.0:
        return float("nan")
    sm = float(np.sum(d3))
    num = (sm**2) - ss
    den = float((s - 1) * ss)
    return float(num / den) if den != 0.0 else float("nan")


def compute_cojump_metrics(cojumps: pd.DataFrame) -> pd.DataFrame:
    if cojumps is None or cojumps.empty:
        return pd.DataFrame()
    out = cojumps.copy()
    out["mean_sign"] = out["returns"].apply(lambda xs: float(np.mean(np.sign(np.asarray(xs, dtype=float)))) if len(xs) else float("nan"))
    out["abs_mean_sign"] = out["mean_sign"].abs()
    if "D1s" in out.columns:
        out["D1_mean"] = out["D1s"].apply(lambda xs: float(np.mean(np.asarray(xs, dtype=float))) if len(xs) else float("nan"))
        out["D1_max"] = out["D1s"].apply(lambda xs: float(np.max(np.asarray(xs, dtype=float))) if len(xs) else float("nan"))
        out["D1_min"] = out["D1s"].apply(lambda xs: float(np.min(np.asarray(xs, dtype=float))) if len(xs) else float("nan"))
        out["D1_std"] = out["D1s"].apply(lambda xs: float(np.std(np.asarray(xs, dtype=float), ddof=0)) if len(xs) else float("nan"))
        sigma_by_size = out.groupby("size")["D1_std"].mean()
        out["sigma_D1_by_size"] = out["size"].map(sigma_by_size).astype(float)
        out["D1_mean_norm"] = out["D1_mean"] / out["sigma_D1_by_size"]
        out["D1_min_norm"] = out["D1_min"] / out["sigma_D1_by_size"]
        out["D1_max_norm"] = out["D1_max"] / out["sigma_D1_by_size"]
    if "D3s" in out.columns:
        out["rho_D3"] = out["D3s"].apply(lambda xs: _rho_from_scores(np.asarray(xs, dtype=float)))
    return out

# %% [markdown]
# ### Plot helpers (Plotly)

# %%

def plot_cojump_scatter(cojumps: pd.DataFrame, title: str, out_path: Path) -> None:
    """
    Scatter: x = date-time bin, y = time-of-day (decimal hours), marker size = cojump size.
    For daily bins, y will be 0.
    """
    if cojumps is None or cojumps.empty:
        return

    t = pd.to_datetime(cojumps["bin"])
    # time-of-day in decimal hours; for daily this becomes 0
    y = t.dt.hour + t.dt.minute / 60.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="markers",
            marker=dict(
                size=np.clip(cojumps["size"].to_numpy(dtype=float) * 2.5, 6, 60),
                color=cojumps["size"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="cojump size"),
                opacity=0.75,
            ),
            customdata=np.stack([cojumps["size"], cojumps["tickers"]], axis=1),
            hovertemplate=(
                "bin=%{x}<br>"
                "tod=%{y:.2f}h<br>"
                "size=%{customdata[0]}<br>"
                "tickers=%{customdata[1]}<extra></extra>"
            ),
            name="cojumps",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time bin",
        yaxis_title="Time of day (hours)",
        template="plotly_white",
        hovermode="closest",
    )
    fig.write_html(out_path)


def plot_size_distribution(cojumps: pd.DataFrame, title: str, out_path: Path) -> None:
    if cojumps is None or cojumps.empty:
        return
    fig = px.histogram(
        cojumps,
        x="size",
        nbins=min(60, max(10, int(cojumps["size"].max()))),
        title=title,
    )
    fig.update_layout(template="plotly_white", xaxis_title="cojump size", yaxis_title="count")
    fig.write_html(out_path)


def plot_size_ccdf(cojumps: pd.DataFrame, title: str, out_path: Path) -> None:
    if cojumps is None or cojumps.empty:
        return
    sizes = cojumps["size"].to_numpy(dtype=float)
    sizes = sizes[np.isfinite(sizes)]
    if sizes.size == 0:
        return
    s_sorted = np.sort(sizes)
    ccdf = 1.0 - (np.arange(1, s_sorted.size + 1) / s_sorted.size)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_sorted, y=ccdf, mode="markers", name="CCDF"))
    fig.update_layout(template="plotly_white", title=title, xaxis_title="S", yaxis_title="P(size > S)")
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    fig.write_html(out_path)


def plot_mean_sign_vs_size(cojumps: pd.DataFrame, title: str, out_path: Path) -> None:
    if cojumps is None or cojumps.empty or "mean_sign" not in cojumps.columns:
        return
    g = cojumps.groupby("size")["mean_sign"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["size"], y=g["mean_sign"], mode="markers+lines", name="E[mean sign | size]"))
    fig.update_layout(template="plotly_white", title=title, xaxis_title="cojump size S", yaxis_title="average sign")
    fig.write_html(out_path)


def plot_mean_vs_min_d1(cojumps: pd.DataFrame, title: str, out_path: Path) -> None:
    if cojumps is None or cojumps.empty or "D1_mean_norm" not in cojumps.columns or "D1_min_norm" not in cojumps.columns:
        return
    df = cojumps.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["D1_mean_norm"],
            y=df["D1_min_norm"],
            mode="markers",
            marker=dict(size=np.clip(df["size"].to_numpy(dtype=float) * 1.5, 6, 60), color=df["size"], colorscale="Viridis", showscale=True),
            text=df["size"],
            hovertemplate="size=%{text}<br>mean(D1)/σ=%{x:.3f}<br>min(D1)/σ=%{y:.3f}<extra></extra>",
            name="cojumps",
        )
    )
    fig.update_layout(template="plotly_white", title=title, xaxis_title="mean(D1) / σ(size)", yaxis_title="min(D1) / σ(size)")
    fig.write_html(out_path)


def plot_rho_vs_size(cojumps: pd.DataFrame, title: str, out_path: Path) -> None:
    if cojumps is None or cojumps.empty or "rho_D3" not in cojumps.columns:
        return
    df = cojumps.copy()
    df = df[np.isfinite(df["rho_D3"].to_numpy(dtype=float))]
    if df.empty:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["size"], y=df["rho_D3"], mode="markers", name="rho"))
    fig.update_layout(template="plotly_white", title=title, xaxis_title="cojump size S", yaxis_title="rho (trend score correlation)")
    fig.write_html(out_path)


def _log_returns_from_close(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")  # we coerce close to numeric
    close = close.replace([np.inf, -np.inf], np.nan).dropna()  # we drop non-finite
    if close.empty:  # we handle empty close
        return close  # we return empty series
    lr = np.log(close).diff().fillna(0.0)  # we compute log returns
    return lr  # we return log returns


def plot_cojump_log_return_profiles(
    dfs: Dict[str, pd.DataFrame],
    cojump_row: pd.Series,
    window_steps: int,
    max_tickers: int,
    title: str,
    out_path: Path,
    show_plots: bool,
) -> None:
    if cojump_row is None or cojump_row.empty:  # we validate input row
        return  # we do nothing
    if "bin" not in cojump_row.index or "tickers" not in cojump_row.index:  # we validate required fields
        return  # we do nothing
    ts = pd.to_datetime(cojump_row["bin"])  # we read cojump timestamp
    tickers = list(cojump_row["tickers"])  # we read tickers list
    if not tickers:  # we handle empty
        return  # we do nothing

    x = np.arange(-int(window_steps), int(window_steps) + 1, dtype=int)  # we build relative time axis
    traces: List[go.Scatter] = []  # we collect traces
    center_lr: List[Tuple[str, float]] = []  # we rank tickers by absolute center move
    series_by_ticker: Dict[str, np.ndarray] = {}  # we keep extracted arrays

    for t in tickers:  # we iterate tickers
        if t not in dfs:  # we skip missing tickers
            continue  # we continue
        df = dfs[t]  # we take dataframe
        if df is None or df.empty or "close" not in df.columns:  # we validate data
            continue  # we continue
        if ts not in df.index:  # we require exact timestamp match for clean alignment
            continue  # we continue
        loc = df.index.get_loc(ts)  # we locate timestamp
        if isinstance(loc, slice):  # we skip ambiguous
            continue  # we continue
        loc_i = int(loc)  # we cast to int
        if loc_i - window_steps < 0 or loc_i + window_steps + 1 > len(df):  # we require enough context
            continue  # we continue
        subset = df.iloc[loc_i - window_steps : loc_i + window_steps + 1]  # we take window
        lr = _log_returns_from_close(subset["close"])  # we compute log returns in window
        if len(lr) != (2 * window_steps + 1):  # we validate length
            continue  # we continue
        arr = lr.to_numpy(dtype=float)  # we store as array
        if not np.all(np.isfinite(arr)):  # we drop non-finite
            continue  # we continue
        series_by_ticker[str(t)] = arr  # we store series
        center_lr.append((str(t), float(abs(arr[window_steps]))))  # we store center magnitude

    if not series_by_ticker:  # we handle no extracted series
        return  # we do nothing

    center_lr.sort(key=lambda kv: kv[1], reverse=True)  # we rank by abs center move
    selected = [t for t, _ in center_lr[: int(max_tickers)]]  # we select top movers if needed

    Y = []  # we collect selected arrays
    for t in selected:  # we build traces
        arr = series_by_ticker[t]  # we take array
        Y.append(arr)  # we collect for mean/median
        traces.append(
            go.Scatter(
                x=x,
                y=arr,
                mode="lines",
                name=t,
                line=dict(width=1),
                opacity=0.35,
            )
        )

    Y_arr = np.asarray(Y, dtype=float)  # we stack arrays
    mean_lr = np.mean(Y_arr, axis=0)  # we compute mean profile
    med_lr = np.median(Y_arr, axis=0)  # we compute median profile
    traces.append(go.Scatter(x=x, y=mean_lr, mode="lines", name="mean", line=dict(width=3, color="black")))  # we add mean trace
    traces.append(go.Scatter(x=x, y=med_lr, mode="lines", name="median", line=dict(width=3, color="gray", dash="dash")))  # we add median trace

    fig = go.Figure(traces)  # we build figure
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.6)  # we mark cojump time
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="steps relative to cojump time",
        yaxis_title="log return",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        height=650,
    )
    fig.write_html(out_path)  # we save html
    if bool(show_plots):  # we optionally show the figure inline in notebooks
        fig.show()  # we display plotly figure

# %% [markdown]
# ### Run analysis

# %%

def run(cfg: CojumpConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if bool(cfg.show_plots):  # we configure renderer only when the user wants inline plots
        configure_plotly_renderer()  # we avoid browser-only rendering
    print(f"Loading Poland data for base_freq={cfg.base_freq} ...")
    dfs = load_poland(cfg.base_freq, min_len=cfg.min_len, max_tickers=cfg.max_tickers)
    dfs = preprocess(cfg.base_freq, dfs, trim_minutes=cfg.trim_intraday_minutes)
    print(f"Loaded {len(dfs)} tickers after filtering.")

    thr = threshold_for_freq(cfg.base_freq)
    print(f"Detecting jumps (threshold={thr}) ...")
    jumps_df = detect_jumps_many(dfs, threshold=thr)
    print(f"Detected {len(jumps_df)} jumps total.")
    if jumps_df is None or jumps_df.empty:
        return pd.DataFrame(), {}
    if len(jumps_df) > int(cfg.max_jumps_total):
        rng = np.random.default_rng(int(cfg.random_seed))
        idx = rng.choice(len(jumps_df), size=int(cfg.max_jumps_total), replace=False)
        jumps_df = jumps_df.iloc[np.sort(idx)].reset_index(drop=True)
        print(f"Subsampled jumps to {len(jumps_df)} for speed.")

    w = window_steps_for_freq(cfg.base_freq)
    X, jumps_scored = extract_windows(dfs, jumps_df, window_steps=w, max_windows=int(cfg.max_windows_total), seed=int(cfg.random_seed))
    scores = compute_scores(
        X,
        include_ss=bool(cfg.include_scattering_spectra),
        J=int(cfg.J),
        n_components=int(cfg.n_components),
        score_mode=cfg.score_mode,
        seed=int(cfg.random_seed),
    )
    if not jumps_scored.empty:
        jumps_scored = jumps_scored.copy()
        jumps_scored["D1"] = scores["D1"]
        jumps_scored["D2"] = scores["D2"]
        jumps_scored["D3"] = scores["D3"]
        jumps_scored.to_csv(out_dir() / f"jumps_with_D1D2D3_{cfg.base_freq}.csv", index=False)
        print(f"Computed D1/D2/D3 for {len(jumps_scored)} jumps (window_steps={w}).")
    else:
        print("No valid jump windows extracted; skipping D1/D2/D3.")
        jumps_scored = jumps_df

    results: Dict[str, pd.DataFrame] = {}
    for b in cfg.bins:
        cj0 = group_cojumps(jumps_scored, bin_freq=b, min_size=cfg.min_cojump_size)
        cj = compute_cojump_metrics(cj0)
        results[b] = cj
        print(f"  bin={b}: {len(cj)} cojumps (min_size={cfg.min_cojump_size})")

        # Save plots
        title_scatter = f"Co-jumps (bin={b}) - base={cfg.base_freq} - threshold={thr}"
        if not bool(cfg.show_plots) or not bool(cfg.show_only_profile_plots):  # we avoid showing everything by default
            plot_cojump_scatter(cj, title=title_scatter, out_path=out_dir() / f"cojumps_scatter_{cfg.base_freq}_{b}.html")

        title_hist = f"Co-jump size distribution (bin={b}) - base={cfg.base_freq}"
        if not bool(cfg.show_plots) or not bool(cfg.show_only_profile_plots):  # we avoid showing everything by default
            plot_size_distribution(cj, title=title_hist, out_path=out_dir() / f"cojumps_size_hist_{cfg.base_freq}_{b}.html")
            plot_size_ccdf(cj, title=f"Co-jump size CCDF (bin={b}) - base={cfg.base_freq}", out_path=out_dir() / f"cojumps_size_ccdf_{cfg.base_freq}_{b}.html")
            plot_mean_sign_vs_size(cj, title=f"Average sign vs size (bin={b}) - base={cfg.base_freq}", out_path=out_dir() / f"cojumps_mean_sign_{cfg.base_freq}_{b}.html")
            plot_mean_vs_min_d1(cj, title=f"Co-jump indicators: mean(D1) vs min(D1) (bin={b}) - base={cfg.base_freq}", out_path=out_dir() / f"cojumps_mean_vs_min_D1_{cfg.base_freq}_{b}.html")
            plot_rho_vs_size(cj, title=f"Trend correlation rho vs size (bin={b}) - base={cfg.base_freq}", out_path=out_dir() / f"cojumps_rho_{cfg.base_freq}_{b}.html")

        # Save a small CSV summary for inspection
        if not cj.empty:
            cj.to_csv(out_dir() / f"cojumps_{cfg.base_freq}_{b}.csv", index=False)

    # plot top-k cojump log-return profiles (overlay of constituents)  # we add requested visualization
    prof_bin = str(cfg.profile_bin)
    if prof_bin in results and results[prof_bin] is not None and not results[prof_bin].empty:
        topk = results[prof_bin].sort_values("size", ascending=False).head(int(cfg.top_k_profiles)).reset_index(drop=True)
        for i, row in topk.iterrows():
            size = int(row.get("size", 0))
            ts = pd.to_datetime(row.get("bin"))
            title = f"Co-jump log-return profiles | base={cfg.base_freq} | bin={prof_bin} | rank={i+1} | size={size} | time={ts}"
            out_path = out_dir() / f"cojump_logret_profiles_{cfg.base_freq}_{prof_bin}_rank{i+1:02d}_S{size}.html"
            plot_cojump_log_return_profiles(
                dfs,
                cojump_row=row,
                window_steps=int(cfg.profile_window_steps),
                max_tickers=int(cfg.max_tickers_per_profile),
                title=title,
                out_path=out_path,
                show_plots=bool(cfg.show_plots),
            )
    else:
        print(f"Skipping profile plots: bin={prof_bin} not available or empty.")  # we report missing bin

    return jumps_scored, results


# %% [markdown]
# ### Execute

# %%
jumps_df, cojumps_by_bin = run(CFG)

# %% [markdown]
# ### Inspect: show the largest co-jumps for each bin

# %%
for b, cj in cojumps_by_bin.items():
    if cj is None or cj.empty:
        print(f"{b}: (no cojumps)")
        continue
    top = cj.sort_values("size", ascending=False).head(10)[["bin", "size", "tickers"]]
    print(f"\nTop co-jumps for bin={b}:")
    print(top.to_string(index=False))


