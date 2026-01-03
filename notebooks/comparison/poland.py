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
# ## Poland ablation studies (Jupytext notebook)
#
# This notebook-style script runs ablations across:
#
# - **Frequencies**: 5-min, hourly, daily
# - **With / without Scattering Spectra (SS)** inside `WaveletModel`
# - **Scores**:
#   - **KPCA**: use (D1, D2, D3) = (embedding[:,0], embedding[:,1], embedding[:,2])
#   - **Handcrafted**: keep D1 from KPCA, but use handcrafted filters for D2/D3:
#     - D2 = x_pre − x_post
#     - D3 = x_pre + x_post
#     (as in `reproduce_single_stock.ipynb`)
#
# It also outputs:
#
# - **Jump-score distribution** + **jump-score time series** (per frequency)
# - **Overlay plots** for D1/D2/D3: average x(t) profiles for **Q0.1** and **Q0.9**,
#   superposed across all configs (Plotly HTML files).

# %%
from __future__ import annotations

# %% [markdown]
# ### Imports

# %%
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data.curating_stooq import (
    curate_stooq_dir_5min,
    curate_stooq_dir_daily,
    curate_stooq_dir_hourly,
)
from utils.data.jump_detection import compute_jump_score, detect_jumps_many
from model.wavelet.wavelet import WaveletModel

Freq = Literal["5min", "hourly", "daily"]
ScoreMode = Literal["kpca", "handcrafted"]

# %% [markdown]
# ### Config
#
# Tune these and re-run the notebook.

# %%
MIN_LEN = 500
MAX_TICKERS = 120
MAX_WINDOWS = 3000
J_SCALES = 3
THRESHOLD = 4.0
SHOW_PLOTS = True


@dataclass(frozen=True)
class AblationConfig:
    freq: Freq
    include_ss: bool
    score_mode: ScoreMode
    threshold: float = 4.0
    J: int = 3
    n_components: int = 3
    window_steps: int = 12

    def label(self) -> str:
        ss = "SS" if self.include_ss else "noSS"
        return f"{self.freq}|{ss}|{self.score_mode}|J={self.J}|w={self.window_steps}"

# %% [markdown]
# ### Helpers: paths + data loading

# %%
def _project_root() -> Path:
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


def _data_dir(freq: Freq) -> Path:
    root = _project_root()
    if freq == "5min":
        return root / "data" / "stooq" / "poland" / "5_min" / "pl" / "wsestocks"
    if freq == "hourly":
        return root / "data" / "stooq" / "poland" / "hourly" / "ncstocks"
    if freq == "daily":
        return root / "data" / "stooq" / "poland" / "daily" / "ncstocks"
    raise ValueError(f"Unknown freq: {freq}")


def load_poland(freq: Freq, min_len: int, max_tickers: int) -> Dict[str, pd.DataFrame]:
    data_dir = _data_dir(freq)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if freq == "5min":
        dfs = curate_stooq_dir_5min(str(data_dir), pattern="*.txt", recursive=True)
    elif freq == "hourly":
        dfs = curate_stooq_dir_hourly(str(data_dir), pattern="*.txt", recursive=True)
    elif freq == "daily":
        dfs = curate_stooq_dir_daily(str(data_dir), pattern="*.txt", recursive=True)
    else:
        raise ValueError(f"Unknown freq: {freq}")

    # Filter by length and keep the longest max_tickers
    tickers = [t for t, d in dfs.items() if d is not None and not d.empty and len(d) >= min_len]
    tickers.sort(key=lambda t: len(dfs[t]), reverse=True)
    tickers = tickers[:max_tickers]
    return {t: dfs[t] for t in tickers}

# %% [markdown]
# ### Helpers: preprocessing (intraday trimming)

# %%
def trim_intraday(df: pd.DataFrame, minutes: int = 60) -> pd.DataFrame:
    """
    Remove the first/last `minutes` of each day (used for intraday 5m/hourly).
    """
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


def preprocess(freq: Freq, dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for t, df in dfs.items():
        if df is None or df.empty:
            continue
        d = df.sort_index()
        d = d[~d.index.duplicated(keep="last")]
        if freq in ("5min", "hourly"):
            d = trim_intraday(d, minutes=60)
        # daily: no trimming
        if not d.empty:
            out[t] = d
    return out

# %% [markdown]
# ### Helpers: jump windows

# %%
def extract_windows(
    dfs: Dict[str, pd.DataFrame],
    jumps_df: pd.DataFrame,
    window_steps: int,
    max_windows: int,
    seed: int = 0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Build X_windows of shape (n_windows, 2*window_steps+1) from jump table.
    Uses constant normalization per window: norm = f(t0)*sigma(t0) from jumps_df at the jump timestamp.
    """
    if jumps_df.empty:
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
    jumps_subset = jumps_df.iloc[valid_rows].reset_index(drop=True)
    return X, jumps_subset

# %% [markdown]
# ### Helpers: compute scores (KPCA vs handcrafted)

# %%
def compute_scores(
    X: np.ndarray,
    include_ss: bool,
    J: int,
    n_components: int,
    score_mode: ScoreMode,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Returns score vectors for D1/D2/D3.
    D1 is always the first KPCA component (optionally flipped so positive correlates with post>pre activity).
    D2/D3 depend on score_mode:
      - kpca: use embedding[:,1], embedding[:,2]
      - handcrafted: D2=x_pre-x_post, D3=x_pre+x_post
    """
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
            raise ValueError(f"Need n_components>=3 for kpca mode, got {emb.shape[1]}")
        d2 = emb[:, 1]
        d3 = emb[:, 2]
    elif score_mode == "handcrafted":
        x_pre = X[:, center - 1]
        x_post = X[:, center + 1]
        d2 = x_pre - x_post
        d3 = x_pre + x_post
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    return {"D1": d1, "D2": np.asarray(d2), "D3": np.asarray(d3)}

# %% [markdown]
# ### Helpers: quantile profiles

# %%
def q_low_high_profiles(X: np.ndarray, score: np.ndarray, q: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return mean profiles for the lowest q and highest q quantiles of score.
    """
    if X.size == 0 or score.size == 0:
        return np.array([]), np.array([])
    n = len(score)
    k = max(1, int(round(q * n)))
    order = np.argsort(score)
    low = np.mean(X[order[:k]], axis=0)
    high = np.mean(X[order[-k:]], axis=0)
    return low, high

# %% [markdown]
# ### Plots: jump-score diagnostics

# %%
def plot_jump_score_diagnostics(
    out_dir: Path,
    freq: Freq,
    dfs: Dict[str, pd.DataFrame],
    threshold: float = 4.0,
    show: bool = True,
) -> None:
    if not dfs:
        return
    ticker, df = max(dfs.items(), key=lambda kv: len(kv[1]))
    scores = compute_jump_score(df, price_col="close")
    if scores.empty:
        return

    fig_hist = px.histogram(
        scores.reset_index(), x="score", nbins=160,
        title=f"Jump score distribution x(t) - {freq} - {ticker}",
    )
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red")
    fig_hist.add_vline(x=-threshold, line_dash="dash", line_color="red")
    fig_hist.update_layout(template="plotly_white")
    fig_hist.write_html(out_dir / f"jump_score_hist_{freq}.html")
    if show:
        fig_hist.show()

    max_points = 10000
    scores_plot = scores if len(scores) <= max_points else scores.iloc[-max_points:]
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=scores_plot.index, y=scores_plot["score"], mode="lines", name="x(t)", line=dict(width=1)))
    fig_ts.add_hline(y=threshold, line_dash="dash", line_color="red")
    fig_ts.add_hline(y=-threshold, line_dash="dash", line_color="red")
    fig_ts.update_layout(
        title=f"Jump score x(t) over time - {freq} - {ticker}",
        xaxis_title="Time",
        yaxis_title="x(t)",
        template="plotly_white",
        hovermode="x unified",
    )
    fig_ts.write_html(out_dir / f"jump_score_ts_{freq}.html")
    if show:
        fig_ts.show()

# %% [markdown]
# ### Plots: overlay Q0.1 / Q0.9 profiles for D1/D2/D3 across configs

# %%
def overlay_profiles_plotly(
    out_dir: Path,
    direction: Literal["D1", "D2", "D3"],
    t_axis: np.ndarray,
    curves: List[Tuple[str, np.ndarray, np.ndarray]],
    show: bool = True,
) -> None:
    """
    curves: list of (label, low_profile, high_profile)
    """
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    for i, (label, low, high) in enumerate(curves):
        color = palette[i % len(palette)]
        if low.size:
            fig.add_trace(go.Scatter(x=t_axis, y=low, mode="lines", name=f"{label} | Q0.1", line=dict(color=color, dash="dot", width=2)))
        if high.size:
            fig.add_trace(go.Scatter(x=t_axis, y=high, mode="lines", name=f"{label} | Q0.9", line=dict(color=color, dash="solid", width=2)))

    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.6)
    fig.update_layout(
        title=f"Overlay: average x(t) profiles for {direction} (Q0.1 vs Q0.9) across configs",
        xaxis_title="Time (steps) relative to jump",
        yaxis_title="Jump-aligned normalized return x(t)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="v"),
    )
    fig.write_html(out_dir / f"overlay_profiles_{direction}.html")
    if show:
        fig.show()

# %% [markdown]
# ### Runner

# %%
def run_ablation(
    min_len: int = 500,
    max_tickers: int = 120,
    max_windows: int = 3000,
    J: int = 3,
) -> Path:
    out_dir = _project_root() / "notebooks" / "comparison" / "outputs" / "poland_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets per frequency once
    datasets: Dict[Freq, Dict[str, pd.DataFrame]] = {}
    windows: Dict[Freq, Tuple[np.ndarray, pd.DataFrame, int]] = {}

    for freq in ("5min", "hourly", "daily"):
        dfs = load_poland(freq, min_len=min_len, max_tickers=max_tickers)
        dfs_p = preprocess(freq, dfs)

        # Jump-score diagnostics per frequency (independent of SS and scoring mode)
        plot_jump_score_diagnostics(out_dir, freq, dfs_p, threshold=THRESHOLD, show=SHOW_PLOTS)

        # Detect jumps and extract windows
        jumps_df = detect_jumps_many(dfs_p, threshold=4.0)

        if freq == "daily":
            window_steps = 20
        else:
            window_steps = 12

        X, jumps_subset = extract_windows(dfs_p, jumps_df, window_steps=window_steps, max_windows=max_windows, seed=0)
        datasets[freq] = dfs_p
        windows[freq] = (X, jumps_subset, window_steps)

    # Build all configs
    configs: List[AblationConfig] = []
    for freq in ("5min", "hourly", "daily"):
        _, _, w = windows[freq]
        for include_ss in (False, True):
            for score_mode in ("kpca", "handcrafted"):
                configs.append(AblationConfig(freq=freq, include_ss=include_ss, score_mode=score_mode, J=J, window_steps=w))

    # Compute profiles for Q0.1 and Q0.9 for each config and each direction
    curves_by_direction: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {"D1": [], "D2": [], "D3": []}
    for cfg in configs:
        X, _, _ = windows[cfg.freq]
        if X.size == 0:
            continue

        scores = compute_scores(
            X,
            include_ss=cfg.include_ss,
            J=cfg.J,
            n_components=cfg.n_components,
            score_mode=cfg.score_mode,
            seed=0,
        )
        for d in ("D1", "D2", "D3"):
            low, high = q_low_high_profiles(X, scores[d], q=0.1)
            curves_by_direction[d].append((cfg.label(), low, high))

    # Overlay plots per direction
    for freq in ("5min", "hourly", "daily"):
        X, _, w = windows[freq]
        if X.size:
            break
    # Use the time axis from the first non-empty frequency; each config uses its own window length in label.
    # (All current windows lengths are 25 for 5min/hourly and 41 for daily; overlay uses the axis matching each curve.)
    # To keep a single overlay axis, we plot each direction separately for each window length.

    # Split by window length so overlays have consistent x-axis
    curves_split: Dict[int, Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]]] = {}
    for cfg in configs:
        wl = 2 * cfg.window_steps + 1
        curves_split.setdefault(wl, {"D1": [], "D2": [], "D3": []})

    for d in ("D1", "D2", "D3"):
        for label, low, high in curves_by_direction[d]:
            # parse window length from label suffix "...|w=NN"
            try:
                w_steps = int(label.split("|w=")[-1])
            except Exception:
                continue
            wl = 2 * w_steps + 1
            curves_split[wl][d].append((label, low, high))

    for wl, by_dir in curves_split.items():
        center = wl // 2
        t_axis = np.arange(-center, center + 1)
        for d in ("D1", "D2", "D3"):
            overlay_profiles_plotly(out_dir, d, t_axis, by_dir[d], show=SHOW_PLOTS)

    return out_dir


# %% [markdown]
# ### Execute

# %%
def main() -> None:
    out_dir = run_ablation(
        min_len=MIN_LEN,
        max_tickers=MAX_TICKERS,
        max_windows=MAX_WINDOWS,
        J=J_SCALES,
    )
    print(f"Saved ablation outputs to: {out_dir}")


if __name__ == "__main__":
    main()

