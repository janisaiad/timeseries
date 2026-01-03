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
# ## Complexity benchmarks (time, memory, FLOPs proxy) — Poland 5‑min pipeline
#
# This notebook benchmarks the main computational blocks used in `notebooks/jump/reproduce_poland.ipynb`:
#
# - Load + curate Stooq 5‑min data
# - Trim open/close
# - Compute jump score series \(x(t)\)
# - Detect jumps (`detect_jumps_many`) for thresholds \(\tau \in \{1,2,3,4,5\}\)
# - Extract windows around detected jumps
# - Wavelet embedding (`WaveletModel.fit_transform`) on extracted windows
#
# It runs the pipeline for **different dataset scales** (different numbers of tickers → different total points)
# and plots **log–log** relationships:
#
# - total points vs wall-time (scatter, log–log)
# - total points vs peak RSS / Python memory (log–log)
#
# Notes:
# - FLOPs are **approximate proxies** (simple per-point/per-window constants) — enough for scaling comparisons,
#   not a microarchitectural truth.
# - Memory is tracked as:
#   - RSS (process resident set size) if `psutil` is available
#   - otherwise `resource.getrusage(...).ru_maxrss` (Linux: KiB)
#   - plus optional `tracemalloc` for Python allocation peaks

# %%
from __future__ import annotations

# %% [markdown]
# ### Imports

# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import time
import gc
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data.curating_stooq import curate_stooq_dir_5min
from utils.data.jump_detection import compute_jump_score, detect_jumps_many
from model.wavelet.wavelet import WaveletModel

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

import tracemalloc
import resource

Freq = Literal["5min"]

# %% [markdown]
# ### Config

# %%
@dataclass(frozen=True)
class BenchConfig:
    # data
    min_len: int = 500
    max_tickers: int = 200
    trim_intraday_minutes: int = 60

    # scaling (subsets of tickers by length, longest first)
    ticker_scales: Tuple[int, ...] = (5, 10, 20, 40, 80, 120, 160, 200)

    # threshold sweep
    thresholds: Tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)

    # windows
    window_steps: int = 12
    max_windows: int = 2000

    # wavelet embedding
    J: int = 3
    n_components: int = 3
    include_scattering_spectra: bool = False

    # benchmarking
    repeat: int = 1  # set >1 for more stable timing (takes longer)
    do_embedding: bool = True
    seed: int = 0


CFG = BenchConfig()

# %% [markdown]
# ### Paths

# %%
def project_root() -> Path:
    try:
        here = Path(__file__).resolve()
        return here.parents[2]  # .../notebooks/complexity/complexity.py -> repo root
    except NameError:
        cwd = Path.cwd().resolve()
        for p in [cwd, *cwd.parents]:
            if (p / "utils").is_dir() and (p / "data").is_dir():
                return p
        return cwd


def data_dir_5min() -> Path:
    return project_root() / "data" / "stooq" / "poland" / "5_min" / "pl" / "wsestocks"


def out_dir() -> Path:
    d = project_root() / "notebooks" / "complexity" / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d

# %% [markdown]
# ### Small utilities: memory + timing

# %%
def rss_bytes() -> int:
    """
    Current process RSS in bytes (best-effort).
    """
    if psutil is not None:
        try:
            return int(psutil.Process().memory_info().rss)
        except Exception:
            pass
    # fallback: ru_maxrss is peak, not current; still useful as a monotone upper bound
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux ru_maxrss is in KiB
        return int(ru.ru_maxrss) * 1024
    except Exception:
        return -1


@dataclass
class StageResult:
    name: str
    seconds: float
    rss_before: int
    rss_after: int
    tracemalloc_peak: int


class stage_timer:
    def __init__(self, name: str):
        self.name = name
        self._t0 = 0.0
        self.rss0 = 0
        self.rss1 = 0
        self.tm_peak = 0

    def __enter__(self) -> "stage_timer":
        gc.collect()
        self.rss0 = rss_bytes()
        tracemalloc.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.rss1 = rss_bytes()
        _, peak = tracemalloc.get_traced_memory()
        self.tm_peak = int(peak)
        tracemalloc.stop()
        self.dt = float(time.perf_counter() - self._t0)

    def result(self) -> StageResult:
        return StageResult(
            name=self.name,
            seconds=float(getattr(self, "dt", 0.0)),
            rss_before=int(self.rss0),
            rss_after=int(self.rss1),
            tracemalloc_peak=int(self.tm_peak),
        )

# %% [markdown]
# ### Data prep (5‑min): load + trim

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


def load_poland_5min(min_len: int, max_tickers: int) -> Dict[str, pd.DataFrame]:
    d = data_dir_5min()
    if not d.exists():
        raise FileNotFoundError(f"Data directory not found: {d}")
    dfs = curate_stooq_dir_5min(str(d), pattern="*.txt", recursive=True)
    tickers = [t for t, df in dfs.items() if df is not None and not df.empty and len(df) >= min_len]
    tickers.sort(key=lambda t: len(dfs[t]), reverse=True)
    tickers = tickers[:max_tickers]

    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = dfs[t].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = trim_intraday(df, minutes=CFG.trim_intraday_minutes)
        if not df.empty:
            out[t] = df
    return out


def subset_by_tickers(dfs: Dict[str, pd.DataFrame], n: int) -> Dict[str, pd.DataFrame]:
    tickers = sorted(dfs.keys(), key=lambda t: len(dfs[t]), reverse=True)
    tickers = tickers[: min(n, len(tickers))]
    return {t: dfs[t] for t in tickers}


def total_points(dfs: Dict[str, pd.DataFrame]) -> int:
    return int(sum(len(df) for df in dfs.values() if df is not None and not df.empty))

# %% [markdown]
# ### Window extraction (jump-aligned profiles)

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
# ### FLOPs proxy model (rough)
#
# We attach a simple *estimated flops* to each stage to get an “effective GFLOPs/s” proxy.

# %%
@dataclass(frozen=True)
class FlopsModel:
    # Per-point “ops” for scoring (pct_change + a few arithmetic ops)
    flops_per_point_score: float = 25.0
    # Per-point ops for jump detection (u-shape + EWM sigma + thresholding): rough
    flops_per_point_detect: float = 80.0
    # Per-window ops to build x_profile (returns + normalize + sign)
    flops_per_window_extract: float = 400.0
    # Per-window ops for wavelet embedding: highly approximate; grows with length*J
    flops_per_window_wavelet: float = 20_000.0


FLOPS = FlopsModel()


def est_flops(stage: str, n_points: int, n_windows: int) -> float:
    if stage == "score":
        return FLOPS.flops_per_point_score * n_points
    if stage == "detect":
        return FLOPS.flops_per_point_detect * n_points
    if stage == "extract":
        return FLOPS.flops_per_window_extract * n_windows
    if stage == "embed":
        return FLOPS.flops_per_window_wavelet * n_windows
    return float("nan")

# %% [markdown]
# ### Benchmark runner

# %%
def run_once(dfs: Dict[str, pd.DataFrame], threshold: float) -> Tuple[Dict[str, StageResult], Dict[str, float]]:
    """
    Returns:
      - per-stage timing/memory
      - summary scalars (points, jumps, windows, est_flops, etc.)
    """
    n_pts = total_points(dfs)

    stages: Dict[str, StageResult] = {}

    # representative score computation (1 ticker only) – fast sanity + keep consistent with other notebooks
    sample_ticker, sample_df = max(dfs.items(), key=lambda kv: len(kv[1]))
    with stage_timer("score") as st:
        scores_df = compute_jump_score(sample_df, price_col="close")
        _ = scores_df["score"].to_numpy(dtype=float)
    stages["score"] = st.result()

    with stage_timer("detect") as st:
        jumps_df = detect_jumps_many(dfs, threshold=threshold)
    stages["detect"] = st.result()

    with stage_timer("extract") as st:
        X = extract_windows(
            dfs,
            jumps_df,
            window_steps=CFG.window_steps,
            max_windows=CFG.max_windows,
            seed=CFG.seed,
        )
    stages["extract"] = st.result()

    if CFG.do_embedding:
        with stage_timer("embed") as st:
            wm = WaveletModel(
                n_layers=0,
                n_neurons=0,
                n_outputs=0,
                J=CFG.J,
                n_components=CFG.n_components,
                include_scattering_spectra=CFG.include_scattering_spectra,
                random_state=CFG.seed,
            )
            _ = wm.fit_transform(X) if len(X) else np.empty((0, CFG.n_components))
        stages["embed"] = st.result()

    summary = {
        "threshold": float(threshold),
        "n_points": float(n_pts),
        "n_tickers": float(len(dfs)),
        "n_jumps": float(len(jumps_df)),
        "n_windows": float(len(X)),
        "rss_bytes_end": float(rss_bytes()),
    }

    # estimated flops + effective rates
    for k in ("score", "detect", "extract", "embed"):
        if k not in stages:
            continue
        f = est_flops(k, n_points=n_pts, n_windows=len(X))
        t = max(1e-9, stages[k].seconds)
        summary[f"est_flops_{k}"] = float(f)
        summary[f"est_gflops_per_s_{k}"] = float(f / t / 1e9)

    return stages, summary


def benchmark(dfs_full: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for n_tickers in CFG.ticker_scales:
        dfs = subset_by_tickers(dfs_full, n_tickers)
        if not dfs:
            continue

        for thr in CFG.thresholds:
            for r in range(CFG.repeat):
                stages, summary = run_once(dfs, threshold=thr)

                # flatten stages
                row: Dict[str, float] = dict(summary)
                row["repeat"] = float(r)
                for name, sr in stages.items():
                    row[f"t_{name}_s"] = float(sr.seconds)
                    row[f"rss_{name}_before"] = float(sr.rss_before)
                    row[f"rss_{name}_after"] = float(sr.rss_after)
                    row[f"tm_peak_{name}_bytes"] = float(sr.tracemalloc_peak)
                rows.append(row)
    return pd.DataFrame(rows)

# %% [markdown]
# ### Run benchmark
#
# This will take time. Start with fewer `ticker_scales` / `repeat` if needed.

# %%
with stage_timer("load_5min") as st:
    dfs_full = load_poland_5min(min_len=CFG.min_len, max_tickers=CFG.max_tickers)
load_stage = st.result()
print(f"Loaded {len(dfs_full)} tickers (5min). Total points={total_points(dfs_full)}. load_time={load_stage.seconds:.2f}s")

# %%
df_bench = benchmark(dfs_full)
df_bench.to_csv(out_dir() / "bench_results.csv", index=False)
print("Saved:", out_dir() / "bench_results.csv")
df_bench.head()

# %% [markdown]
# ### Log–log plots: time vs points (scatter)

# %%
def loglog_scatter_time(stage: str) -> go.Figure:
    col = f"t_{stage}_s"
    sub = df_bench[np.isfinite(df_bench[col]) & (df_bench[col] > 0) & (df_bench["n_points"] > 0)].copy()
    fig = px.scatter(
        sub,
        x="n_points",
        y=col,
        color=sub["threshold"].astype(str),
        symbol=sub["n_tickers"].astype(int).astype(str),
        log_x=True,
        log_y=True,
        title=f"log–log: {stage} time vs number of points (color=threshold, symbol=#tickers)",
        labels={"n_points": "#points (bars)", col: "seconds"},
    )
    fig.update_layout(template="plotly_white")
    return fig


for stage in ("score", "detect", "extract", "embed"):
    if f"t_{stage}_s" not in df_bench.columns:
        continue
    fig = loglog_scatter_time(stage)
    fig.write_html(out_dir() / f"loglog_time_vs_points_{stage}.html")
    fig.show()

# %% [markdown]
# ### Log–log plots: peak RSS / Python peak alloc vs points

# %%
def loglog_scatter_mem(stage: str, which: Literal["rss_after", "tm_peak"]) -> go.Figure:
    if which == "rss_after":
        col = f"rss_{stage}_after"
        ylab = "RSS bytes (after stage)"
    else:
        col = f"tm_peak_{stage}_bytes"
        ylab = "tracemalloc peak bytes (stage)"
    sub = df_bench[np.isfinite(df_bench[col]) & (df_bench[col] > 0) & (df_bench["n_points"] > 0)].copy()
    fig = px.scatter(
        sub,
        x="n_points",
        y=col,
        color=sub["threshold"].astype(str),
        symbol=sub["n_tickers"].astype(int).astype(str),
        log_x=True,
        log_y=True,
        title=f"log–log: {stage} memory vs number of points ({which})",
        labels={"n_points": "#points (bars)", col: ylab},
    )
    fig.update_layout(template="plotly_white")
    return fig


for stage in ("score", "detect", "extract", "embed"):
    if f"rss_{stage}_after" in df_bench.columns:
        fig = loglog_scatter_mem(stage, "rss_after")
        fig.write_html(out_dir() / f"loglog_mem_rss_vs_points_{stage}.html")
        fig.show()
    if f"tm_peak_{stage}_bytes" in df_bench.columns:
        fig = loglog_scatter_mem(stage, "tm_peak")
        fig.write_html(out_dir() / f"loglog_mem_tracemalloc_vs_points_{stage}.html")
        fig.show()

# %% [markdown]
# ### Optional: aggregate summaries (median over repeats)

# %%
group_cols = ["n_tickers", "n_points", "threshold"]
agg_cols = [c for c in df_bench.columns if c.startswith("t_")]
df_med = df_bench.groupby(group_cols, as_index=False)[agg_cols].median(numeric_only=True)
df_med.to_csv(out_dir() / "bench_results_median.csv", index=False)
print("Saved:", out_dir() / "bench_results_median.csv")
df_med.head()


