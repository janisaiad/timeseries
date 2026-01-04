# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %%
"""
Wavelet change / comparison notebook (Jupytext: percent format).

Goals
-----
1) Visualize the wavelet filter in time and frequency domains (real/imag parts),
   as in `refs/texsources/ridingwavelets/main.tex` (Fig. "wavelet_filter").
2) Visualize a few alternative continuous wavelets.
3) Run a Poland daily jump → window extraction → WaveletModel(KernelPCA) pipeline
   for several wavelets and compare the resulting D1 (reflexivity) coordinate.

This file is meant to be synced to an `.ipynb` with:
  uv run jupytext --sync notebooks/wavelet_change/wavelet_change.py
"""

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
# ### Wavelet visualization + D1 comparison across wavelets
#
# This notebook does two things:
# - **Wavelet plots**: show \( \psi(t) \) in time (Re/Im) and its Fourier transform \( \hat{\psi}(\omega) \).
# - **Pipeline comparison**: run the same jump-window dataset through `WaveletModel` with different PyWavelets
#   continuous wavelets, then compare the resulting **D1** (first KPCA coordinate) across wavelets.

# %%
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import plotly.io as pio
except Exception:
    pio = None  # type: ignore[assignment]

try:
    import pywt
except Exception as e:
    pywt = None  # type: ignore[assignment]
    _PYWT_IMPORT_ERROR = e

# Make project root importable when executed as a notebook / script.
try:
    _HERE = Path(__file__).resolve()
    PROJECT_ROOT = _HERE.parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd().resolve()
    # When run from notebook, cwd is typically project root already.

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.data.curating_stooq import curate_stooq_dir_daily

# NOTE: notebooks cache imports; reload so edits to jump_detection.py show up without restarting kernel.
import importlib
import utils.data.jump_detection as jump_detection

jump_detection = importlib.reload(jump_detection)
detect_jumps_many = jump_detection.detect_jumps_many
compute_jump_score = jump_detection.compute_jump_score

from model.wavelet.wavelet import WaveletModel


def configure_plotly_renderer() -> None:
    """
    Make Plotly show figures inline across environments (WSL, VSCode, notebooks).
    """

    if pio is None:
        return
    candidates = ["vscode", "plotly_mimetype", "notebook_connected", "notebook"]
    available = set(getattr(pio, "renderers", {}).keys())
    for r in candidates:
        if r in available:
            pio.renderers.default = r
            return


configure_plotly_renderer()


def _require_pywt() -> None:
    if pywt is None:
        raise ImportError(f"pywt (PyWavelets) is required here but not available: {_PYWT_IMPORT_ERROR}")


# %% [markdown]
# ### 1) Plot wavelet in time + frequency (paper-style)

# %%
def plot_wavelet_time_and_frequency(
    wavelet: str,
    *,
    wavefun_level: int = 10,
    normalize_psi: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plot psi(t) (real/imag) and its Fourier transform.
    """

    _require_pywt()

    cw = pywt.ContinuousWavelet(wavelet)
    psi, t = cw.wavefun(level=int(wavefun_level))
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)

    if normalize_psi:
        # L2 normalize in time domain for comparable amplitude across wavelets.
        dt = float(np.mean(np.diff(t))) if t.size > 1 else 1.0
        norm = float(np.sqrt(np.sum(np.abs(psi) ** 2) * dt))
        if norm > 0:
            psi = psi / norm

    # FFT-based frequency response (discrete approximation).
    dt = float(np.mean(np.diff(t))) if t.size > 1 else 1.0
    Psi = np.fft.fftshift(np.fft.fft(psi))
    w = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(psi.size, d=dt))  # angular frequency

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Time domain: Re/Im ψ(t)", "Frequency domain: Re/Im/|ψ̂(ω)|"),
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Scatter(x=t, y=np.real(psi), mode="lines", name="Re ψ(t)", line=dict(width=2)), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=t, y=np.imag(psi), mode="lines", name="Im ψ(t)", line=dict(width=2, dash="dash")), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=w, y=np.real(Psi), mode="lines", name="Re ψ̂(ω)", line=dict(width=2)), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=w, y=np.imag(Psi), mode="lines", name="Im ψ̂(ω)", line=dict(width=2, dash="dash")), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=w, y=np.abs(Psi), mode="lines", name="|ψ̂(ω)|", line=dict(width=2, color="black")),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="t", row=1, col=1, zeroline=True)
    fig.update_yaxes(title_text="ψ(t)", row=1, col=1, zeroline=True)
    fig.update_xaxes(title_text="ω", row=1, col=2, zeroline=True)
    fig.update_yaxes(title_text="ψ̂(ω)", row=1, col=2, zeroline=True)

    fig.update_layout(
        template="plotly_white",
        title=title or f"Wavelet `{wavelet}` (PyWavelets ContinuousWavelet)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
        height=420,
        margin=dict(l=40, r=20, t=60, b=60),
    )
    return fig


# %%
# Default wavelet used by `WaveletModel` (see `model/wavelet/wavelet.py`).
fig = plot_wavelet_time_and_frequency("cmor1.5-1.0", wavefun_level=10)
fig.show()

# %% [markdown]
# ### 2) Compare a few other continuous wavelets
#
# PyWavelets provides many continuous wavelets; here are a few representative ones:
# - `cmor...` : complex Morlet (parametrized)
# - `cgauN`   : complex Gaussian derivatives
# - `shan...` : Shannon wavelet (parametrized)
# - `fbsp...` : frequency B-spline (parametrized)
# - `morl`    : real Morlet (imag part should be ~0)

# %%
WAVELET_CANDIDATES: Tuple[str, ...] = (
    "cmor1.5-1.0",
    "cmor0.5-1.0",
    "cmor1.0-1.5",
    "cgau1",
    "cgau4",
    "shan1.5-1.0",
    "fbsp1-1.5-1.0",
    "morl",
)

for wname in WAVELET_CANDIDATES:
    plot_wavelet_time_and_frequency(wname, wavefun_level=9, title=f"Wavelet filter: `{wname}`").show()

# %% [markdown]
# ### 3) Poland daily pipeline: compute D1 for many wavelets and compare
#
# We reuse the logic from `notebooks/jump/reproduce_poland_daily.ipynb`:
# - detect jumps using threshold = 2.0 (daily)
# - extract jump-centered windows of length `2*window_steps+1`
# - build aligned return profiles `x(t)` (aligned so jump is positive)
# - fit `WaveletModel` (KernelPCA) and take **embedding[:, 0]** as D1
# - orient D1 sign so it correlates positively with a simple post-vs-pre activity asymmetry

# %%
@dataclass(frozen=True)
class PolandDailyConfig:
    data_dir: Path = Path("/home/janis/4A/timeseries/data/stooq/poland/daily/ncstocks/")
    min_len: int = 500
    threshold: float = 2.0
    window_steps: int = 20
    max_windows: int = 1200  # subsample windows for speed (KernelPCA is O(n^3))
    random_seed: int = 0

    # WaveletModel parameters
    J: int = 3
    n_components: int = 3
    kernel: str = "rbf"
    include_scattering_spectra: bool = False

    # Where to write HTML plots
    out_dir: Path = Path("/home/janis/4A/timeseries/notebooks/wavelet_change/poland_outputs_daily_wavelets/")

    # Profile plots (like `reproduce_poland_daily.ipynb`)
    make_profile_plots: bool = True
    show_profile_plots: bool = True
    profile_quantiles: Tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
    profile_wavelets: Optional[Tuple[str, ...]] = None  # None -> use `wavelets_to_compare`
    profiles_dirname: str = "profiles"


def _prepare_daily_series(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    filtered: Dict[str, pd.DataFrame] = {}
    for ticker, df in dfs.items():
        if df is None or df.empty:
            continue
        d = df.sort_index()
        d = d[~d.index.duplicated(keep="last")]
        filtered[ticker] = d
    return filtered


def _extract_aligned_windows_daily(
    filtered_dfs: Dict[str, pd.DataFrame],
    jumps_df: pd.DataFrame,
    *,
    window_steps: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    windows: List[np.ndarray] = []
    valid_indices: List[int] = []

    for idx, row in jumps_df.iterrows():
        ticker, ts = row["ticker"], row["timestamp"]
        if ticker not in filtered_dfs:
            continue
        df = filtered_dfs[ticker]
        if ts not in df.index:
            continue
        loc = df.index.get_loc(ts)
        if loc - window_steps < 0 or loc + window_steps + 1 > len(df):
            continue

        subset = df.iloc[loc - window_steps : loc + window_steps + 1]

        norm = float(row.get("f", 1.0)) * float(row.get("sigma", 1.0))
        if not np.isfinite(norm) or norm == 0.0:
            norm = 1e-4

        prices = subset["close"].astype(float).clip(lower=1e-12)
        r_window = np.log(prices).diff().fillna(0.0).to_numpy(dtype=float)
        x_profile = r_window / norm

        # Align jump direction to positive at t=0 (center index).
        jump_sign = float(np.sign(x_profile[window_steps]))
        if jump_sign == 0.0:
            jump_sign = 1.0
        windows.append(x_profile * jump_sign)
        valid_indices.append(idx)

    X = np.asarray(windows, dtype=float)
    j = jumps_df.loc[valid_indices].copy()
    return X, j


def _activity_asymmetry(X_windows: np.ndarray, *, center: int) -> np.ndarray:
    act_post = np.sum(np.abs(X_windows[:, center + 1 :]), axis=1)
    act_pre = np.sum(np.abs(X_windows[:, :center]), axis=1)
    return (act_post - act_pre) / (act_post + act_pre + 1e-6)


def _subsample_windows(
    X_windows: np.ndarray,
    jumps_subset: pd.DataFrame,
    *,
    max_windows: int,
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    if max_windows <= 0 or len(X_windows) <= max_windows:
        return X_windows, jumps_subset
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(X_windows), size=int(max_windows), replace=False)
    idx = np.sort(idx)
    return X_windows[idx], jumps_subset.iloc[idx].copy()


def compute_D1_for_wavelet(
    X_windows: np.ndarray,
    asymmetry: np.ndarray,
    *,
    wavelet: str,
    cfg: PolandDailyConfig,
) -> Tuple[np.ndarray, float]:
    wm = WaveletModel(
        n_layers=0,
        n_neurons=0,
        n_outputs=0,
        J=cfg.J,
        wavelet=wavelet,
        kernel=cfg.kernel,
        n_components=cfg.n_components,
        include_scattering_spectra=cfg.include_scattering_spectra,
        random_state=cfg.random_seed,
    )
    emb = wm.fit_transform(X_windows)
    d1 = np.asarray(emb[:, 0], dtype=float)

    corr = float(np.corrcoef(d1, asymmetry)[0, 1]) if len(d1) > 1 else 0.0
    if np.isfinite(corr) and corr < 0:
        d1 = -d1
        corr = -corr
    if not np.isfinite(corr):
        corr = 0.0
    return d1, corr


def _corr_flip(x: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Flip sign of x so corr(x, target) >= 0. Returns (x_oriented, corr_abs).
    """
    if x.size <= 1:
        return x, 0.0
    c = float(np.corrcoef(x, target)[0, 1])
    if not np.isfinite(c):
        return x, 0.0
    if c < 0:
        return -x, -c
    return x, c


def compute_D123_for_wavelet(
    X_windows: np.ndarray,
    *,
    wavelet: str,
    cfg: PolandDailyConfig,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Compute the first 3 KPCA coordinates (D1, D2, D3) for a given wavelet and orient their signs
    using simple, interpretable targets (post/pre activity, mean-reversion, trend).
    """
    center = int(cfg.window_steps)
    asym = _activity_asymmetry(X_windows, center=center)

    # Simple local proxies around the jump (t=-1, +1) on aligned profiles:
    # - mean-reversion proxy: large when x(-1) is high and x(+1) is low
    # - trend proxy: large when both x(-1) and x(+1) are high
    mr = X_windows[:, center - 1] - X_windows[:, center + 1]
    tr = X_windows[:, center - 1] + X_windows[:, center + 1]

    wm = WaveletModel(
        n_layers=0,
        n_neurons=0,
        n_outputs=0,
        J=cfg.J,
        wavelet=wavelet,
        kernel=cfg.kernel,
        n_components=cfg.n_components,
        include_scattering_spectra=cfg.include_scattering_spectra,
        random_state=cfg.random_seed,
    )
    emb = wm.fit_transform(X_windows)
    d1 = np.asarray(emb[:, 0], dtype=float)
    d2 = np.asarray(emb[:, 1], dtype=float) if emb.shape[1] > 1 else np.zeros_like(d1)
    d3 = np.asarray(emb[:, 2], dtype=float) if emb.shape[1] > 2 else np.zeros_like(d1)

    d1, c1 = _corr_flip(d1, asym)
    d2, c2 = _corr_flip(d2, mr)
    d3, c3 = _corr_flip(d3, tr)

    return {"D1": d1, "D2": d2, "D3": d3}, {"D1": c1, "D2": c2, "D3": c3}


def plot_profiles_by_quantiles(
    X_windows: np.ndarray,
    score: np.ndarray,
    *,
    window_steps: int,
    quantiles: Tuple[float, ...],
    title: str,
    yaxis_title: str = "aligned x(t)",
) -> go.Figure:
    """
    Plot average aligned profiles across score quantile bins.
    """
    center = int(window_steps)
    t_axis = np.arange(-center, center + 1)

    order = np.argsort(score)
    X_sorted = X_windows[order]
    n = X_sorted.shape[0]

    colors = px.colors.sequential.Viridis
    fig = go.Figure()
    for i in range(len(quantiles) - 1):
        q_s, q_e = float(quantiles[i]), float(quantiles[i + 1])
        idx_s, idx_e = int(q_s * n), int(q_e * n)
        if idx_e <= idx_s:
            continue
        avg = np.mean(X_sorted[idx_s:idx_e], axis=0)
        color_idx = int(i / max(1, (len(quantiles) - 2)) * (len(colors) - 1))
        fig.add_trace(
            go.Scatter(
                x=t_axis,
                y=avg,
                mode="lines",
                name=f"Q {q_s:.2f}-{q_e:.2f}",
                line=dict(color=colors[color_idx], width=2),
            )
        )

    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="time (steps) relative to jump",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        height=420,
    )
    return fig


def plot_D123_profiles_for_wavelet(
    X_windows: np.ndarray,
    scores: Dict[str, np.ndarray],
    corrs: Dict[str, float],
    *,
    wavelet: str,
    cfg: PolandDailyConfig,
) -> go.Figure:
    """
    One figure per wavelet, with three subplots for D1/D2/D3, each showing quantile-averaged profiles.
    """
    center = int(cfg.window_steps)
    t_axis = np.arange(-center, center + 1)
    colors = px.colors.sequential.Viridis

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"D1 (corr={corrs.get('D1', 0.0):.2f})",
            f"D2 (corr={corrs.get('D2', 0.0):.2f})",
            f"D3 (corr={corrs.get('D3', 0.0):.2f})",
        ),
        horizontal_spacing=0.08,
    )

    for col, key in enumerate(("D1", "D2", "D3"), start=1):
        score = scores[key]
        order = np.argsort(score)
        X_sorted = X_windows[order]
        n = X_sorted.shape[0]

        for i in range(len(cfg.profile_quantiles) - 1):
            q_s, q_e = float(cfg.profile_quantiles[i]), float(cfg.profile_quantiles[i + 1])
            idx_s, idx_e = int(q_s * n), int(q_e * n)
            if idx_e <= idx_s:
                continue
            avg = np.mean(X_sorted[idx_s:idx_e], axis=0)
            color_idx = int(i / max(1, (len(cfg.profile_quantiles) - 2)) * (len(colors) - 1))
            fig.add_trace(
                go.Scatter(
                    x=t_axis,
                    y=avg,
                    mode="lines",
                    name=f"Q {q_s:.2f}-{q_e:.2f}",
                    legendgroup=f"Q{i}",
                    showlegend=(col == 1),  # show legend only once
                    line=dict(color=colors[color_idx], width=2),
                ),
                row=1,
                col=col,
            )
        fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=col)

    fig.update_layout(
        template="plotly_white",
        title=f"Poland daily: average aligned profiles by quantiles — wavelet `{wavelet}`",
        hovermode="x unified",
        height=460,
        margin=dict(l=40, r=20, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
    )
    for col in (1, 2, 3):
        fig.update_xaxes(title_text="time (steps)", row=1, col=col)
        fig.update_yaxes(title_text="aligned x(t)", row=1, col=col)
    return fig


def plot_D1_violin(d1_by_wavelet: Dict[str, np.ndarray], corr_by_wavelet: Dict[str, float]) -> go.Figure:
    rows: List[Dict[str, object]] = []
    for w, d1 in d1_by_wavelet.items():
        for v in d1:
            rows.append({"wavelet": w, "D1": float(v), "corr_asym": float(corr_by_wavelet.get(w, 0.0))})
    df = pd.DataFrame(rows)

    order = list(d1_by_wavelet.keys())
    fig = px.violin(
        df,
        x="wavelet",
        y="D1",
        color="wavelet",
        box=True,
        points="outliers",
        category_orders={"wavelet": order},
        title="D1 distribution across wavelets (D1 sign oriented by post/pre activity asymmetry)",
    )
    # Add correlation values in x tick labels via annotations.
    ann = []
    for i, w in enumerate(order):
        ann.append(
            dict(
                x=w,
                y=float(df["D1"].max()) if not df.empty else 0.0,
                xref="x",
                yref="y",
                text=f"corr={corr_by_wavelet.get(w, 0.0):.2f}",
                showarrow=False,
                yshift=18,
                font=dict(size=11),
            )
        )
    fig.update_layout(template="plotly_white", showlegend=False, annotations=ann, height=520)
    return fig


# %% [markdown]
# #### Run the Poland daily comparison
#
# If this is slow, reduce `CFG.max_windows` (e.g. 400–800) or reduce the number of wavelets.

# %%
CFG = PolandDailyConfig()
CFG.out_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading Poland data from {CFG.data_dir} ...")
all_dfs = curate_stooq_dir_daily(str(CFG.data_dir), pattern="*.txt", recursive=True)
print(f"Loaded {len(all_dfs)} tickers before length filter.")

valid_tickers = [t for t, d in all_dfs.items() if d is not None and len(d) > CFG.min_len]
valid_tickers.sort(key=lambda t: len(all_dfs[t]), reverse=True)
print(f"Found {len(valid_tickers)} valid tickers (len > {CFG.min_len}).")

dfs_all = {t: all_dfs[t] for t in valid_tickers}
filtered_dfs = _prepare_daily_series(dfs_all)

print("Detecting jumps (daily)...")
jumps_df = detect_jumps_many(filtered_dfs, threshold=CFG.threshold)
print(f"Detected {len(jumps_df)} total jumps.")

if len(jumps_df) == 0:
    # Diagnostics (same as reproduce_poland_daily)
    sample_ticker, sample_df = max(filtered_dfs.items(), key=lambda kv: len(kv[1]))
    print(f"No jumps found. Plotting jump-score diagnostics for: {sample_ticker} (n={len(sample_df)})")
    scores = compute_jump_score(sample_df, price_col="close")
    fig_hist = px.histogram(scores.reset_index(), x="score", nbins=120, title=f"Jump score distribution x(t) - {sample_ticker}")
    fig_hist.add_vline(x=CFG.threshold, line_dash="dash", line_color="red")
    fig_hist.add_vline(x=-CFG.threshold, line_dash="dash", line_color="red")
    fig_hist.update_layout(template="plotly_white")
    fig_hist.show()
    raise RuntimeError("No jumps detected; adjust CFG.threshold or inspect data.")

print("Extracting aligned windows...")
X_windows, jumps_subset = _extract_aligned_windows_daily(filtered_dfs, jumps_df, window_steps=CFG.window_steps)
print(f"Extracted {len(X_windows)} valid windows.")
if len(X_windows) < 50:
    raise RuntimeError(f"Not enough windows for KPCA: n={len(X_windows)} (need ~> 50).")

# Optionally subsample for speed
X_windows, jumps_subset = _subsample_windows(X_windows, jumps_subset, max_windows=CFG.max_windows, seed=CFG.random_seed)
print(f"Using n={len(X_windows)} windows for wavelet comparison.")

asym = _activity_asymmetry(X_windows, center=CFG.window_steps)

wavelets_to_compare = list(WAVELET_CANDIDATES)
print("Wavelets:", wavelets_to_compare)

d1_by_wavelet: Dict[str, np.ndarray] = {}
corr_by_wavelet: Dict[str, float] = {}

for w in wavelets_to_compare:
    print(f"Fitting WaveletModel for wavelet={w} ...")
    d1, corr = compute_D1_for_wavelet(X_windows, asym, wavelet=w, cfg=CFG)
    d1_by_wavelet[w] = d1
    corr_by_wavelet[w] = corr
    print(f"  corr(D1, asymmetry) = {corr:.3f}")

fig = plot_D1_violin(d1_by_wavelet, corr_by_wavelet)
out_path = CFG.out_dir / "poland_daily_D1_violin_across_wavelets.html"
fig.write_html(out_path)
print(f"Saved: {out_path}")
fig.show()

# %% [markdown]
# #### Profiles for each wavelet (D1/D2/D3)
#
# For each wavelet, we reproduce the "many average profiles along quantiles" visualization (like in
# `notebooks/jump/reproduce_poland_daily.ipynb`), but for **each** of the 3 KPCA coordinates.

# %%
if CFG.make_profile_plots:
    profiles_dir = CFG.out_dir / CFG.profiles_dirname
    profiles_dir.mkdir(parents=True, exist_ok=True)

    wavelets_for_profiles = list(CFG.profile_wavelets) if CFG.profile_wavelets is not None else wavelets_to_compare
    print("Making profile plots for wavelets:", wavelets_for_profiles)

    for w in wavelets_for_profiles:
        print(f"Profile plots for wavelet={w} ...")
        scores, corrs = compute_D123_for_wavelet(X_windows, wavelet=w, cfg=CFG)
        fig_profiles = plot_D123_profiles_for_wavelet(X_windows, scores, corrs, wavelet=w, cfg=CFG)
        out_path = profiles_dir / f"poland_daily_profiles_{w}.html"
        fig_profiles.write_html(out_path)
        print(f"  Saved: {out_path}")
        if CFG.show_profile_plots:
            fig_profiles.show()


