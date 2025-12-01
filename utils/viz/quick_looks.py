from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # we try to use seaborn if available
except Exception:
    sns = None  # we fallback gracefully


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    '''
    we compute autocorrelation up to max_lag using normalization by variance
    '''
    x = np.asarray(x, dtype=np.float64).reshape(-1)  # we ensure 1d
    x = x[~np.isnan(x)]  # we drop nans
    if x.size == 0:
        return np.zeros(max_lag + 1, dtype=np.float64)  # we handle empty
    x = x - x.mean()  # we demean
    denom = float(np.dot(x, x)) if x.size > 0 else 1.0  # we compute energy
    ac = np.correlate(x, x, mode="full")  # we compute full correlation
    ac = ac[ac.size // 2 : ac.size // 2 + max_lag + 1]  # we take non-negative lags
    ac = ac / denom if denom > 0 else ac  # we normalize
    return ac  # we return acf


def _rolling_vol(x: np.ndarray, win: int) -> np.ndarray:
    '''
    we compute rolling standard deviation over a fixed window
    '''
    x = np.asarray(x, dtype=np.float64).reshape(-1)  # we ensure 1d
    if x.size < win:
        return np.array([], dtype=np.float64)  # we handle short input
    c = np.cumsum(np.insert(x, 0, 0.0))  # we cumulative sum
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))  # we cumulative sum of squares
    s = c[win:] - c[:-win]  # we windowed sum
    s2 = c2[win:] - c2[:-win]  # we windowed sum of squares
    var = (s2 - (s * s) / win) / max(win - 1, 1)  # we compute unbiased variance
    vol = np.sqrt(np.maximum(var, 0.0))  # we guard negative due to fp
    return vol  # we return rolling vol


def _qq_theoretical_empirical(x: np.ndarray, n: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    '''
    we compute qq-plot points against standard normal without scipy
    '''
    x = np.asarray(x, dtype=np.float64).reshape(-1)  # we ensure 1d
    x = x[~np.isnan(x)]  # we drop nans
    if x.size == 0:
        return np.array([]), np.array([])  # we handle empty
    x_sorted = np.sort(x)  # we sort sample
    m = min(n, x_sorted.size)  # we cap points
    q_emp = np.linspace(0.5 / m, 1 - 0.5 / m, m)  # we define probs
    idx = (q_emp * (x_sorted.size - 1)).astype(int)  # we map to indices
    y = x_sorted[idx]  # we take empirical quantiles
    # theoretical normal quantiles via inverse error function approximation  # we compute theoretical q
    p = q_emp  # we alias
    a = 0.147  # we winitzki constant
    t = np.where(p < 0.5, p, 1 - p)  # we fold around 0.5
    s = -2.0 * np.log(2.0 * t)  # we compute helper
    z = np.sign(p - 0.5) * np.sqrt(s - (np.log(s) + np.log(4.0 / np.pi)) / s)  # we approximate inv erfc
    return z, y  # we return theoretical vs empirical


def plot_long_overview(
    x: np.ndarray,
    title: Optional[str] = None,
    sample_n: int = 5000,
    acf_lag: int = 128,
    roll_win: int = 256,
    bins: int = 100,
    figsize: Tuple[int, int] = (14, 8),
) -> Tuple[plt.Figure, np.ndarray]:
    '''
    we plot a 2x3 grid: sample line, histogram+kde, acf, rolling vol, qq-plot, log-ccdf
    '''
    x = np.asarray(x, dtype=np.float64).reshape(-1)  # we ensure 1d
    x = x[~np.isnan(x)]  # we drop nans
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)  # we create grid
    if title:
        fig.suptitle(title)  # we add title

    n = min(sample_n, x.size)  # we cap sample length
    axes[0, 0].plot(np.arange(n), x[:n], lw=0.8, color="#1f77b4")  # we plot sample line
    axes[0, 0].set_title("sample (first N)")  # we annotate
    axes[0, 0].set_xlabel("index")  # we label
    axes[0, 0].set_ylabel("value")  # we label

    if sns is not None:
        sns.histplot(x, bins=bins, stat="density", ax=axes[0, 1], color="#9467bd", edgecolor="none")  # we plot hist
        sns.kdeplot(x, ax=axes[0, 1], color="#d62728", lw=1.2)  # we overlay kde
    else:
        axes[0, 1].hist(x, bins=bins, density=True, color="#9467bd", alpha=0.8)  # we plot hist
    axes[0, 1].set_title("distribution (hist + kde)")  # we annotate
    ac = _acf(x, max_lag=acf_lag)  # we compute acf
    axes[0, 2].stem(np.arange(ac.size), ac, basefmt=" ")  # we plot acf compatibly without use_line_collection
    axes[0, 2].set_title("acf")  # we annotate
    axes[0, 2].set_xlabel("lag")  # we label
    rv = _rolling_vol(x, roll_win)  # we compute rolling vol
    axes[1, 0].plot(np.arange(rv.size), rv, color="#2ca02c", lw=1.0)  # we plot rolling vol
    axes[1, 0].set_title(f"rolling vol (win={roll_win})")  # we annotate
    axes[1, 0].set_xlabel("index")  # we label

    qx, qy = _qq_theoretical_empirical(x)  # we compute qq points
    axes[1, 1].scatter(qx, qy, s=6, alpha=0.6, color="#ff7f0e")  # we scatter qq
    lim = np.nanmax(np.abs([qx, qy])) if qx.size and qy.size else 1.0  # we set symmetric limits
    qq_lim = 15.0  # we set desired qq axis half-range
    axes[1, 1].plot([-qq_lim, qq_lim], [-qq_lim, qq_lim], color="black", lw=1.0, alpha=0.6)  # we add diagonal over fixed range
    axes[1, 1].set_xlim(-qq_lim, qq_lim)  # we enforce qq x-limits [-15, 15]
    axes[1, 1].set_ylim(-qq_lim, qq_lim)  # we enforce qq y-limits [-15, 15]
    axes[1, 1].set_title("qq vs normal")  # we annotate
    axes[1, 1].set_xlabel("normal q")  # we label
    axes[1, 1].set_ylabel("empirical q")  # we label

    xs = np.sort(np.abs(x))  # we sort absolute values
    xs = xs[xs > 0]  # we remove zeros
    if xs.size > 0:
        ccdf = 1.0 - np.arange(1, xs.size + 1) / (xs.size + 1.0)  # we compute ccdf
        axes[1, 2].loglog(xs, ccdf, color="#8c564b", lw=1.0)  # we plot log-ccdf
        axes[1, 2].set_xlim(5e-2, 5e0)  # we enforce x-range for tail plot
    axes[1, 2].set_title("abs(x) tail (log-ccdf)")  # we annotate
    axes[1, 2].set_xlabel("|x|")  # we label
    axes[1, 2].set_ylabel("P(|X| > x)")  # we label

    return fig, axes  # we return figure and axes


def scatter_embedding(
    Z: np.ndarray,
    color: Optional[Sequence[float]] = None,
    title: str = "kpca embedding",
    figsize: Tuple[int, int] = (7, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    '''
    we scatter-plot the first two components of an embedding matrix
    '''
    Z = np.asarray(Z, dtype=np.float64)  # we ensure array
    if Z.ndim != 2 or Z.shape[1] < 2:
        raise ValueError("Z must be of shape (n_samples, >=2)")  # we validate input
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)  # we create figure
    if color is None:
        ax.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.7, c="#1f77b4")  # we plot without color
    else:
        sc = ax.scatter(Z[:, 0], Z[:, 1], s=10, alpha=0.8, c=np.asarray(color), cmap="viridis")  # we plot with colormap
        fig.colorbar(sc, ax=ax, shrink=0.8, label="color")  # we add colorbar
    ax.set_xlabel("comp 1")  # we label
    ax.set_ylabel("comp 2")  # we label
    ax.set_title(title)  # we set title
    return fig, ax  # we return figure and axis