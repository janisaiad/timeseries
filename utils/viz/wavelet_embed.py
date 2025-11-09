from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from model.wavelet.wavelet import WaveletModel  # we import the wavelet model


def embed_series_windows(
    x_windows: np.ndarray,
    J: int = 6,
    kernel: str = "rbf",
    n_components: int = 3,
    random_state: Optional[int] = 0,
) -> Tuple[np.ndarray, WaveletModel]:
    '''
    we compute kpca embedding on a batch of windows using WaveletModel  # we describe purpose
    '''
    x_windows = np.asarray(x_windows, dtype=np.float64)  # we ensure array
    if x_windows.ndim != 2:  # we validate shape
        raise ValueError("x_windows must be 2D with shape (n_windows, T)")  # we guard misuse
    T = x_windows.shape[1]  # we get window length
    center = T // 2  # we choose middle as jump time
    wm = WaveletModel(n_layers=0, n_neurons=0, n_outputs=0, J=J, kernel=kernel, n_components=n_components, center_index=center, random_state=random_state)  # we build model
    Z = wm.fit_transform(x_windows)  # we compute embedding
    return Z, wm  # we return embedding and model


def plot_kpca_components(
    Z: np.ndarray,
    title: Optional[str] = "wavelet kpca components (first three)",
    figsize: Tuple[int, int] = (12, 6),
) -> Tuple[plt.Figure, np.ndarray]:
    '''
    we plot the first three kpca components over window index and pairwise scatters  # we describe purpose
    '''
    Z = np.asarray(Z, dtype=np.float64)  # we ensure numeric
    if Z.ndim != 2 or Z.shape[1] < 3:  # we validate dimensions
        raise ValueError("Z must be of shape (n_samples, >=3)")  # we guard misuse
    n = Z.shape[0]  # we get number of windows
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)  # we create grid
    if title:
        fig.suptitle(title)  # we add title
    idx = np.arange(n)  # we build x-axis
    axes[0, 0].plot(idx, Z[:, 0], lw=0.9, color="#1f77b4")  # we plot comp1
    axes[0, 0].set_title("comp 1")  # we label
    axes[0, 1].plot(idx, Z[:, 1], lw=0.9, color="#ff7f0e")  # we plot comp2
    axes[0, 1].set_title("comp 2")  # we label
    axes[0, 2].plot(idx, Z[:, 2], lw=0.9, color="#2ca02c")  # we plot comp3
    axes[0, 2].set_title("comp 3")  # we label
    axes[1, 0].scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.7, c="#1f77b4")  # we scatter 1 vs 2
    axes[1, 0].set_xlabel("comp 1")  # we label
    axes[1, 0].set_ylabel("comp 2")  # we label
    axes[1, 1].scatter(Z[:, 0], Z[:, 2], s=8, alpha=0.7, c="#ff7f0e")  # we scatter 1 vs 3
    axes[1, 1].set_xlabel("comp 1")  # we label
    axes[1, 1].set_ylabel("comp 3")  # we label
    axes[1, 2].scatter(Z[:, 1], Z[:, 2], s=8, alpha=0.7, c="#2ca02c")  # we scatter 2 vs 3
    axes[1, 2].set_xlabel("comp 2")  # we label
    axes[1, 2].set_ylabel("comp 3")  # we label
    return fig, axes  # we return figure and axes


