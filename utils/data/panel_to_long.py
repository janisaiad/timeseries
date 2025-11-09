from __future__ import annotations
from typing import Dict, Iterable, Literal, Optional, Tuple
import numpy as np
import pandas as pd


def _compute_series(
    df: pd.DataFrame,
    value: Literal["price", "return", "log_return"],
    price_col: str,
    dropna: bool,
) -> pd.Series:
    '''
    we compute a 1d series from a single-ticker dataframe: price/return/log_return
    '''
    if price_col not in df.columns:
        raise KeyError(f"missing column '{price_col}' in dataframe")  # we validate input
    s = df[price_col].astype("float64").copy()  # we select price column
    if value == "price":
        out = s  # we pass-through price
    elif value == "return":
        out = s.pct_change()  # we compute simple returns
    elif value == "log_return":
        out = np.log(s).diff()  # we compute log returns
    else:
        raise ValueError("value must be one of {'price','return','log_return'}")  # we validate param
    return out.dropna() if dropna else out  # we drop nas if requested


def _standardize_series(
    s: pd.Series,
    mode: Optional[Literal["zscore", "unitvar", "demean"]],
    eps: float,
) -> pd.Series:
    '''
    we standardize a series per ticker with different modes
    '''
    if mode is None:
        return s  # we skip standardization
    mu = float(s.mean()) if s.size > 0 else 0.0  # we compute mean
    sd = float(s.std(ddof=1)) if s.size > 1 else 0.0  # we compute std
    if mode == "demean":
        return s - mu  # we remove mean
    if mode in ("zscore", "unitvar"):
        denom = sd if sd > eps else eps  # we avoid division by zero
        return (s - mu) / denom if mode == "zscore" else s / denom  # we scale
    raise ValueError("mode must be None, 'zscore', 'unitvar' or 'demean'")  # we validate param


def panel_to_long_vector(
    dfs: Dict[str, pd.DataFrame],
    value: Literal["price", "return", "log_return"] = "log_return",
    price_col: str = "close",
    standardize: Optional[Literal["zscore", "unitvar", "demean"]] = "zscore",
    ticker_order: Optional[Iterable[str]] = None,
    dropna: bool = True,
    zscore_clip: Optional[Tuple[float, float]] = (-8.0, 8.0),
    eps: float = 1e-12,
) -> np.ndarray:
    '''
    we convert a mapping {ticker: df} into a single concatenated 1d numpy vector

    we do:
      1) compute chosen value per ticker (price/return/log_return)
      2) optionally standardize per ticker
      3) optionally clip extreme standardized values
      4) concatenate by ticker (sorted unless ticker_order supplied)

    returns:
      np.ndarray of shape (N,), concatenating all tickers in order
    '''
    tickers = list(ticker_order) if ticker_order is not None else sorted(dfs.keys())  # we define order
    parts: list[np.ndarray] = []  # we collect per-ticker arrays
    for t in tickers:
        if t not in dfs:
            continue  # we skip missing tickers silently
        s = _compute_series(dfs[t].sort_index(), value=value, price_col=price_col, dropna=dropna)  # we compute series
        s = _standardize_series(s, mode=standardize, eps=eps)  # we standardize per ticker
        if zscore_clip is not None:
            lo, hi = zscore_clip  # we unpack clip bounds
            s = s.clip(lower=lo, upper=hi)  # we clip tails
        parts.append(s.to_numpy(dtype=np.float64))  # we convert to numpy
    if not parts:
        return np.array([], dtype=np.float64)  # we return empty if nothing
    long_vec = np.concatenate(parts, axis=0)  # we concatenate into a long vector
    return long_vec  # we return result


def panel_to_matrix(
    dfs: Dict[str, pd.DataFrame],
    value: Literal["price", "return", "log_return"] = "log_return",
    price_col: str = "close",
    standardize: Optional[Literal["zscore", "unitvar", "demean"]] = "zscore",
    align: Literal["intersection", "union"] = "intersection",
    dropna_after_align: bool = True,
    eps: float = 1e-12,
) -> Tuple[pd.DatetimeIndex, list[str], np.ndarray]:
    '''
    we align tickers on a common datetime index and return a time x tickers matrix for heatmaps

    returns:
      index: datetimeindex
      tickers: list[str]
      X: np.ndarray shape (T, K)
    '''
    tickers = sorted(dfs.keys())  # we define tickers
    series_map: dict[str, pd.Series] = {}  # we build per-ticker series
    for t in tickers:
        s = _compute_series(dfs[t].sort_index(), value=value, price_col=price_col, dropna=False)  # we compute series
        s = _standardize_series(s, mode=standardize, eps=eps)  # we standardize per ticker
        series_map[t] = s  # we store
    if align == "intersection":
        idx = None  # we initialize index
        for s in series_map.values():
            idx = s.index if idx is None else idx.intersection(s.index)  # we compute intersection
    elif align == "union":
        idx = None  # we initialize index
        for s in series_map.values():
            idx = s.index if idx is None else idx.union(s.index)  # we compute union
    else:
        raise ValueError("align must be 'intersection' or 'union'")  # we validate param
    frames = []
    for t in tickers:
        frames.append(series_map[t].reindex(idx))  # we reindex to common index
    mat = pd.concat(frames, axis=1)  # we combine columns
    mat.columns = tickers  # we set column names
    if dropna_after_align:
        mat = mat.dropna(axis=0, how="any")  # we drop rows with any nan
    return mat.index, tickers, mat.to_numpy(dtype=np.float64)  # we return aligned matrix


def windows_from_series(
    x: np.ndarray,
    window_len: int = 119,
    stride: int = 10,
    drop_incomplete: bool = True,
) -> np.ndarray:
    '''
    we slice a 1d array into overlapping windows of fixed length for wavelet embedding  # we describe purpose
    '''
    arr = np.asarray(x, dtype=np.float64).reshape(-1)  # we ensure 1d
    if window_len <= 0 or stride <= 0:  # we validate parameters
        raise ValueError("window_len and stride must be positive")  # we guard invalid inputs
    n = arr.size  # we get length
    if n < window_len:  # we handle short input
        return np.empty((0, window_len), dtype=np.float64)  # we return empty
    starts = np.arange(0, n - window_len + 1, stride, dtype=int)  # we compute start indices
    if starts.size == 0:  # we handle edge case
        return np.empty((0, window_len), dtype=np.float64)  # we return empty
    win = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window_len)  # we build sliding view
    win = win[starts]  # we select by stride
    if not drop_incomplete and (starts[-1] + window_len < n):  # we include remainder if requested
        tail = arr[-window_len:]  # we take last window
        win = np.vstack([win, tail])  # we append
    return win.copy()  # we return contiguous windows