from __future__ import annotations
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd


def _infer_bar_timedelta(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
    """
    Infer a representative bar size from a DatetimeIndex using the median positive time delta.
    Returns None if it can't be inferred robustly.
    """
    if not isinstance(index, pd.DatetimeIndex):
        return None
    if len(index) < 2:
        return None

    # diff() keeps index order; ensure sorted for stability
    idx = index.sort_values()
    deltas = idx.to_series().diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return None
    try:
        td = deltas.median()
    except Exception:
        return None
    if pd.isna(td) or td <= pd.Timedelta(0):
        return None
    return td


def _default_sigma_span(bar_td: Optional[pd.Timedelta], auto_params: bool) -> int:
    """
    Choose an EWM span for sigma(t) based on inferred bar size.
    Keeps historical behavior for 5-minute bars (span ~ 100).
    """
    if not auto_params or bar_td is None:
        return 100
    if bar_td >= pd.Timedelta(days=1):
        return 20
    target_span_time = pd.Timedelta(minutes=500)  # 100 * 5min
    span = int(max(3, round(target_span_time / bar_td)))
    return span if span > 0 else 100


def _default_min_periods(span: int) -> int:
    """
    Choose a stable min_periods for the EWM std.
    For span=100 this yields 10 (previous behavior), for small spans it stays >=3.
    """
    return int(min(10, max(3, round(span * 0.1))))


def compute_u_shape(returns: pd.Series) -> pd.Series:
    """
    Estimate an intraday volatility seasonality profile f(t) using robust statistics
    (median of |returns|) aggregated by time-of-day. Returns f(t) aligned to the input index.

    Notes:
    - For daily-or-slower data (or if there's only one unique time-of-day), intraday seasonality
      is not identifiable; this function returns a vector of ones.
    """
    # ensure we have time info
    if not isinstance(returns.index, pd.DatetimeIndex):
        return pd.Series(1.0, index=returns.index)

    bar_td = _infer_bar_timedelta(returns.index)
    if bar_td is not None and bar_td >= pd.Timedelta(days=1):
        return pd.Series(1.0, index=returns.index)

    # absolute returns
    abs_r = returns.abs()
    
    # grouping key: time of day (HH:MM)
    # for 5-min data this works well. for irregular data, might need binning.
    times = returns.index.strftime('%H:%M')
    # If there is only one time-of-day bucket (common for daily data), seasonality is not identifiable.
    if len(pd.unique(times)) <= 1:
        return pd.Series(1.0, index=returns.index)
    
    # compute median of absolute returns per bin as robust scale proxy
    # f_bin ~ proportional to std dev
    medians = abs_r.groupby(times).median()
    
    # avoid zero medians (if assets don't move much) - clip at some small percentile or epsilon
    # here we just replace 0 with the global median * 0.1 to avoid explosions
    global_median = medians[medians > 0].median()
    if pd.isna(global_median) or global_median == 0:
        global_median = 1e-4
    medians = medians.replace(0, global_median * 0.1)
    
    # normalize so that mean(f^2) approx 1 (standard convention) or mean(f)=1
    # paper doesn't specify exact norm, but f(t)sigma(t) is the total vol.
    # let's normalize so quadratic mean is 1.
    qmean = np.sqrt((medians**2).mean())
    if qmean > 0:
        medians /= qmean
    else:
        medians[:] = 1.0
        
    # map back to series
    f_series = pd.Series(times.map(medians).values, index=returns.index)
    return f_series.fillna(1.0)


def compute_jump_score(
    df: pd.DataFrame,
    price_col: str = "close",
    min_vol: float = 1e-4,
    auto_params: bool = True,
    sigma_span: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute the full jump score time series x(t) = r(t) / (f(t) * sigma(t)).

    Returns a dataframe indexed like df with columns:
      - return: simple return r(t) = pct_change(close)
      - f: intraday seasonality estimate (or ones for daily-like data)
      - sigma: local volatility estimate on deseasonalized returns
      - score: x(t)
    """
    if df.empty:
        return pd.DataFrame()

    prices = df[price_col]
    r = prices.pct_change().fillna(0.0)

    bar_td: Optional[pd.Timedelta] = None
    if isinstance(df.index, pd.DatetimeIndex):
        bar_td = _infer_bar_timedelta(df.index)

    f = compute_u_shape(r)
    r_des = r / f

    span_eff = int(sigma_span) if sigma_span is not None else _default_sigma_span(bar_td, auto_params)
    minp_eff = _default_min_periods(span_eff)

    sigma = (
        r_des.ewm(span=span_eff, min_periods=minp_eff)
        .std()
        .bfill()
        .fillna(min_vol)
        .replace(0, min_vol)
    )

    x = r_des / sigma
    out = pd.DataFrame({"return": r, "f": f, "sigma": sigma, "score": x}, index=df.index)
    return out


def detect_jumps_single(
    df: pd.DataFrame, 
    ticker: str,
    price_col: str = "close", 
    threshold: float = 4.0,
    cluster_window: pd.Timedelta = pd.Timedelta("1h"),
    min_vol: float = 1e-4,
    auto_params: bool = True,
    sigma_span: Optional[int] = None
) -> pd.DataFrame:
    """
    Detect jumps for a single ticker using a k-sigma criterion on deseasonalized returns.

    Output columns: [timestamp, ticker, return, score, f, sigma]

    Frequency handling (hourly/daily):
    - If auto_params=True (default), the volatility estimator span is inferred from bar size:
      - Intraday (<1D): span corresponds to ~500 minutes (so 5-min ≈ 100, 1h ≈ 8)
      - Daily (>=1D): span ≈ 20 bars
    - Intraday seasonality f(t) is automatically disabled for daily-like data.
    """
    if df.empty:
        return pd.DataFrame()
        
    # Try to infer bar size (used for auto parameterization)
    bar_td: Optional[pd.Timedelta] = None
    if isinstance(df.index, pd.DatetimeIndex):
        bar_td = _infer_bar_timedelta(df.index)

    # 1. Returns
    # use log returns or simple? paper says "returns time-series r(t)"
    # usually log returns for additivity, but for 1-min simple is fine.
    # let's use pct_change
    prices = df[price_col]
    r = prices.pct_change().fillna(0.0)
    
    # 2. Intraday Pattern f(t)
    f = compute_u_shape(r)
    
    # 3. Deseasonalize
    r_des = r / f
    
    # 4. Local Volatility sigma(t)
    # Paper: "sigma(t) is an estimator of local volatility"
    # typically computed on r_des. 
    # We use EWM std dev. Span depends on data freq. 
    # If 5-min bars, previous default span=100 (≈ 500 minutes) is kept.
    sigma_span_eff = int(sigma_span) if sigma_span is not None else _default_sigma_span(bar_td, auto_params)
    min_periods_eff = _default_min_periods(sigma_span_eff)

    sigma = (
        r_des.ewm(span=sigma_span_eff, min_periods=min_periods_eff)
        .std()
        .bfill()
        .fillna(min_vol)
    )
    
    # avoid zero sigma
    sigma = sigma.replace(0, min_vol)
    
    # 5. Jump Score x(t)
    # x(t) = r(t) / (f(t) * sigma(t))
    # equivalent to r_des / sigma
    x = r_des / sigma
    
    # 6. Thresholding
    is_jump = x.abs() > threshold
    jump_candidates = x[is_jump]
    
    if jump_candidates.empty:
        return pd.DataFrame()
    
    # 7. Clustering / Deduping
    # "discard all jumps that follow an initial jump" (within a cluster)
    # We iterate and keep first jump, skip subsequent ones within window
    
    kept_times = []
    last_time = None

    # For hourly/daily bars, default 1h clustering can inadvertently suppress consecutive bars.
    # If auto_params is enabled and the bar size is >= 30min, disable clustering by default.
    cluster_window_eff = cluster_window
    if auto_params and bar_td is not None and bar_td >= pd.Timedelta(minutes=30):
        cluster_window_eff = pd.Timedelta(0)
    
    # timestamps are sorted
    for t in jump_candidates.index:
        if last_time is None or (t - last_time) > cluster_window_eff:
            kept_times.append(t)
            last_time = t
        # else: skip replica
        
    if not kept_times:
        return pd.DataFrame()
        
    res = pd.DataFrame({
        "timestamp": kept_times,
        "ticker": ticker,
        "return": r.loc[kept_times].values,
        "score": x.loc[kept_times].values,
        "f": f.loc[kept_times].values,
        "sigma": sigma.loc[kept_times].values
    })
    
    return res


def detect_jumps_many(
    dfs: Dict[str, pd.DataFrame],
    price_col: str = "close",
    threshold: float = 4.0,
    cluster_window: str = "1h",
    auto_params: bool = True,
    sigma_span: Optional[int] = None
) -> pd.DataFrame:
    '''
    we run jump detection on many tickers and combine results
    '''
    all_jumps = []
    win = pd.Timedelta(cluster_window)
    
    for ticker, df in dfs.items():
        jdf = detect_jumps_single(
            df,
            ticker,
            price_col=price_col,
            threshold=threshold,
            cluster_window=win,
            auto_params=auto_params,
            sigma_span=sigma_span,
        )
        if not jdf.empty:
            all_jumps.append(jdf)
            
    if not all_jumps:
        return pd.DataFrame(columns=["timestamp", "ticker", "return", "score", "f", "sigma"])
        
    return pd.concat(all_jumps, ignore_index=True).sort_values("timestamp")


def get_cojumps(
    jumps: pd.DataFrame, 
    min_size: int = 2
) -> pd.DataFrame:
    '''
    we group jumps by timestamp to find co-jumps
    returns a dataframe indexed by timestamp with count and list of tickers
    '''
    if jumps.empty:
        return pd.DataFrame()
        
    g = jumps.groupby("timestamp")
    cojumps = g.agg({
        "ticker": list,
        "score": list,
        "return": list
    })
    cojumps["size"] = cojumps["ticker"].apply(len)
    
    # filter by size
    cojumps = cojumps[cojumps["size"] >= min_size]
    
    return cojumps.sort_index()

