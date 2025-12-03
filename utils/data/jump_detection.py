from __future__ import annotations
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def compute_u_shape(returns: pd.Series) -> pd.Series:
    '''
    we estimate the intraday volatility profile f(t) using robust statistics (median absolute deviation)
    aggregated by time-of-day. returns f(t) aligned to the input index.
    '''
    # ensure we have time info
    if not isinstance(returns.index, pd.DatetimeIndex):
        return pd.Series(1.0, index=returns.index)

    # absolute returns
    abs_r = returns.abs()
    
    # grouping key: time of day (HH:MM)
    # for 5-min data this works well. for irregular data, might need binning.
    times = returns.index.strftime('%H:%M')
    
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


def detect_jumps_single(
    df: pd.DataFrame, 
    ticker: str,
    price_col: str = "close", 
    threshold: float = 4.0,
    cluster_window: pd.Timedelta = pd.Timedelta("1h"),
    min_vol: float = 1e-4
) -> pd.DataFrame:
    '''
    we detect jumps for a single ticker dataframe using the 4-sigma criterion on 
    deseasonalized returns.
    
    returns dataframe with columns: [timestamp, ticker, return, score, f, sigma]
    '''
    if df.empty:
        return pd.DataFrame()
        
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
    # If 5-min bars, 1 day ~ 78 bars (6.5h). 
    # Let's use a span of ~1-2 days to be "local" but stable.
    # say span=100 for 5-min data.
    sigma = r_des.ewm(span=100, min_periods=10).std().bfill().fillna(min_vol)
    
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
    
    # timestamps are sorted
    for t in jump_candidates.index:
        if last_time is None or (t - last_time) > cluster_window:
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
    cluster_window: str = "1h"
) -> pd.DataFrame:
    '''
    we run jump detection on many tickers and combine results
    '''
    all_jumps = []
    win = pd.Timedelta(cluster_window)
    
    for ticker, df in dfs.items():
        jdf = detect_jumps_single(df, ticker, price_col, threshold, win)
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

