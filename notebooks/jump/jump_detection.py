# %%
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Tuple, Optional
import sys
import os

# Adjust paths to project structure
sys.path.append(os.path.abspath("../../"))

from utils.data.curating_stooq import curate_stooq_dir_5min
from utils.data.jump_detection import detect_jumps_many, get_cojumps, compute_u_shape
from model.wavelet.wavelet import WaveletModel

# %%
# ==================================================================================
# 1) LOAD DATA
# ==================================================================================
data_dir = "/home/janis/4A/timeseries/data/stooq/hungary/"
# Note: Paper uses 1-min data. We are using 5-min data here.
dfs = curate_stooq_dir_5min(data_dir, pattern="*.txt", recursive=True, tz=None)
print(f"Loaded {len(dfs)} tickers.")

# %%
# ==================================================================================
# 2) HELPER FUNCTIONS (Time Filtering & Returns)
# ==================================================================================

def filter_trading_hours(df: pd.DataFrame, remove_start_min: int = 0, remove_end_min: int = 0) -> pd.DataFrame:
    """
    Filters the dataframe to remove the first N minutes and last M minutes of each trading day.
    """
    if df.empty: return df
    
    # Efficiently filter by time of day if possible, or iterate by day
    # Iterating by day is safer for varying market hours, but slower.
    # Here we assume standard intraday data.
    
    grouped = df.groupby(df.index.date)
    filtered_days = []
    
    for date, day_df in grouped:
        if len(day_df) <= 2: continue
        day_df = day_df.sort_index()
        
        start_time = day_df.index[0] + pd.Timedelta(minutes=remove_start_min)
        end_time = day_df.index[-1] - pd.Timedelta(minutes=remove_end_min)
        
        mask = (day_df.index >= start_time) & (day_df.index <= end_time)
        filtered_days.append(day_df[mask])
        
    if not filtered_days: return pd.DataFrame()
    return pd.concat(filtered_days)

def get_log_returns(df: pd.DataFrame, price_col="close") -> pd.Series:
    """Computes log returns: ln(P_t / P_{t-1})"""
    return np.log(df[price_col] / df[price_col].shift(1)).dropna()

# %%
# ==================================================================================
# 3) SCENARIO DEFINITIONS & DISTRIBUTION ANALYSIS
# ==================================================================================
# Paper methodology: "considering only what happened between 10:30 and 15:00"
# US Market Open 9:30, Close 16:00. 
# 10:30 is +60min (Skip first hour). 15:00 is -60min (Skip last hour).

scenarios = {
    "Full Day": (0, 0),
    "No Open/Close (30m)": (30, 30),
    "No Open/Close (1h)": (60, 60)  # Approximates Paper's selection
}

# Pick a sample ticker for distribution visualization
sample_ticker = next(iter(dfs)) if dfs else None
for t, d in dfs.items():
    if len(d) > 1000:
        sample_ticker = t
        break

if sample_ticker:
    print(f"Using {sample_ticker} for distribution plots.")
    df_sample = dfs[sample_ticker]
    
    fig_dist = go.Figure()
    
    for name, (trim_start, trim_end) in scenarios.items():
        df_filtered = filter_trading_hours(df_sample, trim_start, trim_end)
        log_rets = get_log_returns(df_filtered)
        
        # Standardize
        z_scores = (log_rets - log_rets.mean()) / log_rets.std()
        
    fig_dist.add_trace(go.Histogram(
            x=z_scores,
            name=f"{name} (std={log_rets.std():.5f})",
        histnorm='probability density',
            opacity=0.5,
            nbinsx=100
    ))
        
    x_range = np.linspace(-6, 6, 200)
    fig_dist.add_trace(go.Scatter(
        x=x_range, y=stats.norm.pdf(x_range, 0, 1),
        mode='lines', name='Normal(0,1)', line=dict(color='black', dash='dash')
    ))
    
    fig_dist.update_layout(
        title=f"Log Return Distribution (Standardized) - {sample_ticker}",
        xaxis_title="Z-Score",
        yaxis_title="Density",
        barmode='overlay',
        template="plotly_white"
    )
    fig_dist.show()

# %%
# ==================================================================================
# 4) JUMP DETECTION ON SCENARIOS & TIMING ANALYSIS
# ==================================================================================
jump_results = {}

for name, (trim_start, trim_end) in scenarios.items():
    print(f"Running detection for: {name}...")
    # Filter all
    filtered_dfs = {t: filter_trading_hours(d, trim_start, trim_end) for t, d in dfs.items() if not d.empty}
    # Detect
    jumps_df = detect_jumps_many(filtered_dfs, threshold=4.0)
    jump_results[name] = jumps_df
    print(f"  -> Detected {len(jumps_df)} jumps.")

# Plot Time of Day Distribution
fig_time = go.Figure()
for name, j_df in jump_results.items():
    if j_df.empty: continue
    # Decimal hour
    times = j_df["timestamp"].dt.hour + j_df["timestamp"].dt.minute / 60.0
    fig_time.add_trace(go.Histogram(
        x=times, name=name, xbins=dict(size=0.25), opacity=0.6
    ))

fig_time.update_layout(
    title="Jump Occurrence by Time of Day",
    xaxis_title="Hour of Day",
    yaxis_title="Count",
    barmode='group',
    template="plotly_white"
)
fig_time.show()

# %%
# ==================================================================================
# 5) TIMING CLARIFICATION (Zoom Plot)
# ==================================================================================
# Clarification: The marker at T is the return (T-1 -> T). 
full_jumps = jump_results.get("Full Day")
if full_jumps is not None and not full_jumps.empty:
    # Pick max score jump
    sample_jump = full_jumps.iloc[full_jumps["score"].abs().argmax()]
    t_jump = sample_jump["timestamp"]
    ticker = sample_jump["ticker"]
    
    df_viz = dfs[ticker]
    if t_jump in df_viz.index:
        loc = df_viz.index.get_loc(t_jump)
        start_loc = max(0, loc - 6)
        end_loc = min(len(df_viz), loc + 7)
        df_zoom = df_viz.iloc[start_loc:end_loc]
        
        fig_zoom = make_subplots(specs=[[{"secondary_y": True}]])
        fig_zoom.add_trace(go.Scatter(x=df_zoom.index, y=df_zoom["close"], mode='lines+markers', name='Price'), secondary_y=False)
        
        # Jump Marker
        fig_zoom.add_trace(
            go.Scatter(x=[t_jump], y=[sample_jump["score"]], mode="markers+text", name="Jump Score",
                       marker=dict(symbol="triangle-up", size=15, color="red"),
                       text=["Jump Detected<br>(Return ending here)"], textposition="top center"),
            secondary_y=True
        )
        fig_zoom.update_layout(title=f"Zoom on Jump: {ticker} at {t_jump}", template="plotly_white")
        fig_zoom.show()

# %%
# ==================================================================================
# 6) PAPER REPRODUCTION: WAVELET CLASSIFICATION
# ==================================================================================
# We use the "No Open/Close (1h)" scenario as it matches the paper's "10:30 - 15:00" window best.
jumps_for_analysis = jump_results.get("No Open/Close (1h)")
if jumps_for_analysis is None or jumps_for_analysis.empty:
    print("No jumps found in 1h-trimmed dataset. Falling back to Full Day.")
    jumps_for_analysis = jump_results.get("Full Day")

# 6a) Extract Windows x(t)
def extract_jump_windows(jumps_df: pd.DataFrame, dfs: Dict[str, pd.DataFrame], window_steps: int = 12):
    windows = []
    valid_indices = []
    for idx, row in jumps_df.iterrows():
        ticker, ts = row["ticker"], row["timestamp"]
        if ticker not in dfs: continue
        df = dfs[ticker]
        if ts not in df.index: continue
        
        loc = df.index.get_loc(ts)
        if loc - window_steps < 0 or loc + window_steps + 1 > len(df): continue
        
        subset = df.iloc[loc - window_steps : loc + window_steps + 1]
        # Normalized return profile: r(t) / (f * sigma)
        # We use the jump's stored f/sigma for normalization to preserve relative shape
        norm = row["f"] * row["sigma"]
        if norm == 0: norm = 1e-4
            
        x_profile = subset["close"].pct_change().fillna(0.0).values / norm
        
        # Align jump direction (center is at index `window_steps`)
        jump_sign = np.sign(x_profile[window_steps])
        if jump_sign == 0: jump_sign = 1
        
        windows.append(x_profile * jump_sign)
        valid_indices.append(idx)
        
    return jumps_df.loc[valid_indices].copy(), np.array(windows)

print("Extracting windows for classification...")
jumps_subset, X_windows = extract_jump_windows(jumps_for_analysis, dfs, window_steps=12)
print(f"Extracted {len(X_windows)} windows.")

# 6b) Compute D1, D2, D3
if len(X_windows) > 0:
    # D1: Reflexivity (Wavelet Model)
    # J=3 is appropriate for T=25 (5-min data) vs J=6 for T=119 (1-min data)
    wm = WaveletModel(n_layers=0, n_neurons=0, n_outputs=0, J=3, n_components=3)
    embedding = wm.fit_transform(X_windows)
    jumps_subset["D1_reflexivity"] = embedding[:, 0]
    
    # Check orientation of D1 (Positive should be Exogenous/Post-activity)
    center = X_windows.shape[1] // 2
    post_activity = np.sum(np.abs(X_windows[:, center+1:]), axis=1)
    pre_activity = np.sum(np.abs(X_windows[:, :center]), axis=1)
    asymmetry = (post_activity - pre_activity) / (post_activity + pre_activity + 1e-6)
    
    corr = np.corrcoef(jumps_subset["D1_reflexivity"], asymmetry)[0, 1]
    if corr < 0:
        jumps_subset["D1_reflexivity"] *= -1
        
    # D2: Mean Reversion & D3: Trend (Handcrafted Filters)
    # Refined based on Paper Section III.C and III.D
    # D2 (Mean Reversion): Captures V-shape.
    # Paper: "positive value of x(-1) and negative value of x(1)" (for jump-aligned x)
    # We use x(center-1) - x(center+1). Positive => Pre-jump Up, Post-jump Down.
    jumps_subset["D2_mean_reversion"] = X_windows[:, center - 1] - X_windows[:, center + 1]

    # D3 (Trend): Captures Persistent Trend.
    # We use x(center-1) + x(center+1). Positive => Pre-jump Up, Post-jump Up.
    jumps_subset["D3_trend"] = X_windows[:, center - 1] + X_windows[:, center + 1]

    # 6c) Plot Projections
    fig_mr = px.scatter(
        jumps_subset, x="D1_reflexivity", y="D2_mean_reversion", color="D2_mean_reversion",
        title="<b>Reflexivity vs Mean-Reversion</b> (Paper Fig 5 equivalent)",
        labels={"D1_reflexivity": "D1 (Reflexivity)", "D2_mean_reversion": "D2 (Mean Reversion)"},
        color_continuous_scale="RdBu", opacity=0.6
    )
    fig_mr.add_vline(x=0, line_dash="dash"); fig_mr.add_hline(y=0, line_dash="dash")
    fig_mr.update_layout(template="plotly_white")
    fig_mr.show()
    
    fig_tr = px.scatter(
        jumps_subset, x="D1_reflexivity", y="D3_trend", color="D3_trend",
        title="<b>Reflexivity vs Trend</b> (Paper Fig 6 equivalent)",
        labels={"D1_reflexivity": "D1 (Reflexivity)", "D3_trend": "D3 (Trend)"},
        color_continuous_scale="Viridis", opacity=0.6
    )
    fig_tr.add_vline(x=0, line_dash="dash"); fig_tr.add_hline(y=0, line_dash="dash")
    fig_tr.update_layout(template="plotly_white")
    fig_tr.show()
