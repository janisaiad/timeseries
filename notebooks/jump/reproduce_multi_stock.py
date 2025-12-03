import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple
import random

# Add project root to path
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
except NameError:
    project_root = os.path.abspath("../../")

if project_root not in sys.path:
    sys.path.append(project_root)

from utils.data.curating_stooq import curate_stooq_dir_5min
from utils.data.jump_detection import detect_jumps_many
from model.wavelet.wavelet import WaveletModel

def plot_profiles(X_windows, jumps_subset, output_dir, name):
    """
    Plots average temporal profiles along PCA directions.
    X-axis is time relative to jump (centered at 0).
    """
    directions = ["D1_reflexivity", "D2_mean_reversion"]
    center = X_windows.shape[1] // 2
    t_axis = np.arange(-center, center + 1)
    
    for dim in directions:
        if dim not in jumps_subset.columns: continue
        
        scores = jumps_subset[dim].values
        sorted_idx = np.argsort(scores)
        X_sorted = X_windows[sorted_idx]
        n = len(scores)
        
        # Quantiles to visualize (Low, Mid, High)
        quantiles = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        fig = go.Figure()
        colors = px.colors.sequential.Viridis
        
        for i in range(len(quantiles)-1):
            q_s, q_e = quantiles[i], quantiles[i+1]
            idx_s, idx_e = int(q_s*n), int(q_e*n)
            if idx_e <= idx_s: continue
            
            avg = np.mean(X_sorted[idx_s:idx_e], axis=0)
            
            color_idx = int(i / (len(quantiles)-1) * (len(colors)-1))
            color = colors[color_idx]
            
            fig.add_trace(go.Scatter(
                x=t_axis, 
                y=avg, 
                mode='lines', 
                name=f"Q {q_s}-{q_e}",
                line=dict(color=color, width=2)
            ))
            
        fig.update_layout(
            title=f"Average Profiles along {dim} - {name}<br>(X-axis: Time relative to Jump)",
            xaxis_title="Time (steps)",
            yaxis_title="Normalized Return x(t)",
            template="plotly_white",
            hovermode="x unified"
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Jump")
        
        out_path = os.path.join(output_dir, f"{name}_profile_{dim}.html")
        fig.write_html(out_path)
        print(f"    Saved profile plot to {out_path}")
        
        # Show in notebook
        fig.show()

def run_analysis_for_subset(
    dfs_subset: dict, 
    subset_name: str, 
    output_dir: str, 
    window_steps: int = 12
):
    print(f"\n=== Running Analysis for {subset_name} ({len(dfs_subset)} stocks) ===")
    
    # 1. Filter Trading Hours (10:30 - 15:00 approx)
    # Remove first/last 60 mins from each day
    print("  Filtering trading hours...")
    filtered_dfs = {}
    total_days = 0
    for ticker, df in dfs_subset.items():
        days = []
        for date, day_df in df.groupby(df.index.date):
            if len(day_df) <= 12: continue
            day_df = day_df.sort_index()
            
            # For 5-min bars, 60 mins = 12 bars
            start = day_df.index[0] + pd.Timedelta(minutes=60)
            end = day_df.index[-1] - pd.Timedelta(minutes=60)
            
            mask = (day_df.index >= start) & (day_df.index <= end)
            if mask.any():
                days.append(day_df[mask])
        
        if days:
            filtered_dfs[ticker] = pd.concat(days)
            total_days += len(days)

    if not filtered_dfs:
        print("  No data remaining after filtering.")
        return

    # 2. Detect Jumps
    print("  Detecting jumps...")
    jumps_df = detect_jumps_many(filtered_dfs, threshold=4.0)
    print(f"  Detected {len(jumps_df)} total jumps.")
    
    if len(jumps_df) < 50:
        print("  Not enough jumps for robust PCA (need > 50).")
        return

    # 3. Extract Windows
    print("  Extracting windows...")
    windows = []
    valid_indices = []
    
    for idx, row in jumps_df.iterrows():
        ticker, ts = row["ticker"], row["timestamp"]
        if ticker not in filtered_dfs: continue
        df = filtered_dfs[ticker]
        
        if ts not in df.index: continue
        loc = df.index.get_loc(ts)
        
        if loc - window_steps < 0 or loc + window_steps + 1 > len(df): continue
        
        subset = df.iloc[loc - window_steps : loc + window_steps + 1]
        
        # Normalization
        norm = row["f"] * row["sigma"]
        if norm == 0: norm = 1e-4
            
        r_window = subset["close"].pct_change().fillna(0.0).values
        x_profile = r_window / norm
        
        # Align Jump Direction (Paper convention: Jump > 0)
        # Center is at index `window_steps`
        jump_sign = np.sign(x_profile[window_steps])
        if jump_sign == 0: jump_sign = 1
        
        windows.append(x_profile * jump_sign)
        valid_indices.append(idx)
        
    X_windows = np.array(windows)
    jumps_subset = jumps_df.loc[valid_indices].copy()
    print(f"  Extracted {len(X_windows)} valid windows.")

    # 4. Wavelet PCA
    print("  Running Wavelet Kernel PCA...")
    wm = WaveletModel(n_layers=0, n_neurons=0, n_outputs=0, J=3, n_components=3)
    embedding = wm.fit_transform(X_windows)
    
    d1 = embedding[:, 0]
    
    # Orient D1 (Reflexivity): Positive should correlate with Post-Jump Activity
    center = window_steps
    act_post = np.sum(np.abs(X_windows[:, center+1:]), axis=1)
    act_pre = np.sum(np.abs(X_windows[:, :center]), axis=1)
    asymmetry = (act_post - act_pre) / (act_post + act_pre + 1e-6)
    
    corr = np.corrcoef(d1, asymmetry)[0, 1]
    if corr < 0:
        print(f"  Flipping D1 sign (correlation was {corr:.2f})")
        d1 *= -1
        
    jumps_subset["D1_reflexivity"] = d1
    
    # Handcrafted Features
    # D2 (Mean Reversion): Pre-Jump - Post-Jump
    # D3 (Trend): Pre-Jump + Post-Jump
    # t = +/- 1 step from center
    jumps_subset["D2_mean_reversion"] = X_windows[:, center - 1] - X_windows[:, center + 1]
    jumps_subset["D3_trend"] = X_windows[:, center - 1] + X_windows[:, center + 1]

    # 5. Generate Plots
    print("  Generating plots...")
    
    # Scatter Plot (Fig 5 equivalent)
    fig_mr = px.scatter(
        jumps_subset, x="D1_reflexivity", y="D2_mean_reversion", color="D2_mean_reversion",
        title=f"Reflexivity vs Mean-Reversion ({subset_name}, N={len(X_windows)})", 
        color_continuous_scale="RdBu", opacity=0.5,
        hover_data=["ticker", "timestamp"]
    )
    fig_mr.add_vline(x=0, line_dash="dash"); fig_mr.add_hline(y=0, line_dash="dash")
    
    out_path_scatter = os.path.join(output_dir, f"{subset_name}_fig5_mr.html")
    fig_mr.write_html(out_path_scatter)
    print(f"    Saved scatter plot to {out_path_scatter}")
    
    # Show in notebook
    fig_mr.show()
    
    # Profile Plots
    plot_profiles(X_windows, jumps_subset, output_dir, subset_name)

def main():
    data_dir = "/home/janis/4A/timeseries/data/stooq/hungary/"
    print(f"Loading data from {data_dir}...")
    all_dfs = curate_stooq_dir_5min(data_dir, pattern="*.txt", recursive=True)
    
    # Filter to stocks with enough data (e.g., > 1000 points) to ensure meaningful jumps
    valid_tickers = [t for t, d in all_dfs.items() if len(d) > 1000]
    print(f"Found {len(valid_tickers)} valid tickers.")
    
    if not valid_tickers:
        print("No valid data found.")
        return
    
    # Sort by data length to pick the best ones
    valid_tickers.sort(key=lambda t: len(all_dfs[t]), reverse=True)
    
    # Select Subsets
    # 1. Small Subset: Top 5 most liquid/long stocks
    tickers_5 = valid_tickers[:5]
    dfs_5 = {t: all_dfs[t] for t in tickers_5}
    
    # 2. Large Subset: Top 30 (or all if less)
    limit_30 = min(30, len(valid_tickers))
    tickers_30 = valid_tickers[:limit_30]
    dfs_30 = {t: all_dfs[t] for t in tickers_30}
    
    # Output Directory
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "outputs_multi")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Comparisons
    run_analysis_for_subset(dfs_5, "5_Stocks", output_dir)
    
    if len(tickers_30) > len(tickers_5):
        run_analysis_for_subset(dfs_30, f"{len(tickers_30)}_Stocks", output_dir)
    else:
        print("Skipping large subset analysis (not enough stocks for distinct comparison).")

if __name__ == "__main__":
    main()
