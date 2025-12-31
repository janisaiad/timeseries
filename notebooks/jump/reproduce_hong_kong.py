# %%
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from typing import List, Tuple
import random
from scipy import stats as scipy_stats
from scipy import stats as scipy_stats

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
from plot_utils import save_plot

# %%
def plot_profiles(X_windows, jumps_subset, output_dir, name):
    """
    Plots average temporal profiles along PCA directions.
    X-axis is time relative to jump (centered at 0).
    """
    directions = ["D1_reflexivity", "D2_mean_reversion", "D3_trend"]
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
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(quantiles)-1))
        
        for i in range(len(quantiles)-1):
            q_s, q_e = quantiles[i], quantiles[i+1]
            idx_s, idx_e = int(q_s*n), int(q_e*n)
            if idx_e <= idx_s: continue
            
            avg = np.mean(X_sorted[idx_s:idx_e], axis=0)
            ax.plot(t_axis, avg, linewidth=2, label=f"Q {q_s}-{q_e}", color=colors[i])
            
        ax.axvline(x=0, linestyle='--', color='red', alpha=0.7, label='Jump')
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Normalized Return x(t)")
        ax.set_title(f"Average Profiles along {dim} - {name}\n(X-axis: Time relative to Jump)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        out_path = os.path.join(output_dir, f"{name}_profile_{dim}.html")
        print(f"    Saved profile plot to {out_path}")
        save_plot(fig, f"reproduce_hong_kong_profile_{dim}", format='pdf')
        plt.close(fig)

# %%
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
    wm = WaveletModel(n_layers=0, n_neurons=0, n_outputs=0, J=3, n_components=3, include_scattering_spectra=False)
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
        corr = -corr  # Update correlation after flipping
        
    jumps_subset["D1_reflexivity"] = d1
    jumps_subset["asymmetry"] = asymmetry
    print(f"  D1-Asymmetry correlation: {corr:.3f}")
    
    # Handcrafted Features
    # D2 (Mean Reversion): Pre-Jump - Post-Jump
    # D3 (Trend): Pre-Jump + Post-Jump
    # t = +/- 1 step from center
    jumps_subset["D2_mean_reversion"] = X_windows[:, center - 1] - X_windows[:, center + 1]
    jumps_subset["D3_trend"] = X_windows[:, center - 1] + X_windows[:, center + 1]

    # 5. Generate Plots
    print("  Generating plots...")
    
    # Scatter Plot: D1 vs Asymmetry (to verify separation)
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        jumps_subset["asymmetry"], jumps_subset["D1_reflexivity"]
    )
    
    fig_asym, ax_asym = plt.subplots(figsize=(10, 8))
    scatter = ax_asym.scatter(jumps_subset["asymmetry"], jumps_subset["D1_reflexivity"], 
                              c=jumps_subset["asymmetry"], cmap='RdBu_r', alpha=0.6, s=20)
    
    # Add regression line
    asym_range = np.linspace(jumps_subset["asymmetry"].min(), jumps_subset["asymmetry"].max(), 100)
    reg_line = slope * asym_range + intercept
    ax_asym.plot(asym_range, reg_line, 'r--', linewidth=2, label=f'Regression (R²={r_value**2:.3f})')
    
    ax_asym.axvline(x=0, linestyle='--', color='gray', alpha=0.5, label='Symmetric')
    ax_asym.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
    plt.colorbar(scatter, ax=ax_asym, label='Asymmetry')
    
    ax_asym.set_xlabel("Asymmetry (Post-Pre)/(Post+Pre)")
    ax_asym.set_ylabel("D1 (Reflexivity)")
    ax_asym.set_title(f"D1 (Reflexivity) vs Asymmetry ({subset_name}, N={len(X_windows)})\nCorrelation: {corr:.3f}, R²: {r_value**2:.3f}")
    ax_asym.legend()
    ax_asym.grid(True, alpha=0.3)
    
    out_path_asym = os.path.join(output_dir, f"{subset_name}_D1_asymmetry.html")
    print(f"    Saved D1-Asymmetry plot to {out_path_asym}")
    save_plot(fig_asym, f"reproduce_hong_kong_D1_asymmetry", format='pdf')
    plt.close(fig_asym)
    
    # Scatter Plot: D1 vs D2 (Mean-Reversion) - Fig 5 equivalent
    fig_mr, ax_mr = plt.subplots(figsize=(10, 8))
    scatter_mr = ax_mr.scatter(jumps_subset["D1_reflexivity"], jumps_subset["D2_mean_reversion"], 
                               c=jumps_subset["D2_mean_reversion"], cmap='RdBu_r', alpha=0.5, s=20)
    ax_mr.axvline(x=0, linestyle='--', color='gray', alpha=0.5)
    ax_mr.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
    plt.colorbar(scatter_mr, ax=ax_mr, label='D2 Mean-Reversion')
    ax_mr.set_xlabel("D1 Reflexivity")
    ax_mr.set_ylabel("D2 Mean-Reversion")
    ax_mr.set_title(f"Reflexivity vs Mean-Reversion ({subset_name}, N={len(X_windows)})")
    ax_mr.grid(True, alpha=0.3)
    
    out_path_scatter = os.path.join(output_dir, f"{subset_name}_fig5_mr.html")
    print(f"    Saved scatter plot to {out_path_scatter}")
    save_plot(fig_mr, f"reproduce_hong_kong_fig5_mr", format='pdf')
    plt.close(fig_mr)
    
    # Scatter Plot: D1 vs D3 (Trend) - Fig 6 equivalent
    fig_tr, ax_tr = plt.subplots(figsize=(10, 8))
    scatter_tr = ax_tr.scatter(jumps_subset["D1_reflexivity"], jumps_subset["D3_trend"], 
                               c=jumps_subset["D3_trend"], cmap='viridis', alpha=0.5, s=20)
    ax_tr.axvline(x=0, linestyle='--', color='gray', alpha=0.5)
    ax_tr.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
    plt.colorbar(scatter_tr, ax=ax_tr, label='D3 Trend')
    ax_tr.set_xlabel("D1 Reflexivity")
    ax_tr.set_ylabel("D3 Trend")
    ax_tr.set_title(f"Reflexivity vs Trend ({subset_name}, N={len(X_windows)})")
    ax_tr.grid(True, alpha=0.3)
    
    out_path_scatter_tr = os.path.join(output_dir, f"{subset_name}_fig6_tr.html")
    print(f"    Saved scatter plot to {out_path_scatter_tr}")
    save_plot(fig_tr, f"reproduce_hong_kong_fig6_tr", format='pdf')
    plt.close(fig_tr)
    
    # Profile Plots (all three directions)
    plot_profiles(X_windows, jumps_subset, output_dir, subset_name)

# %%
def main():
    # Hong Kong data directory
    data_dir = "/home/janis/4A/timeseries/data/stooq/hongkong/5_min/hk/hkex_stocks"
    print(f"Loading Hong Kong data from {data_dir}...")
    all_dfs = curate_stooq_dir_5min(data_dir, pattern="*.txt", recursive=True)
    
    # Filter to stocks with enough data (e.g., > 500 points) to ensure meaningful jumps
    valid_tickers = [t for t, d in all_dfs.items() if len(d) > 500]
    print(f"Found {len(valid_tickers)} valid tickers.")
    
    if not valid_tickers:
        print("No valid data found.")
        return
    
    # Sort by data length to pick the best ones
    valid_tickers.sort(key=lambda t: len(all_dfs[t]), reverse=True)
    
    # Select Subsets
    # 1. Small Subset: Top 5 most liquid/long stocks
    tickers_5 = valid_tickers[:5] if len(valid_tickers) >= 5 else valid_tickers
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
    output_dir = os.path.join(base_dir, "hong_kong_outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run Comparisons
    run_analysis_for_subset(dfs_5, "5_Stocks_HongKong", output_dir)
    
    if len(tickers_30) > len(tickers_5):
        run_analysis_for_subset(dfs_30, f"{len(tickers_30)}_Stocks_HongKong", output_dir)
    else:
        print("Skipping large subset analysis (not enough stocks for distinct comparison).")

# %%
if __name__ == "__main__":
    main()

# %%

