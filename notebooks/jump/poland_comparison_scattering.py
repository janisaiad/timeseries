# %%
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from typing import List, Tuple, Dict
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
def run_analysis(
    dfs_subset: dict, 
    subset_name: str, 
    include_ss: bool,
    window_steps: int = 12
) -> Dict:
    """
    Run wavelet PCA analysis with or without scattering spectra.
    Returns dictionary with results.
    """
    print(f"\n=== Running Analysis: {subset_name} (SS={'ON' if include_ss else 'OFF'}) ===")
    
    # 1. Filter Trading Hours
    print("  Filtering trading hours...")
    filtered_dfs = {}
    for ticker, df in dfs_subset.items():
        days = []
        for date, day_df in df.groupby(df.index.date):
            if len(day_df) <= 12: continue
            day_df = day_df.sort_index()
            start = day_df.index[0] + pd.Timedelta(minutes=60)
            end = day_df.index[-1] - pd.Timedelta(minutes=60)
            mask = (day_df.index >= start) & (day_df.index <= end)
            if mask.any():
                days.append(day_df[mask])
        if days:
            filtered_dfs[ticker] = pd.concat(days)

    if not filtered_dfs:
        print("  No data remaining after filtering.")
        return None

    # 2. Detect Jumps
    print("  Detecting jumps...")
    jumps_df = detect_jumps_many(filtered_dfs, threshold=4.0)
    print(f"  Detected {len(jumps_df)} total jumps.")
    
    if len(jumps_df) < 50:
        print("  Not enough jumps for robust PCA (need > 50).")
        return None

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
        norm = row["f"] * row["sigma"]
        if norm == 0: norm = 1e-4
            
        r_window = subset["close"].pct_change().fillna(0.0).values
        x_profile = r_window / norm
        
        jump_sign = np.sign(x_profile[window_steps])
        if jump_sign == 0: jump_sign = 1
        
        windows.append(x_profile * jump_sign)
        valid_indices.append(idx)
        
    X_windows = np.array(windows)
    jumps_subset = jumps_df.loc[valid_indices].copy()
    print(f"  Extracted {len(X_windows)} valid windows.")

    # 4. Wavelet PCA
    print(f"  Running Wavelet Kernel PCA (SS={'ON' if include_ss else 'OFF'})...")
    wm = WaveletModel(
        n_layers=0, n_neurons=0, n_outputs=0, 
        J=3, n_components=3, 
        include_scattering_spectra=include_ss
    )
    embedding = wm.fit_transform(X_windows)
    
    d1 = embedding[:, 0]
    
    # Orient D1 (Reflexivity)
    center = window_steps
    act_post = np.sum(np.abs(X_windows[:, center+1:]), axis=1)
    act_pre = np.sum(np.abs(X_windows[:, :center]), axis=1)
    asymmetry = (act_post - act_pre) / (act_post + act_pre + 1e-6)
    
    corr = np.corrcoef(d1, asymmetry)[0, 1]
    if corr < 0:
        print(f"  Flipping D1 sign (correlation was {corr:.2f})")
        d1 *= -1
        corr = -corr
        
    jumps_subset["D1_reflexivity"] = d1
    jumps_subset["asymmetry"] = asymmetry
    print(f"  D1-Asymmetry correlation: {corr:.3f}")
    
    # Handcrafted Features
    jumps_subset["D2_mean_reversion"] = X_windows[:, center - 1] - X_windows[:, center + 1]
    jumps_subset["D3_trend"] = X_windows[:, center - 1] + X_windows[:, center + 1]
    
    return {
        'X_windows': X_windows,
        'jumps_subset': jumps_subset,
        'embedding': embedding,
        'wm': wm,
        'correlation': corr
    }

# %%
# Load Poland data
data_dir = "/home/janis/4A/timeseries/data/stooq/poland/5_min/pl/wsestocks"
print(f"Loading Poland data from {data_dir}...")
all_dfs = curate_stooq_dir_5min(data_dir, pattern="*.txt", recursive=True)

# Filter to stocks with enough data
valid_tickers = [t for t, d in all_dfs.items() if len(d) > 500]
print(f"Found {len(valid_tickers)} valid tickers.")

if not valid_tickers:
    print("No valid data found.")
    exit()

# Sort by data length
valid_tickers.sort(key=lambda t: len(all_dfs[t]), reverse=True)

# Select subset (use top 30 or all available)
limit = min(30, len(valid_tickers))
tickers_subset = valid_tickers[:limit]
dfs_subset = {t: all_dfs[t] for t in tickers_subset}

print(f"Using {len(dfs_subset)} stocks for comparison.")

# %%
# Run analysis WITH scattering spectra
results_with_ss = run_analysis(dfs_subset, "Poland_WithSS", include_ss=True)

# %%
# Run analysis WITHOUT scattering spectra
results_without_ss = run_analysis(dfs_subset, "Poland_WithoutSS", include_ss=False)

# %%
# Compare results
if results_with_ss is None or results_without_ss is None:
    print("Cannot compare - one or both analyses failed.")
    exit()

jumps_with = results_with_ss['jumps_subset']
jumps_without = results_without_ss['jumps_subset']

print("\n=== Comparison Summary ===")
print(f"Number of jumps: {len(jumps_with)} (both should be same)")
print(f"\nD1 Statistics:")
print(f"  With SS:    mean={jumps_with['D1_reflexivity'].mean():.3f}, std={jumps_with['D1_reflexivity'].std():.3f}")
print(f"  Without SS: mean={jumps_without['D1_reflexivity'].mean():.3f}, std={jumps_without['D1_reflexivity'].std():.3f}")
print(f"\nD2 Statistics:")
print(f"  With SS:    mean={jumps_with['D2_mean_reversion'].mean():.3f}, std={jumps_with['D2_mean_reversion'].std():.3f}")
print(f"  Without SS: mean={jumps_without['D2_mean_reversion'].mean():.3f}, std={jumps_without['D2_mean_reversion'].std():.3f}")
print(f"\nD3 Statistics:")
print(f"  With SS:    mean={jumps_with['D3_trend'].mean():.3f}, std={jumps_with['D3_trend'].std():.3f}")
print(f"  Without SS: mean={jumps_without['D3_trend'].mean():.3f}, std={jumps_without['D3_trend'].std():.3f}")

# Correlation between directions
d1_corr = np.corrcoef(jumps_with['D1_reflexivity'], jumps_without['D1_reflexivity'])[0, 1]
d2_corr = np.corrcoef(jumps_with['D2_mean_reversion'], jumps_without['D2_mean_reversion'])[0, 1]
d3_corr = np.corrcoef(jumps_with['D3_trend'], jumps_without['D3_trend'])[0, 1]
print(f"\nDirection Correlations (With SS vs Without SS):")
print(f"  D1: {d1_corr:.3f}")
print(f"  D2: {d2_corr:.3f}")
print(f"  D3: {d3_corr:.3f}")

# %%
# Plot 1: Scatter plot comparing D1 values
fig_d1_comp, ax_d1_comp = plt.subplots(figsize=(8, 8))

ax_d1_comp.scatter(jumps_without['D1_reflexivity'], jumps_with['D1_reflexivity'], 
                   s=10, alpha=0.6, color='blue', label='Jumps')

# Perfect correlation line
min_d1 = min(jumps_with['D1_reflexivity'].min(), jumps_without['D1_reflexivity'].min())
max_d1 = max(jumps_with['D1_reflexivity'].max(), jumps_without['D1_reflexivity'].max())
ax_d1_comp.plot([min_d1, max_d1], [min_d1, max_d1], 'r--', linewidth=2, label='Perfect Correlation (y=x)')

ax_d1_comp.set_xlabel("D1 (Without Scattering Spectra)")
ax_d1_comp.set_ylabel("D1 (With Scattering Spectra)")
ax_d1_comp.set_title(f"D1 Reflexivity: With SS vs Without SS\nCorrelation: {d1_corr:.3f}")
ax_d1_comp.legend()
ax_d1_comp.grid(True, alpha=0.3)
ax_d1_comp.set_aspect('equal', adjustable='box')
save_plot(fig_d1_comp, "poland_comparison_D1_scatter", format='pdf')
plt.close(fig_d1_comp)

# %%
# Plot 2: Side-by-side comparison of D1 distributions
fig_d1_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(jumps_with['D1_reflexivity'], bins=30, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(x=0, linestyle='--', color='gray', alpha=0.7)
ax1.set_xlabel("D1 Reflexivity")
ax1.set_ylabel("Count")
ax1.set_title("D1 Distribution (With SS)")
ax1.grid(True, alpha=0.3)

ax2.hist(jumps_without['D1_reflexivity'], bins=30, alpha=0.7, color='red', edgecolor='black')
ax2.axvline(x=0, linestyle='--', color='gray', alpha=0.7)
ax2.set_xlabel("D1 Reflexivity")
ax2.set_ylabel("Count")
ax2.set_title("D1 Distribution (Without SS)")
ax2.grid(True, alpha=0.3)

fig_d1_dist.suptitle("D1 Reflexivity Distribution Comparison")
plt.tight_layout()
save_plot(fig_d1_dist, "poland_comparison_D1_distribution", format='pdf')
plt.close(fig_d1_dist)

# %%
# Plot 2b: D2 comparison scatter plot
fig_d2_comp, ax_d2_comp = plt.subplots(figsize=(8, 8))

ax_d2_comp.scatter(jumps_without['D2_mean_reversion'], jumps_with['D2_mean_reversion'], 
                   s=10, alpha=0.6, color='green', label='Jumps')

min_d2 = min(jumps_with['D2_mean_reversion'].min(), jumps_without['D2_mean_reversion'].min())
max_d2 = max(jumps_with['D2_mean_reversion'].max(), jumps_without['D2_mean_reversion'].max())
ax_d2_comp.plot([min_d2, max_d2], [min_d2, max_d2], 'r--', linewidth=2, label='Perfect Correlation (y=x)')

ax_d2_comp.set_xlabel("D2 (Without Scattering Spectra)")
ax_d2_comp.set_ylabel("D2 (With Scattering Spectra)")
ax_d2_comp.set_title(f"D2 Mean-Reversion: With SS vs Without SS\nCorrelation: {d2_corr:.3f}")
ax_d2_comp.legend()
ax_d2_comp.grid(True, alpha=0.3)
ax_d2_comp.set_aspect('equal', adjustable='box')
save_plot(fig_d2_comp, "poland_comparison_D2_scatter", format='pdf')
plt.close(fig_d2_comp)

# %%
# Plot 2c: D3 comparison scatter plot
fig_d3_comp, ax_d3_comp = plt.subplots(figsize=(8, 8))

ax_d3_comp.scatter(jumps_without['D3_trend'], jumps_with['D3_trend'], 
                   s=10, alpha=0.6, color='purple', label='Jumps')

min_d3 = min(jumps_with['D3_trend'].min(), jumps_without['D3_trend'].min())
max_d3 = max(jumps_with['D3_trend'].max(), jumps_without['D3_trend'].max())
ax_d3_comp.plot([min_d3, max_d3], [min_d3, max_d3], 'r--', linewidth=2, label='Perfect Correlation (y=x)')

ax_d3_comp.set_xlabel("D3 (Without Scattering Spectra)")
ax_d3_comp.set_ylabel("D3 (With Scattering Spectra)")
ax_d3_comp.set_title(f"D3 Trend: With SS vs Without SS\nCorrelation: {d3_corr:.3f}")
ax_d3_comp.legend()
ax_d3_comp.grid(True, alpha=0.3)
ax_d3_comp.set_aspect('equal', adjustable='box')
save_plot(fig_d3_comp, "poland_comparison_D3_scatter", format='pdf')
plt.close(fig_d3_comp)

# %%
# Plot 2d: Side-by-side comparison of D2 distributions
fig_d2_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(jumps_with['D2_mean_reversion'], bins=30, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(x=0, linestyle='--', color='gray', alpha=0.7)
ax1.set_xlabel("D2 Mean-Reversion")
ax1.set_ylabel("Count")
ax1.set_title("D2 Distribution (With SS)")
ax1.grid(True, alpha=0.3)

ax2.hist(jumps_without['D2_mean_reversion'], bins=30, alpha=0.7, color='red', edgecolor='black')
ax2.axvline(x=0, linestyle='--', color='gray', alpha=0.7)
ax2.set_xlabel("D2 Mean-Reversion")
ax2.set_ylabel("Count")
ax2.set_title("D2 Distribution (Without SS)")
ax2.grid(True, alpha=0.3)

fig_d2_dist.suptitle("D2 Mean-Reversion Distribution Comparison")
plt.tight_layout()
save_plot(fig_d2_dist, "poland_comparison_D2_distribution", format='pdf')
plt.close(fig_d2_dist)

# %%
# Plot 2e: Side-by-side comparison of D3 distributions
fig_d3_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(jumps_with['D3_trend'], bins=30, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(x=0, linestyle='--', color='gray', alpha=0.7)
ax1.set_xlabel("D3 Trend")
ax1.set_ylabel("Count")
ax1.set_title("D3 Distribution (With SS)")
ax1.grid(True, alpha=0.3)

ax2.hist(jumps_without['D3_trend'], bins=30, alpha=0.7, color='red', edgecolor='black')
ax2.axvline(x=0, linestyle='--', color='gray', alpha=0.7)
ax2.set_xlabel("D3 Trend")
ax2.set_ylabel("Count")
ax2.set_title("D3 Distribution (Without SS)")
ax2.grid(True, alpha=0.3)

fig_d3_dist.suptitle("D3 Trend Distribution Comparison")
plt.tight_layout()
save_plot(fig_d3_dist, "poland_comparison_D3_distribution", format='pdf')
plt.close(fig_d3_dist)

# %%
# Plot 3: D1 vs D2 scatter plots comparison
fig_d1d2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# With SS
ax1.scatter(jumps_with['D1_reflexivity'], jumps_with['D2_mean_reversion'], 
            s=5, alpha=0.5, color='blue')
ax1.axvline(x=0, linestyle='--', color='gray', alpha=0.5)
ax1.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
ax1.set_xlabel("D1 Reflexivity")
ax1.set_ylabel("D2 Mean-Reversion")
ax1.set_title("D1 vs D2 (With SS)")
ax1.grid(True, alpha=0.3)

# Without SS
ax2.scatter(jumps_without['D1_reflexivity'], jumps_without['D2_mean_reversion'], 
            s=5, alpha=0.5, color='red')
ax2.axvline(x=0, linestyle='--', color='gray', alpha=0.5)
ax2.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
ax2.set_xlabel("D1 Reflexivity")
ax2.set_ylabel("D2 Mean-Reversion")
ax2.set_title("D1 vs D2 (Without SS)")
ax2.grid(True, alpha=0.3)

fig_d1d2.suptitle("D1 vs D2 Comparison: With vs Without Scattering Spectra")
plt.tight_layout()
save_plot(fig_d1d2, "poland_comparison_D1_D2", format='pdf')
plt.close(fig_d1d2)

# %%
# Plot 4: D1 vs D3 scatter plots comparison
fig_d1d3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# With SS
ax1.scatter(jumps_with['D1_reflexivity'], jumps_with['D3_trend'], 
            s=5, alpha=0.5, color='blue')
ax1.axvline(x=0, linestyle='--', color='gray', alpha=0.5)
ax1.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
ax1.set_xlabel("D1 Reflexivity")
ax1.set_ylabel("D3 Trend")
ax1.set_title("D1 vs D3 (With SS)")
ax1.grid(True, alpha=0.3)

# Without SS
ax2.scatter(jumps_without['D1_reflexivity'], jumps_without['D3_trend'],
            s=5, alpha=0.5, color='red')
ax2.axvline(x=0, linestyle='--', color='gray', alpha=0.5)
ax2.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
ax2.set_xlabel("D1 Reflexivity")
ax2.set_ylabel("D3 Trend")
ax2.set_title("D1 vs D3 (Without SS)")
ax2.grid(True, alpha=0.3)

fig_d1d3.suptitle("D1 vs D3 Comparison: With vs Without Scattering Spectra")
plt.tight_layout()
save_plot(fig_d1d3, "poland_comparison_D1_D3", format='pdf')
plt.close(fig_d1d3)

# %%
# Plot 5: Profile comparison for D1 direction
if results_with_ss and results_without_ss:
    X_windows = results_with_ss['X_windows']
    center = X_windows.shape[1] // 2
    t_axis = np.arange(-center, center + 1)
    
    # Sort by D1 for both
    sorted_with = np.argsort(jumps_with['D1_reflexivity'].values)
    sorted_without = np.argsort(jumps_without['D1_reflexivity'].values)
    
    X_sorted_with = X_windows[sorted_with]
    X_sorted_without = X_windows[sorted_without]
    
    n = len(X_windows)
    quantiles = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    fig_profiles, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(quantiles)-1))
    
    for col_idx, (X_sorted, title_suffix, ax) in enumerate([(X_sorted_with, "With SS", ax1), (X_sorted_without, "Without SS", ax2)]):
        for i in range(len(quantiles)-1):
            q_s, q_e = quantiles[i], quantiles[i+1]
            idx_s, idx_e = int(q_s*n), int(q_e*n)
            if idx_e <= idx_s: continue
            
            avg = np.mean(X_sorted[idx_s:idx_e], axis=0)
            ax.plot(t_axis, avg, linewidth=2, label=f"Q {q_s}-{q_e}", color=colors[i])
        
        ax.axvline(x=0, linestyle='--', color='red', alpha=0.7, label='Jump')
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Normalized Return x(t)")
        ax.set_title(f"D1 Profiles ({title_suffix})")
        if col_idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig_profiles.suptitle("Average Profiles along D1: With vs Without Scattering Spectra")
    plt.tight_layout()
    save_plot(fig_profiles, "poland_comparison_profiles_D1", format='pdf')
    plt.close(fig_profiles)

# %%
# Plot 6: Profile comparison for D2 direction
if results_with_ss and results_without_ss:
    X_windows = results_with_ss['X_windows']
    center = X_windows.shape[1] // 2
    t_axis = np.arange(-center, center + 1)
    
    # Sort by D2 for both
    sorted_with = np.argsort(jumps_with['D2_mean_reversion'].values)
    sorted_without = np.argsort(jumps_without['D2_mean_reversion'].values)
    
    X_sorted_with = X_windows[sorted_with]
    X_sorted_without = X_windows[sorted_without]
    
    n = len(X_windows)
    quantiles = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    fig_profiles_d2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(quantiles)-1))
    
    for col_idx, (X_sorted, title_suffix, ax) in enumerate([(X_sorted_with, "With SS", ax1), (X_sorted_without, "Without SS", ax2)]):
        for i in range(len(quantiles)-1):
            q_s, q_e = quantiles[i], quantiles[i+1]
            idx_s, idx_e = int(q_s*n), int(q_e*n)
            if idx_e <= idx_s: continue
            
            avg = np.mean(X_sorted[idx_s:idx_e], axis=0)
            ax.plot(t_axis, avg, linewidth=2, label=f"Q {q_s}-{q_e}", color=colors[i])
        
        ax.axvline(x=0, linestyle='--', color='red', alpha=0.7, label='Jump')
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Normalized Return x(t)")
        ax.set_title(f"D2 Profiles ({title_suffix})")
        if col_idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig_profiles_d2.suptitle("Average Profiles along D2 (Mean-Reversion): With vs Without Scattering Spectra")
    plt.tight_layout()
    save_plot(fig_profiles_d2, "poland_comparison_profiles_D2", format='pdf')
    plt.close(fig_profiles_d2)

# %%
# Plot 7: Profile comparison for D3 direction
if results_with_ss and results_without_ss:
    X_windows = results_with_ss['X_windows']
    center = X_windows.shape[1] // 2
    t_axis = np.arange(-center, center + 1)
    
    # Sort by D3 for both
    sorted_with = np.argsort(jumps_with['D3_trend'].values)
    sorted_without = np.argsort(jumps_without['D3_trend'].values)
    
    X_sorted_with = X_windows[sorted_with]
    X_sorted_without = X_windows[sorted_without]
    
    n = len(X_windows)
    quantiles = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    fig_profiles_d3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(quantiles)-1))
    
    for col_idx, (X_sorted, title_suffix, ax) in enumerate([(X_sorted_with, "With SS", ax1), (X_sorted_without, "Without SS", ax2)]):
        for i in range(len(quantiles)-1):
            q_s, q_e = quantiles[i], quantiles[i+1]
            idx_s, idx_e = int(q_s*n), int(q_e*n)
            if idx_e <= idx_s: continue
            
            avg = np.mean(X_sorted[idx_s:idx_e], axis=0)
            ax.plot(t_axis, avg, linewidth=2, label=f"Q {q_s}-{q_e}", color=colors[i])
        
        ax.axvline(x=0, linestyle='--', color='red', alpha=0.7, label='Jump')
        ax.set_xlabel("Time (steps)")
        ax.set_ylabel("Normalized Return x(t)")
        ax.set_title(f"D3 Profiles ({title_suffix})")
        if col_idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig_profiles_d3.suptitle("Average Profiles along D3 (Trend): With vs Without Scattering Spectra")
    plt.tight_layout()
    save_plot(fig_profiles_d3, "poland_comparison_profiles_D3", format='pdf')
    plt.close(fig_profiles_d3)

# %%
# Summary statistics table
summary_data = {
    'Metric': [
        'D1 Mean',
        'D1 Std',
        'D1 Min',
        'D1 Max',
        'D2 Mean',
        'D2 Std',
        'D2 Min',
        'D2 Max',
        'D3 Mean',
        'D3 Std',
        'D3 Min',
        'D3 Max',
        'D1-Asymmetry Correlation',
        'D1 Correlation (With vs Without)',
        'D2 Correlation (With vs Without)',
        'D3 Correlation (With vs Without)'
    ],
    'With SS': [
        f"{jumps_with['D1_reflexivity'].mean():.3f}",
        f"{jumps_with['D1_reflexivity'].std():.3f}",
        f"{jumps_with['D1_reflexivity'].min():.3f}",
        f"{jumps_with['D1_reflexivity'].max():.3f}",
        f"{jumps_with['D2_mean_reversion'].mean():.3f}",
        f"{jumps_with['D2_mean_reversion'].std():.3f}",
        f"{jumps_with['D2_mean_reversion'].min():.3f}",
        f"{jumps_with['D2_mean_reversion'].max():.3f}",
        f"{jumps_with['D3_trend'].mean():.3f}",
        f"{jumps_with['D3_trend'].std():.3f}",
        f"{jumps_with['D3_trend'].min():.3f}",
        f"{jumps_with['D3_trend'].max():.3f}",
        f"{results_with_ss['correlation']:.3f}",
        f"{d1_corr:.3f}",
        f"{d2_corr:.3f}",
        f"{d3_corr:.3f}"
    ],
    'Without SS': [
        f"{jumps_without['D1_reflexivity'].mean():.3f}",
        f"{jumps_without['D1_reflexivity'].std():.3f}",
        f"{jumps_without['D1_reflexivity'].min():.3f}",
        f"{jumps_without['D1_reflexivity'].max():.3f}",
        f"{jumps_without['D2_mean_reversion'].mean():.3f}",
        f"{jumps_without['D2_mean_reversion'].std():.3f}",
        f"{jumps_without['D2_mean_reversion'].min():.3f}",
        f"{jumps_without['D2_mean_reversion'].max():.3f}",
        f"{jumps_without['D3_trend'].mean():.3f}",
        f"{jumps_without['D3_trend'].std():.3f}",
        f"{jumps_without['D3_trend'].min():.3f}",
        f"{jumps_without['D3_trend'].max():.3f}",
        f"{results_without_ss['correlation']:.3f}",
        "-",
        "-",
        "-"
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n=== Summary Statistics Comparison ===")
print(summary_df.to_string(index=False))

# %%

