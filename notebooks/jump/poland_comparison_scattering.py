# %%
import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Correlation between D1 values
d1_corr = np.corrcoef(jumps_with['D1_reflexivity'], jumps_without['D1_reflexivity'])[0, 1]
print(f"\nD1 Correlation (With SS vs Without SS): {d1_corr:.3f}")

# %%
# Plot 1: Scatter plot comparing D1 values
fig_d1_comp = go.Figure()

fig_d1_comp.add_trace(go.Scatter(
    x=jumps_without['D1_reflexivity'],
    y=jumps_with['D1_reflexivity'],
    mode='markers',
    name='Jumps',
    marker=dict(size=4, opacity=0.6, color='blue')
))

# Perfect correlation line
min_d1 = min(jumps_with['D1_reflexivity'].min(), jumps_without['D1_reflexivity'].min())
max_d1 = max(jumps_with['D1_reflexivity'].max(), jumps_without['D1_reflexivity'].max())
fig_d1_comp.add_trace(go.Scatter(
    x=[min_d1, max_d1],
    y=[min_d1, max_d1],
    mode='lines',
    name='Perfect Correlation (y=x)',
    line=dict(color='red', width=2, dash='dash')
))

fig_d1_comp.update_layout(
    title=f"D1 Reflexivity: With SS vs Without SS<br>Correlation: {d1_corr:.3f}",
    xaxis_title="D1 (Without Scattering Spectra)",
    yaxis_title="D1 (With Scattering Spectra)",
    template="plotly_white",
    height=600
)
fig_d1_comp.show()

# %%
# Plot 2: Side-by-side comparison of D1 distributions
fig_d1_dist = make_subplots(
    rows=1, cols=2,
    subplot_titles=("D1 Distribution (With SS)", "D1 Distribution (Without SS)"),
    horizontal_spacing=0.1
)

fig_d1_dist.add_trace(
    go.Histogram(x=jumps_with['D1_reflexivity'], name='With SS', nbinsx=30, opacity=0.7),
    row=1, col=1
)
fig_d1_dist.add_trace(
    go.Histogram(x=jumps_without['D1_reflexivity'], name='Without SS', nbinsx=30, opacity=0.7),
    row=1, col=2
)

fig_d1_dist.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=1)
fig_d1_dist.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)

fig_d1_dist.update_xaxes(title_text="D1 Reflexivity", row=1, col=1)
fig_d1_dist.update_xaxes(title_text="D1 Reflexivity", row=1, col=2)
fig_d1_dist.update_yaxes(title_text="Count", row=1, col=1)
fig_d1_dist.update_yaxes(title_text="Count", row=1, col=2)

fig_d1_dist.update_layout(
    title="D1 Reflexivity Distribution Comparison",
    template="plotly_white",
    height=500,
    showlegend=False
)
fig_d1_dist.show()

# %%
# Plot 3: D1 vs D2 scatter plots comparison
fig_d1d2 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("D1 vs D2 (With SS)", "D1 vs D2 (Without SS)"),
    horizontal_spacing=0.1
)

# With SS
fig_d1d2.add_trace(
    go.Scatter(
        x=jumps_with['D1_reflexivity'],
        y=jumps_with['D2_mean_reversion'],
        mode='markers',
        name='With SS',
        marker=dict(size=3, opacity=0.5, color='blue')
    ),
    row=1, col=1
)

# Without SS
fig_d1d2.add_trace(
    go.Scatter(
        x=jumps_without['D1_reflexivity'],
        y=jumps_without['D2_mean_reversion'],
        mode='markers',
        name='Without SS',
        marker=dict(size=3, opacity=0.5, color='red')
    ),
    row=1, col=2
)

# Add reference lines
fig_d1d2.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=1)
fig_d1d2.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
fig_d1d2.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)
fig_d1d2.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

fig_d1d2.update_xaxes(title_text="D1 Reflexivity", row=1, col=1)
fig_d1d2.update_xaxes(title_text="D1 Reflexivity", row=1, col=2)
fig_d1d2.update_yaxes(title_text="D2 Mean-Reversion", row=1, col=1)
fig_d1d2.update_yaxes(title_text="D2 Mean-Reversion", row=1, col=2)

fig_d1d2.update_layout(
    title="D1 vs D2 Comparison: With vs Without Scattering Spectra",
    template="plotly_white",
    height=500,
    showlegend=False
)
fig_d1d2.show()

# %%
# Plot 4: D1 vs D3 scatter plots comparison
fig_d1d3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=("D1 vs D3 (With SS)", "D1 vs D3 (Without SS)"),
    horizontal_spacing=0.1
)

# With SS
fig_d1d3.add_trace(
    go.Scatter(
        x=jumps_with['D1_reflexivity'],
        y=jumps_with['D3_trend'],
        mode='markers',
        name='With SS',
        marker=dict(size=3, opacity=0.5, color='blue')
    ),
    row=1, col=1
)

# Without SS
fig_d1d3.add_trace(
    go.Scatter(
        x=jumps_without['D1_reflexivity'],
        y=jumps_without['D3_trend'],
        mode='markers',
        name='Without SS',
        marker=dict(size=3, opacity=0.5, color='red')
    ),
    row=1, col=2
)

# Add reference lines
fig_d1d3.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=1)
fig_d1d3.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
fig_d1d3.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)
fig_d1d3.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)

fig_d1d3.update_xaxes(title_text="D1 Reflexivity", row=1, col=1)
fig_d1d3.update_xaxes(title_text="D1 Reflexivity", row=1, col=2)
fig_d1d3.update_yaxes(title_text="D3 Trend", row=1, col=1)
fig_d1d3.update_yaxes(title_text="D3 Trend", row=1, col=2)

fig_d1d3.update_layout(
    title="D1 vs D3 Comparison: With vs Without Scattering Spectra",
    template="plotly_white",
    height=500,
    showlegend=False
)
fig_d1d3.show()

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
    
    fig_profiles = make_subplots(
        rows=1, cols=2,
        subplot_titles=("D1 Profiles (With SS)", "D1 Profiles (Without SS)"),
        horizontal_spacing=0.1
    )
    
    colors = px.colors.sequential.Viridis
    
    for col_idx, (X_sorted, title_suffix) in enumerate([(X_sorted_with, "With SS"), (X_sorted_without, "Without SS")], 1):
        for i in range(len(quantiles)-1):
            q_s, q_e = quantiles[i], quantiles[i+1]
            idx_s, idx_e = int(q_s*n), int(q_e*n)
            if idx_e <= idx_s: continue
            
            avg = np.mean(X_sorted[idx_s:idx_e], axis=0)
            color_idx = int(i / (len(quantiles)-1) * (len(colors)-1))
            color = colors[color_idx]
            
            fig_profiles.add_trace(
                go.Scatter(
                    x=t_axis, y=avg, mode='lines',
                    name=f"Q {q_s}-{q_e}",
                    line=dict(color=color, width=2),
                    showlegend=(col_idx == 1)
                ),
                row=1, col=col_idx
            )
        
        fig_profiles.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Jump", row=1, col=col_idx)
    
    fig_profiles.update_xaxes(title_text="Time (steps)", row=1, col=1)
    fig_profiles.update_xaxes(title_text="Time (steps)", row=1, col=2)
    fig_profiles.update_yaxes(title_text="Normalized Return x(t)", row=1, col=1)
    fig_profiles.update_yaxes(title_text="Normalized Return x(t)", row=1, col=2)
    
    fig_profiles.update_layout(
        title="Average Profiles along D1: With vs Without Scattering Spectra",
        template="plotly_white",
        height=500
    )
    fig_profiles.show()

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
        'D3 Mean',
        'D3 Std',
        'D1-Asymmetry Correlation',
        'D1 Correlation (With vs Without)'
    ],
    'With SS': [
        f"{jumps_with['D1_reflexivity'].mean():.3f}",
        f"{jumps_with['D1_reflexivity'].std():.3f}",
        f"{jumps_with['D1_reflexivity'].min():.3f}",
        f"{jumps_with['D1_reflexivity'].max():.3f}",
        f"{jumps_with['D2_mean_reversion'].mean():.3f}",
        f"{jumps_with['D2_mean_reversion'].std():.3f}",
        f"{jumps_with['D3_trend'].mean():.3f}",
        f"{jumps_with['D3_trend'].std():.3f}",
        f"{results_with_ss['correlation']:.3f}",
        f"{d1_corr:.3f}"
    ],
    'Without SS': [
        f"{jumps_without['D1_reflexivity'].mean():.3f}",
        f"{jumps_without['D1_reflexivity'].std():.3f}",
        f"{jumps_without['D1_reflexivity'].min():.3f}",
        f"{jumps_without['D1_reflexivity'].max():.3f}",
        f"{jumps_without['D2_mean_reversion'].mean():.3f}",
        f"{jumps_without['D2_mean_reversion'].std():.3f}",
        f"{jumps_without['D3_trend'].mean():.3f}",
        f"{jumps_without['D3_trend'].std():.3f}",
        f"{results_without_ss['correlation']:.3f}",
        "-"
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n=== Summary Statistics Comparison ===")
print(summary_df.to_string(index=False))

# %%

