# %%
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import scipy.stats as stats
from scipy.optimize import curve_fit

# Add project root to path
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
except NameError:
    project_root = os.path.abspath("../../")

if project_root not in sys.path:
    sys.path.append(project_root)

from utils.data.curating_stooq import curate_stooq_dir_5min
from utils.data.jump_detection import detect_jumps_many, compute_u_shape
from model.wavelet.wavelet import WaveletModel
from plot_utils import save_plot

# %%
# Load data
data_dir = "/home/janis/4A/timeseries/data/stooq/poland/"
print(f"Loading data from {data_dir}...")
dfs = curate_stooq_dir_5min(data_dir, pattern="*.txt", recursive=True)

# Filter to stocks with enough data
valid_tickers = [t for t, d in dfs.items() if len(d) > 300]
print(f"Found {len(valid_tickers)} valid tickers.")

# Use ALL tickers
dfs_subset = {t: dfs[t] for t in valid_tickers}
print(f"Analyzing {len(dfs_subset)} stocks (all available).")

# %%
# Compute standardized returns x(t) for ALL data points (not just jumps)
print("Computing standardized returns for all data points...")
all_scores = []
all_abs_scores = []

for ticker, df in dfs_subset.items():
    # Compute returns
    r = df["close"].pct_change().fillna(0.0)
    
    # Compute intraday pattern f(t)
    f = compute_u_shape(r)
    
    # Deseasonalize
    r_des = r / f
    
    # Local volatility
    sigma = r_des.ewm(span=100, min_periods=10).std().bfill().fillna(1e-4)
    sigma = sigma.replace(0, 1e-4)
    
    # Jump score x(t) = r_des / sigma
    x = r_des / sigma
    
    all_scores.extend(x.dropna().values)
    all_abs_scores.extend(np.abs(x.dropna().values))

all_scores = np.array(all_scores)
all_abs_scores = np.array(all_abs_scores)

print(f"Total data points: {len(all_scores)}")
print(f"All scores statistics:")
print(f"  Mean: {all_scores.mean():.3f}")
print(f"  Std: {all_scores.std():.3f}")
print(f"  Min: {all_scores.min():.3f}")
print(f"  Max: {all_scores.max():.3f}")
print(f"  |Score| Mean: {all_abs_scores.mean():.3f}")

# %%
# Detect jumps (for comparison)
print("\nDetecting jumps...")
jumps_df = detect_jumps_many(dfs_subset, threshold=4.0)
print(f"Detected {len(jumps_df)} jumps.")

if not jumps_df.empty:
    jump_scores = jumps_df["score"].values
    abs_jump_scores = np.abs(jump_scores)
    print(f"Jump scores statistics:")
    print(f"  Mean: {jump_scores.mean():.3f}")
    print(f"  |Score| Mean: {abs_jump_scores.mean():.3f}")

# %%
# Fit Gumbel distribution to quantiles 0.28 to 1.0 (upper tail)
# The paper states that |x(t)| converges towards a Gumbel distribution under the null hypothesis
# Gumbel distribution: f(x) = (1/β) * exp(-(x-μ)/β) * exp(-exp(-(x-μ)/β))
# where μ is location parameter and β is scale parameter

print("\nFitting Gumbel distribution to |x(t)| values in quantile range [0.28, 1.0]...")

# Filter to upper tail (quantiles 0.28 to 1.0)
quantile_threshold = 0.28
threshold_value = np.percentile(all_abs_scores, quantile_threshold * 100)
filtered_abs_scores = all_abs_scores[all_abs_scores >= threshold_value]

print(f"Quantile threshold: {quantile_threshold}")
print(f"Value threshold: {threshold_value:.3f}")
print(f"Data points in range: {len(filtered_abs_scores)} ({100*len(filtered_abs_scores)/len(all_abs_scores):.1f}% of total)")

# Using scipy.stats.gumbel_r (right-skewed Gumbel)
# For absolute values, we use gumbel_r
params_gumbel = stats.gumbel_r.fit(filtered_abs_scores)
loc_gumbel, scale_gumbel = params_gumbel

print(f"\nFitted Gumbel parameters (on quantiles 0.28-1.0):")
print(f"  Location (μ): {loc_gumbel:.3f}")
print(f"  Scale (β): {scale_gumbel:.3f}")

# Create fitted distribution
gumbel_dist = stats.gumbel_r(loc=loc_gumbel, scale=scale_gumbel)

# %%
# Plot 1: Histogram of all scores (signed) with normal fit
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Histogram of all signed scores
ax1.hist(all_scores, bins=100, density=True, alpha=0.7, label='All Standardized Returns', color='blue')

# Normal distribution fit for comparison
x_range = np.linspace(all_scores.min(), all_scores.max(), 200)
normal_fit = stats.norm.pdf(x_range, all_scores.mean(), all_scores.std())
ax1.plot(x_range, normal_fit, 'r--', linewidth=2, 
         label=f'Normal Fit (μ={all_scores.mean():.2f}, σ={all_scores.std():.2f})')

# Add threshold lines
ax1.axvline(x=4, linestyle='--', color='orange', alpha=0.7, label='Jump Threshold (+4)')
ax1.axvline(x=-4, linestyle='--', color='orange', alpha=0.7, label='Jump Threshold (-4)')

# Highlight jump region if jumps were detected
if not jumps_df.empty:
    ax1.hist(jump_scores, bins=50, density=True, alpha=0.8, label='Detected Jumps', color='green')

ax1.set_xlabel("Standardized Return x(t)")
ax1.set_ylabel("Density")
ax1.set_title("Distribution of All Standardized Returns x(t) with Normal Distribution Fit")
ax1.legend()
ax1.grid(True, alpha=0.3)
save_plot(fig1, "jump_density_poland_distribution_signed", format='pdf')
plt.close(fig1)

# %%
# Plot 2: Histogram of ALL |x(t)| with Gumbel fit (full distribution)
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Histogram of all absolute scores
ax2.hist(all_abs_scores, bins=100, density=True, alpha=0.7, label='|All Standardized Returns|', color='blue')

# Gumbel distribution fit (only valid in upper tail)
x_range_abs = np.linspace(threshold_value, min(all_abs_scores.max(), 10), 200)
gumbel_pdf = gumbel_dist.pdf(x_range_abs)
ax2.plot(x_range_abs, gumbel_pdf, 'r-', linewidth=3, 
         label=f'Gumbel Fit (Q>0.28, μ={loc_gumbel:.2f}, β={scale_gumbel:.2f})')

# Add threshold lines
ax2.axvline(x=threshold_value, linestyle=':', color='blue', alpha=0.7, 
            label=f'Q=0.28 ({threshold_value:.2f})')
ax2.axvline(x=4, linestyle='--', color='orange', alpha=0.7, label='Jump Threshold (4)')

# Highlight jump region if jumps were detected
if not jumps_df.empty:
    ax2.hist(abs_jump_scores, bins=30, density=True, alpha=0.8, label='Detected Jumps', color='green')

ax2.set_xlabel("|Standardized Return| |x(t)|")
ax2.set_ylabel("Density")
ax2.set_title("Distribution of |x(t)| with Gumbel Fit (Q > 0.28)\nGumbel fit shown only in valid range")
ax2.set_xlim(0, min(10, all_abs_scores.max()))
ax2.legend()
ax2.grid(True, alpha=0.3)
save_plot(fig2, "jump_density_poland_distribution_abs", format='pdf')
plt.close(fig2)

# %%
# Plot 2b: Zoomed view of upper tail (Q > 0.28) with better Gumbel fit visualization
fig2b, ax2b = plt.subplots(figsize=(10, 6))

# Histogram of filtered scores (upper tail only)
ax2b.hist(filtered_abs_scores, bins=80, density=True, alpha=0.7, 
          label=f'|x(t)| (Q > {quantile_threshold})', color='blue')

# Gumbel distribution fit
x_range_tail = np.linspace(threshold_value, min(filtered_abs_scores.max(), 10), 300)
gumbel_pdf_tail = gumbel_dist.pdf(x_range_tail)
ax2b.plot(x_range_tail, gumbel_pdf_tail, 'r-', linewidth=3, 
          label=f'Gumbel Fit (μ={loc_gumbel:.3f}, β={scale_gumbel:.3f})')

# Add threshold line
ax2b.axvline(x=4, linestyle='--', color='orange', alpha=0.7, label='Jump Threshold (4)')

# Highlight jump region if jumps were detected
if not jumps_df.empty:
    jump_tail = abs_jump_scores[abs_jump_scores >= threshold_value]
    if len(jump_tail) > 0:
        ax2b.hist(jump_tail, bins=30, density=True, alpha=0.8, label='Detected Jumps', color='green')

ax2b.set_xlabel("|Standardized Return| |x(t)|")
ax2b.set_ylabel("Density")
ax2b.set_title(f"Upper Tail Distribution (Q > {quantile_threshold})\nwith Gumbel Distribution Fit")
ax2b.set_xlim(threshold_value, min(10, filtered_abs_scores.max()))
ax2b.legend()
ax2b.grid(True, alpha=0.3)
save_plot(fig2b, "jump_density_poland_distribution_tail", format='pdf')
plt.close(fig2b)

# %%
# Plot 3: Q-Q plot (Quantile-Quantile) to assess Gumbel fit quality
# Showing quantiles from 0.28 to 1.0 (where Gumbel is a good fit)
fig3, ax3 = plt.subplots(figsize=(8, 8))

# Quantile range: from 0.28 to 0.99
quantile_range = np.linspace(0.4, 0.99, 150)

# Theoretical quantiles from Gumbel distribution
theoretical_quantiles = gumbel_dist.ppf(quantile_range)
# Empirical quantiles from ALL data
empirical_quantiles = np.percentile(all_abs_scores, quantile_range * 100)

# Q-Q plot
ax3.scatter(theoretical_quantiles, empirical_quantiles, s=10, alpha=0.7, 
            color='blue', label=f'Data Points (Q: {quantile_threshold:.2f} - 1.0)')

# Perfect fit line (y=x)
min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit (y=x)')

# Add reference line for jump threshold
jump_threshold_quantile = (all_abs_scores < 4).mean()
if jump_threshold_quantile < 0.99:
    jump_theoretical = gumbel_dist.ppf(jump_threshold_quantile)
    ax3.axvline(x=jump_theoretical, linestyle=':', color='orange', alpha=0.7, 
                label=f'Jump Threshold (Q≈{jump_threshold_quantile:.2f})')

ax3.set_xlabel("Theoretical Gumbel Quantiles")
ax3.set_ylabel("Empirical Quantiles")
ax3.set_title(f"Q-Q Plot: Gumbel Distribution Fit (Quantiles {quantile_threshold:.2f} - 1.0)\n(Closer to diagonal = better fit)")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal', adjustable='box')
save_plot(fig3, "jump_density_poland_qq_plot", format='pdf')
plt.close(fig3)

# %%
# Statistical test: Kolmogorov-Smirnov test for Gumbel fit (on filtered data)
ks_statistic, ks_pvalue = stats.kstest(filtered_abs_scores, gumbel_dist.cdf)
print(f"\nKolmogorov-Smirnov Test (on quantiles {quantile_threshold:.2f} - 1.0):")
print(f"  KS Statistic: {ks_statistic:.4f}")
print(f"  p-value: {ks_pvalue:.4f}")
if ks_pvalue > 0.05:
    print(f"  → Cannot reject null hypothesis (p > 0.05): Gumbel fit is acceptable")
else:
    print(f"  → Reject null hypothesis (p < 0.05): Gumbel fit may not be good")

# Additional: Compute R² for the fit
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(theoretical_quantiles, empirical_quantiles)
print(f"\nLinear Regression on Q-Q Plot:")
print(f"  R²: {r_value**2:.4f}")
print(f"  Slope: {slope:.4f} (should be close to 1.0)")
print(f"  Intercept: {intercept:.4f} (should be close to 0.0)")

# %%
# Summary statistics table
summary_data = {
    'Statistic': [
        'Total Data Points',
        f'Data Points (Q > {quantile_threshold:.2f})',
        'Number of Jumps',
        'Mean Score (All)',
        'Std Score (All)',
        'Min Score (All)',
        'Max Score (All)',
        'Mean |Score| (All)',
        f'Mean |Score| (Q > {quantile_threshold:.2f})',
        'Gumbel Location (μ)',
        'Gumbel Scale (β)',
        'KS p-value',
        'Q-Q R²'
    ],
    'Value': [
        len(all_scores),
        len(filtered_abs_scores),
        len(jumps_df) if not jumps_df.empty else 0,
        f"{all_scores.mean():.3f}",
        f"{all_scores.std():.3f}",
        f"{all_scores.min():.3f}",
        f"{all_scores.max():.3f}",
        f"{all_abs_scores.mean():.3f}",
        f"{filtered_abs_scores.mean():.3f}",
        f"{loc_gumbel:.3f}",
        f"{scale_gumbel:.3f}",
        f"{ks_pvalue:.4f}",
        f"{r_value**2:.4f}"
    ]
}

if not jumps_df.empty:
    summary_data['Statistic'].extend([
        'Mean Score (Jumps)',
        'Mean |Score| (Jumps)'
    ])
    summary_data['Value'].extend([
        f"{jump_scores.mean():.3f}",
        f"{abs_jump_scores.mean():.3f}"
    ])

summary_df = pd.DataFrame(summary_data)

print("\nSummary Statistics:")
print(summary_df.to_string(index=False))

# %%
# Additional: Probability of exceeding threshold under Gumbel
threshold = 4.0
prob_exceed = 1 - gumbel_dist.cdf(threshold)
print(f"\nProbability of |x(t)| > {threshold} under Gumbel distribution: {prob_exceed:.6f}")
print(f"Expected number of jumps (if all data): {prob_exceed * len(all_abs_scores):.1f}")
if not jumps_df.empty:
    print(f"Actual number of jumps detected: {len(jumps_df)}")


# %%
# Classify jumps as endogenous vs exogenous and show examples
X_windows = None
jumps_subset = None
filtered_dfs = {}
window_steps = 12

if not jumps_df.empty:
    print("\n=== Classifying Jumps (Endogenous vs Exogenous) ===")
    
    # Filter trading hours for window extraction
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
    
    # Extract jump windows
    windows = []
    valid_indices = []
    jump_metadata = []
    
    print("Extracting jump windows...")
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
        jump_metadata.append({'ticker': ticker, 'timestamp': ts, 'score': row['score']})
    
    if len(windows) < 10:
        print(f"Not enough valid windows ({len(windows)}). Need at least 10 for classification.")
        X_windows = None
        jumps_subset = None
    else:
        X_windows = np.array(windows)
        jumps_subset = jumps_df.loc[valid_indices].copy()
        
        # Compute D1 reflexivity using Wavelet PCA
        print("Computing reflexivity scores (D1)...")
        wm = WaveletModel(n_layers=0, n_neurons=0, n_outputs=0, J=3, n_components=3)
        embedding = wm.fit_transform(X_windows)
        
        d1 = embedding[:, 0]
        
        # Orient D1: Positive = Exogenous (post-activity), Negative = Endogenous (pre-activity/symmetric)
        center = window_steps
        act_post = np.sum(np.abs(X_windows[:, center+1:]), axis=1)
        act_pre = np.sum(np.abs(X_windows[:, :center]), axis=1)
        asymmetry = (act_post - act_pre) / (act_post + act_pre + 1e-6)
        
        corr = np.corrcoef(d1, asymmetry)[0, 1]
        if corr < 0:
            d1 *= -1
        
        jumps_subset["D1_reflexivity"] = d1
        
        # Classify: Endogenous (D1 < 0 or close to 0), Exogenous (D1 > 0.5)
        jumps_subset["classification"] = "Endogenous"
        jumps_subset.loc[jumps_subset["D1_reflexivity"] > 0.5, "classification"] = "Exogenous"
        jumps_subset.loc[(jumps_subset["D1_reflexivity"] >= 0) & (jumps_subset["D1_reflexivity"] <= 0.5), "classification"] = "Mixed"
        
        print(f"\nClassification results:")
        print(jumps_subset["classification"].value_counts())

# %%
# Visualize examples from dataset
if X_windows is not None and jumps_subset is not None:
    endogenous_jumps = jumps_subset[jumps_subset["classification"] == "Endogenous"]
    exogenous_jumps = jumps_subset[jumps_subset["classification"] == "Exogenous"]
    
    if len(endogenous_jumps) > 0 and len(exogenous_jumps) > 0:
        # Select representative examples (median D1 in each class)
        endo_example = endogenous_jumps.iloc[np.abs(endogenous_jumps["D1_reflexivity"] - endogenous_jumps["D1_reflexivity"].median()).argmin()]
        exo_example = exogenous_jumps.iloc[np.abs(exogenous_jumps["D1_reflexivity"] - exogenous_jumps["D1_reflexivity"].median()).argmin()]
        
        examples = [
            (endo_example, "Endogenous", "blue"),
            (exo_example, "Exogenous", "red")
        ]
        
        fig_examples, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig_examples.suptitle(f"Example Jumps: Endogenous vs Exogenous\nEndogenous (D1={endo_example['D1_reflexivity']:.2f}) | Exogenous (D1={exo_example['D1_reflexivity']:.2f})", 
                              fontsize=12)
        
        t_axis = np.arange(-window_steps, window_steps + 1)
        
        for col_idx, (ex, label, color) in enumerate(examples):
            # Find the window index
            ex_idx = jumps_subset.index.get_loc(ex.name)
            x_profile = X_windows[ex_idx]
            x_abs = np.abs(x_profile)
            
            # Plot |x(t)|
            axes[0, col_idx].plot(t_axis, x_abs, 'o-', color=color, linewidth=2, markersize=4, label=f'{label} |x(t)|')
            axes[0, col_idx].axvline(x=0, linestyle='--', color='gray', alpha=0.5)
            axes[0, col_idx].set_xlabel("Time (steps)")
            axes[0, col_idx].set_ylabel("|x(t)|")
            axes[0, col_idx].set_title(f"{label} Jump: |x(t)| Profile")
            axes[0, col_idx].grid(True, alpha=0.3)
            axes[0, col_idx].legend()
            
            # Plot x(t)
            axes[1, col_idx].plot(t_axis, x_profile, 'o-', color=color, linewidth=2, markersize=4, label=f'{label} x(t)')
            axes[1, col_idx].axvline(x=0, linestyle='--', color='gray', alpha=0.5)
            axes[1, col_idx].set_xlabel("Time (steps)")
            axes[1, col_idx].set_ylabel("x(t)")
            axes[1, col_idx].set_title(f"{label} Jump: x(t) Profile")
            axes[1, col_idx].grid(True, alpha=0.3)
            axes[1, col_idx].legend()
        
        plt.tight_layout()
        save_plot(fig_examples, "jump_density_poland_examples", format='pdf')
        plt.close(fig_examples)
        
        # Print details
        print(f"\nEndogenous Example (from dataset):")
        print(f"  Ticker: {endo_example['ticker']}")
        print(f"  Timestamp: {endo_example['timestamp']}")
        print(f"  D1 Reflexivity: {endo_example['D1_reflexivity']:.3f}")
        print(f"  Jump Score: {endo_example['score']:.3f}")
        
        print(f"\nExogenous Example (from dataset):")
        print(f"  Ticker: {exo_example['ticker']}")
        print(f"  Timestamp: {exo_example['timestamp']}")
        print(f"  D1 Reflexivity: {exo_example['D1_reflexivity']:.3f}")
        print(f"  Jump Score: {exo_example['score']:.3f}")

# %%
# Show actual price data from the dataset
if X_windows is not None and jumps_subset is not None:
    endogenous_jumps = jumps_subset[jumps_subset["classification"] == "Endogenous"]
    exogenous_jumps = jumps_subset[jumps_subset["classification"] == "Exogenous"]
    
    if len(endogenous_jumps) > 0 and len(exogenous_jumps) > 0:
        endo_example = endogenous_jumps.iloc[np.abs(endogenous_jumps["D1_reflexivity"] - endogenous_jumps["D1_reflexivity"].median()).argmin()]
        exo_example = exogenous_jumps.iloc[np.abs(exogenous_jumps["D1_reflexivity"] - exogenous_jumps["D1_reflexivity"].median()).argmin()]
        examples = [
            (endo_example, "Endogenous", "blue"),
            (exo_example, "Exogenous", "red")
        ]
        
        fig_prices, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig_prices.suptitle(f"Real Examples from Dataset: Actual Price & Returns\nEndogenous: {endo_example['ticker']} at {endo_example['timestamp']} | Exogenous: {exo_example['ticker']} at {exo_example['timestamp']}", 
                            fontsize=12)
        
        for col_idx, (ex, label, color) in enumerate(examples):
            ticker = ex['ticker']
            ts = ex['timestamp']
            
            if ticker not in filtered_dfs: continue
            df = filtered_dfs[ticker]
            if ts not in df.index: continue
            
            loc = df.index.get_loc(ts)
            if loc - window_steps < 0 or loc + window_steps + 1 > len(df): continue
            
            subset = df.iloc[loc - window_steps : loc + window_steps + 1]
            
            # Time axis (actual timestamps)
            time_axis = subset.index
            prices = subset["close"].values
            returns = subset["close"].pct_change().fillna(0.0).values * 100
            
            # Plot price
            axes[0, col_idx].plot(time_axis, prices, 'o-', color=color, linewidth=2, markersize=5, label=f'{label} Price')
            axes[0, col_idx].axvline(x=ts, linestyle='--', color='orange', alpha=0.7, label='Jump')
            axes[0, col_idx].set_xlabel("Time")
            axes[0, col_idx].set_ylabel("Price")
            axes[0, col_idx].set_title(f"{label}: {ticker} - Price")
            axes[0, col_idx].grid(True, alpha=0.3)
            axes[0, col_idx].legend()
            axes[0, col_idx].tick_params(axis='x', rotation=45)
            
            # Plot returns
            axes[1, col_idx].plot(time_axis, returns, 'o-', color=color, linewidth=2, markersize=5, label=f'{label} Returns')
            axes[1, col_idx].axvline(x=ts, linestyle='--', color='orange', alpha=0.7, label='Jump')
            axes[1, col_idx].set_xlabel("Time")
            axes[1, col_idx].set_ylabel("Returns (%)")
            axes[1, col_idx].set_title(f"{label}: {ticker} - Returns")
            axes[1, col_idx].grid(True, alpha=0.3)
            axes[1, col_idx].legend()
            axes[1, col_idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_plot(fig_prices, "jump_density_poland_examples_prices", format='pdf')
        plt.close(fig_prices)

# %%
# Show multiple examples from the dataset
if X_windows is not None and jumps_subset is not None:
    endogenous_jumps = jumps_subset[jumps_subset["classification"] == "Endogenous"]
    exogenous_jumps = jumps_subset[jumps_subset["classification"] == "Exogenous"]
    
    print("\n=== Multiple Examples from Dataset ===")
    
    # Select top 3 examples of each type
    n_examples = 3
    if len(endogenous_jumps) >= n_examples:
        top_endo = endogenous_jumps.nsmallest(n_examples, 'D1_reflexivity')
        print(f"\nTop {n_examples} Endogenous Examples (most negative D1):")
        for idx, (_, row) in enumerate(top_endo.iterrows(), 1):
            print(f"  {idx}. {row['ticker']} at {row['timestamp']} | D1={row['D1_reflexivity']:.3f} | Score={row['score']:.2f}")
    
    if len(exogenous_jumps) >= n_examples:
        top_exo = exogenous_jumps.nlargest(n_examples, 'D1_reflexivity')
        print(f"\nTop {n_examples} Exogenous Examples (most positive D1):")
        for idx, (_, row) in enumerate(top_exo.iterrows(), 1):
            print(f"  {idx}. {row['ticker']} at {row['timestamp']} | D1={row['D1_reflexivity']:.3f} | Score={row['score']:.2f}")

# %%
# Distribution of D1 scores by classification
if X_windows is not None and jumps_subset is not None:
    fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
    
    for classification in ["Endogenous", "Mixed", "Exogenous"]:
        subset = jumps_subset[jumps_subset["classification"] == classification]
        if len(subset) > 0:
            ax_dist.hist(subset["D1_reflexivity"], bins=30, alpha=0.7, label=classification, density=False)
    
    ax_dist.axvline(x=0, linestyle='--', color='gray', alpha=0.7, label='Symmetric')
    ax_dist.axvline(x=0.5, linestyle='--', color='orange', alpha=0.7, label='Threshold')
    
    ax_dist.set_xlabel("D1 Reflexivity Score")
    ax_dist.set_ylabel("Count")
    ax_dist.set_title("Distribution of D1 Reflexivity Scores by Classification")
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)
    save_plot(fig_dist, "jump_density_poland_D1_classification", format='pdf')
    plt.close(fig_dist)

# %%
