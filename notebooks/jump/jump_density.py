# %%
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# %%
# Load data
data_dir = "/home/janis/4A/timeseries/data/stooq/hungary/"
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
fig1 = go.Figure()

# Histogram of all signed scores
fig1.add_trace(go.Histogram(
    x=all_scores,
    name='All Standardized Returns',
    histnorm='probability density',
    nbinsx=100,
    opacity=0.7
))

# Normal distribution fit for comparison
x_range = np.linspace(all_scores.min(), all_scores.max(), 200)
normal_fit = stats.norm.pdf(x_range, all_scores.mean(), all_scores.std())
fig1.add_trace(go.Scatter(
    x=x_range,
    y=normal_fit,
    mode='lines',
    name=f'Normal Fit (μ={all_scores.mean():.2f}, σ={all_scores.std():.2f})',
    line=dict(color='red', width=2, dash='dash')
))

# Add threshold lines
fig1.add_vline(x=4, line_dash="dash", line_color="orange", annotation_text="Jump Threshold (+4)")
fig1.add_vline(x=-4, line_dash="dash", line_color="orange", annotation_text="Jump Threshold (-4)")

# Highlight jump region if jumps were detected
if not jumps_df.empty:
    fig1.add_trace(go.Histogram(
        x=jump_scores,
        name='Detected Jumps',
        histnorm='probability density',
        nbinsx=50,
        opacity=0.8,
        marker_color='green'
    ))

fig1.update_layout(
    title="Distribution of All Standardized Returns x(t)<br>with Normal Distribution Fit",
    xaxis_title="Standardized Return x(t)",
    yaxis_title="Density",
    template="plotly_white",
    height=500
)
fig1.show()

# %%
# Plot 2: Histogram of ALL |x(t)| with Gumbel fit (full distribution)
fig2 = go.Figure()

# Histogram of all absolute scores
fig2.add_trace(go.Histogram(
    x=all_abs_scores,
    name='|All Standardized Returns|',
    histnorm='probability density',
    nbinsx=100,
    opacity=0.7
))

# Gumbel distribution fit (only valid in upper tail)
x_range_abs = np.linspace(threshold_value, min(all_abs_scores.max(), 10), 200)
gumbel_pdf = gumbel_dist.pdf(x_range_abs)
fig2.add_trace(go.Scatter(
    x=x_range_abs,
    y=gumbel_pdf,
    mode='lines',
    name=f'Gumbel Fit (Q>0.28, μ={loc_gumbel:.2f}, β={scale_gumbel:.2f})',
    line=dict(color='red', width=3)
))

# Add threshold lines
fig2.add_vline(x=threshold_value, line_dash="dot", line_color="blue", 
               annotation_text=f"Q=0.28 ({threshold_value:.2f})")
fig2.add_vline(x=4, line_dash="dash", line_color="orange", annotation_text="Jump Threshold (4)")

# Highlight jump region if jumps were detected
if not jumps_df.empty:
    fig2.add_trace(go.Histogram(
        x=abs_jump_scores,
        name='Detected Jumps',
        histnorm='probability density',
        nbinsx=30,
        opacity=0.8,
        marker_color='green'
    ))

fig2.update_layout(
    title="Distribution of |x(t)| with Gumbel Fit (Q > 0.28)<br>Gumbel fit shown only in valid range",
    xaxis_title="|Standardized Return| |x(t)|",
    yaxis_title="Density",
    template="plotly_white",
    height=500,
    xaxis_range=[0, min(10, all_abs_scores.max())]
)
fig2.show()

# %%
# Plot 2b: Zoomed view of upper tail (Q > 0.28) with better Gumbel fit visualization
fig2b = go.Figure()

# Histogram of filtered scores (upper tail only)
fig2b.add_trace(go.Histogram(
    x=filtered_abs_scores,
    name=f'|x(t)| (Q > {quantile_threshold})',
    histnorm='probability density',
    nbinsx=80,
    opacity=0.7
))

# Gumbel distribution fit
x_range_tail = np.linspace(threshold_value, min(filtered_abs_scores.max(), 10), 300)
gumbel_pdf_tail = gumbel_dist.pdf(x_range_tail)
fig2b.add_trace(go.Scatter(
    x=x_range_tail,
    y=gumbel_pdf_tail,
    mode='lines',
    name=f'Gumbel Fit (μ={loc_gumbel:.3f}, β={scale_gumbel:.3f})',
    line=dict(color='red', width=3)
))

# Add threshold line
fig2b.add_vline(x=4, line_dash="dash", line_color="orange", annotation_text="Jump Threshold (4)")

# Highlight jump region if jumps were detected
if not jumps_df.empty:
    jump_tail = abs_jump_scores[abs_jump_scores >= threshold_value]
    if len(jump_tail) > 0:
        fig2b.add_trace(go.Histogram(
            x=jump_tail,
            name='Detected Jumps',
            histnorm='probability density',
            nbinsx=30,
            opacity=0.8,
            marker_color='green'
        ))

fig2b.update_layout(
    title=f"Upper Tail Distribution (Q > {quantile_threshold})<br>with Gumbel Distribution Fit",
    xaxis_title="|Standardized Return| |x(t)|",
    yaxis_title="Density",
    template="plotly_white",
    height=500,
    xaxis_range=[threshold_value, min(10, filtered_abs_scores.max())]
)
fig2b.show()

# %%
# Plot 3: Q-Q plot (Quantile-Quantile) to assess Gumbel fit quality
# Showing quantiles from 0.28 to 1.0 (where Gumbel is a good fit)
fig3 = go.Figure()

# Quantile range: from 0.28 to 0.99
quantile_range = np.linspace(0.4, 0.99, 150)

# Theoretical quantiles from Gumbel distribution
theoretical_quantiles = gumbel_dist.ppf(quantile_range)
# Empirical quantiles from ALL data
empirical_quantiles = np.percentile(all_abs_scores, quantile_range * 100)

# Q-Q plot
fig3.add_trace(go.Scatter(
    x=theoretical_quantiles,
    y=empirical_quantiles,
    mode='markers',
    name=f'Data Points (Q: {quantile_threshold:.2f} - 1.0)',
    marker=dict(size=4, opacity=0.7, color='blue')
))

# Perfect fit line (y=x)
min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
fig3.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Perfect Fit (y=x)',
    line=dict(color='red', width=2, dash='dash')
))

# Add reference line for jump threshold
jump_threshold_quantile = (all_abs_scores < 4).mean()
if jump_threshold_quantile < 0.99:
    jump_theoretical = gumbel_dist.ppf(jump_threshold_quantile)
    fig3.add_vline(x=jump_theoretical, line_dash="dot", line_color="orange", 
                   annotation_text=f"Jump Threshold (Q≈{jump_threshold_quantile:.2f})")

fig3.update_layout(
    title=f"Q-Q Plot: Gumbel Distribution Fit (Quantiles {quantile_threshold:.2f} - 1.0)<br>(Closer to diagonal = better fit)",
    xaxis_title="Theoretical Gumbel Quantiles",
    yaxis_title="Empirical Quantiles",
    template="plotly_white",
    height=500
)
fig3.show()

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
if not jumps_df.empty:
    print("\n=== Classifying Jumps (Endogenous vs Exogenous) ===")
    
    # Filter trading hours for window extraction
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
    
    # Extract jump windows
    window_steps = 12
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
        
        # Select examples
        endogenous_jumps = jumps_subset[jumps_subset["classification"] == "Endogenous"]
        exogenous_jumps = jumps_subset[jumps_subset["classification"] == "Exogenous"]
        
        # %%
        # Visualize examples
        if len(endogenous_jumps) > 0 and len(exogenous_jumps) > 0:
            # Select representative examples (median D1 in each class)
            endo_example = endogenous_jumps.iloc[np.abs(endogenous_jumps["D1_reflexivity"] - endogenous_jumps["D1_reflexivity"].median()).argmin()]
            exo_example = exogenous_jumps.iloc[np.abs(exogenous_jumps["D1_reflexivity"] - exogenous_jumps["D1_reflexivity"].median()).argmin()]
            
            examples = [
                (endo_example, "Endogenous", "blue"),
                (exo_example, "Exogenous", "red")
            ]
            
            fig_examples = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Endogenous Jump: |x(t)| Profile",
                    "Exogenous Jump: |x(t)| Profile",
                    "Endogenous Jump: x(t) Profile",
                    "Exogenous Jump: x(t) Profile"
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            t_axis = np.arange(-window_steps, window_steps + 1)
            
            for col_idx, (ex, label, color) in enumerate(examples, 1):
                # Find the window index
                ex_idx = jumps_subset.index.get_loc(ex.name)
                x_profile = X_windows[ex_idx]
                x_abs = np.abs(x_profile)
                
                # Plot |x(t)|
                fig_examples.add_trace(
                    go.Scatter(
                        x=t_axis, y=x_abs, mode='lines+markers',
                        name=f'{label} |x(t)|',
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    ),
                    row=1, col=col_idx
                )
                
                # Plot x(t)
                fig_examples.add_trace(
                    go.Scatter(
                        x=t_axis, y=x_profile, mode='lines+markers',
                        name=f'{label} x(t)',
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    ),
                    row=2, col=col_idx
                )
                
                # Add jump marker
                fig_examples.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=col_idx)
                fig_examples.add_vline(x=0, line_dash="dash", line_color="gray", row=2, col=col_idx)
                
                # Update axes
                fig_examples.update_xaxes(title_text="Time (steps)", row=1, col=col_idx)
                fig_examples.update_xaxes(title_text="Time (steps)", row=2, col=col_idx)
                fig_examples.update_yaxes(title_text="|x(t)|", row=1, col=col_idx)
                fig_examples.update_yaxes(title_text="x(t)", row=2, col=col_idx)
            
            fig_examples.update_layout(
                title=f"Example Jumps: Endogenous vs Exogenous<br>Endogenous (D1={endo_example['D1_reflexivity']:.2f}) | Exogenous (D1={exo_example['D1_reflexivity']:.2f})",
                template="plotly_white",
                height=700,
                showlegend=False
            )
            fig_examples.show()
            
            # Print details
            print(f"\nEndogenous Example:")
            print(f"  Ticker: {endo_example['ticker']}")
            print(f"  Timestamp: {endo_example['timestamp']}")
            print(f"  D1 Reflexivity: {endo_example['D1_reflexivity']:.3f}")
            print(f"  Jump Score: {endo_example['score']:.3f}")
            
            print(f"\nExogenous Example:")
            print(f"  Ticker: {exo_example['ticker']}")
            print(f"  Timestamp: {exo_example['timestamp']}")
            print(f"  D1 Reflexivity: {exo_example['D1_reflexivity']:.3f}")
            print(f"  Jump Score: {exo_example['score']:.3f}")
        
        # %%
        # Distribution of D1 scores by classification
        fig_dist = go.Figure()
        
        for classification in ["Endogenous", "Mixed", "Exogenous"]:
            subset = jumps_subset[jumps_subset["classification"] == classification]
            if len(subset) > 0:
                fig_dist.add_trace(go.Histogram(
                    x=subset["D1_reflexivity"],
                    name=classification,
                    opacity=0.7,
                    nbinsx=30
                ))
        
        fig_dist.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Symmetric")
        fig_dist.add_vline(x=0.5, line_dash="dash", line_color="orange", annotation_text="Threshold")
        
        fig_dist.update_layout(
            title="Distribution of D1 Reflexivity Scores by Classification",
            xaxis_title="D1 Reflexivity Score",
            yaxis_title="Count",
            template="plotly_white",
            height=500,
            barmode='overlay'
        )
        fig_dist.show()

# %%
