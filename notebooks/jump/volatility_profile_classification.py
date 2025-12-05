# %%
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
except NameError:
    project_root = os.path.abspath("../../")

if project_root not in sys.path:
    sys.path.append(project_root)

from utils.data.curating_stooq import curate_stooq_dir_5min
from utils.data.jump_detection import detect_jumps_many

# %%
def power_law_model(t, N_less, p_less, N_greater, p_greater, t_c, d):
    """
    Power-law model for |x(t)| as in equation (1) of the paper:
    |x(t)| = 1_{t<tc} * N_< / |t-tc|^p_< + 1_{t>tc} * N_> / |t-tc|^p_> + d
    """
    result = np.zeros_like(t, dtype=float)
    mask_less = t < t_c
    mask_greater = t > t_c
    
    # Avoid division by zero
    eps = 1e-6
    t_diff_less = np.abs(t[mask_less] - t_c) + eps
    t_diff_greater = np.abs(t[mask_greater] - t_c) + eps
    
    result[mask_less] = N_less / (t_diff_less ** p_less)
    result[mask_greater] = N_greater / (t_diff_greater ** p_greater)
    result += d
    
    return result

def fit_power_law(x_abs, t_axis, center_idx):
    """
    Fit power-law model to |x(t)|.
    Returns fitted parameters and R² goodness of fit.
    """
    # Initial guess for parameters
    # t_c should be near the center (jump time)
    t_c_init = t_axis[center_idx]
    
    # Estimate N_< and N_> from pre/post jump activity
    pre_activity = np.mean(x_abs[:center_idx]) if center_idx > 0 else 1.0
    post_activity = np.mean(x_abs[center_idx+1:]) if center_idx < len(x_abs)-1 else 1.0
    
    # Initial guesses
    N_less_init = pre_activity * 0.1  # Scale factor
    N_greater_init = post_activity * 0.1
    p_less_init = 0.5  # Typical power-law exponent
    p_greater_init = 0.5
    d_init = np.min(x_abs) * 0.5  # Baseline
    
    # Bounds: ensure positive parameters and reasonable ranges
    bounds = (
        [0, 0.1, 0, 0.1, t_axis[0], 0],  # Lower bounds
        [np.inf, 3.0, np.inf, 3.0, t_axis[-1], np.max(x_abs)]  # Upper bounds
    )
    
    try:
        popt, _ = curve_fit(
            power_law_model,
            t_axis,
            x_abs,
            p0=[N_less_init, p_less_init, N_greater_init, p_greater_init, t_c_init, d_init],
            bounds=bounds,
            maxfev=5000
        )
        
        # Compute R²
        y_pred = power_law_model(t_axis, *popt)
        ss_res = np.sum((x_abs - y_pred) ** 2)
        ss_tot = np.sum((x_abs - np.mean(x_abs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'N_less': popt[0],
            'p_less': popt[1],
            'N_greater': popt[2],
            'p_greater': popt[3],
            't_c': popt[4],
            'd': popt[5],
            'r_squared': r_squared,
            'success': True
        }
    except Exception as e:
        return {
            'N_less': np.nan,
            'p_less': np.nan,
            'N_greater': np.nan,
            'p_greater': np.nan,
            't_c': np.nan,
            'd': np.nan,
            'r_squared': 0.0,
            'success': False,
            'error': str(e)
        }

def compute_asymmetry(x_profile, center_idx):
    """
    Compute asymmetry measure A_jump as in equation (2):
    A_jump = (A_> - A_<) / (A_> + A_<)
    where A_</> = sum_{t<0/t>0} |x(t) - min_{t<0/t>0}(x(t))|
    """
    # Split into pre-jump (t < 0) and post-jump (t > 0)
    x_pre = x_profile[:center_idx]
    x_post = x_profile[center_idx+1:]
    
    if len(x_pre) == 0 or len(x_post) == 0:
        return np.nan
    
    # Compute A_< and A_>
    min_pre = np.min(x_pre) if len(x_pre) > 0 else 0
    min_post = np.min(x_post) if len(x_post) > 0 else 0
    
    A_less = np.sum(np.abs(x_pre - min_pre))
    A_greater = np.sum(np.abs(x_post - min_post))
    
    # Compute asymmetry
    denominator = A_greater + A_less
    if denominator == 0:
        return 0.0
    
    A_jump = (A_greater - A_less) / denominator
    return A_jump

# %%
# Load data and detect jumps
data_dir = "/home/janis/4A/timeseries/data/stooq/hungary/"
print(f"Loading data from {data_dir}...")
dfs = curate_stooq_dir_5min(data_dir, pattern="*.txt", recursive=True)

# Filter to stocks with enough data
valid_tickers = [t for t, d in dfs.items() if len(d) > 1000]
print(f"Found {len(valid_tickers)} valid tickers.")

# Select subset
valid_tickers.sort(key=lambda t: len(dfs[t]), reverse=True)
selected_tickers = valid_tickers[:10]
dfs_subset = {t: dfs[t] for t in selected_tickers}

# Filter trading hours (remove first/last 60 mins)
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

# Detect jumps
print("Detecting jumps...")
jumps_df = detect_jumps_many(filtered_dfs, threshold=4.0)
print(f"Detected {len(jumps_df)} jumps.")

# %%
# Extract jump windows and fit power-law model
window_steps = 12  # 60 minutes for 5-min data
results = []

print("Fitting power-law models to jump profiles...")

for idx, row in jumps_df.iterrows():
    ticker = row["ticker"]
    ts = row["timestamp"]
    
    if ticker not in filtered_dfs: continue
    df = filtered_dfs[ticker]
    if ts not in df.index: continue
    
    loc = df.index.get_loc(ts)
    if loc - window_steps < 0 or loc + window_steps + 1 > len(df): continue
    
    # Extract window
    subset = df.iloc[loc - window_steps : loc + window_steps + 1]
    norm = row["f"] * row["sigma"]
    if norm == 0: norm = 1e-4
    
    r_window = subset["close"].pct_change().fillna(0.0).values
    x_profile = r_window / norm
    
    # Align jump direction
    jump_sign = np.sign(x_profile[window_steps])
    if jump_sign == 0: jump_sign = 1
    x_profile = x_profile * jump_sign
    
    # Time axis (centered at 0)
    center_idx = window_steps
    t_axis = np.arange(-center_idx, center_idx + 1)
    
    # Compute asymmetry
    A_jump = compute_asymmetry(x_profile, center_idx)
    
    # Fit power-law to |x(t)|
    x_abs = np.abs(x_profile)
    fit_result = fit_power_law(x_abs, t_axis, center_idx)
    
    results.append({
        'ticker': ticker,
        'timestamp': ts,
        'A_jump': A_jump,
        **fit_result
    })

results_df = pd.DataFrame(results)
print(f"Successfully fitted {results_df['success'].sum()} out of {len(results_df)} jumps.")

# %%
# Filter to good fits (R² > threshold, as in the paper)
r_squared_threshold = 0.5  # Adjustable threshold
good_fits = results_df[results_df['success'] & (results_df['r_squared'] > r_squared_threshold)].copy()
print(f"Jumps with good fit (R² > {r_squared_threshold}): {len(good_fits)} ({100*len(good_fits)/len(results_df):.1f}%)")

# %%
# Visualization 1: Distribution of A_jump
fig1 = go.Figure()

fig1.add_trace(go.Histogram(
    x=good_fits['A_jump'],
    name='A_jump Distribution',
    histnorm='probability density',
    nbinsx=50,
    opacity=0.7
))

# Add vertical lines for classification
fig1.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Symmetric (A=0)")
fig1.add_vline(x=0.3, line_dash="dash", line_color="orange", annotation_text="Exogenous (A>0)")
fig1.add_vline(x=-0.3, line_dash="dash", line_color="blue", annotation_text="Anticipatory (A<0)")

fig1.update_layout(
    title="Distribution of Asymmetry Measure A_jump<br>(Good fits only, R² > 0.5)",
    xaxis_title="A_jump = (A_> - A_<) / (A_> + A_<)",
    yaxis_title="Density",
    template="plotly_white",
    height=500
)
fig1.show()

# %%
# Visualization 2: Scatter plot of A_jump vs R²
fig2 = px.scatter(
    results_df[results_df['success']],
    x='r_squared',
    y='A_jump',
    color='A_jump',
    title="Asymmetry (A_jump) vs Goodness of Fit (R²)",
    labels={'r_squared': 'R² (Goodness of Fit)', 'A_jump': 'Asymmetry A_jump'},
    color_continuous_scale="RdBu",
    opacity=0.6
)
fig2.add_vline(x=r_squared_threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold ({r_squared_threshold})")
fig2.add_hline(y=0, line_dash="dash", line_color="gray")
fig2.update_layout(template="plotly_white", height=500)
fig2.show()

# %%
# Visualization 3: Example fits for different A_jump values
# Select examples: low A (anticipatory), mid A (endogenous), high A (exogenous)
if len(good_fits) >= 3:
    low_A = good_fits.nsmallest(1, 'A_jump').iloc[0]
    mid_A = good_fits.iloc[np.abs(good_fits['A_jump']).argmin()]
    high_A = good_fits.nlargest(1, 'A_jump').iloc[0]
    
    examples = [
        (low_A, "Anticipatory (A < 0)"),
        (mid_A, "Endogenous (A ≈ 0)"),
        (high_A, "Exogenous (A > 0)")
    ]
    
    fig3 = make_subplots(rows=1, cols=3, subplot_titles=[ex[1] for ex in examples])
    
    for col, (ex, label) in enumerate(examples, 1):
        ticker = ex['ticker']
        ts = ex['timestamp']
        df = filtered_dfs[ticker]
        loc = df.index.get_loc(ts)
        subset = df.iloc[loc - window_steps : loc + window_steps + 1]
        
        # Reconstruct x_profile
        jump_row = jumps_df[(jumps_df['ticker'] == ticker) & (jumps_df['timestamp'] == ts)].iloc[0]
        norm = jump_row["f"] * jump_row["sigma"]
        if norm == 0: norm = 1e-4
        r_window = subset["close"].pct_change().fillna(0.0).values
        x_profile = r_window / norm
        jump_sign = np.sign(x_profile[window_steps])
        if jump_sign == 0: jump_sign = 1
        x_profile = x_profile * jump_sign
        x_abs = np.abs(x_profile)
        
        t_axis = np.arange(-window_steps, window_steps + 1)
        
        # Plot data
        fig3.add_trace(
            go.Scatter(x=t_axis, y=x_abs, mode='lines+markers', name='|x(t)|', line=dict(color='blue')),
            row=1, col=col
        )
        
        # Plot fit
        if ex['success']:
            y_fit = power_law_model(t_axis, ex['N_less'], ex['p_less'], ex['N_greater'], 
                                   ex['p_greater'], ex['t_c'], ex['d'])
            fig3.add_trace(
                go.Scatter(x=t_axis, y=y_fit, mode='lines', name='Fit', line=dict(color='red', dash='dash')),
                row=1, col=col
            )
        
        fig3.update_xaxes(title_text="Time (rel to jump)", row=1, col=col)
        fig3.update_yaxes(title_text="|x(t)|", row=1, col=col)
    
    fig3.update_layout(
        title="Example Power-Law Fits for Different Asymmetry Classes",
        template="plotly_white",
        height=400,
        showlegend=False
    )
    fig3.show()

# %%
# Summary statistics
print("\n=== Summary Statistics ===")
print(f"Total jumps analyzed: {len(results_df)}")
print(f"Successful fits: {results_df['success'].sum()} ({100*results_df['success'].sum()/len(results_df):.1f}%)")
print(f"Good fits (R² > {r_squared_threshold}): {len(good_fits)} ({100*len(good_fits)/len(results_df):.1f}%)")

if len(good_fits) > 0:
    print(f"\nAsymmetry Statistics (good fits only):")
    print(f"  Mean A_jump: {good_fits['A_jump'].mean():.3f}")
    print(f"  Std A_jump: {good_fits['A_jump'].std():.3f}")
    print(f"  Median A_jump: {good_fits['A_jump'].median():.3f}")
    print(f"\nClassification (based on A_jump):")
    print(f"  Anticipatory (A < -0.1): {(good_fits['A_jump'] < -0.1).sum()} ({(good_fits['A_jump'] < -0.1).sum()/len(good_fits)*100:.1f}%)")
    print(f"  Endogenous (-0.1 ≤ A ≤ 0.1): {((good_fits['A_jump'] >= -0.1) & (good_fits['A_jump'] <= 0.1)).sum()} ({((good_fits['A_jump'] >= -0.1) & (good_fits['A_jump'] <= 0.1)).sum()/len(good_fits)*100:.1f}%)")
    print(f"  Exogenous (A > 0.1): {(good_fits['A_jump'] > 0.1).sum()} ({(good_fits['A_jump'] > 0.1).sum()/len(good_fits)*100:.1f}%)")

