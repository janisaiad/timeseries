import sys
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
except NameError:
    project_root = os.path.abspath("../../")

if project_root not in sys.path:
    sys.path.append(project_root)

from utils.data.curating_stooq import curate_stooq_dir_5min
from utils.data.jump_detection import detect_jumps_single, compute_u_shape
from model.wavelet.wavelet import WaveletModel

def plot_pca_directions(X_windows, jumps_subset, output_dir, ticker):
    """
    Visualizes what the PCA directions 'look like' by averaging x(t) profiles
    for extreme quantiles along D1, D2, and D3.
    """
    print("Generating PCA Direction visualizations (Average Profiles)...")
    
    directions = ["D1_reflexivity", "D2_mean_reversion", "D3_trend"]
    # For 5-min windows of length 25, center is at index 12
    center = X_windows.shape[1] // 2
    t_axis = np.arange(-center, center + 1)
    
    for dim in directions:
        if dim not in jumps_subset.columns: continue
        
        # Create bins/quantiles
        scores = jumps_subset[dim].values
        # Define quantiles: Low (0-10%), Mid (45-55%), High (90-100%)
        # Or 5 bins
        quantiles = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        fig = go.Figure()
        
        # Sort X by score
        sorted_indices = np.argsort(scores)
        X_sorted = X_windows[sorted_indices]
        
        n = len(scores)
        
        # Plot average profile for each bin
        colors = px.colors.sequential.Viridis
        
        for i in range(len(quantiles)-1):
            q_start = quantiles[i]
            q_end = quantiles[i+1]
            
            idx_start = int(q_start * n)
            idx_end = int(q_end * n)
            
            if idx_end <= idx_start: continue
            
            # Average profile
            avg_profile = np.mean(X_sorted[idx_start:idx_end], axis=0)
            
            # Color
            color_idx = int(i / (len(quantiles)-1) * (len(colors)-1))
            color = colors[color_idx]
            
            fig.add_trace(go.Scatter(
                x=t_axis,
                y=avg_profile,
                mode='lines',
                name=f"Q {q_start:.1f}-{q_end:.1f}",
                line=dict(color=color, width=2)
            ))
            
        fig.update_layout(
            title=f"Average Jump Profiles along {dim} - {ticker}<br>Low Score (Purple) -> High Score (Yellow)",
            xaxis_title="Time Steps (Rel to Jump)",
            yaxis_title="Normalized Return x(t)",
            template="plotly_white",
            hovermode="x unified"
        )
        
        # Add Jump Marker line
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Jump")
        
        out_path = os.path.join(output_dir, f"{ticker}_profile_{dim}.html")
        fig.write_html(out_path)
        print(f"  Saved profile for {dim} to {out_path}")

def main():
    # ---------------------------------------------------------
    # 1. Load Data for a Single Stock
    # ---------------------------------------------------------
    data_dir = "/home/janis/4A/timeseries/data/stooq/hungary/"
    print(f"Loading data from {data_dir}...")
    
    # We load all to find the best one, or just pick the first valid one
    # For efficiency, let's just find the largest file or load a few and pick best
    dfs = curate_stooq_dir_5min(data_dir, pattern="*.txt", recursive=True)
    
    if not dfs:
        print("No data found.")
        return

    # Pick the ticker with the most data points
    ticker = max(dfs, key=lambda t: len(dfs[t]))
    df = dfs[ticker]
    print(f"Selected Stock: {ticker} ({len(df)} 5-min bars)")

    # ---------------------------------------------------------
    # 2. Filter Trading Hours (10:30 - 15:00 approx)
    # ---------------------------------------------------------
    # Paper uses 10:30 - 15:00. 
    # Assuming standard day, we remove first 60 min and last 60 min.
    print("Filtering trading hours (removing first/last 60 mins)...")
    
    filtered_days = []
    grouped = df.groupby(df.index.date)
    for date, day_df in grouped:
        if len(day_df) <= 12: continue # skip short days
        day_df = day_df.sort_index()
        start = day_df.index[0] + pd.Timedelta(minutes=60)
        end = day_df.index[-1] - pd.Timedelta(minutes=60)
        mask = (day_df.index >= start) & (day_df.index <= end)
        filtered_days.append(day_df[mask])
    
    if not filtered_days:
        print("No data left after filtering.")
        return
        
    df_filtered = pd.concat(filtered_days)
    print(f"Data points after filtering: {len(df_filtered)}")

    # ---------------------------------------------------------
    # 3. Detect Jumps
    # ---------------------------------------------------------
    print("Detecting jumps...")
    # We use the single version since we have one dataframe
    jumps_df = detect_jumps_single(
        df_filtered, 
        ticker=ticker, 
        threshold=4.0, 
        cluster_window=pd.Timedelta("1h")
    )
    
    print(f"Detected {len(jumps_df)} jumps.")
    if len(jumps_df) < 5:
        print("Not enough jumps to perform PCA analysis.")
        return

    # ---------------------------------------------------------
    # 4. Extract Jump Windows (x(t))
    # ---------------------------------------------------------
    # For 5-min data, we use window=12 (approx 60 mins)
    window_steps = 12
    windows = []
    valid_indices = []
    
    print("Extracting jump windows...")
    for idx, row in jumps_df.iterrows():
        ts = row["timestamp"]
        if ts not in df_filtered.index: continue
        
        loc = df_filtered.index.get_loc(ts)
        
        # Check bounds
        if loc - window_steps < 0 or loc + window_steps + 1 > len(df_filtered):
            continue
            
        subset = df_filtered.iloc[loc - window_steps : loc + window_steps + 1]
        
        # Normalized return profile: r(t) / (f * sigma)
        # We use the jump's stored f/sigma for normalization
        norm = row["f"] * row["sigma"]
        if norm == 0: norm = 1e-4
            
        # Compute returns for the window
        # pct_change gives NaN at start, fill 0
        r_window = subset["close"].pct_change().fillna(0.0).values
        
        # Normalize
        x_profile = r_window / norm
        
        # Align jump direction (center is at index `window_steps`)
        # The paper aligns so that the jump itself is positive
        jump_val = x_profile[window_steps]
        jump_sign = np.sign(jump_val) if jump_val != 0 else 1
        
        windows.append(x_profile * jump_sign)
        valid_indices.append(idx)

    if not windows:
        print("Could not extract any valid windows.")
        return
        
    X_windows = np.array(windows)
    jumps_subset = jumps_df.loc[valid_indices].copy()
    print(f"Extracted {len(X_windows)} valid windows. Shape: {X_windows.shape}")

    # ---------------------------------------------------------
    # 5. Wavelet Kernel PCA (D1: Reflexivity)
    # ---------------------------------------------------------
    print("Running Wavelet Kernel PCA...")
    # J=3 for ~25 length series
    wm = WaveletModel(n_layers=0, n_neurons=0, n_outputs=0, J=3, n_components=3)
    embedding = wm.fit_transform(X_windows)
    
    # Extract D1
    d1 = embedding[:, 0]
    
    # Orient D1: Positive should correlate with Asymmetry (Activity After > Activity Before)
    center = window_steps
    # Activity After (t > 0)
    act_post = np.sum(np.abs(X_windows[:, center+1:]), axis=1)
    # Activity Before (t < 0)
    act_pre = np.sum(np.abs(X_windows[:, :center]), axis=1)
    
    asymmetry = (act_post - act_pre) / (act_post + act_pre + 1e-6)
    corr = np.corrcoef(d1, asymmetry)[0, 1]
    
    if corr < 0:
        print(f"Flipping D1 sign (correlation with asymmetry was {corr:.2f})")
        d1 *= -1
    else:
        print(f"D1 sign is correct (correlation with asymmetry was {corr:.2f})")
        
    jumps_subset["D1_reflexivity"] = d1

    # ---------------------------------------------------------
    # 6. Handcrafted Features (D2: Mean-Reversion, D3: Trend)
    # ---------------------------------------------------------
    # As per paper and our notebook refinement:
    # D2 ~ Pre-Jump - Post-Jump (V-shape check)
    # D3 ~ Pre-Jump + Post-Jump (Trend check)
    # Using t = center-1 (pre) and t = center+1 (post)
    
    x_pre = X_windows[:, center - 1]
    x_post = X_windows[:, center + 1]
    
    jumps_subset["D2_mean_reversion"] = x_pre - x_post
    jumps_subset["D3_trend"] = x_pre + x_post

    # ---------------------------------------------------------
    # 7. Reproduce Figures 5 & 6 (Plots)
    # ---------------------------------------------------------
    print("Generating plots...")
    
    # Fig 5: Reflexivity vs Mean-Reversion
    fig_mr = px.scatter(
        jumps_subset,
        x="D1_reflexivity",
        y="D2_mean_reversion",
        color="D2_mean_reversion",
        title=f"<b>Reflexivity vs Mean-Reversion</b> ({ticker})<br>Reproduction of Paper Fig 5",
        labels={
            "D1_reflexivity": "Reflexivity (D1) [Endogenous <-> Exogenous]",
            "D2_mean_reversion": "Mean Reversion (D2)"
        },
        color_continuous_scale="RdBu",
        opacity=0.7,
        hover_data=["timestamp", "score"]
    )
    fig_mr.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_mr.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Fig 6: Reflexivity vs Trend
    fig_tr = px.scatter(
        jumps_subset,
        x="D1_reflexivity",
        y="D3_trend",
        color="D3_trend",
        title=f"<b>Reflexivity vs Trend</b> ({ticker})<br>Reproduction of Paper Fig 6",
        labels={
            "D1_reflexivity": "Reflexivity (D1) [Endogenous <-> Exogenous]",
            "D3_trend": "Trend (D3)"
        },
        color_continuous_scale="Viridis",
        opacity=0.7,
        hover_data=["timestamp", "score"]
    )
    fig_tr.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_tr.add_hline(y=0, line_dash="dash", line_color="gray")

    # Save or Show
    # Since this is a script, we can show() but it might block or fail if headless.
    # We'll try to write to HTML as well.
    try:
        base_dir = os.path.dirname(__file__)
    except NameError:
        base_dir = os.getcwd()

    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    f_mr_path = os.path.join(output_dir, f"{ticker}_fig5_mean_reversion.html")
    f_tr_path = os.path.join(output_dir, f"{ticker}_fig6_trend.html")
    
    fig_mr.write_html(f_mr_path)
    fig_tr.write_html(f_tr_path)
    
    print(f"Plots saved to:\n  - {f_mr_path}\n  - {f_tr_path}")
    
    # ---------------------------------------------------------
    # 8. Generate Profile Plots (Visualizing the Directions)
    # ---------------------------------------------------------
    plot_pca_directions(X_windows, jumps_subset, output_dir, ticker)
    
    # Optionally show if interactive
    # fig_mr.show()
    # fig_tr.show()

if __name__ == "__main__":
    main()
