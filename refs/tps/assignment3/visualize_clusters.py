"""we visualize frequency distributions and bayesian model to identify 4x4 cluster structure"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans


def gaussian_pdf(x, mu, sigma):
    """we compute gaussian probability density"""
    if sigma == 0:
        sigma = 1e-6
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def main():
    """we create comprehensive frequency distribution visualizations"""
    
    # we load data
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    figures_dir = data_dir / "figures"
    
    # we load empirical mapping
    with open(data_dir / "empirical_frequency_mapping.json", 'r') as fp:
        empirical_data = json.load(fp)
    
    # we load bayesian model
    with open(data_dir / "bayesian_regression_results.json", 'r') as fp:
        bayesian_data = json.load(fp)
    
    symbol_stats = empirical_data['symbol_statistics']
    all_pairs = empirical_data['all_pairs']
    likelihoods = bayesian_data['model']['likelihoods']
    
    # we extract all low and high frequencies
    all_low_freqs = []
    all_high_freqs = []
    for sym, stats in symbol_stats.items():
        all_low_freqs.extend(stats['all_low_freqs'])
        all_high_freqs.extend(stats['all_high_freqs'])
    
    all_low_freqs = np.array(all_low_freqs)
    all_high_freqs = np.array(all_high_freqs)
    
    print(f"Total frequency pairs: {len(all_low_freqs)}")
    print(f"Low frequencies: {len(np.unique(all_low_freqs))} unique values")
    print(f"High frequencies: {len(np.unique(all_high_freqs))} unique values")
    
    # we perform k-means clustering to identify the 4 clusters
    kmeans_low = KMeans(n_clusters=4, random_state=0, n_init=10)
    kmeans_high = KMeans(n_clusters=4, random_state=0, n_init=10)
    
    low_clusters = kmeans_low.fit_predict(all_low_freqs.reshape(-1, 1))
    high_clusters = kmeans_high.fit_predict(all_high_freqs.reshape(-1, 1))
    
    low_centers = sorted(kmeans_low.cluster_centers_.flatten())
    high_centers = sorted(kmeans_high.cluster_centers_.flatten())
    
    print(f"\nIdentified 4 low frequency clusters: {[f'{c:.1f} Hz' for c in low_centers]}")
    print(f"Identified 4 high frequency clusters: {[f'{c:.1f} Hz' for c in high_centers]}")
    
    # ========== FIGURE 1: Low and High Frequency Distributions ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # we plot low frequency histogram
    ax = axes[0, 0]
    counts, bins, patches = ax.hist(all_low_freqs, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    for center in low_centers:
        ax.axvline(center, color='red', linestyle='--', linewidth=2, label=f'{center:.1f} Hz')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Low Frequency Distribution (4 clusters identified)', fontsize=14, fontweight='bold')
    ax.legend(title='Cluster Centers', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # we plot high frequency histogram
    ax = axes[0, 1]
    counts, bins, patches = ax.hist(all_high_freqs, bins=50, alpha=0.7, color='coral', edgecolor='black')
    for center in high_centers:
        ax.axvline(center, color='red', linestyle='--', linewidth=2, label=f'{center:.1f} Hz')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('High Frequency Distribution (4 clusters identified)', fontsize=14, fontweight='bold')
    ax.legend(title='Cluster Centers', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # we plot low frequency KDE with cluster markers
    ax = axes[1, 0]
    kde_low = gaussian_kde(all_low_freqs)
    x_low = np.linspace(all_low_freqs.min() - 50, all_low_freqs.max() + 50, 1000)
    y_low = kde_low(x_low)
    ax.plot(x_low, y_low, 'b-', linewidth=2, label='KDE')
    ax.fill_between(x_low, y_low, alpha=0.3)
    for i, center in enumerate(low_centers):
        ax.axvline(center, color='red', linestyle='--', linewidth=2)
        ax.text(center, ax.get_ylim()[1] * 0.9, f'C{i+1}\n{center:.1f}Hz', 
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Low Frequency KDE with 4 Cluster Centers', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # we plot high frequency KDE with cluster markers
    ax = axes[1, 1]
    kde_high = gaussian_kde(all_high_freqs)
    x_high = np.linspace(all_high_freqs.min() - 50, all_high_freqs.max() + 50, 1000)
    y_high = kde_high(x_high)
    ax.plot(x_high, y_high, 'r-', linewidth=2, label='KDE')
    ax.fill_between(x_high, y_high, alpha=0.3, color='coral')
    for i, center in enumerate(high_centers):
        ax.axvline(center, color='red', linestyle='--', linewidth=2)
        ax.text(center, ax.get_ylim()[1] * 0.9, f'C{i+1}\n{center:.1f}Hz', 
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('High Frequency KDE with 4 Cluster Centers', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "frequency_distributions_4clusters.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {figures_dir / 'frequency_distributions_4clusters.png'}")
    
    # ========== FIGURE 2: 2D Frequency Space (4x4 grid) ==========
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # we plot scatter of all frequency pairs
    ax = axes[0]
    colors = plt.cm.tab20(np.linspace(0, 1, 16))
    symbol_list = sorted(symbol_stats.keys())
    
    for i, sym in enumerate(symbol_list):
        stats = symbol_stats[sym]
        low_f = np.array(stats['all_low_freqs'])
        high_f = np.array(stats['all_high_freqs'])
        ax.scatter(low_f, high_f, s=100, alpha=0.6, c=[colors[i]], 
                  edgecolors='black', linewidths=1.5, label=f"'{sym}'")
    
    # we add grid lines at cluster centers
    for center in low_centers:
        ax.axvline(center, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    for center in high_centers:
        ax.axhline(center, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Low Frequency (Hz)', fontsize=12)
    ax.set_ylabel('High Frequency (Hz)', fontsize=12)
    ax.set_title('Frequency Pairs (4x4 Grid Structure)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # we plot symbol means on the 4x4 grid
    ax = axes[1]
    for i, sym in enumerate(symbol_list):
        stats = symbol_stats[sym]
        ax.scatter(stats['low_mean'], stats['high_mean'], s=500, alpha=0.7, 
                  c=[colors[i]], edgecolors='black', linewidths=2)
        ax.text(stats['low_mean'], stats['high_mean'], sym, 
               ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # we add grid lines at cluster centers
    for i, center in enumerate(low_centers):
        ax.axvline(center, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(center, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
               f'Low-{i+1}', ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    for i, center in enumerate(high_centers):
        ax.axhline(center, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02, center,
               f'High-{i+1}', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax.set_xlabel('Low Frequency (Hz)', fontsize=12)
    ax.set_ylabel('High Frequency (Hz)', fontsize=12)
    ax.set_title('Symbol Means on 4x4 DTMF Grid', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "frequency_grid_4x4.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir / 'frequency_grid_4x4.png'}")
    
    # ========== FIGURE 3: Bayesian Gaussian Distributions ==========
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # we plot low frequency gaussians
    ax = axes[0]
    x_range = np.linspace(650, 1000, 1000)
    
    for i, sym in enumerate(sorted(likelihoods.keys())):
        params = likelihoods[sym]
        y = gaussian_pdf(x_range, params['low_mean'], params['low_std'])
        ax.plot(x_range, y, linewidth=2, label=f"'{sym}': N({params['low_mean']:.1f}, {params['low_std']:.1f}²)",
               alpha=0.7)
    
    for center in low_centers:
        ax.axvline(center, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Bayesian Model: Low Frequency Gaussian Distributions (4 clusters)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # we plot high frequency gaussians
    ax = axes[1]
    x_range = np.linspace(1150, 1700, 1000)
    
    for i, sym in enumerate(sorted(likelihoods.keys())):
        params = likelihoods[sym]
        y = gaussian_pdf(x_range, params['high_mean'], params['high_std'])
        ax.plot(x_range, y, linewidth=2, label=f"'{sym}': N({params['high_mean']:.1f}, {params['high_std']:.1f}²)",
               alpha=0.7)
    
    for center in high_centers:
        ax.axvline(center, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('Bayesian Model: High Frequency Gaussian Distributions (4 clusters)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "bayesian_gaussians_4clusters.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir / 'bayesian_gaussians_4clusters.png'}")
    
    # ========== FIGURE 4: 4x4 Grid Heatmap ==========
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # we create a 4x4 grid matrix
    grid = np.zeros((4, 4))
    grid_symbols = [['' for _ in range(4)] for _ in range(4)]
    grid_counts = [[0 for _ in range(4)] for _ in range(4)]
    
    # we assign each symbol to its grid cell
    for sym, stats in symbol_stats.items():
        low_mean = stats['low_mean']
        high_mean = stats['high_mean']
        
        # we find closest cluster centers
        low_idx = int(np.argmin([abs(low_mean - c) for c in low_centers]))
        high_idx = int(np.argmin([abs(high_mean - c) for c in high_centers]))
        
        grid[high_idx, low_idx] += stats['count']
        if grid_symbols[high_idx][low_idx]:
            grid_symbols[high_idx][low_idx] += f",{sym}"
        else:
            grid_symbols[high_idx][low_idx] = sym
        grid_counts[high_idx][low_idx] = grid[high_idx, low_idx]
    
    # we plot heatmap
    im = ax.imshow(grid, cmap='YlOrRd', aspect='auto')
    
    # we add text annotations
    for i in range(4):
        for j in range(4):
            if grid_symbols[i][j]:
                text = ax.text(j, i, f"{grid_symbols[i][j]}\n({int(grid_counts[i][j])})",
                             ha="center", va="center", color="black", fontsize=14, 
                             fontweight='bold')
    
    # we set ticks and labels
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels([f'{c:.1f} Hz' for c in low_centers], fontsize=11)
    ax.set_yticklabels([f'{c:.1f} Hz' for c in high_centers], fontsize=11)
    
    ax.set_xlabel('Low Frequency Cluster', fontsize=13, fontweight='bold')
    ax.set_ylabel('High Frequency Cluster', fontsize=13, fontweight='bold')
    ax.set_title('DTMF 4x4 Grid: Empirical Symbol Distribution\n(symbols and detection counts)', 
                fontsize=14, fontweight='bold')
    
    # we add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Detections', fontsize=12)
    
    # we add grid lines
    ax.set_xticks(np.arange(4) - 0.5, minor=True)
    ax.set_yticks(np.arange(4) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "dtmf_4x4_grid_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir / 'dtmf_4x4_grid_heatmap.png'}")
    
    # ========== FIGURE 5: Combined Overview ==========
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # we plot low freq histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(all_low_freqs, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    for center in low_centers:
        ax1.axvline(center, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Low Frequency (Hz)')
    ax1.set_ylabel('Count')
    ax1.set_title('Low Frequency Distribution', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # we plot high freq histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(all_high_freqs, bins=30, alpha=0.7, color='coral', edgecolor='black')
    for center in high_centers:
        ax2.axvline(center, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('High Frequency (Hz)')
    ax2.set_ylabel('Count')
    ax2.set_title('High Frequency Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # we plot 4x4 heatmap
    ax3 = fig.add_subplot(gs[0, 2])
    im = ax3.imshow(grid, cmap='YlOrRd', aspect='auto')
    for i in range(4):
        for j in range(4):
            if grid_symbols[i][j]:
                ax3.text(j, i, grid_symbols[i][j], ha="center", va="center", 
                        color="black", fontsize=12, fontweight='bold')
    ax3.set_xticks(np.arange(4))
    ax3.set_yticks(np.arange(4))
    ax3.set_xticklabels([f'{c:.0f}' for c in low_centers], fontsize=9)
    ax3.set_yticklabels([f'{c:.0f}' for c in high_centers], fontsize=9)
    ax3.set_xlabel('Low (Hz)')
    ax3.set_ylabel('High (Hz)')
    ax3.set_title('4x4 DTMF Grid', fontweight='bold')
    
    # we plot 2D scatter
    ax4 = fig.add_subplot(gs[1:, :])
    for i, sym in enumerate(symbol_list):
        stats = symbol_stats[sym]
        ax4.scatter(stats['low_mean'], stats['high_mean'], s=800, alpha=0.7, 
                   c=[colors[i]], edgecolors='black', linewidths=3)
        ax4.text(stats['low_mean'], stats['high_mean'], sym, 
                ha='center', va='center', fontsize=16, fontweight='bold', color='white')
    
    for center in low_centers:
        ax4.axvline(center, color='red', linestyle='--', linewidth=2, alpha=0.5)
    for center in high_centers:
        ax4.axhline(center, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    ax4.set_xlabel('Low Frequency (Hz)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('High Frequency (Hz)', fontsize=14, fontweight='bold')
    ax4.set_title('DTMF Frequency Space: 4x4 Cluster Structure from Empirical Data', 
                 fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3, linewidth=1.5)
    
    plt.suptitle('Empirical DTMF Frequency Analysis: 4 Low × 4 High Clusters', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(figures_dir / "dtmf_complete_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir / 'dtmf_complete_analysis.png'}")
    
    # we print summary
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nLow Frequency Clusters (4):")
    for i, center in enumerate(low_centers):
        print(f"  Cluster {i+1}: {center:.1f} Hz")
    
    print(f"\nHigh Frequency Clusters (4):")
    for i, center in enumerate(high_centers):
        print(f"  Cluster {i+1}: {center:.1f} Hz")
    
    print(f"\nTotal 4x4 Grid Cells: 16")
    print(f"Occupied Grid Cells: {sum(1 for row in grid_symbols for cell in row if cell)}")
    
    print(f"\n4x4 Grid Layout:")
    print(f"{'':>10}", end='')
    for c in low_centers:
        print(f"{c:>8.1f}", end='')
    print(" Hz (Low)")
    print("-" * 50)
    for i, high_c in enumerate(high_centers):
        print(f"{high_c:>8.1f} |", end='')
        for j in range(4):
            sym = grid_symbols[i][j] if grid_symbols[i][j] else '--'
            print(f"{sym:>8}", end='')
        print()
    
    print("\n" + "="*60)
    print("All visualization files saved to:", figures_dir)
    print("="*60)


if __name__ == "__main__":
    main()
