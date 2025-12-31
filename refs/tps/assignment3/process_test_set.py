"""we process the test set using the trained bayesian model"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.cluster import KMeans
import ruptures as rpt


# we define constants
FS = 22050
WINDOW_LENGTH = 900
OVERLAP = WINDOW_LENGTH // 2
FRAME_STEP = WINDOW_LENGTH - OVERLAP
DTMF_FREQ_RANGE = (400.0, 2000.0)


def gaussian_pdf(x, mu, sigma):
    """we compute gaussian probability density"""
    if sigma == 0:
        sigma = 1e-6
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def predict_symbol_bayesian(low_freq, high_freq, priors, likelihoods):
    """we predict symbol using bayesian inference"""
    posteriors = {}
    
    for sym in priors.keys():
        prior = priors[sym]
        likelihood_low = gaussian_pdf(low_freq, likelihoods[sym]['low_mean'], likelihoods[sym]['low_std'])
        likelihood_high = gaussian_pdf(high_freq, likelihoods[sym]['high_mean'], likelihoods[sym]['high_std'])
        likelihood = likelihood_low * likelihood_high
        posteriors[sym] = prior * likelihood
    
    total_posterior = sum(posteriors.values())
    if total_posterior > 0:
        posteriors = {sym: prob / total_posterior for sym, prob in posteriors.items()}
    else:
        posteriors = {sym: 1.0 / len(posteriors) for sym in posteriors.keys()}
    
    best_symbol = max(posteriors.items(), key=lambda x: x[1])
    return best_symbol[0], best_symbol[1], posteriors


def process_test_signal(signal, signal_idx, priors, likelihoods, output_dir, figures_dir):
    """we process a test signal and predict symbols"""
    
    log_lines = []
    log_lines.append(f"Processing TEST signal {signal_idx}")
    log_lines.append(f"Signal length: {len(signal)} samples ({len(signal)/FS:.3f} seconds)")
    
    sig_fig_dir = figures_dir / f"test_signal_{signal_idx}"
    sig_fig_dir.mkdir(parents=True, exist_ok=True)
    
    # we compute STFT
    f, t, Zxx = stft(signal, fs=FS, nperseg=WINDOW_LENGTH, noverlap=OVERLAP)
    energy_per_freq = np.sum(np.abs(Zxx)**2, axis=1)
    log_lines.append(f"STFT computed: {Zxx.shape[0]} freq bins x {Zxx.shape[1]} time frames")
    
    # we perform k-means (k=2) to separate signal from noise
    energy_reshaped = energy_per_freq.reshape(-1, 1)
    kmeans_energy = KMeans(n_clusters=2, random_state=0, n_init=10)
    energy_labels = kmeans_energy.fit_predict(energy_reshaped)
    cluster_centers = kmeans_energy.cluster_centers_.flatten()
    signal_cluster_idx = np.argmax(cluster_centers)
    high_energy_freq_indices = np.where(energy_labels == signal_cluster_idx)[0]
    log_lines.append(f"Energy clustering: {len(high_energy_freq_indices)} high-energy bins identified")
    
    # we filter to DTMF band
    high_energy_freqs_all = f[high_energy_freq_indices]
    high_energy_energies_all = energy_per_freq[high_energy_freq_indices]
    band_mask = (high_energy_freqs_all >= DTMF_FREQ_RANGE[0]) & (high_energy_freqs_all <= DTMF_FREQ_RANGE[1])
    high_energy_freqs = high_energy_freqs_all[band_mask]
    high_energy_energies = high_energy_energies_all[band_mask]
    high_energy_indices = high_energy_freq_indices[band_mask]
    n_high_energy = len(high_energy_indices)
    log_lines.append(f"DTMF band filtering: {n_high_energy} bins in [{DTMF_FREQ_RANGE[0]}, {DTMF_FREQ_RANGE[1]}] Hz")
    
    if n_high_energy < 2:
        log_lines.append("Warning: insufficient high-energy bins")
        return {'signal_idx': signal_idx, 'predicted': '', 'error': 'insufficient_bins'}, log_lines
    
    # we perform hierarchical clustering
    km_2 = KMeans(n_clusters=2, random_state=0, n_init=10)
    coarse_labels = km_2.fit_predict(high_energy_freqs.reshape(-1, 1))
    coarse_centers = km_2.cluster_centers_.flatten()
    low_coarse_id = int(np.argmin(coarse_centers))
    high_coarse_id = int(np.argmax(coarse_centers))
    
    low_mask = coarse_labels == low_coarse_id
    high_mask = coarse_labels == high_coarse_id
    low_freqs = high_energy_freqs[low_mask]
    low_energies = high_energy_energies[low_mask]
    low_indices = high_energy_indices[low_mask]
    high_freqs = high_energy_freqs[high_mask]
    high_energies = high_energy_energies[high_mask]
    high_indices = high_energy_indices[high_mask]
    
    log_lines.append(f"Coarse split: low={len(low_freqs)}, high={len(high_freqs)}")
    
    # we run the complete pipeline (same as training)
    from torun import elbow_kmeans_1d, build_all_intervals_for_group, select_topN_non_overlapping
    
    k_low, low_labels, _, _ = elbow_kmeans_1d(low_freqs, k_max=8)
    k_high, high_labels, _, _ = elbow_kmeans_1d(high_freqs, k_max=8)
    log_lines.append(f"Elbow clustering: k_low={k_low}, k_high={k_high}")
    
    # we build frequency clusters
    major_freq_clusters = []
    cluster_id_counter = 0
    
    if (low_labels is not None) and (k_low > 0):
        for cid in range(int(k_low)):
            mask = low_labels == cid
            cfreqs = low_freqs[mask]
            cenergies = low_energies[mask]
            cidx = low_indices[mask]
            if len(cfreqs) == 0:
                continue
            major_freq_clusters.append({
                'cluster_id': cluster_id_counter,
                'band': 'low',
                'center_freq': float(np.mean(cfreqs)),
                'center_energy': float(np.mean(cenergies)),
                'freq_indices': cidx,
                'freqs': cfreqs,
                'energies': cenergies,
            })
            cluster_id_counter += 1
    
    if (high_labels is not None) and (k_high > 0):
        for cid in range(int(k_high)):
            mask = high_labels == cid
            cfreqs = high_freqs[mask]
            cenergies = high_energies[mask]
            cidx = high_indices[mask]
            if len(cfreqs) == 0:
                continue
            major_freq_clusters.append({
                'cluster_id': cluster_id_counter,
                'band': 'high',
                'center_freq': float(np.mean(cfreqs)),
                'center_energy': float(np.mean(cenergies)),
                'freq_indices': cidx,
                'freqs': cfreqs,
                'energies': cenergies,
            })
            cluster_id_counter += 1
    
    major_freq_clusters.sort(key=lambda c: c['center_freq'])
    log_lines.append(f"Frequency clusters: {len(major_freq_clusters)}")
    
    # we run changepoint detection
    t_max = len(t)
    all_changepoints = []
    
    for cluster in major_freq_clusters:
        freq_indices = np.array(cluster['freq_indices'], dtype=int)
        if len(freq_indices) == 0:
            continue
        
        cluster_f = f[freq_indices]
        center_f = float(cluster.get('center_freq', float(np.mean(cluster_f))))
        rel_idx = int(np.argmin(np.abs(cluster_f - center_f)))
        selected_freq_idx = int(freq_indices[rel_idx])
        selected_freq = float(f[selected_freq_idx])
        
        freq_energy = np.abs(Zxx[selected_freq_idx, :])**2
        sigma_est = np.std(freq_energy)
        pen_bic = 2 * sigma_est**2 * np.log(t_max)
        
        algo = rpt.Pelt(model="l2", jump=1)
        predicted_bkps = algo.fit_predict(freq_energy, pen=pen_bic)
        signal_bkps = [min(int(idx * FRAME_STEP), len(signal)) for idx in predicted_bkps[:-1]]
        
        all_changepoints.append({
            'cluster_id': int(cluster['cluster_id']),
            'center_freq': float(cluster.get('center_freq', selected_freq)),
            'selected_freq': selected_freq,
            'selected_freq_idx': selected_freq_idx,
            'changepoints': signal_bkps,
            'frame_bkps': predicted_bkps[:-1],
            'energy_signal': freq_energy,
            'n_cps': int(len(signal_bkps)),
        })
    
    all_bkps = sorted(set([bp for cp_dict in all_changepoints for bp in cp_dict['changepoints']]))
    log_lines.append(f"Changepoints: {len(all_bkps)}")
    
    # we split into low/high and build intervals
    selected_freqs = np.array([float(cp['selected_freq']) for cp in all_changepoints], dtype=float)
    km2 = KMeans(n_clusters=2, random_state=0, n_init=10)
    coarse_labels = km2.fit_predict(selected_freqs.reshape(-1, 1))
    centers = km2.cluster_centers_.flatten()
    low_id = int(np.argmin(centers))
    high_id = int(np.argmax(centers))
    low_freq_clusters = [cp for cp, lab in zip(all_changepoints, coarse_labels) if int(lab) == low_id]
    high_freq_clusters = [cp for cp, lab in zip(all_changepoints, coarse_labels) if int(lab) == high_id]
    
    low_all = build_all_intervals_for_group(low_freq_clusters, len(signal), FRAME_STEP)
    high_all = build_all_intervals_for_group(high_freq_clusters, len(signal), FRAME_STEP)
    
    # we estimate N from intervals
    N = max(len(low_all), len(high_all), 5)
    
    low_ranked, low_topN_time = select_topN_non_overlapping(low_all, N, score_key='mean_energy')
    high_ranked, high_topN_time = select_topN_non_overlapping(high_all, N, score_key='mean_energy')
    
    # we predict using Bayesian model
    n_pairs = min(len(low_topN_time), len(high_topN_time))
    predicted = []
    freq_pairs = []
    
    for i in range(n_pairs):
        low_itv = low_topN_time[i]
        high_itv = high_topN_time[i]
        
        # we use Bayesian inference
        sym, confidence, posteriors = predict_symbol_bayesian(
            low_itv['freq'], high_itv['freq'], priors, likelihoods
        )
        predicted.append(sym)
        freq_pairs.append({
            'low_freq': float(low_itv['freq']),
            'high_freq': float(high_itv['freq']),
            'symbol': sym,
            'confidence': float(confidence),
        })
        log_lines.append(f"Pair {i+1}: ({low_itv['freq']:.1f}, {high_itv['freq']:.1f}) Hz -> '{sym}' (conf={confidence:.4f})")
    
    pred_seq = ''.join(predicted)
    log_lines.append(f"\nPredicted sequence: {pred_seq}")
    
    # we save figures (similar to training)
    spectrogram = np.abs(Zxx)
    
    # figure 1: STFT overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(signal)
    axes[0, 0].set_title(f"TEST Signal {signal_idx}")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)
    
    im1 = axes[0, 1].pcolormesh(t, f, 20 * np.log10(spectrogram + 1e-10), shading='gouraud', cmap='viridis')
    axes[0, 1].set_title("STFT Spectrogram")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Frequency (Hz)")
    axes[0, 1].set_ylim([0, 4000])
    plt.colorbar(im1, ax=axes[0, 1], label="Magnitude (dB)")
    
    axes[1, 0].plot(f, energy_per_freq, 'b-', linewidth=1.5)
    axes[1, 0].set_title("Energy per Frequency Bin")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Energy")
    axes[1, 0].set_xlim([0, 4000])
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(f, energy_per_freq, 'b-', linewidth=1.5)
    axes[1, 1].set_title("Energy per Frequency Bin (Log Scale)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Energy (log scale)")
    axes[1, 1].set_xlim([0, 4000])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(sig_fig_dir / "01_stft_overview.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # figure 2: detected symbols
    plt.figure(figsize=(16, 5))
    plt.plot(np.arange(len(signal)) / FS, signal, 'b-', linewidth=0.8, alpha=0.7)
    y_pos = float(np.max(signal)) * 0.8 if len(signal) > 0 and np.max(signal) != 0 else 0.8
    for i in range(n_pairs):
        s = min(low_topN_time[i]['start_time'], high_topN_time[i]['start_time'])
        e = max(low_topN_time[i]['end_time'], high_topN_time[i]['end_time'])
        plt.axvspan(s/FS, e/FS, alpha=0.15, color='yellow')
        mid_t = (s + e) / (2 * FS)
        plt.text(mid_t, y_pos, predicted[i], ha='center', va='bottom', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    plt.title(f'TEST Signal {signal_idx}: Predicted Sequence = {pred_seq}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.savefig(sig_fig_dir / "02_detected_symbols.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # we prepare results
    results = {
        'signal_idx': signal_idx,
        'predicted': pred_seq,
        'n_symbols': len(predicted),
        'frequency_pairs': freq_pairs,
    }
    
    with open(output_dir / f"test_results_{signal_idx}.json", 'w') as fp:
        json.dump(results, fp, indent=2)
    
    with open(output_dir / f"test_log_{signal_idx}.txt", 'w') as fp:
        fp.write('\n'.join(log_lines))
    
    return results, log_lines


def main():
    """we process test set using trained Bayesian model"""
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    figures_dir = data_dir / "figures"
    
    # we load Bayesian model
    print("Loading trained Bayesian model...")
    with open(data_dir / "bayesian_regression_results.json", 'r') as fp:
        bayesian_data = json.load(fp)
    
    priors = bayesian_data['model']['priors']
    likelihoods = bayesian_data['model']['likelihoods']
    
    print(f"Model loaded: {len(priors)} symbols")
    
    # we load test data
    print("\nLoading test data...")
    X_test = np.load(script_dir / "X_test.npy", allow_pickle=True).tolist()
    n_test = len(X_test)
    print(f"Loaded {n_test} test signals")
    
    # we process all test signals
    all_results = []
    
    for i in range(n_test):
        print(f"\n{'='*60}")
        print(f"Processing TEST signal {i}/{n_test-1}...")
        print(f"{'='*60}")
        
        signal = X_test[i]
        
        try:
            result, log_lines = process_test_signal(signal, i, priors, likelihoods, data_dir, figures_dir)
            all_results.append(result)
            print(f"TEST Signal {i}: predicted = {result['predicted']}")
            
        except Exception as e:
            print(f"Error processing test signal {i}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'signal_idx': i,
                'predicted': '',
                'error': str(e),
            })
    
    # we save summary
    summary = {
        'n_test_signals': n_test,
        'results': all_results,
    }
    
    with open(data_dir / "test_summary.json", 'w') as fp:
        json.dump(summary, fp, indent=2)
    
    print(f"\n{'='*60}")
    print("TEST SET PROCESSING COMPLETE")
    print(f"{'='*60}")
    for i, result in enumerate(all_results):
        if 'error' not in result:
            print(f"TEST Signal {i}: {result['predicted']}")
        else:
            print(f"TEST Signal {i}: ERROR - {result.get('error', 'unknown')}")
    
    print(f"\nTest results saved to: {data_dir}")


if __name__ == "__main__":
    main()
