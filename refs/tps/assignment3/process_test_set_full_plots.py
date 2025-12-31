"""we process the test set with complete plotting (all figures like training set)"""

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


def elbow_kmeans_1d(freqs_1d, k_max=12, random_state=0, n_init=10):
    """we perform elbow method on 1d frequencies"""
    freqs_1d = np.asarray(freqs_1d, dtype=float).reshape(-1, 1)
    n = int(freqs_1d.shape[0])
    if n < 2:
        return 0, None, [], []
    k_max_eff = int(min(max(1, k_max), n))
    k_candidates = list(range(1, k_max_eff + 1))
    inertias = []
    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=int(random_state), n_init=int(n_init))
        km.fit(freqs_1d)
        inertias.append(float(km.inertia_))
    if len(inertias) == 1:
        k_opt = 1
    else:
        ks = np.array(k_candidates, dtype=float)
        ys = np.array(inertias, dtype=float)
        ks_n = (ks - ks.min()) / (ks.max() - ks.min() + 1e-12)
        ys_n = (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
        p1 = np.array([ks_n[0], ys_n[0]])
        p2 = np.array([ks_n[-1], ys_n[-1]])
        v = p2 - p1
        v_norm = np.linalg.norm(v) + 1e-12
        distances = []
        for xi, yi in zip(ks_n, ys_n):
            p = np.array([xi, yi])
            dist = abs(np.cross(v, p - p1)) / v_norm
            distances.append(float(dist))
        k_opt = int(k_candidates[int(np.argmax(distances))])
    km_final = KMeans(n_clusters=int(k_opt), random_state=int(random_state), n_init=int(n_init))
    labels = km_final.fit_predict(freqs_1d)
    return int(k_opt), labels, k_candidates, inertias


def build_all_intervals_for_group(freq_clusters, signal_length, frame_step):
    """we build all intervals for a group of frequency clusters"""
    all_intervals = []
    for cp_dict in freq_clusters:
        bkps = cp_dict['frame_bkps']
        energy_signal = cp_dict['energy_signal']
        freq_val = float(cp_dict['selected_freq'])
        
        intervals = []
        if len(bkps) == 0:
            intervals.append((0, len(energy_signal) - 1))
        else:
            start = 0
            for bp in bkps:
                intervals.append((start, bp))
                start = bp + 1
            intervals.append((start, len(energy_signal) - 1))
        
        for (start_frame, end_frame) in intervals:
            if start_frame >= end_frame:
                continue
            
            segment_energy = energy_signal[start_frame:end_frame + 1]
            mean_energy = float(np.mean(segment_energy))
            total_energy = float(np.sum(segment_energy))
            n_frames = int(len(segment_energy))
            duration = float(n_frames * frame_step) / FS
            
            start_sample = int(start_frame * frame_step)
            end_sample = int(min(end_frame * frame_step, signal_length - 1))
            
            all_intervals.append({
                'freq': freq_val,
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'start_time': int(start_sample),
                'end_time': int(end_sample),
                'mean_energy': mean_energy,
                'total_energy': total_energy,
                'n_frames': n_frames,
                'duration': duration,
            })
    
    return all_intervals


def select_topN_non_overlapping(intervals, N, score_key='mean_energy'):
    """we select top-N non-overlapping intervals"""
    sorted_intervals = sorted(intervals, key=lambda x: x[score_key], reverse=True)
    selected = []
    
    for itv in sorted_intervals:
        overlaps = False
        for sel in selected:
            if not (itv['end_time'] <= sel['start_time'] or itv['start_time'] >= sel['end_time']):
                overlaps = True
                break
        if not overlaps:
            selected.append(itv)
        if len(selected) >= N:
            break
    
    selected_ordered = sorted(selected, key=lambda x: x['start_time'])
    return selected, selected_ordered


def process_test_signal_full_plots(signal, signal_idx, priors, likelihoods, output_dir, figures_dir):
    """we process a test signal with complete plotting"""
    
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
    noise_cluster_idx = 1 - signal_cluster_idx
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
    
    # we perform hierarchical clustering (coarse split)
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
    
    # we run elbow method on each group
    k_low, low_labels, low_k_cand, low_inertias = elbow_kmeans_1d(low_freqs, k_max=8)
    k_high, high_labels, high_k_cand, high_inertias = elbow_kmeans_1d(high_freqs, k_max=8)
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
            'band': cluster['band'],
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
    low_freq_clusters = [cp for cp in all_changepoints if cp['band'] == 'low']
    high_freq_clusters = [cp for cp in all_changepoints if cp['band'] == 'high']
    
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
    
    # =============================================================================
    # PLOTTING SECTION - ALL FIGURES LIKE TRAINING SET
    # =============================================================================
    
    spectrogram = np.abs(Zxx)
    colors_8 = plt.cm.tab10(np.linspace(0, 1, 8))
    
    # figure 1: STFT overview (signal, spectrogram, energy distributions)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].plot(signal, linewidth=0.8)
    axes[0, 0].set_title(f"TEST Signal {signal_idx}", fontweight='bold')
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)
    
    im1 = axes[0, 1].pcolormesh(t, f, 20 * np.log10(spectrogram + 1e-10), shading='gouraud', cmap='viridis')
    axes[0, 1].set_title("STFT Spectrogram", fontweight='bold')
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Frequency (Hz)")
    axes[0, 1].set_ylim([0, 4000])
    plt.colorbar(im1, ax=axes[0, 1], label="Magnitude (dB)")
    
    axes[1, 0].plot(f, energy_per_freq, 'b-', linewidth=1.5)
    axes[1, 0].set_title("Energy per Frequency Bin", fontweight='bold')
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Energy")
    axes[1, 0].set_xlim([0, 4000])
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogy(f, energy_per_freq, 'b-', linewidth=1.5)
    axes[1, 1].set_title("Energy (Log Scale)", fontweight='bold')
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Energy (log)")
    axes[1, 1].set_xlim([0, 4000])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(sig_fig_dir / "01_stft_overview.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # figure 2: coarse split (low/high separation)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].scatter(f, energy_per_freq, c=['red' if lb == signal_cluster_idx else 'gray' for lb in energy_labels],
                       alpha=0.6, s=30)
    axes[0, 0].axhline(y=cluster_centers[signal_cluster_idx], color='red', linestyle='--', linewidth=2,
                       label=f'Signal ({cluster_centers[signal_cluster_idx]:.2e})')
    axes[0, 0].axhline(y=cluster_centers[noise_cluster_idx], color='gray', linestyle='--', linewidth=2,
                       label=f'Noise ({cluster_centers[noise_cluster_idx]:.2e})')
    axes[0, 0].set_title("K-means (k=2): Signal vs Noise", fontweight='bold')
    axes[0, 0].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("Energy")
    axes[0, 0].set_xlim([0, 4000])
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(low_freqs, np.zeros_like(low_freqs), s=80, alpha=0.7, label=f'Low (n={len(low_freqs)})', color='blue')
    axes[0, 1].scatter(high_freqs, np.zeros_like(high_freqs) + 0.05, s=80, alpha=0.7, label=f'High (n={len(high_freqs)})', color='orange')
    axes[0, 1].set_title("Coarse Low/High Split", fontweight='bold')
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_yticks([])
    axes[0, 1].set_xlim([DTMF_FREQ_RANGE[0]-100, DTMF_FREQ_RANGE[1]+100])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # elbow plots
    if len(low_inertias) > 0:
        axes[1, 0].plot(low_k_cand, low_inertias, marker='o', linewidth=2, markersize=8)
        axes[1, 0].axvline(k_low, color='red', linestyle='--', linewidth=2, label=f'k_low={k_low}')
        axes[1, 0].set_title("Elbow Method (Low Group)", fontweight='bold')
        axes[1, 0].set_xlabel("k")
        axes[1, 0].set_ylabel("Inertia")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis('off')
    
    if len(high_inertias) > 0:
        axes[1, 1].plot(high_k_cand, high_inertias, marker='o', linewidth=2, markersize=8)
        axes[1, 1].axvline(k_high, color='red', linestyle='--', linewidth=2, label=f'k_high={k_high}')
        axes[1, 1].set_title("Elbow Method (High Group)", fontweight='bold')
        axes[1, 1].set_xlabel("k")
        axes[1, 1].set_ylabel("Inertia")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(sig_fig_dir / "02_coarse_split_elbow.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # figure 3: fine clustering (refined clusters in low and high groups)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if (low_labels is not None) and (k_low > 0):
        colors_low = plt.cm.tab10(np.linspace(0, 1, int(k_low)))
        for cid in range(int(k_low)):
            mask = low_labels == cid
            axes[0].scatter(low_freqs[mask], low_energies[mask], s=100, alpha=0.7,
                            color=colors_low[cid], edgecolors='black', linewidths=1.2,
                            label=f'c{cid}: {float(np.mean(low_freqs[mask])):.0f}Hz')
        axes[0].set_title(f"Fine Clustering: Low Group (k={k_low})", fontweight='bold')
        axes[0].set_xlabel("Frequency (Hz)")
        axes[0].set_ylabel("Energy")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].axis('off')
    
    if (high_labels is not None) and (k_high > 0):
        colors_high = plt.cm.tab10(np.linspace(0, 1, int(k_high)))
        for cid in range(int(k_high)):
            mask = high_labels == cid
            axes[1].scatter(high_freqs[mask], high_energies[mask], s=100, alpha=0.7,
                            color=colors_high[cid], edgecolors='black', linewidths=1.2,
                            label=f'c{cid}: {float(np.mean(high_freqs[mask])):.0f}Hz')
        axes[1].set_title(f"Fine Clustering: High Group (k={k_high})", fontweight='bold')
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].set_ylabel("Energy")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(sig_fig_dir / "03_fine_clustering.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # figure 4: energy time series with changepoints for each cluster
    n_clusters = len(all_changepoints)
    if n_clusters > 0:
        n_rows = (n_clusters + 1) // 2
        figE, axesE = plt.subplots(n_rows, 2, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axesE = axesE.reshape(1, -1)
        axesE = axesE.flatten()
        
        for idx, cp_dict in enumerate(all_changepoints):
            ax = axesE[idx]
            cid = int(cp_dict['cluster_id'])
            ax.plot(t, cp_dict['energy_signal'], color=colors_8[cid % 8], linewidth=2.0,
                    label=f"f={cp_dict['selected_freq']:.0f} Hz ({cp_dict['n_cps']} CPs)")
            for frame_bp in cp_dict['frame_bkps']:
                if frame_bp < len(t):
                    ax.axvline(x=t[frame_bp], color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            ax.set_title(f"Cluster {cp_dict['cluster_id']} ({cp_dict['band']}): {cp_dict['center_freq']:.0f} Hz", 
                        fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Energy')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        for idx in range(n_clusters, len(axesE)):
            axesE[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(sig_fig_dir / "04_energy_changepoints_per_cluster.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    # figure 5: detected symbols overlay
    plt.figure(figsize=(18, 6))
    plt.plot(np.arange(len(signal)) / FS, signal, 'b-', linewidth=0.8, alpha=0.7, label='Signal')
    
    y_max = float(np.max(np.abs(signal))) if len(signal) > 0 else 1.0
    y_pos = y_max * 0.8
    
    for i in range(n_pairs):
        s = min(low_topN_time[i]['start_time'], high_topN_time[i]['start_time'])
        e = max(low_topN_time[i]['end_time'], high_topN_time[i]['end_time'])
        plt.axvspan(s/FS, e/FS, alpha=0.15, color='yellow')
        mid_t = (s + e) / (2 * FS)
        plt.text(mid_t, y_pos, predicted[i], ha='center', va='bottom', fontsize=14, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='black', linewidth=2))
    
    plt.title(f'TEST Signal {signal_idx}: Predicted = {pred_seq}', fontsize=16, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(sig_fig_dir / "05_detected_symbols.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # figure 6: spectrogram with detected symbols
    plt.figure(figsize=(18, 8))
    im = plt.pcolormesh(t, f, 20 * np.log10(spectrogram + 1e-10), shading='gouraud', cmap='viridis')
    
    # we highlight detected frequency clusters
    for cluster in major_freq_clusters:
        cid = int(cluster['cluster_id'])
        for idx in cluster['freq_indices']:
            plt.axhline(y=f[idx], color=colors_8[cid % 8], linewidth=2.0, alpha=0.7)
    
    # we mark changepoints
    for cp_dict in all_changepoints:
        for frame_bp in cp_dict['frame_bkps']:
            if frame_bp < len(t):
                plt.axvline(x=t[frame_bp], color='red', linestyle='--', linewidth=1.2, alpha=0.6)
    
    plt.title(f'Spectrogram with Detected Clusters & Changepoints (TEST {signal_idx})', fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.ylim([0, 4000])
    plt.colorbar(im, label='Magnitude (dB)')
    plt.tight_layout()
    plt.savefig(sig_fig_dir / "06_spectrogram_clusters.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # we prepare results
    results = {
        'signal_idx': signal_idx,
        'predicted': pred_seq,
        'n_symbols': len(predicted),
        'frequency_pairs': freq_pairs,
        'n_clusters': len(major_freq_clusters),
        'n_changepoints': len(all_bkps),
    }
    
    with open(output_dir / f"test_results_{signal_idx}.json", 'w') as fp:
        json.dump(results, fp, indent=2)
    
    with open(output_dir / f"test_log_{signal_idx}.txt", 'w') as fp:
        fp.write('\n'.join(log_lines))
    
    return results, log_lines


def main():
    """we process test set with complete plotting"""
    
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
    
    # we process all test signals with full plotting
    all_results = []
    
    for i in range(n_test):
        print(f"\n{'='*70}")
        print(f"Processing TEST signal {i}/{n_test-1} with FULL PLOTTING...")
        print(f"{'='*70}")
        
        signal = X_test[i]
        
        try:
            result, log_lines = process_test_signal_full_plots(signal, i, priors, likelihoods, data_dir, figures_dir)
            all_results.append(result)
            print(f"✅ TEST Signal {i}: predicted = '{result['predicted']}' ({result['n_symbols']} symbols)")
            print(f"   Clusters: {result['n_clusters']}, Changepoints: {result['n_changepoints']}")
            
        except Exception as e:
            print(f"❌ Error processing test signal {i}: {e}")
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
    
    with open(data_dir / "test_summary_full.json", 'w') as fp:
        json.dump(summary, fp, indent=2)
    
    print(f"\n{'='*70}")
    print("TEST SET PROCESSING COMPLETE (WITH FULL PLOTS)")
    print(f"{'='*70}")
    for i, result in enumerate(all_results):
        if 'error' not in result:
            print(f"TEST {i}: '{result['predicted']}' ({result['n_symbols']} symbols)")
        else:
            print(f"TEST {i}: ERROR - {result.get('error', 'unknown')}")
    
    print(f"\n✅ All test results and figures saved to: {data_dir}/figures/test_signal_*/")
    print(f"   Generated 6 figures per test signal:")
    print(f"   - 01_stft_overview.png")
    print(f"   - 02_coarse_split_elbow.png")
    print(f"   - 03_fine_clustering.png")
    print(f"   - 04_energy_changepoints_per_cluster.png")
    print(f"   - 05_detected_symbols.png")
    print(f"   - 06_spectrogram_clusters.png")


if __name__ == "__main__":
    main()
