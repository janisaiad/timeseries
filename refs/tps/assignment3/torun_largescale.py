"""we run the complete DTMF detection pipeline for all training signals and save outputs"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # we use non-interactive backend for batch processing
import matplotlib.pyplot as plt
from scipy.signal import stft
from sklearn.cluster import KMeans
import ruptures as rpt

# we define constants
FS = 22050  # sampling frequency (Hz)
WINDOW_LENGTH = 900
OVERLAP = WINDOW_LENGTH // 2
FRAME_STEP = WINDOW_LENGTH - OVERLAP
DTMF_FREQ_RANGE = (400.0, 2000.0)
DTMF_LOW_FREQS = [697, 770, 852, 941]  # Hz
DTMF_HIGH_FREQS = [1209, 1336, 1477, 1633]  # Hz
DTMF_SYMBOLS = [
    ['1', '2', '3', 'A'],
    ['4', '5', '6', 'B'],
    ['7', '8', '9', 'C'],
    ['*', '0', '#', 'D'],
]


def elbow_kmeans_1d(freqs_1d, k_max=12, random_state=0, n_init=10):
    """we perform elbow method on 1d frequency data"""
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


def freq_to_dtmf_index(freq, freq_list, tol_hz=50.0):
    """we map frequency to DTMF index"""
    distances = np.abs(np.array(freq_list, dtype=float) - float(freq))
    min_idx = int(np.argmin(distances))
    min_dist = float(distances[min_idx])
    if min_dist <= float(tol_hz):
        return min_idx, min_dist
    return None, min_dist


def intervals_from_changepoints(changepoints, start_sample, end_sample):
    """we extract intervals from changepoint boundaries"""
    cps = [int(cp) for cp in changepoints if cp is not None]
    cps = [cp for cp in cps if int(start_sample) < cp < int(end_sample)]
    cps_sorted = sorted(set(cps))
    boundaries = [int(start_sample)] + cps_sorted + [int(end_sample)]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1) if boundaries[i] < boundaries[i + 1]]


def segment_energy_stats(energy_frames, start_sample, end_sample, frame_step):
    """we compute energy statistics for a segment"""
    start_frame = max(0, int(start_sample / frame_step))
    end_frame = min(len(energy_frames), int(end_sample / frame_step) + 1)
    if start_frame >= end_frame:
        return 0.0, 0.0, 0
    seg = energy_frames[start_frame:end_frame]
    if len(seg) == 0:
        return 0.0, 0.0, 0
    mean_e = float(np.mean(seg))
    n_frames = int(len(seg))
    total_e = float(mean_e * n_frames)
    return mean_e, total_e, n_frames


def build_all_intervals_for_group(cp_dicts, signal_len, frame_step, min_duration_samples=100):
    """we build all intervals for a frequency group"""
    intervals = []
    for cp in cp_dicts:
        segs = intervals_from_changepoints(cp['changepoints'], 0, signal_len)
        for s, e in segs:
            if (e - s) < int(min_duration_samples):
                continue
            mean_e, total_e, n_frames = segment_energy_stats(cp['energy_signal'], s, e, frame_step)
            intervals.append({
                'cluster_id': int(cp['cluster_id']),
                'freq': float(cp['selected_freq']),
                'start_time': int(s),
                'end_time': int(e),
                'duration_samples': int(e - s),
                'duration_seconds': float((e - s) / FS),
                'mean_energy': float(mean_e),
                'total_energy': float(total_e),
                'n_frames': int(n_frames),
            })
    return intervals


def select_topN_non_overlapping(intervals, N, score_key='mean_energy'):
    """we select top-N non-overlapping intervals by score"""
    ranked = sorted(intervals, key=lambda x: x[score_key], reverse=True)
    selected = []
    for itv in ranked:
        overlap = False
        for s in selected:
            if not (itv['end_time'] <= s['start_time'] or itv['start_time'] >= s['end_time']):
                overlap = True
                break
        if not overlap:
            selected.append(itv)
        if len(selected) >= int(max(1, N)):
            break
    selected_time = sorted(selected, key=lambda x: x['start_time'])
    return ranked, selected_time


def process_signal(signal, signal_idx, ground_truth, output_dir, figures_dir):
    """we process a single DTMF signal and save all outputs"""
    
    log_lines = []  # we collect log output
    log_lines.append(f"Processing signal {signal_idx}")
    log_lines.append(f"Signal length: {len(signal)} samples ({len(signal)/FS:.3f} seconds)")
    
    # we extract ground truth
    gt_seq = ground_truth if isinstance(ground_truth, str) else ''.join(list(ground_truth))
    N = int(len(gt_seq))
    log_lines.append(f"Ground truth: {gt_seq} (N={N} symbols)")
    
    # we create signal-specific figure directory
    sig_fig_dir = figures_dir / f"signal_{signal_idx}"
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
        log_lines.append("Warning: insufficient high-energy bins for clustering")
        # we save empty results
        results = {
            'signal_idx': signal_idx,
            'ground_truth': gt_seq,
            'predicted': '',
            'accuracy': 0.0,
            'error': 'insufficient_high_energy_bins',
        }
        with open(output_dir / f"results_{signal_idx}.json", 'w') as fp:
            json.dump(results, fp, indent=2)
        with open(output_dir / f"log_{signal_idx}.txt", 'w') as fp:
            fp.write('\n'.join(log_lines))
        return results
    
    # we perform hierarchical clustering: coarse k=2 split then elbow in each group
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
    
    log_lines.append(f"Coarse split: low group n={len(low_freqs)} mean={float(np.mean(low_freqs)) if len(low_freqs) else 0:.1f} Hz, high group n={len(high_freqs)} mean={float(np.mean(high_freqs)) if len(high_freqs) else 0:.1f} Hz")
    
    # we cluster within each group using elbow method
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
                'center_energy': float(np.mean(cenergies)) if len(cenergies) > 0 else 0.0,
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
                'center_energy': float(np.mean(cenergies)) if len(cenergies) > 0 else 0.0,
                'freq_indices': cidx,
                'freqs': cfreqs,
                'energies': cenergies,
            })
            cluster_id_counter += 1
    
    major_freq_clusters.sort(key=lambda c: c['center_freq'])
    log_lines.append(f"Frequency clusters: {len(major_freq_clusters)} total")
    for c in major_freq_clusters:
        log_lines.append(f"  Cluster {c['cluster_id']}: band={c['band']}, center={c['center_freq']:.1f} Hz, n_bins={len(c['freq_indices'])}")
    
    # we run changepoint detection on each cluster
    t_max = len(t)
    all_changepoints = []
    
    for cluster in major_freq_clusters:
        freq_indices = np.array(cluster['freq_indices'], dtype=int)
        if len(freq_indices) == 0:
            continue
        
        # we select frequency bin closest to cluster center
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
    log_lines.append(f"Changepoint detection: {len(all_bkps)} unique changepoints across all clusters")
    
    # we split clusters into low/high for interval pairing
    selected_freqs = np.array([float(cp['selected_freq']) for cp in all_changepoints], dtype=float)
    km2 = KMeans(n_clusters=2, random_state=0, n_init=10)
    coarse_labels = km2.fit_predict(selected_freqs.reshape(-1, 1))
    centers = km2.cluster_centers_.flatten()
    low_id = int(np.argmin(centers))
    high_id = int(np.argmax(centers))
    low_freq_clusters = [cp for cp, lab in zip(all_changepoints, coarse_labels) if int(lab) == low_id]
    high_freq_clusters = [cp for cp, lab in zip(all_changepoints, coarse_labels) if int(lab) == high_id]
    
    log_lines.append(f"Interval grouping: {len(low_freq_clusters)} low clusters, {len(high_freq_clusters)} high clusters")
    
    # we build all intervals per cluster in each group
    low_all = build_all_intervals_for_group(low_freq_clusters, len(signal), FRAME_STEP)
    high_all = build_all_intervals_for_group(high_freq_clusters, len(signal), FRAME_STEP)
    log_lines.append(f"Intervals built: {len(low_all)} low, {len(high_all)} high")
    
    # we rank by mean energy and keep top-N non-overlapping, then order by start time
    low_ranked, low_topN_time = select_topN_non_overlapping(low_all, N, score_key='mean_energy')
    high_ranked, high_topN_time = select_topN_non_overlapping(high_all, N, score_key='mean_energy')
    log_lines.append(f"Top-N selection: {len(low_topN_time)} low intervals, {len(high_topN_time)} high intervals (N={N})")
    
    # we pair by order index and map to DTMF symbols
    n_pairs = min(len(low_topN_time), len(high_topN_time), N)
    predicted = []
    for i in range(n_pairs):
        low_itv = low_topN_time[i]
        high_itv = high_topN_time[i]
        li, low_dist = freq_to_dtmf_index(low_itv['freq'], DTMF_LOW_FREQS)
        hi, high_dist = freq_to_dtmf_index(high_itv['freq'], DTMF_HIGH_FREQS)
        sym = '?'
        if (li is not None) and (hi is not None):
            sym = DTMF_SYMBOLS[li][hi]
        predicted.append(sym)
    
    pred_seq = ''.join(predicted)
    n_cmp = min(len(gt_seq), len(pred_seq))
    acc = sum(1 for a, b in zip(gt_seq[:n_cmp], pred_seq[:n_cmp]) if a == b) / n_cmp if n_cmp > 0 else 0.0
    
    log_lines.append(f"Prediction: {pred_seq}")
    log_lines.append(f"Ground truth: {gt_seq}")
    log_lines.append(f"Accuracy: {acc:.3f} ({int(acc*n_cmp)}/{n_cmp} correct)")
    
    # we save figures
    # figure 1: STFT overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].plot(signal)
    axes[0, 0].set_title("Original Signal")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(True, alpha=0.3)
    
    spectrogram = np.abs(Zxx)
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
    
    # figure 2: signal with changepoints
    plt.figure(figsize=(16, 5))
    plt.plot(np.arange(len(signal)) / FS, signal, 'b-', linewidth=0.8, alpha=0.7)
    for bp in all_bkps:
        plt.axvline(x=bp/FS, color='red', linestyle='--', linewidth=1.2, alpha=0.6)
    plt.title(f'Signal {signal_idx} with Detected Changepoints')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.savefig(sig_fig_dir / "02_signal_changepoints.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # figure 3: spectrogram with changepoints
    plt.figure(figsize=(14, 6))
    im = plt.pcolormesh(t, f, 20 * np.log10(spectrogram + 1e-10), shading='gouraud', cmap='viridis')
    for cp_dict in all_changepoints:
        for frame_bp in cp_dict['frame_bkps']:
            if frame_bp < len(t):
                plt.axvline(x=t[frame_bp], color='red', linestyle='--', linewidth=1.2, alpha=0.6)
    plt.title(f'Signal {signal_idx} Spectrogram with Changepoints')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim([0, 4000])
    plt.colorbar(im, label='Magnitude (dB)')
    plt.savefig(sig_fig_dir / "03_spectrogram_changepoints.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # figure 4: energy time series with changepoints for each cluster
    n_clusters = len(all_changepoints)
    if n_clusters > 0:
        n_rows = (n_clusters + 1) // 2
        colors_8 = plt.cm.tab10(np.linspace(0, 1, 8))
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
            ax.set_title(f"Cluster {cp_dict['cluster_id']}: center {cp_dict['center_freq']:.0f} Hz", fontweight='bold')
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
    plt.title(f'Signal {signal_idx}: Detected Symbols (pred={pred_seq}, gt={gt_seq})')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.savefig(sig_fig_dir / "05_detected_symbols.png", dpi=100, bbox_inches='tight')
    plt.close()
    
    # we prepare results
    results = {
        'signal_idx': signal_idx,
        'ground_truth': gt_seq,
        'predicted': pred_seq,
        'accuracy': float(acc),
        'n_correct': int(acc * n_cmp) if n_cmp > 0 else 0,
        'n_total': n_cmp,
        'n_clusters': len(major_freq_clusters),
        'n_changepoints': len(all_bkps),
        'low_intervals': len(low_topN_time),
        'high_intervals': len(high_topN_time),
        'pairs': [
            {
                'idx': i,
                'symbol': predicted[i],
                'low_freq': float(low_topN_time[i]['freq']),
                'high_freq': float(high_topN_time[i]['freq']),
                'low_interval': [int(low_topN_time[i]['start_time']), int(low_topN_time[i]['end_time'])],
                'high_interval': [int(high_topN_time[i]['start_time']), int(high_topN_time[i]['end_time'])],
            } for i in range(n_pairs)
        ],
    }
    
    # we save results
    with open(output_dir / f"results_{signal_idx}.json", 'w') as fp:
        json.dump(results, fp, indent=2)
    
    # we save log
    with open(output_dir / f"log_{signal_idx}.txt", 'w') as fp:
        fp.write('\n'.join(log_lines))
    
    return results


def main():
    """we process all training signals"""
    
    # we set up directories
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    figures_dir = data_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # we load data
    print("Loading training data...")
    X_train = np.load(script_dir / "X_train.npy", allow_pickle=True).tolist()
    y_train = np.load(script_dir / "y_train.npy", allow_pickle=True).tolist()
    n_signals = len(X_train)
    print(f"Loaded {n_signals} training signals")
    
    # we process all signals
    all_results = []
    freq_dataset = []
    
    for i in range(n_signals):
        print(f"\n{'='*60}")
        print(f"Processing signal {i}/{n_signals-1}...")
        print(f"{'='*60}")
        
        signal = X_train[i]
        ground_truth = y_train[i]
        
        try:
            result = process_signal(signal, i, ground_truth, data_dir, figures_dir)
            all_results.append(result)
            
            # we build frequency dataset entry
            gt_seq = ground_truth if isinstance(ground_truth, str) else ''.join(list(ground_truth))
            freq_dataset.append({
                'signal_idx': i,
                'ground_truth': gt_seq,
                'predicted': result['predicted'],
                'success': result['accuracy'] == 1.0,
                'accuracy': result['accuracy'],
            })
            
            print(f"Signal {i}: predicted={result['predicted']}, gt={gt_seq}, acc={result['accuracy']:.3f}")
            
        except Exception as e:
            print(f"Error processing signal {i}: {e}")
            all_results.append({
                'signal_idx': i,
                'error': str(e),
                'ground_truth': str(ground_truth),
                'predicted': '',
                'accuracy': 0.0,
            })
            freq_dataset.append({
                'signal_idx': i,
                'ground_truth': str(ground_truth),
                'predicted': '',
                'success': False,
                'accuracy': 0.0,
                'error': str(e),
            })
    
    # we compute summary statistics
    successful = [r for r in all_results if 'error' not in r and r['accuracy'] == 1.0]
    total_acc = np.mean([r['accuracy'] for r in all_results if 'error' not in r])
    
    summary = {
        'n_signals': n_signals,
        'n_successful': len(successful),
        'success_rate': len(successful) / n_signals if n_signals > 0 else 0.0,
        'mean_accuracy': float(total_acc),
        'results': all_results,
    }
    
    # we save summary
    with open(data_dir / "summary.json", 'w') as fp:
        json.dump(summary, fp, indent=2)
    
    # we save frequency dataset
    with open(data_dir / "freq_dataset.json", 'w') as fp:
        json.dump(freq_dataset, fp, indent=2)
    
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total signals: {n_signals}")
    print(f"Successful (100% accuracy): {len(successful)}")
    print(f"Success rate: {len(successful)/n_signals*100:.1f}%")
    print(f"Mean accuracy: {total_acc:.3f}")
    print(f"\nOutputs saved to: {data_dir}")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()
