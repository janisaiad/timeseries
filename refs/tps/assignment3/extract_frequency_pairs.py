"""we extract frequency pairs and symbols from detection results to build empirical DTMF mapping"""

import json
from pathlib import Path
import numpy as np


def main():
    """we extract all detected frequency pairs and their symbols"""
    
    # we locate the data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    # we collect all frequency pairs
    all_pairs = []
    
    print("Extracting frequency pairs from results...")
    
    # we iterate through all result files
    for i in range(100):  # we assume 100 signals
        results_file = data_dir / f"results_{i}.json"
        if not results_file.exists():
            continue
        
        # we load results
        with open(results_file, 'r') as fp:
            result = json.load(fp)
        
        # we skip errors
        if 'error' in result:
            continue
        
        # we extract pairs
        if 'pairs' in result:
            for pair in result['pairs']:
                symbol = pair['symbol']
                low_freq = pair['low_freq']
                high_freq = pair['high_freq']
                
                # we record the pair
                all_pairs.append({
                    'signal_idx': i,
                    'symbol': symbol,
                    'low_freq': float(low_freq),
                    'high_freq': float(high_freq),
                    'ground_truth': result.get('ground_truth', ''),
                    'predicted': result.get('predicted', ''),
                })
    
    print(f"Extracted {len(all_pairs)} frequency pairs from {len(set([p['signal_idx'] for p in all_pairs]))} signals")
    
    # we group by symbol
    symbol_freqs = {}
    for pair in all_pairs:
        sym = pair['symbol']
        if sym == '?':  # we skip unmatched symbols
            continue
        if sym not in symbol_freqs:
            symbol_freqs[sym] = {'low': [], 'high': []}
        symbol_freqs[sym]['low'].append(pair['low_freq'])
        symbol_freqs[sym]['high'].append(pair['high_freq'])
    
    # we compute statistics per symbol
    symbol_stats = {}
    for sym in sorted(symbol_freqs.keys()):
        low_freqs = np.array(symbol_freqs[sym]['low'])
        high_freqs = np.array(symbol_freqs[sym]['high'])
        
        symbol_stats[sym] = {
            'count': len(low_freqs),
            'low_mean': float(np.mean(low_freqs)),
            'low_std': float(np.std(low_freqs)),
            'low_min': float(np.min(low_freqs)),
            'low_max': float(np.max(low_freqs)),
            'high_mean': float(np.mean(high_freqs)),
            'high_std': float(np.std(high_freqs)),
            'high_min': float(np.min(high_freqs)),
            'high_max': float(np.max(high_freqs)),
            'all_low_freqs': low_freqs.tolist(),
            'all_high_freqs': high_freqs.tolist(),
        }
    
    # we create output text file
    output_file = data_dir / "empirical_frequency_mapping.txt"
    
    with open(output_file, 'w') as fp:
        fp.write("=" * 80 + "\n")
        fp.write("EMPIRICAL DTMF FREQUENCY MAPPING\n")
        fp.write("Frequency pairs learned from clustering (no prior DTMF knowledge)\n")
        fp.write("=" * 80 + "\n\n")
        
        fp.write(f"Total detections: {len(all_pairs)}\n")
        fp.write(f"Valid symbol pairs: {sum(len(v['low']) for v in symbol_freqs.values())}\n")
        fp.write(f"Unique symbols found: {len(symbol_stats)}\n\n")
        
        fp.write("=" * 80 + "\n")
        fp.write("FREQUENCY STATISTICS PER SYMBOL\n")
        fp.write("=" * 80 + "\n\n")
        
        for sym in sorted(symbol_stats.keys()):
            stats = symbol_stats[sym]
            fp.write(f"Symbol: '{sym}'\n")
            fp.write(f"  Count: {stats['count']} detections\n")
            fp.write(f"  Low frequency:  {stats['low_mean']:7.1f} Hz  (±{stats['low_std']:5.1f})  [{stats['low_min']:.1f} - {stats['low_max']:.1f}]\n")
            fp.write(f"  High frequency: {stats['high_mean']:7.1f} Hz  (±{stats['high_std']:5.1f})  [{stats['high_min']:.1f} - {stats['high_max']:.1f}]\n")
            fp.write(f"  Pair: ({stats['low_mean']:.1f} Hz, {stats['high_mean']:.1f} Hz)\n")
            fp.write("\n")
        
        fp.write("\n" + "=" * 80 + "\n")
        fp.write("ALL DETECTED FREQUENCY PAIRS (RAW DATA)\n")
        fp.write("=" * 80 + "\n\n")
        fp.write(f"{'Signal':<8} {'Symbol':<8} {'Low Freq (Hz)':<15} {'High Freq (Hz)':<15} {'Ground Truth':<15} {'Predicted':<15}\n")
        fp.write("-" * 80 + "\n")
        
        for pair in all_pairs:
            if pair['symbol'] == '?':
                continue
            fp.write(f"{pair['signal_idx']:<8} {pair['symbol']:<8} {pair['low_freq']:<15.2f} {pair['high_freq']:<15.2f} {pair['ground_truth']:<15} {pair['predicted']:<15}\n")
        
        fp.write("\n" + "=" * 80 + "\n")
        fp.write("SUMMARY: LEARNED FREQUENCY PAIRS\n")
        fp.write("=" * 80 + "\n\n")
        fp.write("Symbol -> (Low Freq, High Freq) pairs learned from data:\n\n")
        
        for sym in sorted(symbol_stats.keys()):
            stats = symbol_stats[sym]
            fp.write(f"  '{sym}' -> ({stats['low_mean']:6.1f} Hz, {stats['high_mean']:6.1f} Hz)  [n={stats['count']}]\n")
        
        fp.write("\n" + "=" * 80 + "\n")
        fp.write("CLUSTERING-BASED FREQUENCY MAPPING (no DTMF prior)\n")
        fp.write("=" * 80 + "\n\n")
        
        # we group by rounded frequencies to show the empirical frequency table
        freq_groups = {}
        for sym in sorted(symbol_stats.keys()):
            stats = symbol_stats[sym]
            low_round = round(stats['low_mean'] / 10) * 10  # we round to nearest 10 Hz
            high_round = round(stats['high_mean'] / 10) * 10
            key = (low_round, high_round)
            if key not in freq_groups:
                freq_groups[key] = []
            freq_groups[key].append((sym, stats['count']))
        
        fp.write("Empirical frequency grid (rounded to 10 Hz):\n\n")
        
        # we extract unique low and high frequencies
        low_freqs_unique = sorted(set([k[0] for k in freq_groups.keys()]))
        high_freqs_unique = sorted(set([k[1] for k in freq_groups.keys()]))
        
        fp.write(f"Low frequencies:  {', '.join([f'{f} Hz' for f in low_freqs_unique])}\n")
        fp.write(f"High frequencies: {', '.join([f'{f} Hz' for f in high_freqs_unique])}\n\n")
        
        fp.write("Grid (Low x High):\n\n")
        header = "Low\\High"
        fp.write(f"{header:<12}")
        for hf in high_freqs_unique:
            fp.write(f"{hf:>10} Hz")
        fp.write("\n" + "-" * (12 + 14 * len(high_freqs_unique)) + "\n")
        
        for lf in low_freqs_unique:
            fp.write(f"{lf:>6} Hz    ")
            for hf in high_freqs_unique:
                key = (lf, hf)
                if key in freq_groups:
                    syms = ', '.join([s[0] for s in freq_groups[key]])
                    fp.write(f"{syms:>12}  ")
                else:
                    fp.write(f"{'--':>12}  ")
            fp.write("\n")
    
    # we save JSON version
    json_output = data_dir / "empirical_frequency_mapping.json"
    with open(json_output, 'w') as fp:
        json.dump({
            'symbol_statistics': symbol_stats,
            'all_pairs': all_pairs,
            'summary': {
                'total_detections': len(all_pairs),
                'valid_pairs': sum(len(v['low']) for v in symbol_freqs.values()),
                'unique_symbols': len(symbol_stats),
            }
        }, fp, indent=2)
    
    print(f"\nOutputs saved:")
    print(f"  Text file: {output_file}")
    print(f"  JSON file: {json_output}")
    print(f"\nSummary:")
    print(f"  Unique symbols: {len(symbol_stats)}")
    print(f"  Total valid pairs: {sum(len(v['low']) for v in symbol_freqs.values())}")


if __name__ == "__main__":
    main()
