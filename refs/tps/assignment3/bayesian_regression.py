"""we perform bayesian inference to map frequency pairs to symbols"""

import json
from pathlib import Path
import numpy as np
from collections import defaultdict


def gaussian_pdf(x, mu, sigma):
    """we compute gaussian probability density"""
    if sigma == 0:
        sigma = 1e-6  # we avoid division by zero
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def build_bayesian_model(symbol_stats):
    """we build bayesian model from symbol statistics"""
    
    # we compute prior probabilities P(symbol)
    total_count = sum(stats['count'] for stats in symbol_stats.values())
    priors = {}
    for sym, stats in symbol_stats.items():
        priors[sym] = stats['count'] / total_count
    
    # we store likelihood parameters (gaussian for each frequency)
    likelihoods = {}
    for sym, stats in symbol_stats.items():
        likelihoods[sym] = {
            'low_mean': stats['low_mean'],
            'low_std': max(stats['low_std'], 1.0),  # we set minimum std to avoid overfitting
            'high_mean': stats['high_mean'],
            'high_std': max(stats['high_std'], 1.0),
        }
    
    return priors, likelihoods


def predict_symbol_bayesian(low_freq, high_freq, priors, likelihoods):
    """we predict symbol using bayesian inference"""
    
    # we compute posterior P(symbol | low_freq, high_freq) for each symbol
    posteriors = {}
    
    for sym in priors.keys():
        # we compute prior
        prior = priors[sym]
        
        # we compute likelihoods (assuming independence between low and high frequencies)
        likelihood_low = gaussian_pdf(low_freq, 
                                     likelihoods[sym]['low_mean'], 
                                     likelihoods[sym]['low_std'])
        likelihood_high = gaussian_pdf(high_freq, 
                                       likelihoods[sym]['high_mean'], 
                                       likelihoods[sym]['high_std'])
        
        # we compute joint likelihood
        likelihood = likelihood_low * likelihood_high
        
        # we compute unnormalized posterior
        posteriors[sym] = prior * likelihood
    
    # we normalize posteriors
    total_posterior = sum(posteriors.values())
    if total_posterior > 0:
        posteriors = {sym: prob / total_posterior for sym, prob in posteriors.items()}
    else:
        # we fall back to uniform if all posteriors are zero
        posteriors = {sym: 1.0 / len(posteriors) for sym in posteriors.keys()}
    
    # we return symbol with highest posterior probability
    best_symbol = max(posteriors.items(), key=lambda x: x[1])
    return best_symbol[0], best_symbol[1], posteriors


def evaluate_bayesian_regression(all_pairs, priors, likelihoods):
    """we evaluate bayesian regression on all detected pairs"""
    
    results = []
    correct = 0
    total = 0
    
    for pair in all_pairs:
        if pair['symbol'] == '?':
            continue
        
        low_freq = pair['low_freq']
        high_freq = pair['high_freq']
        true_symbol = pair['symbol']
        
        # we predict using bayesian inference
        pred_symbol, confidence, posteriors = predict_symbol_bayesian(
            low_freq, high_freq, priors, likelihoods
        )
        
        # we record result
        is_correct = (pred_symbol == true_symbol)
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'signal_idx': pair['signal_idx'],
            'low_freq': float(low_freq),
            'high_freq': float(high_freq),
            'true_symbol': true_symbol,
            'predicted_symbol': pred_symbol,
            'confidence': float(confidence),
            'correct': is_correct,
            'posteriors': {k: float(v) for k, v in posteriors.items()},
        })
    
    accuracy = correct / total if total > 0 else 0.0
    
    return results, accuracy, correct, total


def analyze_confusion_matrix(results):
    """we build confusion matrix from results"""
    
    # we collect all symbols
    symbols = sorted(set([r['true_symbol'] for r in results] + 
                        [r['predicted_symbol'] for r in results]))
    
    # we build confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for result in results:
        confusion[result['true_symbol']][result['predicted_symbol']] += 1
    
    return confusion, symbols


def main():
    """we perform bayesian regression on frequency pairs"""
    
    # we load empirical frequency mapping
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    input_file = data_dir / "empirical_frequency_mapping.json"
    
    print("Loading empirical frequency mapping...")
    with open(input_file, 'r') as fp:
        data = json.load(fp)
    
    symbol_stats = data['symbol_statistics']
    all_pairs = data['all_pairs']
    
    print(f"Loaded {len(all_pairs)} frequency pairs")
    print(f"Symbol statistics for {len(symbol_stats)} symbols")
    
    # we build bayesian model
    print("\nBuilding Bayesian model...")
    priors, likelihoods = build_bayesian_model(symbol_stats)
    
    print("\nPrior probabilities P(symbol):")
    for sym in sorted(priors.keys()):
        print(f"  P('{sym}') = {priors[sym]:.4f}")
    
    print("\nLikelihood parameters (Gaussian):")
    for sym in sorted(likelihoods.keys()):
        params = likelihoods[sym]
        print(f"  '{sym}': low ~ N({params['low_mean']:.1f}, {params['low_std']:.1f}²), "
              f"high ~ N({params['high_mean']:.1f}, {params['high_std']:.1f}²)")
    
    # we perform bayesian regression
    print("\nPerforming Bayesian inference on all pairs...")
    results, accuracy, correct, total = evaluate_bayesian_regression(
        all_pairs, priors, likelihoods
    )
    
    print(f"\nBayesian Regression Results:")
    print(f"  Total predictions: {total}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # we analyze confusion matrix
    confusion, symbols = analyze_confusion_matrix(results)
    
    # we compute per-symbol accuracy
    per_symbol_accuracy = {}
    for sym in symbols:
        total_sym = sum(confusion[sym].values())
        correct_sym = confusion[sym][sym]
        per_symbol_accuracy[sym] = correct_sym / total_sym if total_sym > 0 else 0.0
    
    # we save results to text file
    output_txt = data_dir / "bayesian_regression_results.txt"
    
    with open(output_txt, 'w') as fp:
        fp.write("=" * 80 + "\n")
        fp.write("BAYESIAN REGRESSION FOR DTMF SYMBOL PREDICTION\n")
        fp.write("Frequency pairs → Symbols using Bayesian inference\n")
        fp.write("=" * 80 + "\n\n")
        
        fp.write(f"Total frequency pairs: {total}\n")
        fp.write(f"Correct predictions: {correct}\n")
        fp.write(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        
        fp.write("=" * 80 + "\n")
        fp.write("BAYESIAN MODEL PARAMETERS\n")
        fp.write("=" * 80 + "\n\n")
        
        fp.write("Prior probabilities P(symbol):\n\n")
        for sym in sorted(priors.keys()):
            fp.write(f"  P('{sym}') = {priors[sym]:.4f}\n")
        
        fp.write("\n" + "-" * 80 + "\n\n")
        fp.write("Likelihood parameters (Gaussian distributions):\n\n")
        fp.write(f"{'Symbol':<10} {'Low Freq Mean':<15} {'Low Freq Std':<15} {'High Freq Mean':<15} {'High Freq Std':<15}\n")
        fp.write("-" * 80 + "\n")
        for sym in sorted(likelihoods.keys()):
            params = likelihoods[sym]
            fp.write(f"'{sym}'        {params['low_mean']:<15.2f} {params['low_std']:<15.2f} "
                    f"{params['high_mean']:<15.2f} {params['high_std']:<15.2f}\n")
        
        fp.write("\n" + "=" * 80 + "\n")
        fp.write("CONFUSION MATRIX\n")
        fp.write("=" * 80 + "\n\n")
        
        header = "True\\Pred"
        fp.write(f"{header:<10}")
        for sym in symbols:
            fp.write(f"{sym:>5}")
        fp.write(f"{'Total':>8}{'Accuracy':>12}\n")
        fp.write("-" * (10 + 5 * len(symbols) + 20) + "\n")
        
        for true_sym in symbols:
            fp.write(f"'{true_sym}'        ")
            row_total = sum(confusion[true_sym].values())
            for pred_sym in symbols:
                count = confusion[true_sym][pred_sym]
                fp.write(f"{count:>5}")
            acc = per_symbol_accuracy[true_sym]
            fp.write(f"{row_total:>8}{acc:>11.2%}\n")
        
        fp.write("\n" + "=" * 80 + "\n")
        fp.write("PER-SYMBOL ACCURACY\n")
        fp.write("=" * 80 + "\n\n")
        
        for sym in sorted(per_symbol_accuracy.keys(), key=lambda x: per_symbol_accuracy[x], reverse=True):
            total_sym = sum(confusion[sym].values())
            correct_sym = confusion[sym][sym]
            acc = per_symbol_accuracy[sym]
            fp.write(f"  '{sym}': {acc:>6.2%}  ({correct_sym}/{total_sym} correct)\n")
        
        fp.write("\n" + "=" * 80 + "\n")
        fp.write("PREDICTION DETAILS (first 50 predictions)\n")
        fp.write("=" * 80 + "\n\n")
        
        fp.write(f"{'Signal':<8} {'Low Freq':<12} {'High Freq':<12} {'True':<6} {'Pred':<6} {'Conf':<10} {'Status':<10}\n")
        fp.write("-" * 80 + "\n")
        
        for i, result in enumerate(results[:50]):
            status = "✓" if result['correct'] else "✗"
            fp.write(f"{result['signal_idx']:<8} {result['low_freq']:<12.2f} {result['high_freq']:<12.2f} "
                    f"'{result['true_symbol']}'     '{result['predicted_symbol']}'     "
                    f"{result['confidence']:<10.4f} {status:<10}\n")
        
        if len(results) > 50:
            fp.write(f"\n... ({len(results) - 50} more predictions)\n")
        
        fp.write("\n" + "=" * 80 + "\n")
        fp.write("MISCLASSIFICATION ANALYSIS\n")
        fp.write("=" * 80 + "\n\n")
        
        misclassified = [r for r in results if not r['correct']]
        fp.write(f"Total misclassifications: {len(misclassified)}\n\n")
        
        if misclassified:
            fp.write(f"{'Signal':<8} {'Low Freq':<12} {'High Freq':<12} {'True':<6} {'Pred':<6} {'Confidence':<12}\n")
            fp.write("-" * 80 + "\n")
            for result in misclassified[:30]:
                fp.write(f"{result['signal_idx']:<8} {result['low_freq']:<12.2f} {result['high_freq']:<12.2f} "
                        f"'{result['true_symbol']}'     '{result['predicted_symbol']}'     "
                        f"{result['confidence']:<12.4f}\n")
            
            if len(misclassified) > 30:
                fp.write(f"\n... ({len(misclassified) - 30} more misclassifications)\n")
    
    # we save results to JSON
    output_json = data_dir / "bayesian_regression_results.json"
    
    with open(output_json, 'w') as fp:
        json.dump({
            'model': {
                'priors': priors,
                'likelihoods': likelihoods,
            },
            'evaluation': {
                'accuracy': float(accuracy),
                'correct': correct,
                'total': total,
                'per_symbol_accuracy': per_symbol_accuracy,
            },
            'predictions': results,
            'confusion_matrix': {true_sym: dict(pred_dict) 
                                for true_sym, pred_dict in confusion.items()},
        }, fp, indent=2)
    
    print(f"\nOutputs saved:")
    print(f"  Text file: {output_txt}")
    print(f"  JSON file: {output_json}")
    print(f"\nTop 3 symbols by accuracy:")
    for sym, acc in sorted(per_symbol_accuracy.items(), key=lambda x: x[1], reverse=True)[:3]:
        total_sym = sum(confusion[sym].values())
        correct_sym = confusion[sym][sym]
        print(f"  '{sym}': {acc:.2%} ({correct_sym}/{total_sym})")
    
    print(f"\nBottom 3 symbols by accuracy:")
    for sym, acc in sorted(per_symbol_accuracy.items(), key=lambda x: x[1])[:3]:
        total_sym = sum(confusion[sym].values())
        correct_sym = confusion[sym][sym]
        print(f"  '{sym}': {acc:.2%} ({correct_sym}/{total_sym})")


if __name__ == "__main__":
    main()
