import csv
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from results_lib import mannwhitney_pair, bonferroni_threshold

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# Established ranking: Random < Greedy < MCTS < DQN < Beam < DQNSearch
_RANKING = ['random', 'greedy', 'mcts', 'dqn', 'beam', 'dqnsearch']
_ADJACENT_PAIRS = set(zip(_RANKING, _RANKING[1:]))
_ALL_PAIRS = list(itertools.combinations(_RANKING, 2))


def load_scores(agent_name, games=500):
    path = os.path.join(_RESULTS_DIR, f'final_benchmark_{agent_name}_{games}games.csv')
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return [int(r['final_score']) for r in rows]


def main():
    # Bonferroni correction covers all 10 pairwise tests actually run (the plan's
    # literal "each agent pair" requirement) -- the 4 adjacent pairs remain the
    # primary evidence for the ranking narrative, flagged via is_adjacent below.
    n_tests = len(_ALL_PAIRS)
    threshold = bonferroni_threshold(alpha=0.05, n_tests=n_tests)

    results = []
    for a, b in _ALL_PAIRS:
        scores_a = load_scores(a)
        scores_b = load_scores(b)
        u_stat, p_value, rank_biserial = mannwhitney_pair(scores_a, scores_b)
        significant = p_value < threshold
        results.append({
            'comparison': f'{a} vs {b}',
            'is_adjacent': (a, b) in _ADJACENT_PAIRS,
            'n_a': len(scores_a),
            'n_b': len(scores_b),
            'u_statistic': u_stat,
            'p_value': p_value,
            'bonferroni_threshold': threshold,
            'significant': significant,
            'rank_biserial_effect_size': rank_biserial,
        })

    out_path = os.path.join(_RESULTS_DIR, 'final_benchmark_significance.csv')
    fieldnames = ['comparison', 'is_adjacent', 'n_a', 'n_b', 'u_statistic', 'p_value',
                  'bonferroni_threshold', 'significant', 'rank_biserial_effect_size']
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in results:
            writer.writerow([row[k] for k in fieldnames])

    print(f"Bonferroni-corrected significance threshold (alpha=0.05, {n_tests} tests): {threshold:.5f}\n")
    for row in results:
        sig = 'SIGNIFICANT' if row['significant'] else 'not significant'
        tag = '[adjacent]' if row['is_adjacent'] else '[full-matrix]'
        print(f"{row['comparison']:>16} {tag:>14} | U={row['u_statistic']:>10.1f} | "
              f"p={row['p_value']:.6g} | effect size (rank-biserial)={row['rank_biserial_effect_size']:.4f} | {sig}")

    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
