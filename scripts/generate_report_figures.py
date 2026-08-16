import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
_CHARTS_DIR = os.path.join(_RESULTS_DIR, 'charts')

_AGENTS = ['random', 'greedy', 'mcts', 'dqn', 'beam']
_AGENT_LABELS = {'random': 'Random', 'greedy': 'Greedy', 'mcts': 'MCTS', 'dqn': 'DQN', 'beam': 'Beam'}
_AGENT_COLORS = {'random': '#9e9e9e', 'greedy': '#4c78a8', 'mcts': '#72b7b2',
                 'dqn': '#f58518', 'beam': '#54a24b'}


def _ensure_charts_dir():
    os.makedirs(_CHARTS_DIR, exist_ok=True)


def _load_summary():
    path = os.path.join(_RESULTS_DIR, 'final_benchmark_summary.csv')
    with open(path, newline='') as f:
        rows = {r['agent']: r for r in csv.DictReader(f)}
    return rows


def _load_scores(agent, games=500):
    path = os.path.join(_RESULTS_DIR, f'final_benchmark_{agent}_{games}games.csv')
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return rows


def chart_bar_mean_ci():
    summary = _load_summary()
    means = [float(summary[a]['mean_score']) for a in _AGENTS]
    lo = [float(summary[a]['mean_score']) - float(summary[a]['ci95_low']) for a in _AGENTS]
    hi = [float(summary[a]['ci95_high']) - float(summary[a]['mean_score']) for a in _AGENTS]
    colors = [_AGENT_COLORS[a] for a in _AGENTS]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([_AGENT_LABELS[a] for a in _AGENTS], means, yerr=[lo, hi], capsize=5, color=colors)
    ax.set_ylabel('Mean score (95% CI)')
    ax.set_title('Final Benchmark: Mean Score by Agent (500 games, seeds 5000-5499)')
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'bar_mean_score_ci.png'), dpi=150)
    plt.close(fig)


def chart_box_distributions():
    data = [[int(r['final_score']) for r in _load_scores(a)] for a in _AGENTS]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=[_AGENT_LABELS[a] for a in _AGENTS], patch_artist=True, showfliers=True)
    for patch, a in zip(bp['boxes'], _AGENTS):
        patch.set_facecolor(_AGENT_COLORS[a])
    ax.set_ylabel('Final score')
    ax.set_yscale('log')
    ax.set_title('Final Benchmark: Score Distribution by Agent (log scale)')
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'box_score_distributions.png'), dpi=150)
    plt.close(fig)


def chart_score_vs_survival_scatter():
    fig, ax = plt.subplots(figsize=(8, 6))
    for a in _AGENTS:
        rows = _load_scores(a)
        scores = [int(r['final_score']) for r in rows]
        hands = [int(r['hands_played']) for r in rows]
        ax.scatter(hands, scores, s=12, alpha=0.5, label=_AGENT_LABELS[a], color=_AGENT_COLORS[a])
    ax.set_xlabel('Hands survived')
    ax.set_ylabel('Final score')
    ax.set_yscale('log')
    ax.set_title('Score vs. Survival Time, All Agents (500 games each)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'scatter_score_vs_survival.png'), dpi=150)
    plt.close(fig)


def chart_placement_heatmaps():
    fig, axes = plt.subplots(1, len(_AGENTS), figsize=(4 * len(_AGENTS), 4))
    for ax, a in zip(axes, _AGENTS):
        heatmap = np.load(os.path.join(_RESULTS_DIR, f'final_benchmark_{a}_heatmap.npy'))
        im = ax.imshow(heatmap, cmap='viridis')
        ax.set_title(_AGENT_LABELS[a])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Board Placement Heatmaps (piece anchor position, 500 games each)')
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'heatmaps_placement.png'), dpi=150)
    plt.close(fig)


# Documented sweep results from Tuning Log.md (not recomputed here)
_BEAM_WIDTH_SWEEP = {4: (15133.7, 6196.0), 8: (30796.2, 13199.0), 16: (62317.8, 22011.0), 32: (65219.5, 26398.0)}
_BEAM_DEPTH_SWEEP = {0: (6167.2, 3344.0), 1: (62317.8, 22011.0), 2: (30286.5, 21292.0)}
_BEAM_SAMPLES_SWEEP = {2: (30370.0, 11682.0), 4: (62317.8, 22011.0), 8: (70598.6, 58047.0)}
_MCTS_NSIM_SWEEP = {50: 3526.9, 100: 5738.5, 200: 9740.4, 500: 13678.1, 1000: 10699.1}
_MCTS_ROLLOUT_COMPARISON = {'Heuristic rollout': 839.3, 'Random rollout': 501.8}

_DQN_MILESTONES = [
    ('v3\n(old arch)', 121.3),
    ('v9\n(afterstate)', 160.4),
    ('v11\n(stability fix)', 428.2),
    ('v15\n(LR decay, 80k)', 524.7),
    ('v21\n(combo/pwc fix)', 658.8),
    ('v22prod\n(200k)', 1033.0),
    ('v22prod\n+order-search', 19983.3),
    ('deployed\n(order-search trained)', 25227.0),
]


def chart_beam_sweeps():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (title, sweep, adopted) in zip(axes, [
        ('Beam width sweep', _BEAM_WIDTH_SWEEP, 16),
        ('Lookahead depth sweep (width=16)', _BEAM_DEPTH_SWEEP, 1),
        ('Samples sweep (width=16, depth=1)', _BEAM_SAMPLES_SWEEP, 8),
    ]):
        keys = sorted(sweep.keys())
        medians = [sweep[k][1] for k in keys]
        colors = ['#54a24b' if k == adopted else '#c7c7c7' for k in keys]
        ax.bar([str(k) for k in keys], medians, color=colors)
        ax.set_title(title)
        ax.set_ylabel('Median score')

    fig.suptitle('Beam Search Hyperparameter Sweeps (adopted value in green)')
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'beam_hyperparameter_sweeps.png'), dpi=150)
    plt.close(fig)


def chart_mcts_budget():
    keys = sorted(_MCTS_NSIM_SWEEP.keys())
    values = [_MCTS_NSIM_SWEEP[k] for k in keys]
    colors = ['#72b7b2' if k == 500 else '#c7c7c7' for k in keys]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([str(k) for k in keys], values, color=colors)
    ax.set_xlabel('n_simulations')
    ax.set_ylabel('Mean score')
    ax.set_title('MCTS Simulation Budget Sweep (adopted value in teal)')
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'mcts_simulation_budget.png'), dpi=150)
    plt.close(fig)


def chart_mcts_rollout_comparison():
    labels = list(_MCTS_ROLLOUT_COMPARISON.keys())
    values = list(_MCTS_ROLLOUT_COMPARISON.values())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, values, color=['#72b7b2', '#c7c7c7'])
    ax.set_ylabel('Mean score (n_sim=500)')
    ax.set_title('MCTS Rollout Policy Comparison (+67% for heuristic)')
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'mcts_rollout_comparison.png'), dpi=150)
    plt.close(fig)


def chart_dqn_milestones():
    labels = [m[0] for m in _DQN_MILESTONES]
    values = [m[1] for m in _DQN_MILESTONES]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(labels, values, marker='o', color='#f58518', linewidth=2)
    ax.set_yscale('log')
    ax.set_ylabel('Mean benchmark score (log scale)')
    ax.set_title('DQN Milestone Journey: Ancestor Lineage of the Deployed Checkpoint')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'dqn_learning_curve_milestones.png'), dpi=150)
    plt.close(fig)


def chart_dqn_final_lineage():
    episodes, scores = [], []
    for fname in ['dqn_v22prod_training_log.csv', 'dqn_vv24_ordersearch_diag_training_log.csv']:
        path = os.path.join(_RESULTS_DIR, fname)
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                episodes.append(int(row['episode']))
                scores.append(float(row['score']))

    window = 500
    smoothed = np.convolve(scores, np.ones(window) / window, mode='valid')
    smoothed_episodes = episodes[window - 1:]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, scores, color='#f58518', alpha=0.15, linewidth=0.5, label='raw per-episode score')
    ax.plot(smoothed_episodes, smoothed, color='#f58518', linewidth=2, label=f'{window}-episode rolling mean')
    ax.axvline(200000, color='gray', linestyle='--', linewidth=1, label='order-search training begins')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Training score')
    ax.set_title('DQN Learning Curve: Final Deployed Lineage (v22prod -> order-search training)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'dqn_learning_curve_final_lineage.png'), dpi=150)
    plt.close(fig)


def main():
    _ensure_charts_dir()
    chart_bar_mean_ci()
    chart_box_distributions()
    chart_score_vs_survival_scatter()
    chart_placement_heatmaps()
    chart_beam_sweeps()
    chart_mcts_budget()
    chart_mcts_rollout_comparison()
    chart_dqn_milestones()
    chart_dqn_final_lineage()
    print(f"Saved 9 charts to {_CHARTS_DIR}")


if __name__ == '__main__':
    main()
