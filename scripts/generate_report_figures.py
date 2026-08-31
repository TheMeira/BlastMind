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

_AGENTS = ['random', 'greedy', 'mcts', 'dqn', 'beam', 'dqnsearch']
_AGENT_LABELS = {'random': 'Random', 'greedy': 'Greedy', 'mcts': 'MCTS', 'dqn': 'DQN',
                 'beam': 'Beam', 'dqnsearch': 'DQNSearch'}
_AGENT_COLORS = {'random': '#9e9e9e', 'greedy': '#4c78a8', 'mcts': '#72b7b2',
                 'dqn': '#f58518', 'beam': '#54a24b', 'dqnsearch': '#b279a2'}


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


def chart_decision_time_vs_score():
    summary = _load_summary()

    fig, ax = plt.subplots(figsize=(8, 6))
    for a in _AGENTS:
        time_ms = float(summary[a]['mean_decision_time_sec']) * 1000
        score = float(summary[a]['mean_score'])
        ax.scatter([max(time_ms, 0.01)], [score], s=180, color=_AGENT_COLORS[a],
                   label=_AGENT_LABELS[a], zorder=3)
        _off = {'greedy': (-52, -16), 'mcts': (10, 6), 'beam': (10, -14)}
        ax.annotate(_AGENT_LABELS[a], (max(time_ms, 0.01), score),
                    textcoords='offset points', xytext=_off.get(a, (8, 6)),
                    fontsize=10)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Mean decision time per move (ms, log scale)')
    ax.set_ylabel('Mean score (log scale)')
    ax.set_title('Computational Cost vs. Score: Efficiency Trade-off')
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'decision_time_vs_score.png'), dpi=150)
    plt.close(fig)


def chart_density_over_time(min_games=10):
    fig, ax = plt.subplots(figsize=(9, 6))
    for a in _AGENTS:
        path = os.path.join(_RESULTS_DIR, f'final_benchmark_{a}_density.csv')
        by_turn = {}
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                by_turn.setdefault(int(row['turn_index']), []).append(float(row['density']))

        turns = sorted(t for t, vals in by_turn.items() if len(vals) >= min_games)
        means = [sum(by_turn[t]) / len(by_turn[t]) for t in turns]
        ax.plot(turns, means, color=_AGENT_COLORS[a], linewidth=2, label=_AGENT_LABELS[a])

    ax.set_xlabel(f'Turn index (shown while at least {min_games} games still in progress)')
    ax.set_ylabel('Mean board density (fraction of cells filled)')
    ax.set_title('Board Density Over Time by Agent (500 games each)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'density_over_time.png'), dpi=150)
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


def chart_beam_sweeps_line():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (title, sweep, adopted) in zip(axes, [
        ('Beam width sweep', _BEAM_WIDTH_SWEEP, 16),
        ('Lookahead depth sweep (width=16)', _BEAM_DEPTH_SWEEP, 1),
        ('Samples sweep (width=16, depth=1)', _BEAM_SAMPLES_SWEEP, 8),
    ]):
        keys = sorted(sweep.keys())
        medians = [sweep[k][1] for k in keys]
        ax.plot(keys, medians, marker='o', color='#54a24b', linewidth=2)
        adopted_idx = keys.index(adopted)
        ax.scatter([keys[adopted_idx]], [medians[adopted_idx]], color='#2d5a1f', s=100, zorder=5)
        ax.set_title(title)
        ax.set_ylabel('Median score')
        ax.set_xticks(keys)

    fig.suptitle('Beam Search Hyperparameter Sweeps (line variant, adopted value marked)')
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'beam_hyperparameter_sweeps_line.png'), dpi=150)
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


def chart_mcts_budget_line():
    keys = sorted(_MCTS_NSIM_SWEEP.keys())
    values = [_MCTS_NSIM_SWEEP[k] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(keys, values, marker='o', color='#72b7b2', linewidth=2)
    adopted_idx = keys.index(500)
    ax.scatter([keys[adopted_idx]], [values[adopted_idx]], color='#2c6e68', s=100, zorder=5)
    ax.set_xlabel('n_simulations')
    ax.set_ylabel('Mean score')
    ax.set_title('MCTS Simulation Budget Sweep (line variant, adopted value marked)')
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'mcts_simulation_budget_line.png'), dpi=150)
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
    ax.set_yscale('log')
    ax.set_ylabel('Training score (log scale)')
    ax.set_title('DQN Learning Curve: Final Deployed Lineage (v22prod -> order-search training)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'dqn_learning_curve_final_lineage.png'), dpi=150)
    plt.close(fig)


def main():
    _ensure_charts_dir()
    chart_bar_mean_ci()
    chart_box_distributions()
    chart_score_strip()
    chart_score_vs_survival_scatter()
    chart_decision_time_vs_score()
    chart_density_over_time()
    chart_placement_heatmaps()
    chart_beam_sweeps()
    chart_beam_sweeps_line()
    chart_mcts_budget()
    chart_mcts_budget_line()
    chart_mcts_rollout_comparison()
    chart_dqn_milestones()
    chart_dqn_final_lineage()
    chart_dqn_training_diagnostics()
    chart_dqn_loss_curves()
    chart_loss_vs_performance()
    chart_evaluator_search_grid()
    chart_survival_curves()
    chart_score_cdf()
    chart_followup_journey()
    chart_occupancy_heatmaps()
    chart_placement_heatmaps_diff()
    print(f"Saved charts to {_CHARTS_DIR}")


_DQN_LOSS_VERSIONS = [
    ('v3 (old arch)', 'dqn_v3', '#c7c7c7'),
    ('v9 (afterstate)', 'dqn_v9', '#9edae5'),
    ('v11 (stability fix)', 'dqn_v11', '#aec7e8'),
    ('v15 (LR decay)', 'dqn_v15', '#4c78a8'),
    ('v21 (combo/pwc fix)', 'dqn_v21', '#72b7b2'),
    ('v22prod (200k)', 'dqn_v22prod', '#f58518'),
    ('deployed (order-search)', 'dqn_vv24_ordersearch_diag', '#54a24b'),
]

_LOSS_VS_SCORE = [
    ('v11', 'dqn_v11', 428.2),
    ('v15', 'dqn_v15', 524.7),
    ('v17fix', 'dqn_v17fix', 540.8),
    ('v20 (divergence)', 'dqn_v20', 202.0),
    ('v21', 'dqn_v21', 658.8),
    ('v22prod', 'dqn_v22prod', 1033.0),
    ('v23 (n-step)', 'dqn_v23', 793.5),
    ('deployed', 'dqn_vv24_ordersearch_diag', 25227.0),
    ('PopArt scratch', 'dqn_vpopart_scratch_phase1', 106.3),
]


def _read_log(stem):
    path = os.path.join(_RESULTS_DIR, f'{stem}_training_log.csv')
    ep, loss, avg_v, score = [], [], [], []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            try:
                e = int(row['episode']); l = float(row['loss'])
            except (ValueError, KeyError, TypeError):
                continue
            ep.append(e); loss.append(l)
            try: avg_v.append(float(row['avg_v']))
            except (ValueError, KeyError, TypeError): avg_v.append(float('nan'))
            try: score.append(float(row['score']))
            except (ValueError, KeyError, TypeError): score.append(float('nan'))
    return ep, loss, avg_v, score


def _smooth(v, w):
    if len(v) < 2:
        return v
    w = max(1, min(w, len(v) // 5 or 1))
    a = np.asarray(v, dtype=float)
    valid = ~np.isnan(a)
    if valid.sum() == 0:
        return v
    a = np.interp(np.arange(len(a)), np.flatnonzero(valid), a[valid])
    k = np.ones(w) / w
    return np.convolve(a, k, mode='same')


def chart_dqn_training_diagnostics():
    """Loss, avg_v and score across the deployed lineage (v22prod -> order-search)."""
    ep_all, loss_all, v_all, sc_all, boundary = [], [], [], [], None
    for stem in ['dqn_v22prod', 'dqn_vv24_ordersearch_diag']:
        ep, loss, avg_v, score = _read_log(stem)
        if boundary is None and stem == 'dqn_v22prod' and ep:
            boundary = max(ep)
        ep_all += ep; loss_all += loss; v_all += avg_v; sc_all += score

    order = np.argsort(ep_all)
    ep_all = np.asarray(ep_all)[order]
    loss_s = _smooth(np.asarray(loss_all)[order], 500)
    v_s = _smooth(np.asarray(v_all)[order], 500)
    sc_s = _smooth(np.asarray(sc_all)[order], 500)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(ep_all, loss_s, color='#e45756', linewidth=1.4)
    axes[0].set_ylabel('Loss (smoothed)'); axes[0].set_yscale('log')
    axes[1].plot(ep_all, v_s, color='#4c78a8', linewidth=1.4)
    axes[1].axhline(3000, color='#e45756', linestyle=':', linewidth=1.3)
    axes[1].annotate('V_CLIP_MAX = 3000 — never reached during fixed-order training',
                     xy=(0.02, 3000), xycoords=('axes fraction', 'data'),
                     xytext=(0, 6), textcoords='offset points',
                     fontsize=8.5, color='#e45756')
    axes[1].set_ylim(top=3300)
    axes[1].set_ylabel('avg_v (smoothed)')
    axes[2].plot(ep_all, sc_s, color='#f58518', linewidth=1.4)
    axes[2].set_ylabel('Episode score (smoothed)'); axes[2].set_yscale('log')
    axes[2].set_xlabel('Training episode')

    for ax in axes:
        ax.grid(alpha=0.25)
        if boundary:
            ax.axvline(boundary, color='#54a24b', linestyle='--', linewidth=1.3)
    axes[0].annotate('order-search fine-tune begins', xy=(boundary, axes[0].get_ylim()[1]),
                     xytext=(-10, -14), textcoords='offset points', ha='right',
                     fontsize=9, color='#54a24b')
    axes[0].set_title('DQN training diagnostics across the deployed lineage\n'
                      'Loss falls and stabilises while score rises ~30x at the order-search switch',
                      fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'dqn_training_diagnostics.png'), dpi=150)
    plt.close(fig)


def chart_dqn_loss_curves():
    """Loss curves for the milestone lineage, raw and normalised by avg_v."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, stem, colour in _DQN_LOSS_VERSIONS:
        path = os.path.join(_RESULTS_DIR, f'{stem}_training_log.csv')
        if not os.path.exists(path):
            continue
        ep, loss, avg_v, _ = _read_log(stem)
        if not ep:
            continue
        w = max(1, len(ep) // 100)
        axes[0].plot(ep, _smooth(loss, w), label=label, color=colour, linewidth=1.4)
        ratio = [l / abs(v) if v and not np.isnan(v) and abs(v) > 1e-6 else np.nan
                 for l, v in zip(loss, avg_v)]
        axes[1].plot(ep, _smooth(ratio, w), label=label, color=colour, linewidth=1.4)

    axes[0].set_yscale('log'); axes[0].set_ylabel('Loss (smoothed, log scale)')
    axes[0].set_title('Raw loss — not comparable across versions')
    axes[1].set_yscale('log'); axes[1].set_ylabel('Loss / |avg_v| (smoothed, log scale)')
    axes[1].set_title('Scale-normalised loss — comparable')
    for ax in axes:
        ax.set_xlabel('Training episode'); ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc='upper right')
    fig.suptitle('DQN loss across the milestone lineage', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'dqn_loss_curves.png'), dpi=150)
    plt.close(fig)


def chart_loss_vs_performance():
    """Final training loss against benchmark score — the decoupling."""
    pts = []
    for label, stem, score in _LOSS_VS_SCORE:
        path = os.path.join(_RESULTS_DIR, f'{stem}_training_log.csv')
        if not os.path.exists(path):
            continue
        _, loss, _, _ = _read_log(stem)
        if not loss:
            continue
        tail = [l for l in loss[-max(1, len(loss) // 20):] if not np.isnan(l)]
        if tail:
            pts.append((label, float(np.mean(tail)), score))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for label, loss, score in pts:
        popart = 'PopArt' in label
        ax.scatter(loss, score, s=110, zorder=3,
                   color='#e45756' if popart else '#4c78a8',
                   marker='^' if popart else 'o',
                   edgecolor='white', linewidth=1.2)
        offsets = {'v15': (8, -14), 'v17fix': (-52, 6), 'v11': (8, -14),
                   'PopArt scratch': (14, 4), 'v20 (divergence)': (-96, 8)}
        ax.annotate(label, (loss, score), textcoords='offset points',
                    xytext=offsets.get(label, (8, 5)), fontsize=9)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('Final training loss (mean of last 5% of episodes, log scale)')
    ax.set_ylabel('Mean benchmark score (log scale)')
    ax.set_title('Training loss does not predict playing strength\n'
                 'The lowest-loss run is the worst-performing agent built',
                 fontsize=11)
    ax.grid(alpha=0.25, which='both')
    ax.annotate('PopArt normalises its value targets, so its loss\n'
                'sits on a different scale and is not directly comparable',
                xy=(0.30, 0.03), xycoords='axes fraction', fontsize=8.5,
                color='#e45756', va='bottom')
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'dqn_loss_vs_performance.png'), dpi=150)
    plt.close(fig)

_EVAL_SEARCH_GRID = [
    ('hand-crafted heuristic', 'greedy selection', 'greedy', 'Greedy'),
    ('hand-crafted heuristic', 'search', 'beam', 'Beam'),
    ('learned value function', 'greedy selection', 'dqn', 'DQN'),
    ('learned value function', 'search', 'dqnsearch', 'DQNSearch'),
]

# Follow-ups with a full benchmark result. Those killed early on checkpoint
# evidence alone (4, 5, 6, 7, 9) are excluded and covered in the caption.
_FOLLOWUP_JOURNEY = [
    ('Order-search scale-up\n(the regression)', 15404.7, 500, 'rejected'),
    ('FU1 epsilon floor', 20077.8, 500, 'rejected'),
    ('FU2 buffer persistence', 18817.1, 500, 'rejected'),
    ('FU3 head reset', 12256.0, 500, 'rejected'),
    ('FU8 PopArt from scratch', 106.3, 100, 'rejected'),
    ('FU10 ranking loss', 21997.0, 100, 'parity'),
    ('FU11 DQNPool (hybrid)', 81260.5, 100, 'improved'),
    ('FU12 DQNSearch', None, 500, 'adopted'),
]

_VERDICT_COLORS = {'rejected': '#e45756', 'parity': '#f58518',
                   'improved': '#4c78a8', 'adopted': '#54a24b'}

_DEPLOYED_BASELINE = 25227.0


def chart_evaluator_search_grid():
    """The 2x2 that separates evaluator quality from search depth."""
    summary = _load_summary()
    evaluators = ['hand-crafted heuristic', 'learned value function']
    selections = ['greedy selection', 'search']

    vals = {}
    for ev, sel, agent, label in _EVAL_SEARCH_GRID:
        if agent not in summary:
            print(f"  [skip] {agent} missing from summary -- run the benchmark first")
            return
        vals[(ev, sel)] = (float(summary[agent]['mean_score']), label)

    x = np.arange(len(selections))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for i, ev in enumerate(evaluators):
        heights = [vals[(ev, sel)][0] for sel in selections]
        labels = [vals[(ev, sel)][1] for sel in selections]
        colour = '#4c78a8' if i == 0 else '#f58518'
        bars = ax.bar(x + (i - 0.5) * width, heights, width, label=ev, color=colour)
        for b, h, lab in zip(bars, heights, labels):
            ax.annotate(f'{lab}\n{h:,.0f}', (b.get_x() + b.get_width() / 2, h),
                        textcoords='offset points', xytext=(0, 4),
                        ha='center', fontsize=9)

    g = vals[('hand-crafted heuristic', 'greedy selection')][0]
    b = vals[('hand-crafted heuristic', 'search')][0]
    d = vals[('learned value function', 'greedy selection')][0]
    ds = vals[('learned value function', 'search')][0]

    ax.set_xticks(x); ax.set_xticklabels(selections)
    ax.set_yscale('log')
    ax.set_ylabel('Mean benchmark score (log scale)')
    ax.set_title('Evaluator quality vs search depth\n'
                 f'Search multiplies both evaluators ({b/g:.1f}x heuristic, {ds/d:.1f}x learned)\n'
                 f'but the learned value wins at each level ({d/g:.1f}x under greedy, {ds/b:.1f}x under search)',
                 fontsize=10.5)
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    ax.legend(title='evaluator', fontsize=9)
    ax.grid(alpha=0.25, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'evaluator_vs_search.png'), dpi=150)
    plt.close(fig)


def chart_survival_curves():
    """Fraction of games still alive at each turn -- the mechanism behind score."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for agent in _AGENTS:
        try:
            rows = _load_scores(agent)
        except FileNotFoundError:
            print(f"  [skip] {agent} detail CSV missing -- run the benchmark first")
            return
        hands = np.array([int(r['hands_played']) for r in rows])
        grid = np.arange(0, hands.max() + 1)
        alive = [(hands >= t).mean() for t in grid]
        ax.plot(grid, alive, label=_AGENT_LABELS[agent],
                color=_AGENT_COLORS[agent], linewidth=1.8)

    ax.set_xscale('symlog')
    ax.set_xlabel('Turn number (hands played, symlog scale)')
    ax.set_ylabel('Fraction of games still alive')
    ax.set_title('Survival curves: how long each agent keeps the board playable')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'survival_curves.png'), dpi=150)
    plt.close(fig)


def chart_score_cdf():
    """Cumulative score distribution -- why the mean misleads in this game."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for agent in _AGENTS:
        try:
            rows = _load_scores(agent)
        except FileNotFoundError:
            print(f"  [skip] {agent} detail CSV missing -- run the benchmark first")
            return
        sc = np.sort(np.array([int(r['final_score']) for r in rows], dtype=float))
        frac = np.arange(1, len(sc) + 1) / len(sc)
        ax.step(sc, frac, where='post', label=_AGENT_LABELS[agent],
                color=_AGENT_COLORS[agent], linewidth=1.8)
        median = float(np.median(sc)); mean = float(sc.mean())
        ax.scatter([median], [0.5], color=_AGENT_COLORS[agent], s=28, zorder=4)
        ax.scatter([mean], [(sc <= mean).mean()], color=_AGENT_COLORS[agent],
                   s=42, marker='x', zorder=4)

    ax.set_xscale('log')
    ax.set_xlabel('Final score (log scale)')
    ax.set_ylabel('Fraction of games at or below this score')
    ax.set_title('Score distributions are heavy-tailed\n'
                 'Dots mark the median; crosses mark the mean -- the mean sits well above '
                 'the middle of the distribution',
                 fontsize=11)
    ax.axhline(0.5, color='#666666', linestyle=':', linewidth=1)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.25, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'score_cdf.png'), dpi=150)
    plt.close(fig)


def chart_followup_journey():
    """Every DQN follow-up with a full benchmark, against the deployed baseline."""
    summary = _load_summary()
    entries = []
    for label, value, n, verdict in _FOLLOWUP_JOURNEY:
        if value is None:
            if 'dqnsearch' not in summary:
                print("  [skip] dqnsearch missing from summary -- run the benchmark first")
                return
            value = float(summary['dqnsearch']['mean_score'])
        entries.append((label, value, n, verdict))

    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]
    ypos = np.arange(len(entries))[::-1]

    fig, ax = plt.subplots(figsize=(12.5, 6))
    for y, (label, value, n, verdict) in zip(ypos, entries):
        ax.scatter(value, y, s=130 if n == 500 else 90,
                   marker='o' if n == 500 else 's',
                   color=_VERDICT_COLORS[verdict], zorder=3,
                   edgecolor='white', linewidth=1.2)
        ax.hlines(y, min(values) * 0.8, value, color=_VERDICT_COLORS[verdict],
                  alpha=0.3, linewidth=1.5)

    ax.axvline(_DEPLOYED_BASELINE, color='#333333', linestyle='--', linewidth=1.4)
    ax.annotate(f'deployed baseline\n{_DEPLOYED_BASELINE:,.0f}',
                xy=(_DEPLOYED_BASELINE, ypos.max()), xytext=(6, -4),
                textcoords='offset points', fontsize=9, va='top')

    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('Mean benchmark score (log scale)')
    ax.set_title('The DQN improvement campaign: eight rejections before the breakthrough\n'
                 'Circles = 500-game benchmark, squares = 100-game',
                 fontsize=11)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=c, label=v)
               for v, c in _VERDICT_COLORS.items()]
    ax.legend(handles=handles, fontsize=9, title='verdict',
              loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_xlim(right=max(values) * 2.6)
    ax.grid(alpha=0.25, axis='x', which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'dqn_followup_journey.png'), dpi=150)
    plt.close(fig)

def chart_occupancy_heatmaps():
    """Time-averaged board occupancy -- where each agent actually keeps cells filled.

    Unlike the anchor heatmap, this reflects strategy rather than piece geometry:
    it measures how often each cell is occupied, averaged over every turn.
    """
    occ = {}
    for agent in _AGENTS:
        path = os.path.join(_RESULTS_DIR, f'final_benchmark_{agent}_occupancy.npy')
        if not os.path.exists(path):
            print(f"  [skip] {agent} occupancy missing -- run scripts/collect_occupancy.py")
            return
        occ[agent] = np.load(path)

    mean_map = np.mean([occ[a] for a in _AGENTS], axis=0)

    fig, axes = plt.subplots(2, len(_AGENTS), figsize=(3.0 * len(_AGENTS), 6.4))
    vmax = max(o.max() for o in occ.values())
    for j, agent in enumerate(_AGENTS):
        im0 = axes[0, j].imshow(occ[agent], cmap='viridis', vmin=0, vmax=vmax)
        axes[0, j].set_title(_AGENT_LABELS[agent], fontsize=10)
        diff = occ[agent] - mean_map
        lim = float(np.abs([occ[a] - mean_map for a in _AGENTS]).max())
        im1 = axes[1, j].imshow(diff, cmap='coolwarm', vmin=-lim, vmax=lim)
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xticks([]); ax.set_yticks([])

    axes[0, 0].set_ylabel('occupancy', fontsize=10)
    axes[1, 0].set_ylabel('deviation from\nall-agent mean', fontsize=10)
    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.02, pad=0.01)
    fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01)
    fig.suptitle('Time-averaged board occupancy by agent\n'
                 'Top: how often each cell is filled. Bottom: deviation from the all-agent mean',
                 fontsize=11)
    fig.savefig(os.path.join(_CHARTS_DIR, 'occupancy_heatmaps.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def chart_placement_heatmaps_diff():
    """Anchor heatmaps as a log-ratio against the pooled all-agent map.

    Raw anchor counts confound two things: where the agent chose to place, and
    where placement was geometrically possible at all (a 3x3 piece can only
    anchor in rows/cols 0-5). Every agent draws from the same piece
    distribution, so the pooled map approximates that geometric baseline;
    dividing by it isolates genuine preference. A log2 ratio is used so that
    "twice as often" and "half as often" are equal and opposite, and so
    low-count edge cells remain readable.
    """
    maps = {}
    for agent in _AGENTS:
        path = os.path.join(_RESULTS_DIR, f'final_benchmark_{agent}_heatmap.npy')
        if not os.path.exists(path):
            print(f"  [skip] {agent} heatmap missing")
            return
        h = np.load(path).astype(float)
        maps[agent] = h / h.sum()

    pooled = np.mean([maps[a] for a in _AGENTS], axis=0)
    floor = pooled[pooled > 0].min() * 0.5 if (pooled > 0).any() else 1e-9
    ratios = {a: np.log2(np.maximum(maps[a], floor) / np.maximum(pooled, floor))
              for a in _AGENTS}
    lim = float(max(np.abs(r).max() for r in ratios.values()))

    fig, axes = plt.subplots(1, len(_AGENTS), figsize=(3.0 * len(_AGENTS), 3.6))
    for ax, agent in zip(np.atleast_1d(axes), _AGENTS):
        im = ax.imshow(ratios[agent], cmap='coolwarm', vmin=-lim, vmax=lim)
        ax.set_title(_AGENT_LABELS[agent], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), fraction=0.02, pad=0.01)
    cb.set_label('log2(agent / all-agent mean)')
    fig.suptitle('Placement preference, controlling for piece geometry\n'
                 'Red = anchors here more often than the average agent, blue = less often',
                 fontsize=11)
    fig.savefig(os.path.join(_CHARTS_DIR, 'placement_heatmaps_diff.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def chart_score_strip():
    """Every game as one dot, with median and quartile markers.

    Alternative to the box plot: nothing to decode, since each dot is a single
    game. Shows the full 500-game spread rather than compressing it into five
    summary statistics, and makes the heavy upper tail directly visible.
    """
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, agent in enumerate(_AGENTS):
        rows = _load_scores(agent)
        sc = np.array([int(r['final_score']) for r in rows], dtype=float)
        jitter = rng.uniform(-0.28, 0.28, size=len(sc))
        ax.scatter(np.full(len(sc), i) + jitter, sc, s=7, alpha=0.28,
                   color=_AGENT_COLORS[agent], linewidths=0, zorder=2)

        q1, med, q3 = np.percentile(sc, [25, 50, 75])
        ax.hlines(med, i - 0.40, i + 0.40, color='#222222', linewidth=2.4, zorder=4)
        ax.hlines([q1, q3], i - 0.28, i + 0.28, color='#222222',
                  linewidth=1.1, alpha=0.75, zorder=4)
        ax.annotate(f'{med:,.0f}', (i, med), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=9, fontweight='bold',
                    zorder=5)

    ax.set_xticks(range(len(_AGENTS)))
    ax.set_xticklabels([_AGENT_LABELS[a] for a in _AGENTS])
    ax.set_yscale('log')
    ax.set_ylabel('Final score (log scale)')
    ax.set_title('Final Benchmark: Every Game Plotted (500 games per agent)\n'
                 'Each dot is one game; thick line = median, thin lines = middle 50%',
                 fontsize=11)
    ax.grid(alpha=0.2, axis='y', which='both')
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(_CHARTS_DIR, 'score_distributions_strip.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
