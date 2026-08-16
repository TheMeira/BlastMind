import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.game.game_engine import GameEngine
from src.ai.random_agent import RandomAgent
from src.ai.greedy import GreedyAgent
from src.ai.beam import BeamAgent
from src.ai.mcts import MCTSAgent
from src.ai.dqn import DQNAgent
from results_lib import summary_stats

_BOARD = 8
_AGENT_SEED = 42


def build_agents():
    return {
        'random': RandomAgent(seed=_AGENT_SEED),
        'greedy': GreedyAgent(search_orderings=True),
        'beam': BeamAgent(beam_width=16, lookahead_depth=1, samples=8,
                           search_orderings=True, seed=_AGENT_SEED),
        'mcts': MCTSAgent(n_simulations=500, time_limit=30.0, seed=_AGENT_SEED),
        'dqn': DQNAgent(checkpoint_path='models/dqn_vv24_ordersearch_diag_ep205000.pt',
                         device='cpu', search_orderings=True),
    }


def _random_choose_moves(agent, state, engine):
    moves = []
    sim = state
    for pid in list(state.pieces):
        placements = engine.get_valid_placements(sim, pid)
        if not placements:
            break
        row, col = agent.choose(sim, pid, placements)
        moves.append((pid, row, col))
        sim = engine.apply_placement(sim, pid, row, col)
    return moves


def run_game_detailed(agent, seed, agent_name, heatmap):
    engine = GameEngine(seed=seed)
    state = engine.new_game()

    hands_played = 0
    pieces_placed = 0
    max_combo = 0
    move_times = []

    while not state.game_over:
        t0 = time.perf_counter()
        if agent_name == 'random':
            moves = _random_choose_moves(agent, state, engine)
        else:
            moves = agent.choose_moves(state, engine)
        move_times.append(time.perf_counter() - t0)

        if not moves:
            state.game_over = True
            break

        for pid, row, col in moves:
            heatmap[row, col] += 1
            state = engine.apply_placement(state, pid, row, col)
            pieces_placed += 1
            max_combo = max(max_combo, state.combo_count)

        hands_played += 1

        if len(state.pieces) == 0:
            state = engine.start_new_turn(state)

    avg_move_time = sum(move_times) / len(move_times) if move_times else 0.0

    return {
        'seed': seed,
        'final_score': state.score,
        'lines_cleared': state.lines_cleared_total,
        'max_combo': max_combo,
        'hands_played': hands_played,
        'pieces_placed': pieces_placed,
        'avg_move_time_sec': avg_move_time,
        'total_move_time_sec': sum(move_times),
    }


def write_detailed_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ['seed', 'final_score', 'lines_cleared', 'max_combo', 'hands_played',
                  'pieces_placed', 'avg_move_time_sec', 'total_move_time_sec']
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([row[k] for k in fieldnames])


def run_agent_benchmark(agent_name, agent, games, seed_base, results_dir):
    heatmap = np.zeros((_BOARD, _BOARD), dtype=np.int64)
    rows = []

    for i in range(games):
        seed = seed_base + i
        result = run_game_detailed(agent, seed, agent_name, heatmap)
        rows.append(result)
        print(f"[{agent_name}] game {i + 1}/{games} seed={seed} "
              f"score={result['final_score']} lines={result['lines_cleared']} "
              f"max_combo={result['max_combo']} hands={result['hands_played']} "
              f"avg_move_time={result['avg_move_time_sec']:.4f}s")

    detail_path = os.path.join(results_dir, f'final_benchmark_{agent_name}_{games}games.csv')
    write_detailed_csv(rows, detail_path)
    print(f"[{agent_name}] saved per-game data to {detail_path}")

    heatmap_path = os.path.join(results_dir, f'final_benchmark_{agent_name}_heatmap.npy')
    np.save(heatmap_path, heatmap)
    print(f"[{agent_name}] saved placement heatmap to {heatmap_path}")

    return rows


def write_summary_csv(all_stats, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ['agent', 'n', 'mean_score', 'median_score', 'mean_excl_top_outlier',
                  'stdev', 'ci95_low', 'ci95_high', 'mean_turns', 'mean_lines_per_game',
                  'mean_decision_time_sec']
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in all_stats:
            writer.writerow([row[k] for k in fieldnames])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=500)
    p.add_argument('--seed-base', type=int, default=5000)
    p.add_argument('--agents', type=str, default='random,greedy,beam,mcts,dqn',
                   help='comma-separated subset of agents to run, in order')
    args = p.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    all_agents = build_agents()
    agent_order = args.agents.split(',')

    all_stats = []
    for agent_name in agent_order:
        agent = all_agents[agent_name]
        t0 = time.time()
        rows = run_agent_benchmark(agent_name, agent, args.games, args.seed_base, results_dir)
        elapsed = time.time() - t0

        scores = [r['final_score'] for r in rows]
        stats = summary_stats(scores)
        mean_turns = sum(r['hands_played'] for r in rows) / len(rows)
        mean_lines = sum(r['lines_cleared'] for r in rows) / len(rows)
        mean_decision_time = sum(r['avg_move_time_sec'] for r in rows) / len(rows)

        all_stats.append({
            'agent': agent_name,
            'n': stats['n'],
            'mean_score': stats['mean'],
            'median_score': stats['median'],
            'mean_excl_top_outlier': stats['mean_excl_top_outlier'],
            'stdev': stats['stdev'],
            'ci95_low': stats['ci95_low'],
            'ci95_high': stats['ci95_high'],
            'mean_turns': mean_turns,
            'mean_lines_per_game': mean_lines,
            'mean_decision_time_sec': mean_decision_time,
        })
        print(f"[{agent_name}] {args.games} games in {elapsed:.1f}s | "
              f"mean={stats['mean']:.1f} median={stats['median']:.1f} "
              f"mean_excl_top_outlier={stats['mean_excl_top_outlier']:.1f}\n")

    summary_path = os.path.join(results_dir, 'final_benchmark_summary.csv')
    write_summary_csv(all_stats, summary_path)
    print(f"Saved consolidated summary to {summary_path}")


if __name__ == '__main__':
    main()
