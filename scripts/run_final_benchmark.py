import argparse
import csv
import os
import pickle
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
from src.ai.dqn_search import DQNSearchAgent
from results_lib import summary_stats

_BOARD = 8
_AGENT_SEED = 42

# Anchored to the project root so the script runs from any working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DQN_CHECKPOINT = os.path.join(_PROJECT_ROOT, 'models',
                               'dqn_vv24_ordersearch_diag_ep205000.pt')

_DETAIL_FIELDNAMES = ['seed', 'final_score', 'lines_cleared', 'max_combo', 'hands_played',
                      'pieces_placed', 'avg_move_time_sec', 'total_move_time_sec']
_DENSITY_FIELDNAMES = ['seed', 'turn_index', 'density']


def build_agents():
    return {
        'random': RandomAgent(seed=_AGENT_SEED),
        'greedy': GreedyAgent(search_orderings=True),
        'beam': BeamAgent(beam_width=16, lookahead_depth=1, samples=8,
                           search_orderings=True, seed=_AGENT_SEED),
        'mcts': MCTSAgent(n_simulations=500, time_limit=30.0, seed=_AGENT_SEED),
        'dqn': DQNAgent(checkpoint_path=_DQN_CHECKPOINT,
                         device='cpu', search_orderings=True),
        'dqnsearch': DQNSearchAgent(checkpoint_path=_DQN_CHECKPOINT,
                                     device='cpu', seed=_AGENT_SEED),
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


def _rng_state_path(results_dir, agent_name):
    return os.path.join(results_dir, f'final_benchmark_{agent_name}_rngstate.pkl')


def _save_rng_state(agent, path):
    """Persist the agent's RNG so an interrupted run resumes bit-exactly."""
    rng = getattr(agent, '_rng', None)
    if rng is None:
        return
    with open(path, 'wb') as f:
        pickle.dump(rng.getstate(), f)


def _restore_rng_state(agent, path):
    """Restore RNG state saved by a previous session. Returns True if restored."""
    rng = getattr(agent, '_rng', None)
    if rng is None or not os.path.exists(path):
        return False
    try:
        with open(path, 'rb') as f:
            rng.setstate(pickle.load(f))
        return True
    except Exception as exc:
        print(f"  WARNING: could not restore RNG state ({exc}); continuing from a fresh seed")
        return False


def run_game_detailed(agent, seed, agent_name, heatmap, occupancy):
    engine = GameEngine(seed=seed)
    state = engine.new_game()

    hands_played = 0
    pieces_placed = 0
    max_combo = 0
    move_times = []
    density_sequence = []
    game_occupancy = np.zeros((_BOARD, _BOARD), dtype=np.float64)
    occupancy_turns = 0

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
        density_sequence.append(float(state.board.grid.sum()) / (_BOARD * _BOARD))
        game_occupancy += state.board.grid
        occupancy_turns += 1

        if len(state.pieces) == 0:
            state = engine.start_new_turn(state)

    avg_move_time = sum(move_times) / len(move_times) if move_times else 0.0
    if occupancy_turns:
        occupancy += game_occupancy / occupancy_turns

    result = {
        'seed': seed,
        'final_score': state.score,
        'lines_cleared': state.lines_cleared_total,
        'max_combo': max_combo,
        'hands_played': hands_played,
        'pieces_placed': pieces_placed,
        'avg_move_time_sec': avg_move_time,
        'total_move_time_sec': sum(move_times),
    }
    return result, density_sequence


def _completed_seeds(detail_path):
    if not os.path.exists(detail_path):
        return set()
    with open(detail_path, newline='') as f:
        return {int(r['seed']) for r in csv.DictReader(f)}


def run_agent_benchmark(agent_name, agent, games, seed_base, results_dir):
    """Resumable: safe to interrupt (Ctrl-C or kill) at any point and re-run the
    identical command later -- already-completed seeds are skipped, in-progress
    games are simply re-run from scratch (nothing partial is ever written)."""
    detail_path = os.path.join(results_dir, f'final_benchmark_{agent_name}_{games}games.csv')
    heatmap_path = os.path.join(results_dir, f'final_benchmark_{agent_name}_heatmap.npy')
    occupancy_path = os.path.join(results_dir, f'final_benchmark_{agent_name}_occupancy_sum.npy')
    density_path = os.path.join(results_dir, f'final_benchmark_{agent_name}_density.csv')

    done_seeds = _completed_seeds(detail_path)
    rng_path = _rng_state_path(results_dir, agent_name)
    if done_seeds:
        if _restore_rng_state(agent, rng_path):
            print(f"[{agent_name}] restored RNG state -- resuming the original random sequence")
        else:
            print(f"[{agent_name}] NOTE: no saved RNG state; games after this point "
                  f"use a fresh random sequence")
    elif os.path.exists(rng_path):
        os.remove(rng_path)
    heatmap = np.load(heatmap_path) if os.path.exists(heatmap_path) else np.zeros((_BOARD, _BOARD), dtype=np.int64)
    occupancy = (np.load(occupancy_path) if os.path.exists(occupancy_path)
                 else np.zeros((_BOARD, _BOARD), dtype=np.float64))

    detail_mode = 'a' if done_seeds else 'w'
    density_mode = 'a' if os.path.exists(density_path) else 'w'

    with open(detail_path, detail_mode, newline='') as detail_f, \
         open(density_path, density_mode, newline='') as density_f:
        detail_writer = csv.writer(detail_f)
        density_writer = csv.writer(density_f)
        if detail_mode == 'w':
            detail_writer.writerow(_DETAIL_FIELDNAMES)
        if density_mode == 'w':
            density_writer.writerow(_DENSITY_FIELDNAMES)

        remaining = games - len(done_seeds)
        if done_seeds:
            print(f"[{agent_name}] resuming: {len(done_seeds)}/{games} already done, {remaining} remaining")

        completed_this_run = 0
        for i in range(games):
            seed = seed_base + i
            if seed in done_seeds:
                continue

            result, density_sequence = run_game_detailed(agent, seed, agent_name,
                                                         heatmap, occupancy)
            detail_writer.writerow([result[k] for k in _DETAIL_FIELDNAMES])
            for turn_index, density in enumerate(density_sequence):
                density_writer.writerow([seed, turn_index, density])
            detail_f.flush()
            density_f.flush()
            np.save(heatmap_path, heatmap)
            np.save(occupancy_path, occupancy)
            _save_rng_state(agent, rng_path)
            games_done = len(done_seeds) + completed_this_run + 1
            np.save(occupancy_path.replace('_occupancy_sum.npy', '_occupancy.npy'),
                    occupancy / games_done)

            completed_this_run += 1
            print(f"[{agent_name}] game {len(done_seeds) + completed_this_run}/{games} seed={seed} "
                  f"score={result['final_score']} lines={result['lines_cleared']} "
                  f"max_combo={result['max_combo']} hands={result['hands_played']} "
                  f"avg_move_time={result['avg_move_time_sec']:.4f}s")

    print(f"[{agent_name}] saved per-game data to {detail_path}")
    print(f"[{agent_name}] saved placement heatmap to {heatmap_path}")
    print(f"[{agent_name}] saved density data to {density_path}")
    print(f"[{agent_name}] saved occupancy sum to {occupancy_path}")


def write_summary_csv(all_stats, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ['agent', 'n', 'mean_score', 'median_score', 'mean_excl_top_outlier',
                  'min_score', 'max_score', 'stdev', 'ci95_low', 'ci95_high', 'mean_turns',
                  'mean_lines_per_game', 'mean_decision_time_sec']
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in all_stats:
            writer.writerow([row[k] for k in fieldnames])


def stats_row_from_detail_csv(agent_name, games, results_dir):
    path = os.path.join(results_dir, f'final_benchmark_{agent_name}_{games}games.csv')
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    scores = [int(r['final_score']) for r in rows]
    stats = summary_stats(scores)
    return {
        'agent': agent_name,
        'n': stats['n'],
        'mean_score': stats['mean'],
        'median_score': stats['median'],
        'mean_excl_top_outlier': stats['mean_excl_top_outlier'],
        'min_score': stats['min'],
        'max_score': stats['max'],
        'stdev': stats['stdev'],
        'ci95_low': stats['ci95_low'],
        'ci95_high': stats['ci95_high'],
        'mean_turns': sum(int(r['hands_played']) for r in rows) / len(rows),
        'mean_lines_per_game': sum(int(r['lines_cleared']) for r in rows) / len(rows),
        'mean_decision_time_sec': sum(float(r['avg_move_time_sec']) for r in rows) / len(rows),
    }


def print_progress(games, results_dir, agents='random,greedy,beam,mcts,dqn,dqnsearch'):
    print(f"Progress toward {games} games/agent:")
    for agent_name in agents.split(','):
        detail_path = os.path.join(results_dir, f'final_benchmark_{agent_name}_{games}games.csv')
        n_done = len(_completed_seeds(detail_path))
        status = 'COMPLETE' if n_done >= games else 'in progress'
        print(f"  {agent_name:>8}: {n_done}/{games} ({status})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=500)
    p.add_argument('--seed-base', type=int, default=5000)
    p.add_argument('--agents', type=str, default='random,greedy,beam,mcts,dqn,dqnsearch',
                   help='comma-separated subset of agents to run, in order')
    p.add_argument('--recompute-summary-only', action='store_true',
                   help='skip gameplay entirely; rebuild final_benchmark_summary.csv from existing per-game CSVs')
    p.add_argument('--progress', action='store_true',
                   help='print how many games/agent are already completed, then exit (no gameplay)')
    args = p.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    agent_order = args.agents.split(',')

    if args.progress:
        print_progress(args.games, results_dir, args.agents)
        return

    all_stats = []
    if args.recompute_summary_only:
        for agent_name in agent_order:
            row = stats_row_from_detail_csv(agent_name, args.games, results_dir)
            all_stats.append(row)
            print(f"[{agent_name}] recomputed from existing CSV | "
                  f"mean={row['mean_score']:.1f} median={row['median_score']:.1f} "
                  f"min={row['min_score']:.0f} max={row['max_score']:.0f}")
    else:
        all_agents = build_agents()
        for agent_name in agent_order:
            agent = all_agents[agent_name]
            t0 = time.time()
            run_agent_benchmark(agent_name, agent, args.games, args.seed_base, results_dir)
            elapsed = time.time() - t0
            row = stats_row_from_detail_csv(agent_name, args.games, results_dir)
            all_stats.append(row)
            print(f"[{agent_name}] {elapsed:.1f}s this run | "
                  f"mean={row['mean_score']:.1f} median={row['median_score']:.1f} "
                  f"min={row['min_score']:.0f} max={row['max_score']:.0f} (n={row['n']})\n")

    summary_path = os.path.join(results_dir, 'final_benchmark_summary.csv')
    write_summary_csv(all_stats, summary_path)
    print(f"Saved consolidated summary to {summary_path}")


if __name__ == '__main__':
    main()
