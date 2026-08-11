import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.game.game_engine import GameEngine
from src.ai.random_agent import RandomAgent
from src.ai.greedy import GreedyAgent
from src.ai.beam import BeamAgent
from src.ai.mcts import MCTSAgent
from src.ai.dqn import DQNAgent


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


def build_agent(args):
    if args.agent == 'random':
        return RandomAgent()
    if args.agent == 'greedy':
        return GreedyAgent()
    if args.agent == 'beam':
        return BeamAgent(
            beam_width=args.beam_width,
            lookahead_depth=args.lookahead_depth,
            samples=args.samples,
            search_orderings=args.beam_search_orderings,
            seed=args.agent_seed,
        )
    if args.agent == 'mcts':
        return MCTSAgent(
            n_simulations=args.n_simulations,
            time_limit=args.time_limit,
            exploration_constant=args.exploration_constant,
            rollout_policy=args.rollout_policy,
            rollout_hands=args.rollout_hands,
            max_tree_hands=args.max_tree_hands,
            root_top_k=args.root_top_k,
            enable_rave=args.enable_rave,
            rave_k=args.rave_k,
            seed=args.agent_seed,
        )
    if args.agent == 'dqn':
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --agent dqn")
        return DQNAgent(checkpoint_path=args.checkpoint, device=args.device,
                         search_orderings=args.dqn_search_orderings)
    raise ValueError(f"Unknown agent: {args.agent}")


def run_game(agent, seed, agent_type):
    engine = GameEngine(seed=seed)
    state = engine.new_game()

    hands_played = 0
    pieces_placed = 0
    max_combo = 0
    move_times = []

    while not state.game_over:
        t0 = time.perf_counter()
        if agent_type == 'random':
            moves = _random_choose_moves(agent, state, engine)
        else:
            moves = agent.choose_moves(state, engine)
        move_times.append(time.perf_counter() - t0)

        if not moves:
            state.game_over = True
            break

        for pid, row, col in moves:
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
        'max_combo': max_combo,
        'hands_played': hands_played,
        'pieces_placed': pieces_placed,
        'avg_move_time_sec': avg_move_time,
        'total_move_time_sec': sum(move_times),
    }


def run_benchmark(args):
    agent = build_agent(args)
    rows = []

    for i in range(args.games):
        seed = args.seed_base + i
        result = run_game(agent, seed, args.agent)
        rows.append(result)
        print(f"[{args.agent}] game {i + 1}/{args.games} seed={seed} "
              f"score={result['final_score']} max_combo={result['max_combo']} "
              f"hands={result['hands_played']} avg_move_time={result['avg_move_time_sec']:.4f}s")

    return rows


def write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ['seed', 'final_score', 'max_combo', 'hands_played',
                  'pieces_placed', 'avg_move_time_sec', 'total_move_time_sec']
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for row in rows:
            writer.writerow([row[k] for k in fieldnames])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--agent', required=True, choices=['random', 'greedy', 'beam', 'mcts', 'dqn'])
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--dqn-search-orderings', action='store_true')
    p.add_argument('--beam-width', type=int, default=8)
    p.add_argument('--lookahead-depth', type=int, default=1)
    p.add_argument('--samples', type=int, default=4)
    p.add_argument('--beam-search-orderings', action='store_true')
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--seed-base', type=int, default=1000)
    p.add_argument('--agent-seed', type=int, default=42)
    p.add_argument('--out', type=str, default=None)

    p.add_argument('--n-simulations', type=int, default=500)
    p.add_argument('--time-limit', type=float, default=30.0)
    p.add_argument('--exploration-constant', type=float, default=1.41421356)
    p.add_argument('--rollout-policy', type=str, default='heuristic', choices=['heuristic', 'random'])
    p.add_argument('--rollout-hands', type=int, default=2)
    p.add_argument('--max-tree-hands', type=int, default=2)
    p.add_argument('--root-top-k', type=int, default=8)
    p.add_argument('--enable-rave', action='store_true')
    p.add_argument('--rave-k', type=float, default=1000)

    p.add_argument('--label', type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    rows = run_benchmark(args)

    scores = [r['final_score'] for r in rows]
    print(f"\n{args.agent} over {args.games} games: "
          f"mean_score={sum(scores) / len(scores):.1f} "
          f"max_score={max(scores)} min_score={min(scores)}")

    label = args.label or args.agent
    out_path = args.out or os.path.join(os.path.dirname(__file__), '..', 'results',
                                         f'{label}_{args.games}games.csv')
    write_csv(rows, out_path)
    print(f"Saved to {out_path}")
