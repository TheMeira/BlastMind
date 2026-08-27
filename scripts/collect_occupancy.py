import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from src.game.game_engine import GameEngine
from run_final_benchmark import build_agents, _random_choose_moves

_BOARD = 8


def collect_agent(agent_name, agent, games, seed_base, results_dir):
    out_path = os.path.join(results_dir, f'final_benchmark_{agent_name}_occupancy.npy')
    accum = np.zeros((_BOARD, _BOARD), dtype=np.float64)
    games_done = 0

    for i in range(games):
        seed = seed_base + i
        engine = GameEngine(seed=seed)
        state = engine.new_game()
        game_occupancy = np.zeros((_BOARD, _BOARD), dtype=np.float64)
        turns = 0

        while not state.game_over:
            if agent_name == 'random':
                moves = _random_choose_moves(agent, state, engine)
            else:
                moves = agent.choose_moves(state, engine)
            if not moves:
                break

            for pid, row, col in moves:
                state = engine.apply_placement(state, pid, row, col)

            game_occupancy += state.board.grid
            turns += 1

            if len(state.pieces) == 0:
                state = engine.start_new_turn(state)

        if turns:
            accum += game_occupancy / turns
            games_done += 1

        if (i + 1) % 5 == 0 or i == 0:
            print(f"[{agent_name}] game {i + 1}/{games} turns={turns}", flush=True)

    occupancy = accum / max(1, games_done)
    np.save(out_path, occupancy)
    print(f"[{agent_name}] mean occupancy={occupancy.mean():.3f} "
          f"min={occupancy.min():.3f} max={occupancy.max():.3f} -> {out_path}")
    return occupancy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--agents', type=str,
                   default='random,greedy,beam,mcts,dqn')
    p.add_argument('--games', type=int, default=40)
    p.add_argument('--seed-base', type=int, default=5000)
    args = p.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    all_agents = build_agents()

    for name in [a.strip() for a in args.agents.split(',') if a.strip()]:
        if name not in all_agents:
            print(f"[{name}] unknown agent, skipping")
            continue
        collect_agent(name, all_agents[name], args.games, args.seed_base, results_dir)


if __name__ == '__main__':
    main()
