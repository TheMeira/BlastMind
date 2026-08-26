import argparse
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from src.ai.beam import BeamAgent
from src.ai.greedy import GreedyAgent
from src.game.game_engine import GameEngine
from src.game.pieces import PIECES, PIECE_IDS

_G = {}


def init_worker(beam_width, k_candidates):
    _G['beam'] = BeamAgent(beam_width=beam_width, lookahead_depth=0,
                           search_orderings=True, seed=42)
    _G['scorer'] = GreedyAgent()
    _G['k'] = k_candidates


def fast_rollout(grid, combo, pwc, hands, scorer):
    g, cb, pc = grid, combo, pwc
    total = 0.0
    for hand in hands:
        for pid in hand:
            piece = PIECES[pid]
            ph, pw = piece.shape
            best_val = float('-inf')
            best = None
            for row, col in scorer._valid(g, piece, ph, pw):
                ng, add, ncb, npc = scorer._place(g, piece, ph, pw, row, col, cb, pc)
                val = scorer._eval(ng, add, 0)
                if val > best_val:
                    best_val = val
                    best = (ng, add, ncb, npc)
            if best is None:
                return total
            g, add, cb, pc = best
            total += add
    return total


def run_game(args):
    gi, n_rollouts, horizon, max_decisions, stride = args
    beam = _G['beam']
    scorer = _G['scorer']
    k = _G['k']
    seq_rng = random.Random(900000 + gi)

    engine = GameEngine(seed=20000 + gi)
    state = engine.new_game()
    out = []
    hand_idx = 0

    while not state.game_over and len(out) < max_decisions:
        if hand_idx % stride == 0:
            grid = state.board.grid.copy()
            finalists = beam._topk_turn_orderings(grid, state.combo_count,
                                                  state.placements_without_clear,
                                                  list(state.pieces), k)
            if len(finalists) >= 3:
                hands_sets = [[[seq_rng.choice(PIECE_IDS) for _ in range(3)]
                               for _ in range(horizon)]
                              for _ in range(n_rollouts)]
                cands = []
                for val, g, cb, pc, moves in finalists:
                    s2 = state.copy()
                    for pid, row, col in moves:
                        s2 = engine.apply_placement(s2, pid, row, col)
                    immediate = s2.score - state.score
                    futures = [fast_rollout(s2.board.grid, s2.combo_count,
                                            s2.placements_without_clear, hs, scorer)
                               for hs in hands_sets]
                    cands.append({
                        'grid': s2.board.grid.copy().astype(np.uint8),
                        'combo': int(s2.combo_count),
                        'pwc': int(s2.placements_without_clear),
                        'truth': float(immediate + float(np.mean(futures))),
                    })
                out.append({'cands': cands, 'filled': int(grid.sum()),
                            'hand': hand_idx, 'game': gi})

        moves = beam.choose_moves(state, engine)
        if not moves:
            break
        for pid, row, col in moves:
            state = engine.apply_placement(state, pid, row, col)
        if not state.pieces:
            state = engine.start_new_turn(state)
        hand_idx += 1

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=200)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--rollouts', type=int, default=6)
    p.add_argument('--horizon', type=int, default=10)
    p.add_argument('--candidates', type=int, default=6)
    p.add_argument('--beam-width', type=int, default=12)
    p.add_argument('--max-decisions', type=int, default=40)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--out', type=str, required=True)
    args = p.parse_args()

    t0 = time.time()
    records = []
    tasks = [(gi, args.rollouts, args.horizon, args.max_decisions, args.stride)
             for gi in range(args.games)]

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(args.beam_width, args.candidates)) as ex:
        for i, res in enumerate(ex.map(run_game, tasks)):
            records.extend(res)
            if (i + 1) % 5 == 0 or i == 0:
                el = time.time() - t0
                print(f"game {i+1}/{args.games} decisions={len(records)} "
                      f"elapsed={el:.0f}s rate={len(records)/max(1,el):.2f}/s", flush=True)

    with open(args.out, 'wb') as f:
        pickle.dump(records, f)
    print(f"saved {len(records)} decisions to {args.out} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
