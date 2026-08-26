import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

GAMMA = 0.99
PLACE_BONUS = 0.5
GAME_OVER_PENALTY = 50.0

_G = {}


def init_worker(ckpt, device):
    import torch
    torch.set_num_threads(1)
    from src.ai.dqn import DQNNet
    from src.ai.greedy import GreedyAgent
    net = DQNNet().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    net.eval()
    _G['net'] = net
    _G['device'] = device
    _G['scorer'] = GreedyAgent()


def fast_rollout(grid, combo, pwc, hands, scorer):
    from src.game.pieces import PIECES
    g, cb, pc = grid, combo, pwc
    disc, total, alive = 1.0, 0.0, True
    for hand in hands:
        for pid in hand:
            piece = PIECES[pid]
            ph, pw = piece.shape
            best_val, best = float('-inf'), None
            for row, col in scorer._valid(g, piece, ph, pw):
                ng, add, ncb, npc = scorer._place(g, piece, ph, pw, row, col, cb, pc)
                v = scorer._eval(ng, add, 0)
                if v > best_val:
                    best_val, best = v, (ng, add, ncb, npc)
            if best is None:
                total += disc * (-GAME_OVER_PENALTY)
                alive = False
                break
            g, add, cb, pc = best
            total += disc * (add + PLACE_BONUS)
            disc *= GAMMA
        if not alive:
            break
    return total


def run_game(args):
    gi, weights, n_rollouts, horizon, max_decisions = args
    from src.ai.dqn import best_order_placement, simulate_placement
    from src.game.piece_generator import PieceGenerator
    from src.game.pieces import PIECE_IDS

    net = _G['net']
    device = _G['device']
    scorer = _G['scorer']
    seq_rng = random.Random(310000 + gi)
    gen = PieceGenerator(seed=52000 + gi)

    grid = np.zeros((8, 8), dtype=np.int8)
    combo, pwc = 0, 0
    out = []

    while len(out) < max_decisions:
        hand = gen.generate(3)
        hands_sets = [[[seq_rng.choice(PIECE_IDS) for _ in range(3)]
                       for _ in range(horizon)]
                      for _ in range(n_rollouts)]

        rec = {}
        base_moves = None
        for w in weights:
            moves = best_order_placement(net, device, grid, list(hand), combo, pwc, w)
            if not moves:
                rec = None
                break
            g, cb, pc = grid, combo, pwc
            imm = 0.0
            for pid, r, c in moves:
                g, sg, cb, pc = simulate_placement(g, pid, r, c, cb, pc)
                imm += float(sg) + PLACE_BONUS
            fut = float(np.mean([fast_rollout(g, cb, pc, hs, scorer) for hs in hands_sets]))
            rec[w] = imm + GAMMA * fut
            if w == 1.0:
                base_moves = (moves, g, cb, pc)
        if rec is None or base_moves is None:
            break

        out.append(rec)
        _, grid, combo, pwc = base_moves

    return out


def ci(v, seed=0):
    rs = np.random.default_rng(seed)
    m = [rs.choice(v, len(v), replace=True).mean() for _ in range(6000)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--weights', type=str, default='0.25,0.5,0.75,1.0,1.5,2.0,3.0')
    p.add_argument('--games', type=int, default=100)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--rollouts', type=int, default=10)
    p.add_argument('--horizon', type=int, default=30)
    p.add_argument('--max-decisions', type=int, default=12)
    p.add_argument('--device', type=str, default='cpu')
    args = p.parse_args()

    weights = [float(x) for x in args.weights.split(',')]
    if 1.0 not in weights:
        weights.append(1.0)

    t0 = time.time()
    recs = []
    tasks = [(gi, weights, args.rollouts, args.horizon, args.max_decisions)
             for gi in range(args.games)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(args.checkpoint, args.device)) as ex:
        for res in ex.map(run_game, tasks):
            recs.extend(res)

    print(f"checkpoint: {os.path.basename(args.checkpoint)}")
    print(f"collected {len(recs)} hand decisions ({time.time()-t0:.0f}s)\n")

    base = np.array([r[1.0] for r in recs])
    print(f"{'weight':>8} {'mean hand value':>16} {'delta vs w=1.0':>16} {'95% CI':>22}")
    for w in sorted(weights):
        v = np.array([r[w] for r in recs])
        d = v - base
        lo, hi = ci(d)
        flag = '  <-- baseline' if w == 1.0 else ''
        print(f"{w:>8.2f} {v.mean():>16.2f} {d.mean():>+16.2f}   [{lo:+7.2f}, {hi:+7.2f}]{flag}")


if __name__ == '__main__':
    main()
