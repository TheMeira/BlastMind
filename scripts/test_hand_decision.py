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


def init_worker(ckpt_a, ckpt_b, device):
    import torch
    torch.set_num_threads(1)
    from src.ai.dqn import DQNNet
    from src.ai.greedy import GreedyAgent
    nets = {}
    for name, path in (('deployed', ckpt_a), ('ranked', ckpt_b)):
        n = DQNNet().to(device)
        n.load_state_dict(torch.load(path, map_location=device), strict=False)
        n.eval()
        nets[name] = n
    _G['nets'] = nets
    _G['device'] = device
    _G['scorer'] = GreedyAgent()
    _G['torch'] = torch


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


def true_value(grid, combo, pwc, hands_sets, scorer):
    return float(np.mean([fast_rollout(grid, combo, pwc, hs, scorer) for hs in hands_sets]))


def apply_moves(grid, combo, pwc, moves):
    from src.ai.dqn import simulate_placement
    g, cb, pc = grid, combo, pwc
    total = 0.0
    for pid, r, c in moves:
        g, sg, cb, pc = simulate_placement(g, pid, r, c, cb, pc)
        total += float(sg) + PLACE_BONUS
    return g, cb, pc, total


def run_game(args):
    gi, n_rollouts, horizon, max_decisions = args
    from src.ai.dqn import best_order_placement
    from src.game.piece_generator import PieceGenerator
    from src.game.pieces import PIECE_IDS

    nets = _G['nets']
    device = _G['device']
    scorer = _G['scorer']
    seq_rng = random.Random(660000 + gi)
    gen = PieceGenerator(seed=41000 + gi)

    grid = np.zeros((8, 8), dtype=np.int8)
    combo, pwc = 0, 0
    out = []

    while len(out) < max_decisions:
        hand = gen.generate(3)
        hands_sets = [[[seq_rng.choice(PIECE_IDS) for _ in range(3)]
                       for _ in range(horizon)]
                      for _ in range(n_rollouts)]

        rec = {}
        chosen = {}
        for name, net in nets.items():
            moves = best_order_placement(net, device, grid, list(hand), combo, pwc)
            if not moves:
                rec = None
                break
            g2, cb2, pc2, imm = apply_moves(grid, combo, pwc, moves)
            fut = true_value(g2, cb2, pc2, hands_sets, scorer)
            rec[name] = imm + GAMMA * fut
            rec[name + '_imm'] = imm
            chosen[name] = (moves, g2, cb2, pc2)
        if rec is None:
            break

        rec['filled'] = int(grid.sum())
        rec['same'] = int(chosen['deployed'][0] == chosen['ranked'][0])
        out.append(rec)

        moves, grid, combo, pwc = chosen['deployed']
        if len(out) >= max_decisions:
            break

    return out


def ci(v, seed=0):
    rs = np.random.default_rng(seed)
    m = [rs.choice(v, len(v), replace=True).mean() for _ in range(6000)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=120)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--rollouts', type=int, default=10)
    p.add_argument('--horizon', type=int, default=30)
    p.add_argument('--max-decisions', type=int, default=12)
    p.add_argument('--deployed', type=str, required=True)
    p.add_argument('--ranked', type=str, required=True)
    p.add_argument('--device', type=str, default='cpu')
    args = p.parse_args()

    t0 = time.time()
    recs = []
    tasks = [(gi, args.rollouts, args.horizon, args.max_decisions)
             for gi in range(args.games)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(args.deployed, args.ranked, args.device)) as ex:
        for res in ex.map(run_game, tasks):
            recs.extend(res)
    print(f"collected {len(recs)} hand decisions ({time.time()-t0:.0f}s)")

    d = np.array([r['deployed'] for r in recs])
    k = np.array([r['ranked'] for r in recs])
    same = np.array([r['same'] for r in recs])
    diff = k - d

    print(f"identical hand choice: {same.mean():.1%} of decisions")
    print()
    print(f"deployed  true hand value: mean={d.mean():.2f}")
    print(f"ranked    true hand value: mean={k.mean():.2f}")
    lo, hi = ci(diff)
    print(f"mean delta (ranked - deployed) = {diff.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")

    md = diff[same == 0]
    if len(md):
        lo2, hi2 = ci(md)
        print(f"\nrestricted to decisions where they DIFFER (n={len(md)}):")
        print(f"  mean delta = {md.mean():+.2f}  95% CI [{lo2:+.2f}, {hi2:+.2f}]")
        print(f"  ranked better in {float((md > 0).mean()):.1%} of differing decisions")

    di = np.array([r['deployed_imm'] for r in recs])
    ki = np.array([r['ranked_imm'] for r in recs])
    print(f"\nimmediate hand score: deployed={di.mean():.2f} ranked={ki.mean():.2f} "
          f"(delta {ki.mean()-di.mean():+.2f})")


if __name__ == '__main__':
    main()
