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


def init_worker(ckpt, device, pool_k, beam_width, samples, lookahead):
    import torch
    torch.set_num_threads(1)
    from src.ai.dqn import DQNNet
    from src.ai.beam import BeamAgent
    from src.ai.greedy import GreedyAgent
    net = DQNNet().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    net.eval()
    _G['net'] = net
    _G['device'] = device
    _G['torch'] = torch
    _G['scorer'] = GreedyAgent()
    _G['BeamAgent'] = BeamAgent
    _G['pool_k'] = pool_k
    _G['beam_width'] = beam_width
    _G['samples'] = samples
    _G['lookahead'] = lookahead


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


def immediate_of(grid, combo, pwc, moves):
    from src.ai.dqn import simulate_placement
    g, cb, pc = grid, combo, pwc
    tot = 0.0
    for pid, r, c in moves:
        g, sg, cb, pc = simulate_placement(g, pid, r, c, cb, pc)
        tot += float(sg) + PLACE_BONUS
    return g, cb, pc, tot


def run_game(args):
    gi, n_rollouts, horizon, max_decisions = args
    from src.ai.dqn import encode_state
    from src.game.piece_generator import PieceGenerator
    from src.game.pieces import PIECE_IDS

    net = _G['net']
    device = _G['device']
    torch = _G['torch']
    scorer = _G['scorer']
    pool_k = _G['pool_k']
    beam = _G['BeamAgent'](beam_width=_G['beam_width'], lookahead_depth=_G['lookahead'],
                           samples=_G['samples'], search_orderings=True, seed=1000 + gi)

    seq_rng = random.Random(220000 + gi)
    gen = PieceGenerator(seed=63000 + gi)

    grid = np.zeros((8, 8), dtype=np.int8)
    combo, pwc = 0, 0
    out = []

    while len(out) < max_decisions:
        hand = gen.generate(3)
        pool = beam._topk_turn_orderings(grid, combo, pwc, list(hand), pool_k)
        if len(pool) < 2:
            break

        hands_sets = [[[seq_rng.choice(PIECE_IDS) for _ in range(3)]
                       for _ in range(horizon)]
                      for _ in range(n_rollouts)]

        imms, finals = [], []
        for val, g, cb, pc, moves in pool:
            g2, cb2, pc2, imm = immediate_of(grid, combo, pwc, moves)
            imms.append(imm)
            finals.append((g2, cb2, pc2))

        boards, pieces = [], []
        for g2, cb2, pc2 in finals:
            b, p = encode_state(g2, [], cb2, pc2)
            boards.append(b)
            pieces.append(p)
        with torch.no_grad():
            vs = net.unnormalize(net(torch.stack(boards).to(device),
                                     torch.stack(pieces).to(device)).squeeze(1)).cpu().numpy()

        dqn_quality = [imms[i] + GAMMA * float(vs[i]) for i in range(len(pool))]
        heur_quality = [pool[i][0] for i in range(len(pool))]

        idx_dqn = int(np.argmax(dqn_quality))
        idx_heur = int(np.argmax(heur_quality))

        truths = {}
        for name, idx in (('dqn', idx_dqn), ('heur', idx_heur)):
            if idx in truths:
                continue
            g2, cb2, pc2 = finals[idx]
            fut = float(np.mean([fast_rollout(g2, cb2, pc2, hs, scorer) for hs in hands_sets]))
            truths[idx] = imms[idx] + GAMMA * fut

        rec = {'dqn': truths[idx_dqn], 'heur': truths[idx_heur],
               'same': int(idx_dqn == idx_heur), 'pool': len(pool),
               'filled': int(grid.sum())}
        out.append(rec)

        grid, combo, pwc = finals[idx_heur]

    return out


def ci(v, seed=0):
    rs = np.random.default_rng(seed)
    m = [rs.choice(v, len(v), replace=True).mean() for _ in range(6000)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--pool-k', type=int, default=50)
    p.add_argument('--beam-width', type=int, default=16)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--lookahead', type=int, default=0)
    p.add_argument('--games', type=int, default=110)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--rollouts', type=int, default=10)
    p.add_argument('--horizon', type=int, default=30)
    p.add_argument('--max-decisions', type=int, default=12)
    p.add_argument('--device', type=str, default='cpu')
    args = p.parse_args()

    t0 = time.time()
    recs = []
    tasks = [(gi, args.rollouts, args.horizon, args.max_decisions) for gi in range(args.games)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(args.checkpoint, args.device, args.pool_k,
                                       args.beam_width, args.samples, args.lookahead)) as ex:
        for res in ex.map(run_game, tasks):
            recs.extend(res)

    d = np.array([r['dqn'] for r in recs])
    h = np.array([r['heur'] for r in recs])
    same = np.array([r['same'] for r in recs])
    diff = d - h
    lo, hi = ci(diff)

    print(f"checkpoint: {os.path.basename(args.checkpoint)}  pool_k={args.pool_k}")
    print(f"collected {len(recs)} hand decisions ({time.time()-t0:.0f}s)")
    print(f"mean pool size: {np.mean([r['pool'] for r in recs]):.1f}")
    print(f"identical pick: {same.mean():.1%}\n")
    print(f"  heuristic-picked  true hand value: {h.mean():.2f}")
    print(f"  DQN-picked        true hand value: {d.mean():.2f}")
    print(f"  delta (DQN - heuristic) = {diff.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]")

    md = diff[same == 0]
    if len(md):
        lo2, hi2 = ci(md)
        print(f"\n  restricted to differing picks (n={len(md)}):")
        print(f"    delta = {md.mean():+.2f}  95% CI [{lo2:+.2f}, {hi2:+.2f}]")
        print(f"    DQN better in {float((md > 0).mean()):.1%}")


if __name__ == '__main__':
    main()
