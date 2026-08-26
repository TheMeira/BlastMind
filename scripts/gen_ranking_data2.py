import argparse
import os
import pickle
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
    _G['torch'] = torch


def fast_rollout(grid, combo, pwc, hands, scorer):
    from src.ai.dqn import simulate_placement
    from src.game.pieces import PIECES
    g, cb, pc = grid, combo, pwc
    disc = 1.0
    total = 0.0
    steps = 0
    alive = True
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
                total += disc * (-GAME_OVER_PENALTY)
                alive = False
                break
            g, add, cb, pc = best
            total += disc * (add + PLACE_BONUS)
            disc *= GAMMA
            steps += 1
        if not alive:
            break
    return total, g, cb, pc, steps, alive


def truth_for(grid, combo, pwc, hands_sets, scorer, net, device, torch):
    from src.ai.dqn import encode_state
    outs = []
    finals = []
    for hands in hands_sets:
        tot, g, cb, pc, steps, alive = fast_rollout(grid, combo, pwc, hands, scorer)
        outs.append((tot, steps, alive))
        finals.append((g, cb, pc, steps, alive))

    boards, pieces = [], []
    for g, cb, pc, steps, alive in finals:
        b, p = encode_state(g, [], cb, pc)
        boards.append(b)
        pieces.append(p)
    with torch.no_grad():
        bt = torch.stack(boards).to(device)
        pt = torch.stack(pieces).to(device)
        vs = net.unnormalize(net(bt, pt).squeeze(1)).cpu().numpy()

    vals = []
    for (tot, steps, alive), v in zip(outs, vs):
        boot = (GAMMA ** steps) * float(v) if alive else 0.0
        vals.append(tot + boot)
    return float(np.mean(vals))


def run_game(args):
    gi, n_rollouts, horizon, max_decisions, k_cand = args
    from src.ai.dqn import enumerate_placements, simulate_placement
    from src.game.piece_generator import PieceGenerator
    from src.game.pieces import PIECE_IDS

    net = _G['net']
    device = _G['device']
    scorer = _G['scorer']
    torch = _G['torch']

    rng = random.Random(770000 + gi)
    seq_rng = random.Random(880000 + gi)
    gen = PieceGenerator(seed=30000 + gi)

    grid = np.zeros((8, 8), dtype=np.int8)
    combo, pwc = 0, 0
    out = []

    while len(out) < max_decisions:
        hand = gen.generate(3)
        order = list(hand)
        rng.shuffle(order)
        dead = False

        for step, pid in enumerate(order):
            queue_after = order[step + 1:]
            placements = enumerate_placements(grid, pid)
            if not placements:
                dead = True
                break

            sampled = placements if len(placements) <= k_cand else rng.sample(placements, k_cand)
            hands_sets = [[[seq_rng.choice(PIECE_IDS) for _ in range(3)]
                           for _ in range(horizon)]
                          for _ in range(n_rollouts)]

            cands = []
            for (r, c) in sampled:
                ng, sg, ncb, npc = simulate_placement(grid, pid, r, c, combo, pwc)
                imm = float(sg) + PLACE_BONUS
                fut = truth_for(ng, ncb, npc, hands_sets, scorer, net, device, torch)
                cands.append({
                    'grid': ng.copy().astype(np.int8),
                    'queue': list(queue_after),
                    'combo': int(ncb),
                    'pwc': int(npc),
                    'truth': imm + GAMMA * fut,
                })

            if len(cands) >= 3:
                out.append({'cands': cands, 'filled': int(grid.sum()),
                            'queue_len': len(queue_after), 'game': gi})

            best = max(range(len(cands)), key=lambda i: cands[i]['truth'])
            r, c = sampled[best]
            grid, sg, combo, pwc = simulate_placement(grid, pid, r, c, combo, pwc)

            if len(out) >= max_decisions:
                break

        if dead:
            break

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=200)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--rollouts', type=int, default=8)
    p.add_argument('--horizon', type=int, default=6)
    p.add_argument('--candidates', type=int, default=8)
    p.add_argument('--max-decisions', type=int, default=30)
    p.add_argument('--bootstrap-checkpoint', type=str, required=True)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--out', type=str, required=True)
    args = p.parse_args()

    t0 = time.time()
    records = []
    tasks = [(gi, args.rollouts, args.horizon, args.max_decisions, args.candidates)
             for gi in range(args.games)]

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(args.bootstrap_checkpoint, args.device)) as ex:
        for i, res in enumerate(ex.map(run_game, tasks)):
            records.extend(res)
            if (i + 1) % 10 == 0 or i == 0:
                el = time.time() - t0
                print(f"game {i+1}/{args.games} decisions={len(records)} "
                      f"elapsed={el:.0f}s rate={len(records)/max(1,el):.2f}/s", flush=True)

    qs = {}
    for r in records:
        qs[r['queue_len']] = qs.get(r['queue_len'], 0) + 1
    print(f"queue_len distribution: {qs}")

    with open(args.out, 'wb') as f:
        pickle.dump(records, f)
    print(f"saved {len(records)} decisions to {args.out} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
