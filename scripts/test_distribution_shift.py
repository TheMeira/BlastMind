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
    disc, total, steps, alive = 1.0, 0.0, 0, True
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
            steps += 1
        if not alive:
            break
    return total, g, cb, pc, steps, alive


def truth_for(grid, combo, pwc, hands_sets, scorer, net, device, torch, bootstrap=True):
    from src.ai.dqn import encode_state
    outs, finals = [], []
    for hands in hands_sets:
        tot, g, cb, pc, steps, alive = fast_rollout(grid, combo, pwc, hands, scorer)
        outs.append((tot, steps, alive))
        finals.append((g, cb, pc))
    boards, pieces = [], []
    for g, cb, pc in finals:
        b, p = encode_state(g, [], cb, pc)
        boards.append(b)
        pieces.append(p)
    with torch.no_grad():
        vs = net.unnormalize(net(torch.stack(boards).to(device),
                                 torch.stack(pieces).to(device)).squeeze(1)).cpu().numpy()
    vals = []
    for (tot, steps, alive), v in zip(outs, vs):
        boot = ((GAMMA ** steps) * float(v) if alive else 0.0) if bootstrap else 0.0
        vals.append(tot + boot)
    return float(np.mean(vals))


def net_values(net, cands, device, torch):
    from src.ai.dqn import encode_state
    boards, pieces = [], []
    for c in cands:
        b, p = encode_state(c['grid'], c['queue'], c['combo'], c['pwc'])
        boards.append(b)
        pieces.append(p)
    with torch.no_grad():
        return net.unnormalize(net(torch.stack(boards).to(device),
                                   torch.stack(pieces).to(device)).squeeze(1)).cpu().numpy()


def run_game(args):
    gi, policy, n_rollouts, horizon, max_decisions, k_cand, bootstrap = args
    from src.ai.dqn import enumerate_placements, simulate_placement, best_order_placement
    from src.game.piece_generator import PieceGenerator
    from src.game.pieces import PIECE_IDS

    nets = _G['nets']
    device = _G['device']
    scorer = _G['scorer']
    torch = _G['torch']
    boot_net = nets['deployed']

    rng = random.Random(770000 + gi)
    seq_rng = random.Random(880000 + gi)
    gen = PieceGenerator(seed=30000 + gi)

    grid = np.zeros((8, 8), dtype=np.int8)
    combo, pwc = 0, 0
    out = []

    while len(out) < max_decisions:
        hand = gen.generate(3)

        if policy == 'dqn':
            moves = best_order_placement(boot_net, device, grid, list(hand), combo, pwc)
            if not moves:
                break
            order = [m[0] for m in moves]
            forced = {m[0]: (m[1], m[2]) for m in moves}
        else:
            order = list(hand)
            rng.shuffle(order)
            forced = None

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
                fut = truth_for(ng, ncb, npc, hands_sets, scorer, boot_net, device, torch, bootstrap)
                cands.append({'grid': ng.copy().astype(np.int8), 'queue': list(queue_after),
                              'combo': int(ncb), 'pwc': int(npc), 'sg': float(sg),
                              'truth': float(sg) + PLACE_BONUS + GAMMA * fut})

            if len(cands) >= 3:
                rec = {'truth': [c['truth'] for c in cands],
                       'sg': [c['sg'] for c in cands],
                       'filled': int(grid.sum()), 'policy': policy}
                for name, net in nets.items():
                    vv = [float(v) for v in net_values(net, cands, device, torch)]
                    rec[name] = vv
                    rec[name + '_decision'] = [c['sg'] + PLACE_BONUS + GAMMA * v
                                               for c, v in zip(cands, vv)]
                out.append(rec)

            if forced is not None and pid in forced:
                r, c = forced[pid]
            else:
                r, c = sampled[max(range(len(cands)), key=lambda i: cands[i]['truth'])]
            grid, sg, combo, pwc = simulate_placement(grid, pid, r, c, combo, pwc)

            if len(out) >= max_decisions:
                break
        if dead:
            break
    return out


def metrics(recs, key):
    accs, regs, tops = [], [], []
    for r in recs:
        pred, truth = r[key], r['truth']
        n = len(truth)
        a = t = 0
        for i in range(n):
            for j in range(i + 1, n):
                if truth[i] == truth[j]:
                    continue
                t += 1
                if (pred[i] - pred[j]) * (truth[i] - truth[j]) > 0:
                    a += 1
        if t:
            accs.append(a / t)
        bi = int(np.argmax(pred))
        best, worst = max(truth), min(truth)
        regs.append((best - truth[bi]) / (best - worst) if best > worst else 0.0)
        tops.append(1.0 if bi == int(np.argmax(truth)) else 0.0)
    return np.array(accs), np.array(regs), np.array(tops)


def ci(v, seed=0):
    rs = np.random.default_rng(seed)
    m = [rs.choice(v, len(v), replace=True).mean() for _ in range(4000)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games', type=int, default=120)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--rollouts', type=int, default=8)
    p.add_argument('--horizon', type=int, default=6)
    p.add_argument('--candidates', type=int, default=8)
    p.add_argument('--max-decisions', type=int, default=15)
    p.add_argument('--deployed', type=str, required=True)
    p.add_argument('--ranked', type=str, required=True)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--no-bootstrap', action='store_true')
    p.add_argument('--out', type=str, default=None)
    args = p.parse_args()

    t0 = time.time()
    all_recs = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(args.deployed, args.ranked, args.device)) as ex:
        for policy in ('generator', 'dqn'):
            tasks = [(gi, policy, args.rollouts, args.horizon, args.max_decisions,
                      args.candidates, not args.no_bootstrap) for gi in range(args.games)]
            recs = []
            for res in ex.map(run_game, tasks):
                recs.extend(res)
            all_recs[policy] = recs
            print(f"[{policy}] collected {len(recs)} decisions ({time.time()-t0:.0f}s)", flush=True)

    print()
    for policy in ('generator', 'dqn'):
        recs = all_recs[policy]
        label = ("GENERATOR distribution (in-distribution for ranked net)"
                 if policy == 'generator' else
                 "DQN-OWN-PLAY distribution (what inference actually sees)")
        print(f"### {label}  n={len(recs)}")
        for name in ('deployed', 'ranked', 'deployed_decision', 'ranked_decision'):
            a, r, t = metrics(recs, name)
            alo, ahi = ci(a)
            rlo, rhi = ci(r)
            print(f"  {name:20s} acc={a.mean():.4f} [{alo:.3f},{ahi:.3f}]  "
                  f"top1={t.mean():.3f}  regret={r.mean():.4f} [{rlo:.3f},{rhi:.3f}]")
        print()

    if args.out:
        with open(args.out, 'wb') as f:
            pickle.dump(all_recs, f)


if __name__ == '__main__':
    main()
