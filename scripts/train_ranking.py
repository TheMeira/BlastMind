import argparse
import os
import pickle
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn

from src.ai.dqn import DQNNet, encode_state


def encode_decision(dec, device):
    boards, pieces, truths = [], [], []
    for c in dec['cands']:
        b, p = encode_state(np.asarray(c['grid']), c.get('queue', []), c['combo'], c['pwc'])
        boards.append(b)
        pieces.append(p)
        truths.append(c['truth'])
    return (torch.stack(boards).to(device),
            torch.stack(pieces).to(device),
            torch.tensor(truths, dtype=torch.float32, device=device))


def ranking_loss(v, y, margin_scale, max_margin):
    n = v.shape[0]
    if n < 2:
        return v.sum() * 0.0
    y_std = y.std()
    if y_std < 1e-6:
        return v.sum() * 0.0
    yn = (y - y.mean()) / y_std
    vn = (v - v.mean()) / (v.std() + 1e-6)

    yi = yn.unsqueeze(1) - yn.unsqueeze(0)
    vi = vn.unsqueeze(1) - vn.unsqueeze(0)
    mask = torch.triu(torch.ones(n, n, device=v.device, dtype=torch.bool), diagonal=1)
    yd = yi[mask]
    vd = vi[mask]

    margins = torch.clamp(margin_scale * yd.abs(), max=max_margin)
    viol = torch.clamp(margins - torch.sign(yd) * vd, min=0.0)
    return viol.mean()


def evaluate(net, data, device):
    net.eval()
    accs, regrets, tops = [], [], []
    with torch.no_grad():
        for dec in data:
            b, p, y = encode_decision(dec, device)
            v = net.unnormalize(net(b, p).squeeze(1))
            yv = y.cpu().numpy()
            vv = v.cpu().numpy()
            n = len(yv)
            a = t = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if yv[i] == yv[j]:
                        continue
                    t += 1
                    if (vv[i] - vv[j]) * (yv[i] - yv[j]) > 0:
                        a += 1
            if t:
                accs.append(a / t)
            bi = int(np.argmax(vv))
            best, worst = float(np.max(yv)), float(np.min(yv))
            regrets.append((best - yv[bi]) / (best - worst) if best > worst else 0.0)
            tops.append(1.0 if bi == int(np.argmax(yv)) else 0.0)
    net.train()
    return float(np.mean(accs)), float(np.mean(tops)), float(np.mean(regrets))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--init-checkpoint', type=str, default=None)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lambda-rank', type=float, default=1.0)
    p.add_argument('--alpha-reg', type=float, default=1.0)
    p.add_argument('--margin-scale', type=float, default=0.5)
    p.add_argument('--max-margin', type=float, default=1.0)
    p.add_argument('--val-frac', type=float, default=0.15)
    p.add_argument('--decisions-per-step', type=int, default=8)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--match-scale', action='store_true', default=True)
    p.add_argument('--no-match-scale', dest='match_scale', action='store_false')
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.data, 'rb') as f:
        records = pickle.load(f)
    random.shuffle(records)
    n_val = max(20, int(len(records) * args.val_frac))
    val, train = records[:n_val], records[n_val:]
    print(f"train decisions={len(train)} val decisions={len(val)} device={device}")

    net = DQNNet().to(device)
    if args.init_checkpoint:
        net.load_state_dict(torch.load(args.init_checkpoint, map_location=device), strict=False)
        print(f"initialised from {args.init_checkpoint}")
    net.train()

    if args.match_scale:
        net.eval()
        all_t, all_p = [], []
        with torch.no_grad():
            for dec in train:
                b, pc, y = encode_decision(dec, device)
                v = net.unnormalize(net(b, pc).squeeze(1))
                all_t.append(y.cpu().numpy())
                all_p.append(v.cpu().numpy())
        net.train()
        tt = np.concatenate(all_t)
        pp = np.concatenate(all_p)
        scale_a = float(pp.std() / (tt.std() + 1e-9))
        scale_b = float(pp.mean() - scale_a * tt.mean())
        print(f"truth->value scale map: v = {scale_a:.4f} * truth + {scale_b:.2f} "
              f"(truth mean={tt.mean():.1f}, net pred mean={pp.mean():.1f})")
        for dec in records:
            for c in dec['cands']:
                c['truth'] = scale_a * c['truth'] + scale_b

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    a0, t0_, r0 = evaluate(net, val, device)
    print(f"[epoch 0 / init] val pairwise_acc={a0:.4f} top1={t0_:.4f} regret={r0:.4f}")

    best_regret = r0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        run_reg = run_rank = 0.0
        nsteps = 0
        for i in range(0, len(train), args.decisions_per_step):
            chunk = train[i:i + args.decisions_per_step]
            opt.zero_grad()
            loss_reg_tot = 0.0
            loss_rank_tot = 0.0
            for dec in chunk:
                b, pc, y = encode_decision(dec, device)
                v = net.unnormalize(net(b, pc).squeeze(1))
                loss_reg_tot = loss_reg_tot + nn.functional.smooth_l1_loss(v, y)
                loss_rank_tot = loss_rank_tot + ranking_loss(v, y, args.margin_scale,
                                                             args.max_margin)
            loss_reg_tot = loss_reg_tot / len(chunk)
            loss_rank_tot = loss_rank_tot / len(chunk)
            loss = args.alpha_reg * loss_reg_tot + args.lambda_rank * loss_rank_tot
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            run_reg += float(loss_reg_tot.detach())
            run_rank += float(loss_rank_tot.detach())
            nsteps += 1

        acc, top1, regret = evaluate(net, val, device)
        print(f"[epoch {epoch}] reg={run_reg/nsteps:.2f} rank={run_rank/nsteps:.4f} "
              f"| val pairwise_acc={acc:.4f} top1={top1:.4f} regret={regret:.4f} "
              f"({time.time()-t_start:.0f}s)", flush=True)

        if regret < best_regret:
            best_regret = regret
            torch.save(net.state_dict(), args.out)
            print(f"  saved improved checkpoint (regret {regret:.4f}) -> {args.out}")

    print(f"done. best val regret={best_regret:.4f} (init {r0:.4f})")


if __name__ == '__main__':
    main()
