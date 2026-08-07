import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.ppo import PPONet, masked_distribution, valid_action_mask
from src.ai.dqn_env import BlockBlastEnv

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
EPOCHS_PER_UPDATE = 4
MINIBATCH_SIZE = 256
VALUE_COEF = 0.5
ENTROPY_COEF = 0.05
MAX_GRAD_NORM = 0.5

PLACE_BONUS = 0.5
GAME_OVER_PENALTY = 50.0


def collect_rollout(env, net, device, n_steps):
    boards, pieces, actions, log_probs, values, rewards, dones, masks = [], [], [], [], [], [], [], []
    episode_scores, episode_steps = [], []

    if env.state is None or env.state.game_over:
        env.reset()

    ep_steps = 0
    for _ in range(n_steps):
        pid = env.current_piece()
        mask = valid_action_mask(env.state.board.grid, pid)

        if not mask.any():
            episode_scores.append(env.state.score)
            episode_steps.append(ep_steps)
            ep_steps = 0
            env.reset()
            continue

        board_t, pieces_t = env.observe()

        with torch.no_grad():
            logits, value = net(board_t.unsqueeze(0).to(device), pieces_t.unsqueeze(0).to(device))
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            dist = masked_distribution(logits, mask_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        action_i = int(action.item())
        row, col = divmod(action_i, 8)
        _, score_gained, done = env.step(row, col)
        ep_steps += 1

        reward = float(score_gained) + PLACE_BONUS - (GAME_OVER_PENALTY if done else 0.0)

        boards.append(board_t)
        pieces.append(pieces_t)
        actions.append(action_i)
        log_probs.append(log_prob.item())
        values.append(value.item())
        rewards.append(reward)
        dones.append(done)
        masks.append(mask)

        if done:
            episode_scores.append(env.state.score)
            episode_steps.append(ep_steps)
            ep_steps = 0
            env.reset()

    if dones[-1]:
        last_value = 0.0
    else:
        board_t, pieces_t = env.observe()
        with torch.no_grad():
            _, v = net(board_t.unsqueeze(0).to(device), pieces_t.unsqueeze(0).to(device))
        last_value = v.item()

    batch = dict(boards=boards, pieces=pieces, actions=actions, log_probs=log_probs,
                 values=values, rewards=rewards, dones=dones, masks=masks)
    return batch, episode_scores, episode_steps, last_value


def compute_gae(rewards, values, dones, last_value):
    advantages = np.zeros(len(rewards), dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        next_non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + GAMMA * next_value * next_non_terminal - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


def ppo_update(net, optimizer, batch, advantages, returns, device):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    boards_t = torch.stack(batch['boards']).to(device)
    pieces_t = torch.stack(batch['pieces']).to(device)
    actions_t = torch.tensor(batch['actions'], dtype=torch.long, device=device)
    old_log_probs_t = torch.tensor(batch['log_probs'], dtype=torch.float32, device=device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    masks_t = torch.tensor(np.stack(batch['masks']), dtype=torch.bool, device=device)

    n = len(batch['actions'])
    indices = np.arange(n)
    total_policy_loss, total_value_loss, total_entropy = 0.0, 0.0, 0.0
    n_updates = 0

    for _ in range(EPOCHS_PER_UPDATE):
        np.random.shuffle(indices)
        for start in range(0, n, MINIBATCH_SIZE):
            mb_idx = torch.tensor(indices[start:start + MINIBATCH_SIZE], dtype=torch.long, device=device)

            logits, values = net(boards_t[mb_idx], pieces_t[mb_idx])
            dist = masked_distribution(logits, masks_t[mb_idx])
            new_log_probs = dist.log_prob(actions_t[mb_idx])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_t[mb_idx])
            surr1 = ratio * advantages_t[mb_idx]
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages_t[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, returns_t[mb_idx])

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    return total_policy_loss / n_updates, total_value_loss / n_updates, total_entropy / n_updates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-steps', type=int, default=500_000)
    parser.add_argument('--rollout-steps', type=int, default=2048)
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--out', type=str, default='models')
    parser.add_argument('--results', type=str, default='results')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    net = PPONet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    env = BlockBlastEnv(seed=args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    results_dir = Path(args.results)
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / f'ppo_v{args.version}_training_log.csv'

    recent_scores, recent_steps = deque(maxlen=100), deque(maxlen=100)
    total_steps_done = 0
    rollout_idx = 0
    t0 = time.time()

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rollout', 'total_steps', 'avg_score', 'avg_steps', 'policy_loss', 'value_loss', 'entropy'])

        while total_steps_done < args.total_steps:
            batch, ep_scores, ep_steps, last_value = collect_rollout(env, net, device, args.rollout_steps)
            advantages, returns = compute_gae(batch['rewards'], batch['values'], batch['dones'], last_value)
            policy_loss, value_loss, entropy = ppo_update(net, optimizer, batch, advantages, returns, device)

            total_steps_done += args.rollout_steps
            rollout_idx += 1
            recent_scores.extend(ep_scores)
            recent_steps.extend(ep_steps)

            avg_score = float(np.mean(recent_scores)) if recent_scores else 0.0
            avg_steps = float(np.mean(recent_steps)) if recent_steps else 0.0
            writer.writerow([rollout_idx, total_steps_done, f'{avg_score:.2f}', f'{avg_steps:.2f}',
                              f'{policy_loss:.4f}', f'{value_loss:.4f}', f'{entropy:.4f}'])

            if rollout_idx % args.log_every == 0:
                f.flush()
                elapsed = time.time() - t0
                print(f'rollout {rollout_idx:>5} | steps {total_steps_done:>9} | avg_score {avg_score:>8.1f} '
                      f'| avg_steps {avg_steps:>5.1f} | pol_loss {policy_loss:.4f} | val_loss {value_loss:.4f} '
                      f'| entropy {entropy:.4f} | {elapsed:.0f}s')

            if rollout_idx % args.checkpoint_every == 0 or total_steps_done >= args.total_steps:
                ckpt_path = out_dir / f'ppo_v{args.version}_step{total_steps_done}.pt'
                torch.save(net.state_dict(), ckpt_path)
                print(f'saved checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()
