import argparse
import csv
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.dqn import DQNNet, GAMMA, best_placement, enumerate_placements, simulate_placement, encode_state
from src.ai.dqn_env import BlockBlastEnv
from src.game.game_engine import GameEngine

LR_START = 5e-5
LR_END = 5e-6
BUFFER_SIZE = 100_000
BATCH_SIZE = 32
TARGET_SYNC_STEPS = 500
EPS_START = 1.0
EPS_END = 0.10
MAX_GRAD_NORM = 0.5

PLACE_BONUS = 0.5
GAME_OVER_PENALTY = 50.0
V_CLIP_MAX = 3000.0

PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_EPS = 0.1


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=PER_ALPHA, eps=PER_EPS):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.data = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float64)
        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

    def push(self, transition):
        self.data[self.pos] = transition
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta):
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, replace=True, p=probs)

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.data[i] for i in indices]
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices, td_errors):
        new_priorities = np.abs(td_errors) + self.eps
        self.priorities[indices] = new_priorities
        self.max_priority = max(self.max_priority, float(new_priorities.max()))

    def __len__(self):
        return self.size


def epsilon_at(episode, total_episodes, eps_end=EPS_END):
    decay_episodes = max(1, total_episodes // 2)
    if episode >= decay_episodes:
        return eps_end
    return EPS_START + (eps_end - EPS_START) * (episode / decay_episodes)


def lr_at(episode, total_episodes):
    frac = min(1.0, episode / max(1, total_episodes))
    return LR_START + (LR_END - LR_START) * frac


def beta_at(episode, total_episodes):
    frac = min(1.0, episode / max(1, total_episodes))
    return PER_BETA_START + (1.0 - PER_BETA_START) * frac


def compute_loss(online, target, batch, device, value_clip=V_CLIP_MAX):
    boards      = torch.stack([b[0] for b in batch]).to(device)
    pieces      = torch.stack([b[1] for b in batch]).to(device)
    rewards     = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
    next_boards = torch.stack([b[3] for b in batch]).to(device)
    next_pieces = torch.stack([b[4] for b in batch]).to(device)
    dones       = torch.tensor([b[5] for b in batch], dtype=torch.float32, device=device)
    gammas      = torch.tensor([b[6] for b in batch], dtype=torch.float32, device=device)

    v_pred = online(boards, pieces).squeeze(1)

    with torch.no_grad():
        v_next = target(next_boards, next_pieces).squeeze(1)
        target_v = rewards + gammas * (1 - dones) * v_next
        target_v = torch.clamp(target_v, max=value_clip)

    return nn.functional.smooth_l1_loss(v_pred, target_v)


def compute_loss_per(online, target, batch, weights, device, value_clip=V_CLIP_MAX):
    boards      = torch.stack([b[0] for b in batch]).to(device)
    pieces      = torch.stack([b[1] for b in batch]).to(device)
    rewards     = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
    next_boards = torch.stack([b[3] for b in batch]).to(device)
    next_pieces = torch.stack([b[4] for b in batch]).to(device)
    dones       = torch.tensor([b[5] for b in batch], dtype=torch.float32, device=device)
    gammas      = torch.tensor([b[6] for b in batch], dtype=torch.float32, device=device)
    weights_t   = torch.tensor(weights, dtype=torch.float32, device=device)

    v_pred = online(boards, pieces).squeeze(1)

    with torch.no_grad():
        v_next = target(next_boards, next_pieces).squeeze(1)
        target_v = rewards + gammas * (1 - dones) * v_next
        target_v = torch.clamp(target_v, max=value_clip)

    td_errors = target_v - v_pred
    per_sample_loss = nn.functional.smooth_l1_loss(v_pred, target_v, reduction='none')
    weighted_loss = (weights_t * per_sample_loss).mean()

    return weighted_loss, td_errors.detach().abs().cpu().numpy()


def make_n_step_transitions(raw_steps, n_step):
    transitions = []
    T = len(raw_steps)
    for i in range(T):
        board_t, pieces_t = raw_steps[i][0], raw_steps[i][1]
        reward_sum = 0.0
        k = 0
        end_next_board, end_next_pieces, end_done = None, None, None
        for j in range(i, min(i + n_step, T)):
            reward_sum += (GAMMA ** k) * raw_steps[j][2]
            k += 1
            end_next_board, end_next_pieces, end_done = raw_steps[j][3], raw_steps[j][4], raw_steps[j][5]
            if end_done:
                break
        gamma_k = GAMMA ** k
        transitions.append((board_t, pieces_t, reward_sum, end_next_board, end_next_pieces, end_done, gamma_k))
    return transitions


def best_placement_batch(net, device, env_infos):
    all_boards, all_pieces, all_scores, all_results, counts = [], [], [], [], []
    for grid, piece_ids, combo, pwc in env_infos:
        pid = piece_ids[0]
        candidates = enumerate_placements(grid, pid)
        counts.append(len(candidates))
        queue_after = piece_ids[1:3]
        for row, col in candidates:
            next_grid, score_gained, new_combo, new_pwc = simulate_placement(grid, pid, row, col, combo, pwc)
            nb, np_ = encode_state(next_grid, queue_after, new_combo, new_pwc)
            all_boards.append(nb)
            all_pieces.append(np_)
            all_scores.append(score_gained)
            all_results.append((row, col, score_gained, new_combo, new_pwc))

    if not all_boards:
        return [None] * len(env_infos)

    with torch.no_grad():
        nb_batch = torch.stack(all_boards).to(device)
        np_batch = torch.stack(all_pieces).to(device)
        v_vals = net(nb_batch, np_batch).squeeze(1).cpu().numpy()

    combined = np.array(all_scores, dtype=np.float32) + GAMMA * v_vals

    outputs = []
    offset = 0
    for n_cand in counts:
        if n_cand == 0:
            outputs.append(None)
            continue
        local_combined = combined[offset:offset + n_cand]
        best_idx = int(np.argmax(local_combined))
        row, col, score_gained, new_combo, new_pwc = all_results[offset + best_idx]
        outputs.append((row, col, score_gained, new_combo, new_pwc, float(local_combined[best_idx])))
        offset += n_cand
    return outputs


def run_episode(env, online, target, optimizer, buffer, device, epsilon, step_counter, train=True,
                 grad_clip=False, per=False, beta=None, value_clip=V_CLIP_MAX, n_step=1):
    env.reset()
    steps = 0
    losses = []
    v_values = []
    grad_norms = []
    clear_fracs = []
    raw_steps = []

    while True:
        candidates = env.candidates()
        if not candidates:
            break
        board_t, pieces_t = env.observe()

        result = best_placement(online, device, env.state.board.grid, list(env.state.pieces),
                                 env.state.combo_count, env.state.placements_without_clear)
        row, col, _, _, _, _, combined_val = result
        v_values.append(combined_val)

        if train and random.random() < epsilon:
            row, col = random.choice(candidates)

        _, score_gained, done = env.step(row, col)
        steps += 1

        next_board_t, next_pieces_t = env.observe()
        reward = float(score_gained) + PLACE_BONUS - (GAME_OVER_PENALTY if done else 0.0)

        if train:
            raw_steps.append((board_t, pieces_t, reward, next_board_t, next_pieces_t, done))

        if done:
            break

    if train:
        for board_t, pieces_t, n_reward, nb, npc, is_done, gamma_k in make_n_step_transitions(raw_steps, n_step):
            buffer.push((board_t, pieces_t, n_reward, nb, npc, is_done, gamma_k))
            if len(buffer) >= BATCH_SIZE:
                if per:
                    batch, indices, weights = buffer.sample(BATCH_SIZE, beta)
                    loss, td_errors = compute_loss_per(online, target, batch, weights, device, value_clip)
                else:
                    batch = buffer.sample(BATCH_SIZE)
                    loss = compute_loss(online, target, batch, device, value_clip)

                batch_rewards = [b[2] for b in batch]
                clear_frac = sum(1 for r in batch_rewards if r > PLACE_BONUS + 1) / len(batch_rewards)
                clear_fracs.append(clear_frac)

                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    norm = nn.utils.clip_grad_norm_(online.parameters(), MAX_GRAD_NORM)
                    grad_norms.append(float(norm))
                optimizer.step()
                if per:
                    buffer.update_priorities(indices, td_errors)
                losses.append(loss.item())
                step_counter[0] += 1
                if step_counter[0] % TARGET_SYNC_STEPS == 0:
                    target.load_state_dict(online.state_dict())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    avg_v = float(np.mean(v_values)) if v_values else 0.0
    avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
    avg_clear_frac = float(np.mean(clear_fracs)) if clear_fracs else 0.0
    return env.state.score, steps, avg_loss, avg_v, avg_grad_norm, avg_clear_frac


def save_resume_bundle(path, online, target, optimizer, step_counter, counter):
    torch.save({
        'online': online.state_dict(),
        'target': target.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step_counter': step_counter[0],
        'counter': counter,
    }, path)


def train_parallel(args, online, target, optimizer, buffer, device, step_counter, out_dir, log_path,
                    decay_horizon, start_games_completed=0, log_mode='w'):
    n_envs = args.parallel_envs
    envs = [BlockBlastEnv(seed=(args.seed + i if args.seed is not None else None)) for i in range(n_envs)]
    for env in envs:
        env.reset()

    per = args.per
    grad_clip = args.grad_clip
    resume_path = out_dir / f'dqn_v{args.version}_resume.pt'

    env_losses = [[] for _ in range(n_envs)]
    env_v_values = [[] for _ in range(n_envs)]
    env_grad_norms = [[] for _ in range(n_envs)]
    env_clear_fracs = [[] for _ in range(n_envs)]
    env_raw_steps = [[] for _ in range(n_envs)]
    env_steps = [0] * n_envs

    scores, steps_hist = deque(maxlen=100), deque(maxlen=100)
    games_completed = start_games_completed
    t0 = time.time()

    with open(log_path, log_mode, newline='') as f:
        writer = csv.writer(f)
        if log_mode == 'w':
            writer.writerow(['episode', 'score', 'steps', 'epsilon', 'lr', 'beta', 'loss', 'avg_v',
                              'grad_norm', 'clear_frac'])

        while games_completed < args.episodes:
            epsilon = epsilon_at(games_completed, decay_horizon, args.eps_end)
            current_lr = lr_at(games_completed, decay_horizon) if args.lr_decay else LR_START
            current_beta = beta_at(games_completed, decay_horizon) if per else 0.0
            for g in optimizer.param_groups:
                g['lr'] = current_lr

            env_infos = [(env.state.board.grid, list(env.state.pieces),
                          env.state.combo_count, env.state.placements_without_clear) for env in envs]
            results = best_placement_batch(online, device, env_infos)
            board_pieces_before = [env.observe() for env in envs]

            for i, env in enumerate(envs):
                result = results[i]
                if result is None:
                    env.reset()
                    board_pieces_before[i] = env.observe()
                    info = (env.state.board.grid, list(env.state.pieces),
                            env.state.combo_count, env.state.placements_without_clear)
                    result = best_placement_batch(online, device, [info])[0]

                row, col, _, _, _, combined_val = result
                env_v_values[i].append(combined_val)

                candidates = env.candidates()
                if random.random() < epsilon:
                    row, col = random.choice(candidates)

                board_t, pieces_t = board_pieces_before[i]
                _, score_gained, done = env.step(row, col)
                env_steps[i] += 1

                next_board_t, next_pieces_t = env.observe()
                reward = float(score_gained) + PLACE_BONUS - (GAME_OVER_PENALTY if done else 0.0)
                env_raw_steps[i].append((board_t, pieces_t, reward, next_board_t, next_pieces_t, done))

                if done:
                    for bt, pt, n_reward, nb, npc, is_done, gamma_k in make_n_step_transitions(
                            env_raw_steps[i], args.n_step):
                        buffer.push((bt, pt, n_reward, nb, npc, is_done, gamma_k))
                        if len(buffer) >= BATCH_SIZE:
                            if per:
                                batch, indices, weights = buffer.sample(BATCH_SIZE, current_beta)
                                loss, td_errors = compute_loss_per(online, target, batch, weights, device,
                                                                    args.value_clip)
                            else:
                                batch = buffer.sample(BATCH_SIZE)
                                loss = compute_loss(online, target, batch, device, args.value_clip)

                            batch_rewards = [b[2] for b in batch]
                            clear_frac = sum(1 for r in batch_rewards if r > PLACE_BONUS + 1) / len(batch_rewards)
                            env_clear_fracs[i].append(clear_frac)

                            optimizer.zero_grad()
                            loss.backward()
                            if grad_clip:
                                norm = nn.utils.clip_grad_norm_(online.parameters(), MAX_GRAD_NORM)
                                env_grad_norms[i].append(float(norm))
                            optimizer.step()
                            if per:
                                buffer.update_priorities(indices, td_errors)
                            env_losses[i].append(loss.item())
                            step_counter[0] += 1
                            if step_counter[0] % TARGET_SYNC_STEPS == 0:
                                target.load_state_dict(online.state_dict())

                    games_completed += 1
                    final_score = env.state.score
                    final_steps = env_steps[i]
                    avg_loss = float(np.mean(env_losses[i])) if env_losses[i] else 0.0
                    avg_v = float(np.mean(env_v_values[i])) if env_v_values[i] else 0.0
                    avg_grad_norm = float(np.mean(env_grad_norms[i])) if env_grad_norms[i] else 0.0
                    avg_clear_frac = float(np.mean(env_clear_fracs[i])) if env_clear_fracs[i] else 0.0

                    scores.append(final_score)
                    steps_hist.append(final_steps)
                    writer.writerow([games_completed, final_score, final_steps, f'{epsilon:.4f}',
                                      f'{current_lr:.8f}', f'{current_beta:.4f}', f'{avg_loss:.4f}',
                                      f'{avg_v:.4f}', f'{avg_grad_norm:.4f}', f'{avg_clear_frac:.4f}'])

                    if games_completed % args.log_every == 0:
                        f.flush()
                        elapsed = time.time() - t0
                        print(f'ep {games_completed:>6} | avg_score {np.mean(scores):>7.1f} | '
                              f'avg_steps {np.mean(steps_hist):>5.1f} | eps {epsilon:.3f} | lr {current_lr:.2e} '
                              f'| beta {current_beta:.3f} | loss {avg_loss:.4f} | avg_v {avg_v:.2f} '
                              f'| grad_norm {avg_grad_norm:.3f} | clear_frac {avg_clear_frac:.3f} | {elapsed:.0f}s')

                    if games_completed % args.checkpoint_every == 0 or games_completed >= args.episodes:
                        ckpt_path = out_dir / f'dqn_v{args.version}_ep{games_completed}.pt'
                        torch.save(online.state_dict(), ckpt_path)
                        print(f'saved checkpoint: {ckpt_path}')
                        save_resume_bundle(resume_path, online, target, optimizer, step_counter, games_completed)
                        print(f'saved resume bundle: {resume_path}')

                    env.reset()
                    env_steps[i] = 0
                    env_losses[i] = []
                    env_v_values[i] = []
                    env_grad_norms[i] = []
                    env_clear_fracs[i] = []
                    env_raw_steps[i] = []

    return scores, steps_hist


def random_baseline(n_games, seed=0):
    steps_list = []
    for i in range(n_games):
        engine = GameEngine(seed=seed + i)
        state = engine.new_game()
        steps = 0
        while not state.game_over:
            placed_any = False
            for pid in list(state.pieces):
                placements = engine.get_valid_placements(state, pid)
                if not placements:
                    continue
                row, col = random.choice(placements)
                state = engine.apply_placement(state, pid, row, col)
                placed_any = True
                steps += 1
            if not placed_any:
                break
            if len(state.pieces) == 0:
                state = engine.start_new_turn(state)
        steps_list.append(steps)
    return float(np.mean(steps_list))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--version', type=str, default='1')
    parser.add_argument('--out', type=str, default='models')
    parser.add_argument('--results', type=str, default='results')
    parser.add_argument('--sanity', action='store_true', help='500-episode sanity check vs random baseline')
    parser.add_argument('--grad-clip', action='store_true', help='enable gradient norm clipping')
    parser.add_argument('--lr-decay', action='store_true', help='enable LR decay over the full run')
    parser.add_argument('--per', action='store_true', help='enable prioritized experience replay')
    parser.add_argument('--parallel-envs', type=int, default=1, help='number of environments to run in parallel')
    parser.add_argument('--resume-from', type=str, default=None, help='path to a resume bundle to continue from')
    parser.add_argument('--decay-episodes', type=int, default=None,
                         help='episode horizon for epsilon/lr/beta decay; defaults to --episodes if unset')
    parser.add_argument('--buffer-size', type=int, default=BUFFER_SIZE, help='replay buffer capacity (transitions)')
    parser.add_argument('--eps-end', type=float, default=EPS_END, help='epsilon floor value after decay')
    parser.add_argument('--value-clip', type=float, default=V_CLIP_MAX, help='max value for the TD bootstrap target')
    parser.add_argument('--n-step', type=int, default=1, help='number of steps to sum before bootstrapping')
    args = parser.parse_args()

    if args.sanity:
        args.episodes = 500
        args.log_every = 25
        args.checkpoint_every = 10_000

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    online = DQNNet().to(device)
    target = DQNNet().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = optim.Adam(online.parameters(), lr=LR_START)
    buffer = PrioritizedReplayBuffer(args.buffer_size) if args.per else ReplayBuffer(args.buffer_size)
    step_counter = [0]

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    results_dir = Path(args.results)
    results_dir.mkdir(exist_ok=True)
    log_path = results_dir / f'dqn_v{args.version}_training_log.csv'

    start_counter = 0
    if args.resume_from:
        resume_bundle = torch.load(args.resume_from, map_location=device)
        online.load_state_dict(resume_bundle['online'])
        target.load_state_dict(resume_bundle['target'])
        optimizer.load_state_dict(resume_bundle['optimizer'])
        step_counter[0] = resume_bundle['step_counter']
        start_counter = resume_bundle['counter']
        print(f'resumed from {args.resume_from} at counter {start_counter}')
    log_mode = 'a' if (args.resume_from and log_path.exists()) else 'w'
    decay_horizon = args.decay_episodes if args.decay_episodes else args.episodes

    if args.parallel_envs > 1:
        scores, steps_hist = train_parallel(args, online, target, optimizer, buffer, device,
                                             step_counter, out_dir, log_path, decay_horizon,
                                             start_games_completed=start_counter, log_mode=log_mode)
    else:
        env = BlockBlastEnv(seed=args.seed)
        scores, steps_hist = deque(maxlen=100), deque(maxlen=100)
        t0 = time.time()
        resume_path = out_dir / f'dqn_v{args.version}_resume.pt'

        with open(log_path, log_mode, newline='') as f:
            writer = csv.writer(f)
            if log_mode == 'w':
                writer.writerow(['episode', 'score', 'steps', 'epsilon', 'lr', 'beta', 'loss', 'avg_v',
                                  'grad_norm', 'clear_frac'])

            for ep in range(start_counter + 1, args.episodes + 1):
                epsilon = epsilon_at(ep, decay_horizon, args.eps_end)
                current_lr = lr_at(ep, decay_horizon) if args.lr_decay else LR_START
                current_beta = beta_at(ep, decay_horizon) if args.per else 0.0
                for g in optimizer.param_groups:
                    g['lr'] = current_lr
                score, steps, loss, avg_v, grad_norm, clear_frac = run_episode(
                    env, online, target, optimizer, buffer, device, epsilon, step_counter,
                    train=True, grad_clip=args.grad_clip, per=args.per, beta=current_beta,
                    value_clip=args.value_clip, n_step=args.n_step)
                scores.append(score)
                steps_hist.append(steps)
                writer.writerow([ep, score, steps, f'{epsilon:.4f}', f'{current_lr:.8f}', f'{current_beta:.4f}',
                                  f'{loss:.4f}', f'{avg_v:.4f}', f'{grad_norm:.4f}', f'{clear_frac:.4f}'])

                if ep % args.log_every == 0:
                    f.flush()
                    elapsed = time.time() - t0
                    print(f'ep {ep:>6} | avg_score {np.mean(scores):>7.1f} | avg_steps {np.mean(steps_hist):>5.1f} '
                          f'| eps {epsilon:.3f} | lr {current_lr:.2e} | beta {current_beta:.3f} | loss {loss:.4f} '
                          f'| avg_v {avg_v:.2f} | grad_norm {grad_norm:.3f} | clear_frac {clear_frac:.3f} | {elapsed:.0f}s')

                if ep % args.checkpoint_every == 0 or ep == args.episodes:
                    ckpt_path = out_dir / f'dqn_v{args.version}_ep{ep}.pt'
                    torch.save(online.state_dict(), ckpt_path)
                    print(f'saved checkpoint: {ckpt_path}')
                    save_resume_bundle(resume_path, online, target, optimizer, step_counter, ep)
                    print(f'saved resume bundle: {resume_path}')

    if args.sanity:
        print('\nrunning random-agent baseline (100 games) for comparison...')
        baseline_steps = random_baseline(100, seed=(args.seed or 0) + 10_000)
        dqn_steps = float(np.mean(steps_hist))
        print(f'random baseline avg steps: {baseline_steps:.1f}')
        print(f'DQN (last 100 ep) avg steps: {dqn_steps:.1f}')
        verdict = 'PASS' if dqn_steps > baseline_steps else 'FAIL'
        print(f'sanity check: {verdict}')


if __name__ == '__main__':
    main()
