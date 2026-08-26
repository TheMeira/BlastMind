import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_dqn import run_episode, ReplayBuffer
from src.ai.dqn import DQNNet
from src.ai.dqn_env import BlockBlastEnv


def _make_training_stack(seed=42):
    torch.manual_seed(seed)
    online = DQNNet()
    target = DQNNet()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    buffer = ReplayBuffer(capacity=1000)
    step_counter = [0]
    return online, target, optimizer, buffer, step_counter


def _run_fixed_episode(seed=123, **kwargs):
    online, target, optimizer, buffer, step_counter = _make_training_stack(seed=42)
    env = BlockBlastEnv(seed=seed)
    run_episode(env, online, target, optimizer, buffer, 'cpu', epsilon=0.0, step_counter=step_counter,
                train=True, behavior_policy='fixed', **kwargs)
    return list(buffer.buf)


def test_reward_scale_multiplies_stored_rewards_by_exact_ratio():
    full = _run_fixed_episode(reward_scale=1.0)
    scaled = _run_fixed_episode(reward_scale=0.05)

    assert len(full) == len(scaled)
    assert len(full) > 0
    for f, s in zip(full, scaled):
        assert s[2] == pytest.approx(f[2] * 0.05, rel=1e-6)


def test_reward_scale_default_matches_explicit_1_0():
    default = _run_fixed_episode()
    explicit = _run_fixed_episode(reward_scale=1.0)

    assert len(default) == len(explicit)
    assert len(default) > 0
    for d, e in zip(default, explicit):
        assert d[2] == pytest.approx(e[2], rel=1e-9)


def test_clear_frac_reflects_real_clears_under_reward_scale():
    online, target, optimizer, buffer, step_counter = _make_training_stack()

    board = torch.zeros(5, 8, 8)
    pieces = torch.zeros(3, 5, 5)
    for _ in range(31):
        buffer.push((board.clone(), pieces.clone(), 0.01, board.clone(), pieces.clone(), False, 0.99))

    env = BlockBlastEnv(seed=1)
    env.reset()
    grid = np.ones((8, 8), dtype=np.int8)
    grid[0, 7] = 0
    env.state.board.grid[:, :] = grid
    env.state.pieces = ['P01', 'P01', 'P01']
    env.reset = lambda: env.state

    result = run_episode(env, online, target, optimizer, buffer, 'cpu', epsilon=0.0, step_counter=step_counter,
                          train=True, behavior_policy='fixed', reward_scale=0.05)
    avg_clear_frac = result[5]

    assert avg_clear_frac > 0
