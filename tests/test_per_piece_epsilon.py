import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train_dqn
from train_dqn import random_single_placement, run_episode, ReplayBuffer
from src.ai.dqn import DQNNet, best_order_placement_train
from src.ai.dqn_env import BlockBlastEnv


def test_random_single_placement_picks_the_only_placeable_piece():
    grid = np.ones((8, 8), dtype=np.int8)
    grid[0, 0] = 0
    result = random_single_placement(grid, ['P02', 'P01'])
    assert result == ('P01', 0, 0)


def test_random_single_placement_all_blocked_returns_none():
    grid = np.ones((8, 8), dtype=np.int8)
    assert random_single_placement(grid, ['P01', 'P02']) is None


def test_random_single_placement_near_uniform_across_candidate_pieces():
    grid = np.ones((8, 8), dtype=np.int8)
    grid[0, :] = 0
    seen = set()
    for i in range(200):
        pid, row, col = random_single_placement(grid, ['P01', 'P02'])
        seen.add(pid)
        assert row == 0
    assert seen == {'P01', 'P02'}


def _make_training_stack():
    online = DQNNet()
    target = DQNNet()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    buffer = ReplayBuffer(capacity=1000)
    step_counter = [0]
    return online, target, optimizer, buffer, step_counter


def test_deviation_triggers_recompute_against_post_deviation_board():
    online, target, optimizer, buffer, step_counter = _make_training_stack()

    env = BlockBlastEnv(seed=1)
    env.reset()
    env.state.board.grid[:, :] = 0
    env.state.pieces = ['P01', 'P01', 'P01']
    env.reset = lambda: env.state

    fixed_deviation = ('P01', 5, 5)
    recorded_grids = []
    random_call_count = {'n': 0}

    def tracking_side_effect(net, device, grid, piece_ids, combo, pwc):
        recorded_grids.append(grid.copy())
        if len(recorded_grids) <= 2:
            return best_order_placement_train(net, device, grid, piece_ids, combo, pwc)
        return [], 0.0

    def fake_random():
        random_call_count['n'] += 1
        return 0.0 if random_call_count['n'] == 1 else 1.0

    with patch('train_dqn.random_single_placement', return_value=fixed_deviation) as mock_random_single, \
         patch('train_dqn.random.random', side_effect=fake_random), \
         patch('train_dqn.best_order_placement_train', side_effect=tracking_side_effect):
        run_episode(env, online, target, optimizer, buffer, 'cpu', epsilon=0.5, step_counter=step_counter,
                    train=True, behavior_policy='order_search', per_piece_epsilon=True)

    mock_random_single.assert_called_once()
    assert len(recorded_grids) >= 2
    assert recorded_grids[1][5, 5] == 1


def test_random_single_placement_none_mid_plan_ends_episode_not_loop():
    online, target, optimizer, buffer, step_counter = _make_training_stack()

    env = BlockBlastEnv(seed=1)

    with patch('train_dqn.random_single_placement', return_value=None), \
         patch('train_dqn.random.random', return_value=0.0):
        score, steps, loss, avg_v, grad_norm, clear_frac = run_episode(
            env, online, target, optimizer, buffer, 'cpu', epsilon=1.0, step_counter=step_counter,
            train=True, behavior_policy='order_search', per_piece_epsilon=True)

    assert steps == 0


def test_per_piece_epsilon_off_never_calls_random_single_placement():
    online, target, optimizer, buffer, step_counter = _make_training_stack()

    env = BlockBlastEnv(seed=1)
    env.reset()

    with patch('train_dqn.random_single_placement') as mock_random_single:
        run_episode(env, online, target, optimizer, buffer, 'cpu', epsilon=0.9, step_counter=step_counter,
                    train=True, behavior_policy='order_search', per_piece_epsilon=False)

    mock_random_single.assert_not_called()


def test_per_piece_epsilon_flag_requires_order_search_behavior_policy():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts' / 'train_dqn.py'),
         '--episodes', '10', '--behavior-policy', 'fixed', '--per-piece-epsilon', '--version', 'test_guard'],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert 'per-piece-epsilon' in result.stderr or 'per-piece-epsilon' in result.stdout


def test_per_piece_epsilon_smoke_end_to_end():
    online, target, optimizer, buffer, step_counter = _make_training_stack()
    env = BlockBlastEnv(seed=7)

    for _ in range(20):
        score, steps, loss, avg_v, grad_norm, clear_frac = run_episode(
            env, online, target, optimizer, buffer, 'cpu', epsilon=0.15, step_counter=step_counter,
            train=True, behavior_policy='order_search', per_piece_epsilon=True)
        assert steps >= 0
        assert np.isfinite(score)
