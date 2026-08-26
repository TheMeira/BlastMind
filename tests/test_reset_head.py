import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.dqn import DQNNet, reset_head, BOARD_N, PIECE_N


def _make_net_and_optimizer():
    online = DQNNet()
    target = DQNNet()
    target.load_state_dict(online.state_dict())
    optimizer = optim.Adam(online.parameters(), lr=5e-5)
    return online, target, optimizer


def _dummy_batch():
    board = torch.randn(4, 5, BOARD_N, BOARD_N)
    pieces = torch.randn(4, 3, PIECE_N, PIECE_N)
    return board, pieces


def test_conv_layers_untouched_by_reset():
    online, target, optimizer = _make_net_and_optimizer()
    board_conv_before = [p.clone() for p in online.board_conv.parameters()]
    piece_conv_before = [p.clone() for p in online.piece_conv.parameters()]

    reset_head(online, target, optimizer)

    for before, after in zip(board_conv_before, online.board_conv.parameters()):
        assert torch.equal(before, after)
    for before, after in zip(piece_conv_before, online.piece_conv.parameters()):
        assert torch.equal(before, after)


def test_fc_parameters_change_after_reset():
    online, target, optimizer = _make_net_and_optimizer()
    fc_before = [p.clone() for p in online.fc.parameters()]

    reset_head(online, target, optimizer)

    fc_after = list(online.fc.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(fc_before, fc_after))


def test_target_synced_immediately_after_reset():
    online, target, optimizer = _make_net_and_optimizer()
    reset_head(online, target, optimizer)

    for p_online, p_target in zip(online.state_dict().values(), target.state_dict().values()):
        assert torch.equal(p_online, p_target)


def test_optimizer_still_tracks_fc_after_reset():
    online, target, optimizer = _make_net_and_optimizer()
    reset_head(online, target, optimizer)

    fc_after_reset = [p.clone() for p in online.fc.parameters()]

    board, pieces = _dummy_batch()
    output = online(board, pieces)
    loss = output.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    fc_after_step = list(online.fc.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(fc_after_reset, fc_after_step))


def test_conv_optimizer_momentum_untouched_by_reset():
    online, target, optimizer = _make_net_and_optimizer()

    board, pieces = _dummy_batch()
    output = online(board, pieces)
    loss = output.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    conv_param = next(online.board_conv.parameters())
    momentum_before = optimizer.state[conv_param]['exp_avg'].clone()

    reset_head(online, target, optimizer)

    momentum_after = optimizer.state[conv_param]['exp_avg']
    assert torch.equal(momentum_before, momentum_after)


def test_fc_optimizer_state_cleared_by_reset():
    online, target, optimizer = _make_net_and_optimizer()

    board, pieces = _dummy_batch()
    output = online(board, pieces)
    loss = output.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    fc_param = next(online.fc.parameters())
    assert len(optimizer.state[fc_param]) > 0

    reset_head(online, target, optimizer)

    assert len(optimizer.state[fc_param]) == 0


def test_reset_head_at_leq_start_counter_raises(tmp_path):
    from src.ai.dqn import DQNNet as _DQNNet

    online = _DQNNet()
    target = _DQNNet()
    target.load_state_dict(online.state_dict())
    optimizer = optim.Adam(online.parameters(), lr=5e-5)
    step_counter = [0]

    bundle_path = tmp_path / 'resume.pt'
    torch.save({
        'online': online.state_dict(),
        'target': target.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step_counter': step_counter[0],
        'counter': 205000,
    }, bundle_path)

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts' / 'train_dqn.py'),
         '--resume-from', str(bundle_path), '--episodes', '205010',
         '--reset-head-at', '205000', '--version', 'test_guard'],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert 'reset-head-at' in result.stderr or 'reset-head-at' in result.stdout
