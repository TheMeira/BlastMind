import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ai.dqn import DQNNet, best_placement, state_value, BOARD_N, PIECE_N, GAMMA
from train_dqn import compute_loss, compute_loss_per, V_CLIP_MAX


def _make_net_and_optimizer():
    online = DQNNet()
    target = DQNNet()
    target.load_state_dict(online.state_dict())
    optimizer = optim.Adam(online.parameters(), lr=5e-5)
    return online, target, optimizer


def _dummy_batch(n=4):
    board = torch.randn(n, 5, BOARD_N, BOARD_N)
    pieces = torch.randn(n, 3, PIECE_N, PIECE_N)
    return board, pieces


def test_default_buffers_are_identity_transform():
    net = DQNNet()
    x = torch.tensor([1.5, -3.2, 100.0])
    assert torch.allclose(net.normalize(x), x)
    assert torch.allclose(net.unnormalize(x), x)


def test_update_popart_stats_preserves_unnormalized_output_for_held_out_inputs():
    net = DQNNet()
    board, pieces = _dummy_batch(3)
    with torch.no_grad():
        before = net.unnormalize(net(board, pieces).squeeze(1))

    targets = torch.tensor([500.0, 520.0, 480.0, 510.0])
    net.update_popart_stats(targets, beta=0.3)

    with torch.no_grad():
        after = net.unnormalize(net(board, pieces).squeeze(1))

    assert torch.allclose(before, after, atol=1e-4)
    assert not torch.allclose(net.popart_nu, torch.tensor(0.0))


def test_calibration_style_hard_set_reproduces_old_checkpoint_output():
    net = DQNNet()
    board, pieces = _dummy_batch(3)
    with torch.no_grad():
        before = net.unnormalize(net(board, pieces).squeeze(1))

    calib_targets = torch.tensor([2900.0, 2950.0, 2980.0, 2920.0, 2905.0])
    net.update_popart_stats(calib_targets, beta=1.0)

    assert net.popart_nu.item() == pytest.approx(calib_targets.mean().item(), rel=1e-4)

    with torch.no_grad():
        after = net.unnormalize(net(board, pieces).squeeze(1))

    assert torch.allclose(before, after, atol=1e-3, rtol=1e-2)


def test_old_checkpoint_loads_with_strict_false_and_keeps_identity_defaults():
    donor = DQNNet()
    old_format_dict = {k: v for k, v in donor.state_dict().items() if 'popart' not in k}
    assert 'popart_nu' not in old_format_dict

    net = DQNNet()
    net.load_state_dict(old_format_dict, strict=False)
    assert net.popart_nu.item() == 0.0
    assert net.popart_omega.item() == 1.0

    with pytest.raises(RuntimeError):
        DQNNet().load_state_dict(old_format_dict, strict=True)


def test_target_sync_carries_popart_buffers():
    online, target, _ = _make_net_and_optimizer()
    online.update_popart_stats(torch.tensor([1000.0, 1100.0, 900.0]), beta=1.0)

    assert not torch.allclose(online.popart_nu, target.popart_nu)

    target.load_state_dict(online.state_dict())
    assert torch.allclose(online.popart_nu, target.popart_nu)
    assert torch.allclose(online.popart_omega, target.popart_omega)


def test_ema_update_matches_expected_moments():
    net = DQNNet()
    beta = 0.25
    targets = torch.tensor([10.0, 20.0, 30.0])

    expected_nu = (1 - beta) * 0.0 + beta * targets.mean().item()
    expected_omega = (1 - beta) * 1.0 + beta * (targets ** 2).mean().item()

    net.update_popart_stats(targets, beta=beta)

    assert net.popart_nu.item() == pytest.approx(expected_nu, rel=1e-5)
    assert net.popart_omega.item() == pytest.approx(expected_omega, rel=1e-5)


def test_sigma_floor_prevents_blowup():
    net = DQNNet()
    targets = torch.full((8,), 5.0)
    net.update_popart_stats(targets, beta=1.0)

    for p in net.fc[-1].parameters():
        assert torch.isfinite(p).all()
    sigma = net._popart_sigma()
    assert sigma.item() >= 1e-2 - 1e-9


def test_value_clip_bypassed_when_popart_active():
    online, target, optimizer = _make_net_and_optimizer()
    board, pieces = _dummy_batch(4)
    next_board, next_pieces = _dummy_batch(4)

    huge_reward = 50_000.0
    batch = [
        (board[i], pieces[i], huge_reward, next_board[i], next_pieces[i], False, GAMMA)
        for i in range(4)
    ]

    nu_before = online.popart_nu.item()
    loss = compute_loss(online, target, batch, 'cpu', value_clip=V_CLIP_MAX, popart=True, popart_beta=1.0)
    assert torch.isfinite(loss)
    assert online.popart_nu.item() > nu_before + 1000


def test_per_td_errors_use_normalized_space():
    online, target, optimizer = _make_net_and_optimizer()
    board, pieces = _dummy_batch(4)
    next_board, next_pieces = _dummy_batch(4)
    batch = [
        (board[i], pieces[i], 100.0, next_board[i], next_pieces[i], False, GAMMA)
        for i in range(4)
    ]
    weights = [1.0, 1.0, 1.0, 1.0]

    with torch.no_grad():
        v_next_real = target.unnormalize(target(next_board, next_pieces).squeeze(1))
        target_real = torch.tensor([100.0] * 4) + GAMMA * v_next_real

    online.update_popart_stats(target_real, beta=1.0)
    target_norm_expected = online.normalize(target_real)
    with torch.no_grad():
        v_pred_norm_expected = online(board, pieces).squeeze(1)
    expected_td = (target_norm_expected - v_pred_norm_expected).abs().numpy()

    online2, target2, _ = _make_net_and_optimizer()
    online2.load_state_dict(online.state_dict())
    target2.load_state_dict(target.state_dict())
    _, td_errors = compute_loss_per(online2, target2, batch, weights, 'cpu', popart=True, popart_beta=1.0)

    assert td_errors == pytest.approx(expected_td, rel=1e-3)


def test_best_placement_and_state_value_unnormalize():
    net = DQNNet()
    net.eval()
    net.update_popart_stats(torch.full((4,), 500.0), beta=1.0)
    assert net.popart_nu.item() != 0.0 or net.popart_omega.item() != 1.0

    import numpy as np
    grid = np.zeros((BOARD_N, BOARD_N), dtype=np.int8)

    v = state_value(net, 'cpu', grid, 0, 0)

    with torch.no_grad():
        board_t = torch.zeros(1, 5, BOARD_N, BOARD_N)
        board_t[0, 0] = torch.from_numpy(grid.astype('float32'))
        pieces_t = torch.zeros(1, 3, PIECE_N, PIECE_N)
        raw = net(board_t, pieces_t).squeeze(1)
        expected = net.unnormalize(raw).item()

    assert v == pytest.approx(expected, rel=1e-3)


def test_compute_loss_uses_online_stats_not_target_stats():
    online, target, optimizer = _make_net_and_optimizer()
    online.update_popart_stats(torch.tensor([200.0, 210.0, 190.0]), beta=1.0)
    target.update_popart_stats(torch.tensor([9000.0, 9100.0, 8900.0]), beta=1.0)

    assert online.popart_nu.item() != target.popart_nu.item()

    board, pieces = _dummy_batch(4)
    next_board, next_pieces = _dummy_batch(4)
    batch = [
        (board[i], pieces[i], 5.0, next_board[i], next_pieces[i], False, GAMMA)
        for i in range(4)
    ]

    online_copy = DQNNet()
    online_copy.load_state_dict(online.state_dict())
    target_copy = DQNNet()
    target_copy.load_state_dict(target.state_dict())

    with torch.no_grad():
        v_next_real = target_copy.unnormalize(target_copy(next_board, next_pieces).squeeze(1))
        target_real = torch.tensor([5.0] * 4) + GAMMA * v_next_real
    online_copy.update_popart_stats(target_real, beta=1.0)
    expected_target_norm = online_copy.normalize(target_real)

    online2 = DQNNet()
    online2.load_state_dict(online.state_dict())
    target2 = DQNNet()
    target2.load_state_dict(target.state_dict())
    with torch.no_grad():
        v_next_real_2 = target2.unnormalize(target2(next_board, next_pieces).squeeze(1))
        target_real_2 = torch.tensor([5.0] * 4) + GAMMA * v_next_real_2
    online2.update_popart_stats(target_real_2, beta=1.0)
    with torch.no_grad():
        v_pred_norm_2 = online2(board, pieces).squeeze(1)
    loss_manual = nn.functional.smooth_l1_loss(v_pred_norm_2, expected_target_norm)

    online3 = DQNNet()
    online3.load_state_dict(online.state_dict())
    target3 = DQNNet()
    target3.load_state_dict(target.state_dict())
    loss_actual = compute_loss(online3, target3, batch, 'cpu', popart=True, popart_beta=1.0)

    assert loss_actual.item() == pytest.approx(loss_manual.item(), rel=1e-3)
