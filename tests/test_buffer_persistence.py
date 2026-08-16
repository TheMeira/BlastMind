import io
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from train_dqn import (ReplayBuffer, PrioritizedReplayBuffer, save_resume_bundle,
                        load_buffer_from_bundle)


class _Args:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size


def _make_transition(i):
    board = torch.full((5, 8, 8), float(i))
    pieces = torch.full((3, 5, 5), float(i))
    next_board = torch.full((5, 8, 8), float(i + 1))
    next_pieces = torch.full((3, 5, 5), float(i + 1))
    return (board, pieces, float(i) * 1.5, next_board, next_pieces, i % 2 == 0, 0.99)


def _transitions_equal(a, b):
    return (torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == pytest.approx(b[2])
            and torch.equal(a[3], b[3]) and torch.equal(a[4], b[4]) and a[5] == b[5]
            and a[6] == pytest.approx(b[6]))


class _Stub:
    def state_dict(self):
        return {}


def _dummy_net_state():
    return _Stub()


def test_uniform_buffer_round_trip(tmp_path):
    buf = ReplayBuffer(capacity=50)
    for i in range(30):
        buf.push(_make_transition(i))

    step_counter = [7]
    path = tmp_path / 'resume.pt'
    save_resume_bundle(str(path), _dummy_net_state(), _dummy_net_state(), _dummy_net_state(),
                        step_counter, 30, buffer=buf)
    bundle = torch.load(str(path), map_location='cpu')

    restored = ReplayBuffer(capacity=50)
    load_buffer_from_bundle(bundle, restored, _Args(buffer_size=50))

    assert len(restored) == 30
    for orig, loaded in zip(buf.buf, restored.buf):
        assert _transitions_equal(orig, loaded)


def test_uniform_buffer_capacity_change_keeps_newest(tmp_path):
    buf = ReplayBuffer(capacity=50)
    for i in range(50):
        buf.push(_make_transition(i))

    path = tmp_path / 'resume.pt'
    save_resume_bundle(str(path), _dummy_net_state(), _dummy_net_state(), _dummy_net_state(),
                        [0], 50, buffer=buf)
    bundle = torch.load(str(path), map_location='cpu')

    restored = ReplayBuffer(capacity=20)
    load_buffer_from_bundle(bundle, restored, _Args(buffer_size=20))

    assert len(restored) == 20
    assert restored.buf[-1][2] == pytest.approx(49 * 1.5)


def test_per_buffer_round_trip(tmp_path):
    buf = PrioritizedReplayBuffer(capacity=50)
    for i in range(30):
        buf.push(_make_transition(i))
    buf.update_priorities([0, 1, 2], [5.0, 3.0, 1.0])

    path = tmp_path / 'resume.pt'
    save_resume_bundle(str(path), _dummy_net_state(), _dummy_net_state(), _dummy_net_state(),
                        [0], 30, buffer=buf)
    bundle = torch.load(str(path), map_location='cpu')

    restored = PrioritizedReplayBuffer(capacity=50)
    load_buffer_from_bundle(bundle, restored, _Args(buffer_size=50))

    assert len(restored) == 30
    assert restored.pos == buf.pos
    assert restored.max_priority == buf.max_priority
    for i in range(30):
        assert _transitions_equal(buf.data[i], restored.data[i])
        assert restored.priorities[i] == pytest.approx(buf.priorities[i])


def test_per_buffer_capacity_mismatch_raises(tmp_path):
    buf = PrioritizedReplayBuffer(capacity=50)
    for i in range(10):
        buf.push(_make_transition(i))

    path = tmp_path / 'resume.pt'
    save_resume_bundle(str(path), _dummy_net_state(), _dummy_net_state(), _dummy_net_state(),
                        [0], 10, buffer=buf)
    bundle = torch.load(str(path), map_location='cpu')

    restored = PrioritizedReplayBuffer(capacity=100)
    with pytest.raises(SystemExit):
        load_buffer_from_bundle(bundle, restored, _Args(buffer_size=100))


def test_old_format_bundle_falls_back_to_empty(tmp_path):
    path = tmp_path / 'resume.pt'
    save_resume_bundle(str(path), _dummy_net_state(), _dummy_net_state(), _dummy_net_state(),
                        [0], 5)
    bundle = torch.load(str(path), map_location='cpu')

    restored = ReplayBuffer(capacity=50)
    load_buffer_from_bundle(bundle, restored, _Args(buffer_size=50))

    assert len(restored) == 0
