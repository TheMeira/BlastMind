import random

import numpy as np
import pytest
import torch

from src.ai.dqn import GAMMA, encode_state, simulate_placement
from src.ai.dqn_search import DQNSearchAgent
from src.game.game_engine import GameEngine
from src.game.pieces import PIECE_IDS, PIECES


@pytest.fixture
def engine():
    return GameEngine()


@pytest.fixture
def agent():
    return DQNSearchAgent(device='cpu', seed=1)


def dense_state(engine, rng, fill=0.45):
    state = engine.new_game()
    for r in range(8):
        for c in range(8):
            state.board.grid[r, c] = 1 if rng.random() < fill else 0
    state.pieces = [rng.choice(PIECE_IDS) for _ in range(3)]
    return state


def test_no_heuristic_dependency():
    import src.ai.dqn_search as mod
    src = open(mod.__file__).read()
    for banned in ('beam', 'Beam', 'greedy', 'Greedy', '_eval'):
        assert banned not in src


def test_moves_are_legal_and_use_each_piece_once(engine, agent):
    rng = random.Random(21)
    for _ in range(6):
        state = dense_state(engine, rng)
        moves = agent.choose_moves(state, engine)
        remaining = list(state.pieces)
        g = state.board.grid.copy()
        cb, pc = state.combo_count, state.placements_without_clear
        for pid, row, col in moves:
            assert pid in remaining
            remaining.remove(pid)
            piece = PIECES[pid]
            ph, pw = piece.shape
            assert row + ph <= 8 and col + pw <= 8
            assert not np.any(g[row:row + ph, col:col + pw] & piece)
            g, _, cb, pc = simulate_placement(g, pid, row, col, cb, pc)


def test_returns_empty_on_dead_board(engine, agent):
    state = engine.new_game()
    state.board.grid[:, :] = 1
    state.pieces = ["P01", "P02", "P04"]
    assert agent.choose_moves(state, engine) == []


def test_enumeration_is_deduplicated(engine, agent):
    rng = random.Random(4)
    state = dense_state(engine, rng, fill=0.4)
    cands = agent._enumerate(state.board.grid.copy(), state.combo_count,
                             state.placements_without_clear, list(state.pieces))
    keys = set()
    for _, g, cb, pc, _ in cands:
        key = (g.tobytes(), cb, pc)
        assert key not in keys
        keys.add(key)


def test_enumeration_covers_all_orderings(engine, agent):
    grid = np.zeros((8, 8), dtype=np.int8)
    grid[:, :6] = 1
    cands = agent._enumerate(grid, 0, 0, ["P01", "P01", "P01"])
    assert len(cands) >= 1
    for gained, g, cb, pc, moves in cands:
        assert len(moves) == 3


def test_picks_argmax_of_immediate_plus_discounted_value(engine, agent):
    rng = random.Random(9)
    state = dense_state(engine, rng, fill=0.45)
    grid = state.board.grid.copy()
    cands = agent._enumerate(grid, state.combo_count, state.placements_without_clear,
                             list(state.pieces))
    if len(cands) < 2:
        pytest.skip("too few candidates")

    qualities = []
    for gained, g, cb, pc, _ in cands:
        b, p = encode_state(g, [], cb, pc)
        with torch.no_grad():
            v = agent.net.unnormalize(agent.net(b.unsqueeze(0), p.unsqueeze(0)).squeeze(1))
        qualities.append(float(gained) + GAMMA * float(v.item()))

    expected = cands[int(np.argmax(qualities))][4]
    assert agent.choose_moves(state, engine) == expected


def test_value_weight_zero_maximises_immediate_score(engine):
    a = DQNSearchAgent(device='cpu', value_weight=0.0, seed=2)
    rng = random.Random(13)
    state = dense_state(engine, rng, fill=0.45)
    grid = state.board.grid.copy()
    cands = a._enumerate(grid, state.combo_count, state.placements_without_clear,
                         list(state.pieces))
    if len(cands) < 2:
        pytest.skip("too few candidates")
    best_gain = max(c[0] for c in cands)
    chosen = a.choose_moves(state, engine)
    gain_of_chosen = next(c[0] for c in cands if c[4] == chosen)
    assert gain_of_chosen == pytest.approx(best_gain)


def test_batching_matches_single_pass(engine):
    big = DQNSearchAgent(device='cpu', batch_size=4096, seed=3)
    small = DQNSearchAgent(device='cpu', batch_size=7, seed=3)
    small.net.load_state_dict(big.net.state_dict())
    rng = random.Random(17)
    for _ in range(3):
        state = dense_state(engine, rng, fill=0.45)
        assert big.choose_moves(state.copy(), engine) == small.choose_moves(state.copy(), engine)


def test_subsample_cap_is_respected(engine):
    a = DQNSearchAgent(device='cpu', max_candidates=5, seed=4)
    rng = random.Random(23)
    state = dense_state(engine, rng, fill=0.3)
    moves = a.choose_moves(state, engine)
    assert moves
    assert len(moves) == 3
