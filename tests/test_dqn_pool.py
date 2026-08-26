import random

import numpy as np
import pytest
import torch

from src.ai.dqn import DQNNet, GAMMA, encode_state, simulate_placement
from src.ai.dqn_pool import DQNPoolAgent
from src.game.board import Board
from src.game.game_engine import GameEngine
from src.game.pieces import PIECE_IDS, PIECES


@pytest.fixture
def engine():
    return GameEngine()


@pytest.fixture
def agent():
    return DQNPoolAgent(device='cpu', pool_k=20)


def test_moves_are_legal_and_use_each_piece_once(engine, agent):
    rng = random.Random(11)
    for _ in range(8):
        state = engine.new_game()
        fill = rng.random() * 0.5
        for r in range(8):
            for c in range(8):
                state.board.grid[r, c] = 1 if rng.random() < fill else 0
        state.pieces = [rng.choice(PIECE_IDS) for _ in range(3)]

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


def test_full_hand_placed_on_open_board(engine, agent):
    state = engine.new_game()
    moves = agent.choose_moves(state, engine)
    assert len(moves) == 3
    assert sorted(pid for pid, _, _ in moves) == sorted(state.pieces)


def test_determinism_same_state(engine):
    a = DQNPoolAgent(device='cpu', pool_k=20)
    b = DQNPoolAgent(device='cpu', pool_k=20)
    a.net.load_state_dict(b.net.state_dict())
    state = engine.new_game()
    assert a.choose_moves(state.copy(), engine) == b.choose_moves(state.copy(), engine)


def test_picks_argmax_of_immediate_plus_discounted_value(engine, agent):
    rng = random.Random(5)
    state = engine.new_game()
    for r in range(8):
        for c in range(8):
            state.board.grid[r, c] = 1 if rng.random() < 0.35 else 0
    state.pieces = [rng.choice(PIECE_IDS) for _ in range(3)]

    grid = state.board.grid.copy()
    combo, pwc = state.combo_count, state.placements_without_clear
    pool = agent._proposer._topk_turn_orderings(grid, combo, pwc, list(state.pieces),
                                                 agent.pool_k)
    if len(pool) < 2:
        pytest.skip("pool too small on this board")

    qualities = []
    for _, _, _, _, moves in pool:
        g, cb, pc = grid, combo, pwc
        imm = 0.0
        for pid, r, c in moves:
            g, sg, cb, pc = simulate_placement(g, pid, r, c, cb, pc)
            imm += float(sg)
        b, p = encode_state(g, [], cb, pc)
        with torch.no_grad():
            v = agent.net.unnormalize(agent.net(b.unsqueeze(0), p.unsqueeze(0)).squeeze(1))
        qualities.append(imm + GAMMA * float(v.item()))

    expected = pool[int(np.argmax(qualities))][4]
    assert agent.choose_moves(state, engine) == expected


def test_value_weight_zero_reduces_to_immediate_score(engine):
    a = DQNPoolAgent(device='cpu', pool_k=20, value_weight=0.0)
    rng = random.Random(7)
    state = engine.new_game()
    for r in range(8):
        for c in range(8):
            state.board.grid[r, c] = 1 if rng.random() < 0.3 else 0
    state.pieces = [rng.choice(PIECE_IDS) for _ in range(3)]

    grid = state.board.grid.copy()
    combo, pwc = state.combo_count, state.placements_without_clear
    pool = a._proposer._topk_turn_orderings(grid, combo, pwc, list(state.pieces), a.pool_k)
    if len(pool) < 2:
        pytest.skip("pool too small on this board")

    imms = []
    for _, _, _, _, moves in pool:
        g, cb, pc = grid, combo, pwc
        tot = 0.0
        for pid, r, c in moves:
            g, sg, cb, pc = simulate_placement(g, pid, r, c, cb, pc)
            tot += float(sg)
        imms.append(tot)

    chosen = a.choose_moves(state, engine)
    assert imms[[m[4] for m in pool].index(chosen)] == pytest.approx(max(imms))


def test_larger_pool_never_reduces_best_available_immediate(engine):
    rng = random.Random(3)
    small = DQNPoolAgent(device='cpu', pool_k=5)
    large = DQNPoolAgent(device='cpu', pool_k=60)
    large.net.load_state_dict(small.net.state_dict())

    state = engine.new_game()
    for r in range(8):
        for c in range(8):
            state.board.grid[r, c] = 1 if rng.random() < 0.3 else 0
    state.pieces = [rng.choice(PIECE_IDS) for _ in range(3)]
    grid = state.board.grid.copy()

    p_small = small._proposer._topk_turn_orderings(grid, state.combo_count,
                                                    state.placements_without_clear,
                                                    list(state.pieces), 5)
    p_large = large._proposer._topk_turn_orderings(grid, state.combo_count,
                                                    state.placements_without_clear,
                                                    list(state.pieces), 60)
    assert len(p_large) >= len(p_small)
    if p_small and p_large:
        assert p_large[0][0] >= p_small[0][0] - 1e-9
