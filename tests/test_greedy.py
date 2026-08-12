import random

import pytest
from src.game.board import Board
from src.game.game_engine import GameEngine
from src.game.pieces import PIECE_IDS, PIECES
from src.ai.greedy import GreedyAgent


@pytest.fixture
def engine():
    return GameEngine()


def test_search_orderings_weakly_dominates_fixed_order(engine):
    agent = GreedyAgent()
    rng = random.Random(654)
    improved = 0

    for _ in range(15):
        state = engine.new_game()
        fill_prob = 0.5 + rng.random() * 0.3
        for r in range(8):
            for c in range(8):
                state.board.grid[r, c] = 1 if rng.random() < fill_prob else 0
        state.pieces = [rng.choice(PIECE_IDS) for _ in range(3)]
        grid = state.board.grid.copy()

        fixed_val, fixed_moves = agent._search(grid, state.combo_count,
                                                 state.placements_without_clear,
                                                 list(state.pieces), [], 0, 0)
        ordered_val, ordered_moves = agent._search_orderings(grid, state.combo_count,
                                                               state.placements_without_clear,
                                                               list(state.pieces), 0)

        if fixed_moves is None:
            if ordered_moves is not None:
                improved += 1
            continue

        assert ordered_moves is not None
        assert ordered_val >= fixed_val - 1e-9
        if ordered_val > fixed_val + 1e-9:
            improved += 1

    assert improved > 0


def test_search_orderings_empty_on_dead_hand(engine):
    b = Board()
    b.grid[:, :] = 1
    agent = GreedyAgent()
    val, moves = agent._search_orderings(b.grid.copy(), 0, 0, ["P01", "P02", "P04"], 0)
    assert val == float('-inf')
    assert moves is None


def test_search_orderings_determinism_same_state(engine):
    state = engine.new_game()
    agent_a = GreedyAgent(search_orderings=True)
    agent_b = GreedyAgent(search_orderings=True)
    moves_a = agent_a.choose_moves(state.copy(), engine)
    moves_b = agent_b.choose_moves(state.copy(), engine)
    assert moves_a == moves_b


def test_search_orderings_moves_are_legal_across_random_boards(engine):
    rng = random.Random(77)
    agent = GreedyAgent(search_orderings=True)

    for _ in range(15):
        state = engine.new_game()
        fill_prob = rng.random() * 0.5
        for r in range(8):
            for c in range(8):
                state.board.grid[r, c] = 1 if rng.random() < fill_prob else 0
        state.pieces = [rng.choice(PIECE_IDS) for _ in range(3)]

        moves = agent.choose_moves(state, engine)
        sim_grid = state.board.grid.copy()
        sim_combo, sim_pwc = state.combo_count, state.placements_without_clear
        remaining_pieces = list(state.pieces)
        for pid, row, col in moves:
            assert pid in remaining_pieces
            remaining_pieces.remove(pid)
            piece = PIECES[pid]
            candidates = agent._valid(sim_grid, piece, *piece.shape)
            assert (row, col) in candidates
            sim_grid, _, sim_combo, sim_pwc = agent._place(sim_grid, piece, *piece.shape, row, col, sim_combo, sim_pwc)


def test_search_orderings_returns_full_hand_on_open_board(engine):
    state = engine.new_game()
    agent = GreedyAgent(search_orderings=True)
    moves = agent.choose_moves(state, engine)
    assert len(moves) == 3
    placed_ids = sorted(pid for pid, _, _ in moves)
    assert placed_ids == sorted(state.pieces)


def test_search_orderings_default_is_false():
    assert GreedyAgent().search_orderings is False


def test_search_orderings_off_matches_existing_behavior(engine):
    rng = random.Random(999)
    for _ in range(5):
        state = engine.new_game()
        state.pieces = [rng.choice(PIECE_IDS) for _ in range(3)]
        agent_off = GreedyAgent(search_orderings=False)
        agent_default = GreedyAgent()
        assert agent_off.choose_moves(state.copy(), engine) == agent_default.choose_moves(state.copy(), engine)
