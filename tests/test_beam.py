import random

import pytest
from src.game.board import Board
from src.game.game_engine import GameEngine, GameState
from src.game.pieces import PIECE_IDS, PIECES
from src.ai.beam import BeamAgent
from src.ai.greedy import GreedyAgent


@pytest.fixture
def engine():
    return GameEngine()


def test_topk_turn_matches_greedy_search(engine):
    beam = BeamAgent(seed=1)
    greedy = GreedyAgent()
    rng = random.Random(321)

    for _ in range(10):
        state = engine.new_game()
        state.pieces = [rng.choice(PIECE_IDS) for _ in range(3)]
        grid = state.board.grid.copy()

        finalists = beam._topk_turn(grid, state.combo_count,
                                     state.placements_without_clear,
                                     list(state.pieces), k=8)
        greedy_val, greedy_moves = greedy._search(grid, state.combo_count,
                                                    state.placements_without_clear,
                                                    list(state.pieces), [], 0, 0)

        if greedy_moves is None:
            assert finalists == []
        else:
            assert finalists
            assert finalists[0][0] == pytest.approx(greedy_val)


def test_topk_turn_orderings_weakly_dominates_fixed_order(engine):
    beam = BeamAgent(seed=2)
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

        fixed = beam._topk_turn(grid, state.combo_count,
                                 state.placements_without_clear,
                                 list(state.pieces), k=8)
        ordered = beam._topk_turn_orderings(grid, state.combo_count,
                                             state.placements_without_clear,
                                             list(state.pieces), k=8)

        if not fixed:
            if ordered:
                improved += 1
            continue

        assert ordered
        assert ordered[0][0] >= fixed[0][0] - 1e-9
        if ordered[0][0] > fixed[0][0] + 1e-9:
            improved += 1

    assert improved > 0


def test_topk_turn_orderings_no_duplicate_states(engine):
    b = Board()
    beam = BeamAgent(seed=3)

    finalists = beam._topk_turn_orderings(b.grid.copy(), 0, 0, ["P01", "P01", "P02"], k=20)

    keys = set()
    for _, g, cb, pc, _ in finalists:
        key = (g.tobytes(), cb, pc)
        assert key not in keys
        keys.add(key)


def test_topk_turn_orderings_empty_on_dead_hand(engine):
    b = Board()
    b.grid[:, :] = 1
    beam = BeamAgent(seed=4)
    finalists = beam._topk_turn_orderings(b.grid.copy(), 0, 0, ["P01", "P02", "P04"], k=8)
    assert finalists == []


def test_search_orderings_determinism_same_seed_same_state(engine):
    state = engine.new_game()
    agent_a = BeamAgent(search_orderings=True, seed=42)
    agent_b = BeamAgent(search_orderings=True, seed=42)
    moves_a = agent_a.choose_moves(state.copy(), engine)
    moves_b = agent_b.choose_moves(state.copy(), engine)
    assert moves_a == moves_b


def test_search_orderings_moves_are_legal_across_random_boards(engine):
    rng = random.Random(77)
    agent = BeamAgent(search_orderings=True, seed=5)

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
    agent = BeamAgent(search_orderings=True, seed=6)
    moves = agent.choose_moves(state, engine)
    assert len(moves) == 3
    placed_ids = sorted(pid for pid, _, _ in moves)
    assert placed_ids == sorted(state.pieces)


def test_search_orderings_default_is_false():
    assert BeamAgent().search_orderings is False


def test_search_orderings_off_matches_existing_behavior(engine):
    rng = random.Random(999)
    for _ in range(5):
        state = engine.new_game()
        state.pieces = [rng.choice(PIECE_IDS) for _ in range(3)]
        agent_off = BeamAgent(search_orderings=False, seed=10)
        agent_default = BeamAgent(seed=10)
        assert agent_off.choose_moves(state.copy(), engine) == agent_default.choose_moves(state.copy(), engine)
