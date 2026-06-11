import numpy as np
import pytest
from src.game.board import Board
from src.game.pieces import PIECES
from src.game.game_engine import GameEngine, GameState


@pytest.fixture
def engine():
    return GameEngine()


@pytest.fixture
def empty_state(engine):
    return engine.new_game()


def test_new_game_has_empty_board(engine):
    state = engine.new_game()
    assert state.board.is_empty()


def test_new_game_has_three_pieces(engine):
    state = engine.new_game()
    assert len(state.pieces) == 3


def test_new_game_score_is_zero(engine):
    state = engine.new_game()
    assert state.score == 0


def test_apply_placement_scores_blocks_placed(engine):
    state = engine.new_game()
    state.pieces = ["P01"]
    result = engine.apply_placement(state, "P01", 0, 0)
    assert result.score >= 1


def test_apply_placement_removes_piece(engine):
    state = engine.new_game()
    state.pieces = ["P01", "P02", "P04"]
    result = engine.apply_placement(state, "P01", 0, 0)
    assert "P01" not in result.pieces
    assert len(result.pieces) == 2


def test_line_clear_bonus_single(engine):
    b = Board()
    b.grid[0, 1:] = 1
    b.grid[7, 0] = 1
    state = GameState(board=b, pieces=["P02"], score=0, combo_count=0, placements_without_clear=0)
    result = engine.apply_placement(state, "P02", 0, 0)
    assert result.score == 2 + 10


def test_line_clear_bonus_double(engine):
    b = Board()
    b.grid[0, 2:] = 1
    b.grid[1, 2:] = 1
    b.grid[7, 0] = 1
    state = GameState(board=b, pieces=["P11"], score=0, combo_count=0, placements_without_clear=0)
    result = engine.apply_placement(state, "P11", 0, 0)
    assert result.score == 6 + 20


def test_line_clear_bonus_triple(engine):
    b = Board()
    b.grid[0, 3:] = 1
    b.grid[1, 3:] = 1
    b.grid[:, 2] = 1
    b.grid[0, 2] = 0
    b.grid[1, 2] = 0
    b.grid[0, :2] = 0
    b.grid[1, :2] = 0
    b = Board()
    b.grid[0, 1:] = 1
    b.grid[1, 1:] = 1
    b.grid[2:, 0] = 1
    state = GameState(board=b, pieces=["P12"], score=0, combo_count=0, placements_without_clear=0)
    result = engine.apply_placement(state, "P12", 0, 0)
    assert result.score == 6 + 60


def test_line_clear_bonus_formula(engine):
    assert engine._line_clear_bonus(1, 0) == 10
    assert engine._line_clear_bonus(2, 0) == 20
    assert engine._line_clear_bonus(3, 0) == 60
    assert engine._line_clear_bonus(4, 0) == 120
    assert engine._line_clear_bonus(5, 0) == 200
    assert engine._line_clear_bonus(6, 0) == 300


def test_combo_increments_on_consecutive_clears(engine):
    b = Board()
    b.grid[0, 1:] = 1
    state = GameState(board=b, pieces=["P02"], score=0, combo_count=0, placements_without_clear=0)
    result = engine.apply_placement(state, "P02", 0, 0)
    assert result.combo_count == 1


def test_combo_resets_after_three_placements_without_clear(engine):
    state = engine.new_game()
    state.pieces = ["P01", "P01", "P01"]
    state.combo_count = 5
    s1 = engine.apply_placement(state, "P01", 0, 0)
    s1.pieces = ["P01", "P01"]
    s2 = engine.apply_placement(s1, "P01", 2, 0)
    s2.pieces = ["P01"]
    s3 = engine.apply_placement(s2, "P01", 4, 0)
    assert s3.combo_count == 0


def test_board_clear_bonus(engine):
    b = Board()
    b.grid[:, :] = 1
    b.grid[0, 0] = 0
    state = GameState(board=b, pieces=["P01"], score=0, combo_count=0, placements_without_clear=0)
    result = engine.apply_placement(state, "P01", 0, 0)
    assert result.board.is_empty()
    assert result.score >= 360


def test_game_over_when_no_valid_placements(engine):
    b = Board()
    b.grid[:, :] = 1
    state = GameState(board=b, pieces=["P01", "P02", "P04"], score=0)
    assert engine.check_game_over(state) is True


def test_not_game_over_when_any_piece_fits(engine):
    b = Board()
    b.grid[:, :] = 1
    b.grid[0, 0] = 0
    state = GameState(board=b, pieces=["P01", "P13"], score=0)
    assert engine.check_game_over(state) is False


def test_game_state_copy_independence(engine):
    state = engine.new_game()
    copy = state.copy()
    copy.board.grid[0, 0] = 1
    assert state.board.grid[0, 0] == 0


def test_get_valid_placements_delegates_to_board(engine):
    state = engine.new_game()
    state.pieces = ["P01"]
    placements = engine.get_valid_placements(state, "P01")
    assert len(placements) == 64
