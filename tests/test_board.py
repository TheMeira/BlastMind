import numpy as np
import pytest
from src.game.board import Board
from src.game.pieces import PIECES


def test_initial_board_is_empty():
    b = Board()
    assert np.all(b.grid == 0)
    assert b.grid.shape == (8, 8)


def test_can_place_on_empty_board():
    b = Board()
    assert b.can_place(PIECES["P04"], 0, 0) is True


def test_can_place_out_of_bounds_row():
    b = Board()
    assert b.can_place(PIECES["P05"], 6, 0) is False


def test_can_place_out_of_bounds_col():
    b = Board()
    assert b.can_place(PIECES["P04"], 0, 6) is False


def test_can_place_overlap():
    b = Board()
    b.grid[0, 0] = 1
    assert b.can_place(PIECES["P02"], 0, 0) is False


def test_place_updates_grid():
    b = Board()
    piece = PIECES["P10"]
    b.place(piece, 0, 0)
    assert b.grid[0, 0] == 1
    assert b.grid[0, 1] == 1
    assert b.grid[1, 0] == 1
    assert b.grid[1, 1] == 1
    assert b.grid[0, 2] == 0


def test_place_returns_cell_count():
    b = Board()
    count = b.place(PIECES["P13"], 0, 0)
    assert count == 9


def test_clear_single_row():
    b = Board()
    b.grid[3, :] = 1
    lines = b.clear_lines()
    assert lines == 1
    assert np.all(b.grid[3, :] == 0)


def test_clear_single_col():
    b = Board()
    b.grid[:, 5] = 1
    lines = b.clear_lines()
    assert lines == 1
    assert np.all(b.grid[:, 5] == 0)


def test_clear_multiple_simultaneous():
    b = Board()
    b.grid[0, :] = 1
    b.grid[1, :] = 1
    b.grid[:, 0] = 1
    lines = b.clear_lines()
    assert lines == 3
    assert np.all(b.grid[0, :] == 0)
    assert np.all(b.grid[1, :] == 0)
    assert np.all(b.grid[:, 0] == 0)


def test_clear_returns_zero_when_no_lines():
    b = Board()
    b.grid[0, 0] = 1
    lines = b.clear_lines()
    assert lines == 0


def test_board_copy_independence():
    b = Board()
    b.grid[0, 0] = 1
    c = b.copy()
    c.grid[1, 1] = 1
    assert b.grid[1, 1] == 0


def test_is_empty_true():
    b = Board()
    assert b.is_empty() is True


def test_is_empty_false():
    b = Board()
    b.grid[0, 0] = 1
    assert b.is_empty() is False


def test_get_valid_placements_1x1():
    b = Board()
    placements = b.get_valid_placements(PIECES["P01"])
    assert len(placements) == 64


def test_get_valid_placements_reduces_on_occupied():
    b = Board()
    b.grid[0, 0] = 1
    placements = b.get_valid_placements(PIECES["P01"])
    assert (0, 0) not in placements
    assert len(placements) == 63


def test_get_valid_placements_5x1_horizontal():
    b = Board()
    placements = b.get_valid_placements(PIECES["P08"])
    assert len(placements) == 8 * 4


def test_get_valid_placements_3x3():
    b = Board()
    placements = b.get_valid_placements(PIECES["P13"])
    assert len(placements) == 6 * 6
