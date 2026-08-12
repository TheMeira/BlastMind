import itertools

import numpy as np

from src.game.game_engine import GameEngine, GameState
from src.game.pieces import PIECES

_BOARD = 8


class GreedyAgent:

    W_SCORE  =  1.0
    W_HOLES  = -3.0
    W_BUMPY  = -1.0
    W_HEIGHT = -0.5

    def __init__(self, search_orderings=False, w_score=None, w_holes=None, w_bumpy=None, w_height=None):
        self.search_orderings = search_orderings
        if w_score is not None:
            self.W_SCORE = w_score
        if w_holes is not None:
            self.W_HOLES = w_holes
        if w_bumpy is not None:
            self.W_BUMPY = w_bumpy
        if w_height is not None:
            self.W_HEIGHT = w_height

    def choose_moves(self, state: GameState, engine: GameEngine):
        piece_ids = list(state.pieces)
        grid = state.board.grid.copy()
        initial_score = state.score
        combo = state.combo_count
        pwc = state.placements_without_clear

        if self.search_orderings:
            _, moves = self._search_orderings(grid, combo, pwc, piece_ids, initial_score)
        else:
            _, moves = self._search(grid, combo, pwc, piece_ids, [], initial_score, initial_score)
        return moves or []

    def _search_orderings(self, grid, combo, pwc, piece_ids, initial_score):
        best_val = float('-inf')
        best_moves = None

        for order in dict.fromkeys(itertools.permutations(piece_ids)):
            val, moves = self._search(grid, combo, pwc, list(order), [], initial_score, initial_score)
            if moves is not None and val > best_val:
                best_val = val
                best_moves = moves

        return best_val, best_moves

    def _search(self, grid, combo, pwc, remaining, moves, initial_score, cur_score):
        if not remaining:
            return self._eval(grid, cur_score, initial_score), moves

        pid = remaining[0]
        piece = PIECES[pid]
        ph, pw = piece.shape
        placements = self._valid(grid, piece, ph, pw)

        if not placements:
            return float('-inf'), None

        best_val = float('-inf')
        best_moves = None

        for row, col in placements:
            ng, gained, new_combo, new_pwc = self._place(grid, piece, ph, pw, row, col, combo, pwc)
            new_score = cur_score + gained

            val, ms = self._search(ng, new_combo, new_pwc, remaining[1:],
                                   moves + [(pid, row, col)], initial_score, new_score)
            if ms is not None and val > best_val:
                best_val = val
                best_moves = ms

        return best_val, best_moves

    def _place(self, grid, piece, ph, pw, row, col, combo, pwc):
        ng = grid.copy()
        ng[row:row + ph, col:col + pw] |= piece
        cells = int(np.sum(piece))

        rows_done = np.where(np.all(ng == 1, axis=1))[0]
        cols_done = np.where(np.all(ng == 1, axis=0))[0]
        ng[rows_done, :] = 0
        ng[:, cols_done] = 0
        lines = len(rows_done) + len(cols_done)

        if lines > 0:
            bonus = lines * 10 * (combo + 1)
            if lines > 2:
                bonus *= (lines - 1)
            if not np.any(ng):
                bonus += 360
            new_combo = combo + lines
            new_pwc = 0
        else:
            bonus = 0
            new_pwc = pwc + 1
            new_combo = 0 if new_pwc >= 3 else combo

        return ng, cells + bonus, new_combo, new_pwc

    def _valid(self, grid, piece, ph, pw):
        windows = np.lib.stride_tricks.sliding_window_view(grid, (ph, pw))
        free = (windows * piece).sum(axis=(2, 3)) == 0
        return [(int(r), int(c)) for r, c in np.argwhere(free)]

    def _eval(self, grid, cur_score, initial_score):
        score_gained = cur_score - initial_score
        filled = grid == 1
        occupied = filled.any(axis=0)
        heights = np.where(occupied, _BOARD - np.argmax(filled, axis=0), 0)
        covered = np.maximum.accumulate(filled, axis=0)
        holes = int(np.sum(covered & ~filled))
        bumpy = int(np.abs(np.diff(heights)).sum())
        total_h = int(heights.sum())

        return (score_gained * self.W_SCORE +
                holes       * self.W_HOLES +
                bumpy       * self.W_BUMPY +
                total_h     * self.W_HEIGHT)
