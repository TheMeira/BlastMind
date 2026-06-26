import numpy as np

from src.game.game_engine import GameEngine, GameState
from src.game.pieces import PIECES

_BOARD = 8


class GreedyAgent:

    W_SCORE  =  1.0
    W_HOLES  = -3.0
    W_BUMPY  = -1.0
    W_HEIGHT = -0.5

    def choose_moves(self, state: GameState, engine: GameEngine):
        piece_ids = list(state.pieces)
        grid = state.board.grid.copy()
        initial_score = state.score
        combo = state.combo_count
        pwc = state.placements_without_clear

        _, moves = self._search(grid, combo, pwc, piece_ids, [], initial_score, initial_score)
        return moves or []

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

            new_score = cur_score + cells + bonus

            val, ms = self._search(ng, new_combo, new_pwc, remaining[1:],
                                   moves + [(pid, row, col)], initial_score, new_score)
            if ms is not None and val > best_val:
                best_val = val
                best_moves = ms

        return best_val, best_moves

    def _valid(self, grid, piece, ph, pw):
        out = []
        for r in range(_BOARD - ph + 1):
            for c in range(_BOARD - pw + 1):
                if not np.any((piece == 1) & (grid[r:r + ph, c:c + pw] == 1)):
                    out.append((r, c))
        return out

    def _eval(self, grid, cur_score, initial_score):
        score_gained = cur_score - initial_score
        heights = self._col_heights(grid)
        holes = self._holes(grid, heights)
        bumpy = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
        total_h = sum(heights)

        return (score_gained * self.W_SCORE +
                holes       * self.W_HOLES +
                bumpy       * self.W_BUMPY +
                total_h     * self.W_HEIGHT)

    def _col_heights(self, grid):
        heights = []
        for c in range(_BOARD):
            filled = np.where(grid[:, c] == 1)[0]
            heights.append(_BOARD - filled[0] if len(filled) else 0)
        return heights

    def _holes(self, grid, heights):
        holes = 0
        for c, h in enumerate(heights):
            if h == 0:
                continue
            top = _BOARD - h
            for r in range(top + 1, _BOARD):
                if grid[r, c] == 0:
                    holes += 1
        return holes
