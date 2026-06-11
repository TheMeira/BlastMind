import numpy as np

BOARD_SIZE = 8


class Board:
    def __init__(self):
        self.grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

    def copy(self):
        b = Board.__new__(Board)
        b.grid = self.grid.copy()
        return b

    def can_place(self, piece, row, col):
        ph, pw = piece.shape
        if row + ph > BOARD_SIZE or col + pw > BOARD_SIZE:
            return False
        region = self.grid[row:row + ph, col:col + pw]
        return not np.any((piece == 1) & (region == 1))

    def place(self, piece, row, col):
        ph, pw = piece.shape
        self.grid[row:row + ph, col:col + pw] |= piece
        return int(np.sum(piece))

    def clear_lines(self):
        completed_rows = np.where(np.all(self.grid == 1, axis=1))[0]
        completed_cols = np.where(np.all(self.grid == 1, axis=0))[0]
        self.grid[completed_rows, :] = 0
        self.grid[:, completed_cols] = 0
        return len(completed_rows) + len(completed_cols)

    def is_empty(self):
        return not np.any(self.grid)

    def get_valid_placements(self, piece):
        ph, pw = piece.shape
        placements = []
        for r in range(BOARD_SIZE - ph + 1):
            for c in range(BOARD_SIZE - pw + 1):
                if self.can_place(piece, r, c):
                    placements.append((r, c))
        return placements
