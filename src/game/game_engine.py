from dataclasses import dataclass, field
from typing import List, Tuple

from src.game.board import Board
from src.game.pieces import PIECES
from src.game.piece_generator import PieceGenerator


@dataclass
class GameState:
    board: Board
    pieces: List[str]
    score: int = 0
    combo_count: int = 0
    placements_without_clear: int = 0
    game_over: bool = False

    def copy(self):
        return GameState(
            board=self.board.copy(),
            pieces=list(self.pieces),
            score=self.score,
            combo_count=self.combo_count,
            placements_without_clear=self.placements_without_clear,
            game_over=self.game_over,
        )


class GameEngine:
    def __init__(self, seed=None):
        self._generator = PieceGenerator(seed=seed)

    def new_game(self) -> GameState:
        return GameState(
            board=Board(),
            pieces=self._generator.generate(3),
        )

    def get_valid_placements(self, state: GameState, piece_id: str) -> List[Tuple[int, int]]:
        return state.board.get_valid_placements(PIECES[piece_id])

    def apply_placement(self, state: GameState, piece_id: str, row: int, col: int) -> GameState:
        new_state = state.copy()
        piece = PIECES[piece_id]

        cells_placed = new_state.board.place(piece, row, col)
        new_state.score += cells_placed

        lines_cleared = new_state.board.clear_lines()
        new_state.score += self._line_clear_bonus(lines_cleared, new_state.combo_count)

        if lines_cleared > 0:
            new_state.combo_count += lines_cleared
            new_state.placements_without_clear = 0
            if new_state.board.is_empty():
                new_state.score += 360
        else:
            new_state.placements_without_clear += 1
            if new_state.placements_without_clear >= 3:
                new_state.combo_count = 0

        pieces = list(new_state.pieces)
        pieces.remove(piece_id)
        new_state.pieces = pieces

        return new_state

    def start_new_turn(self, state: GameState) -> GameState:
        new_state = state.copy()
        new_state.pieces = self._generator.generate(3)
        new_state.game_over = self._is_game_over(new_state)
        return new_state

    def check_game_over(self, state: GameState) -> bool:
        return self._is_game_over(state)

    def _is_game_over(self, state: GameState) -> bool:
        for piece_id in state.pieces:
            if state.board.get_valid_placements(PIECES[piece_id]):
                return False
        return True

    def _line_clear_bonus(self, lines_cleared: int, combo_count: int) -> int:
        if lines_cleared == 0:
            return 0
        bonus = lines_cleared * 10 * (combo_count + 1)
        if lines_cleared > 2:
            bonus *= (lines_cleared - 1)
        return bonus
