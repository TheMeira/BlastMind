from src.ai.dqn import encode_state, enumerate_placements
from src.game.game_engine import GameEngine


class BlockBlastEnv:
    def __init__(self, seed=None):
        self.engine = GameEngine(seed=seed)
        self.state = None

    def reset(self):
        self.state = self.engine.new_game()
        return self.state

    def current_piece(self):
        return self.state.pieces[0]

    def piece_queue(self):
        return list(self.state.pieces[:3])

    def observe(self):
        return encode_state(self.state.board.grid, self.piece_queue(),
                             self.state.combo_count, self.state.placements_without_clear)

    def candidates(self):
        return enumerate_placements(self.state.board.grid, self.current_piece())

    def step(self, row, col, pid=None):
        pid = pid or self.current_piece()
        score_before = self.state.score

        self.state = self.engine.apply_placement(self.state, pid, row, col)
        score_gained = self.state.score - score_before

        if not self.state.pieces:
            self.state = self.engine.start_new_turn(self.state)

        done = self.engine.check_game_over(self.state)
        self.state.game_over = done
        return self.state, score_gained, done
