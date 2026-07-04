import heapq
import itertools
import random

from src.game.game_engine import GameEngine, GameState
from src.game.pieces import PIECES, PIECE_IDS
from src.ai.greedy import GreedyAgent

_DEAD_END_PENALTY = -500.0


class BeamAgent(GreedyAgent):

    def __init__(self, beam_width=8, lookahead_depth=1, samples=4, seed=None):
        self.beam_width = beam_width
        self.lookahead_depth = lookahead_depth
        self.samples = samples
        self._rng = random.Random(seed)

    def choose_moves(self, state: GameState, engine: GameEngine):
        piece_ids = list(state.pieces)
        grid = state.board.grid.copy()

        finalists = self._topk_turn(grid, state.combo_count,
                                    state.placements_without_clear,
                                    piece_ids, self.beam_width)
        if not finalists:
            return []
        if self.lookahead_depth == 0 or len(finalists) == 1:
            return finalists[0][4]

        futures = [
            [[self._rng.choice(PIECE_IDS) for _ in range(3)]
             for _ in range(self.lookahead_depth)]
            for _ in range(self.samples)
        ]

        best_val = float('-inf')
        best_moves = finalists[0][4]

        for val, g, combo, pwc, moves in finalists:
            future_val = sum(self._rollout(g, combo, pwc, future)
                             for future in futures) / self.samples
            total = val + future_val
            if total > best_val:
                best_val = total
                best_moves = moves

        return best_moves

    def _topk_turn(self, grid, combo, pwc, piece_ids, k):
        heap = []
        tiebreak = itertools.count()

        def expand(g, cb, pc, remaining, gained, moves):
            if not remaining:
                val = self._eval(g, gained, 0)
                if len(heap) < k:
                    heapq.heappush(heap, (val, next(tiebreak), g, cb, pc, moves))
                elif val > heap[0][0]:
                    heapq.heapreplace(heap, (val, next(tiebreak), g, cb, pc, moves))
                return

            pid = remaining[0]
            piece = PIECES[pid]
            ph, pw = piece.shape

            for row, col in self._valid(g, piece, ph, pw):
                ng, add, ncb, npc = self._place(g, piece, ph, pw, row, col, cb, pc)
                expand(ng, ncb, npc, remaining[1:], gained + add,
                       moves + [(pid, row, col)])

        expand(grid, combo, pwc, piece_ids, 0, [])

        finalists = [(val, g, cb, pc, moves) for val, _, g, cb, pc, moves in heap]
        finalists.sort(key=lambda t: t[0], reverse=True)
        return finalists

    def _beam_turn(self, grid, combo, pwc, piece_ids, width):
        beam = [(0.0, grid, combo, pwc, 0)]

        for pid in piece_ids:
            piece = PIECES[pid]
            ph, pw = piece.shape
            next_beam = []

            for _, g, cb, pc, gained in beam:
                for row, col in self._valid(g, piece, ph, pw):
                    ng, add, ncb, npc = self._place(g, piece, ph, pw, row, col, cb, pc)
                    total_gained = gained + add
                    val = self._eval(ng, total_gained, 0)
                    next_beam.append((val, ng, ncb, npc, total_gained))

            if not next_beam:
                return []

            next_beam.sort(key=lambda t: t[0], reverse=True)
            beam = next_beam[:width]

        return beam

    def _rollout(self, grid, combo, pwc, future):
        width = max(2, self.beam_width // 2)
        total = 0.0
        g, cb, pc = grid, combo, pwc

        for piece_ids in future:
            beam = self._beam_turn(g, cb, pc, piece_ids, width)
            if not beam:
                return total + _DEAD_END_PENALTY
            val, g, cb, pc, _ = beam[0]
            total += val

        return total
