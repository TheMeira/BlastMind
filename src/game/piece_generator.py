import random
from src.game.pieces import PIECE_IDS


class PieceGenerator:
    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def generate(self, n=3):
        return [self._rng.choice(PIECE_IDS) for _ in range(n)]
