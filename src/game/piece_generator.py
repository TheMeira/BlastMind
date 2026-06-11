import random
from src.game.pieces import PIECE_IDS


class PieceGenerator:
    def generate(self, n=3):
        return [random.choice(PIECE_IDS) for _ in range(n)]
