import random


class RandomAgent:
    def __init__(self, seed=None):
        self._rng = random.Random(seed)

    def choose(self, state, piece_id, placements):
        return self._rng.choice(placements)
