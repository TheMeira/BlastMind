import random


class RandomAgent:
    def choose(self, state, piece_id, placements):
        return random.choice(placements)
