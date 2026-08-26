import itertools
import random

import numpy as np
import torch

from src.ai.dqn import (DQNNet, GAMMA, encode_state, enumerate_placements,
                        simulate_placement)


class DQNSearchAgent:

    def __init__(self, checkpoint_path=None, device=None, max_candidates=20000,
                 value_weight=1.0, batch_size=4096, seed=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = DQNNet().to(self.device)
        if checkpoint_path:
            self.net.load_state_dict(torch.load(checkpoint_path, map_location=self.device),
                                     strict=False)
        self.net.eval()
        self.max_candidates = max_candidates
        self.value_weight = value_weight
        self.batch_size = batch_size
        self._rng = random.Random(seed)

    def _enumerate(self, grid, combo, pwc, piece_ids):
        seen = {}

        def rec(g, cb, pc, remaining, gained, moves):
            if not remaining:
                key = (g.tobytes(), cb, pc)
                prev = seen.get(key)
                if prev is None or gained > prev[0]:
                    seen[key] = (gained, g, cb, pc, moves)
                return
            pid = remaining[0]
            for row, col in enumerate_placements(g, pid):
                ng, sg, ncb, npc = simulate_placement(g, pid, row, col, cb, pc)
                rec(ng, ncb, npc, remaining[1:], gained + sg,
                    moves + [(pid, row, col)])

        for order in dict.fromkeys(itertools.permutations(piece_ids)):
            rec(grid, combo, pwc, list(order), 0, [])

        return list(seen.values())

    def choose_moves(self, state, engine):
        grid = state.board.grid.copy()
        combo = state.combo_count
        pwc = state.placements_without_clear
        piece_ids = list(state.pieces)

        cands = self._enumerate(grid, combo, pwc, piece_ids)
        if not cands:
            return []
        if len(cands) == 1:
            return cands[0][4]

        if len(cands) > self.max_candidates:
            cands = self._rng.sample(cands, self.max_candidates)

        boards, pieces, gains = [], [], []
        for gained, g, cb, pc, _ in cands:
            b, p = encode_state(g, [], cb, pc)
            boards.append(b)
            pieces.append(p)
            gains.append(float(gained))

        values = np.empty(len(cands), dtype=np.float64)
        with torch.no_grad():
            for i in range(0, len(cands), self.batch_size):
                bb = torch.stack(boards[i:i + self.batch_size]).to(self.device)
                pp = torch.stack(pieces[i:i + self.batch_size]).to(self.device)
                v = self.net.unnormalize(self.net(bb, pp).squeeze(1))
                values[i:i + self.batch_size] = v.cpu().numpy()

        quality = np.asarray(gains) + self.value_weight * GAMMA * values
        return cands[int(np.argmax(quality))][4]
