import numpy as np
import torch

from src.ai.beam import BeamAgent
from src.ai.dqn import DQNNet, GAMMA, encode_state, simulate_placement


class DQNPoolAgent:

    def __init__(self, checkpoint_path=None, device=None, pool_k=50, beam_width=16,
                 value_weight=1.0):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = DQNNet().to(self.device)
        if checkpoint_path:
            self.net.load_state_dict(torch.load(checkpoint_path, map_location=self.device),
                                     strict=False)
        self.net.eval()
        self.pool_k = pool_k
        self.value_weight = value_weight
        self._proposer = BeamAgent(beam_width=beam_width, lookahead_depth=0,
                                   search_orderings=True)

    def choose_moves(self, state, engine):
        grid = state.board.grid.copy()
        combo = state.combo_count
        pwc = state.placements_without_clear
        piece_ids = list(state.pieces)

        pool = self._proposer._topk_turn_orderings(grid, combo, pwc, piece_ids, self.pool_k)
        if not pool:
            return []
        if len(pool) == 1:
            return pool[0][4]

        imms, boards, pieces = [], [], []
        for _, g, cb, pc, moves in pool:
            sim_g, sim_cb, sim_pc = grid, combo, pwc
            total = 0.0
            for pid, r, c in moves:
                sim_g, sg, sim_cb, sim_pc = simulate_placement(sim_g, pid, r, c, sim_cb, sim_pc)
                total += float(sg)
            imms.append(total)
            b, p = encode_state(sim_g, [], sim_cb, sim_pc)
            boards.append(b)
            pieces.append(p)

        with torch.no_grad():
            vs = self.net.unnormalize(
                self.net(torch.stack(boards).to(self.device),
                         torch.stack(pieces).to(self.device)).squeeze(1)).cpu().numpy()

        quality = np.asarray(imms, dtype=np.float64) + self.value_weight * GAMMA * vs
        return pool[int(np.argmax(quality))][4]
