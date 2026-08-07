import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.ai.dqn import BOARD_N, PIECE_N, encode_state
from src.game.pieces import PIECES

MASK_FILL = -1e8


class PPONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.board_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.piece_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        )
        board_flat = 64 * BOARD_N * BOARD_N
        piece_flat = 32 * PIECE_N * PIECE_N
        self.trunk = nn.Sequential(
            nn.Linear(board_flat + piece_flat, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.actor_head = nn.Linear(256, BOARD_N * BOARD_N)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, board, pieces):
        b = self.board_conv(board).flatten(1)
        p = self.piece_conv(pieces).flatten(1)
        x = self.trunk(torch.cat([b, p], dim=1))
        return self.actor_head(x), self.critic_head(x).squeeze(-1)


def valid_action_mask(grid, piece_id):
    piece = PIECES[piece_id]
    ph, pw = piece.shape
    windows = np.lib.stride_tricks.sliding_window_view(grid, (ph, pw))
    free = (windows * piece).sum(axis=(2, 3)) == 0
    mask = np.zeros((BOARD_N, BOARD_N), dtype=bool)
    mask[:free.shape[0], :free.shape[1]] = free
    return mask.flatten()


def masked_distribution(logits, mask_t):
    masked_logits = logits.masked_fill(~mask_t, MASK_FILL)
    return Categorical(logits=masked_logits)


class PPOAgent:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = PPONet().to(self.device)
        if checkpoint_path:
            self.net.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.net.eval()

    def choose_moves(self, state, engine):
        grid = state.board.grid.copy()
        piece_ids = list(state.pieces)
        moves = []

        for i, pid in enumerate(piece_ids):
            mask = valid_action_mask(grid, pid)
            if not mask.any():
                break

            queue = piece_ids[i:i + 3]
            board_t, pieces_t = encode_state(grid, queue)
            with torch.no_grad():
                logits, _ = self.net(
                    board_t.unsqueeze(0).to(self.device),
                    pieces_t.unsqueeze(0).to(self.device),
                )
            logits = logits[0].cpu().numpy()
            logits = np.where(mask, logits, MASK_FILL)
            action = int(np.argmax(logits))
            row, col = divmod(action, BOARD_N)

            piece = PIECES[pid]
            ph, pw = piece.shape
            grid[row:row + ph, col:col + pw] |= piece
            rows_done = np.where(np.all(grid == 1, axis=1))[0]
            cols_done = np.where(np.all(grid == 1, axis=0))[0]
            grid[rows_done, :] = 0
            grid[:, cols_done] = 0

            moves.append((pid, row, col))

        return moves
