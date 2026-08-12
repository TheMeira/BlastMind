import itertools

import numpy as np
import torch
import torch.nn as nn

from src.game.pieces import PIECES

BOARD_N = 8
PIECE_N = 5
GAMMA = 0.99


class DQNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.board_conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.piece_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        )
        board_flat = 64 * BOARD_N * BOARD_N
        piece_flat = 32 * PIECE_N * PIECE_N
        self.fc = nn.Sequential(
            nn.Linear(board_flat + piece_flat, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, board, pieces):
        b = self.board_conv(board).flatten(1)
        p = self.piece_conv(pieces).flatten(1)
        return self.fc(torch.cat([b, p], dim=1))


def encode_board(grid, combo, pwc):
    occupancy = grid.astype(np.float32)
    row_fill = grid.sum(axis=1, keepdims=True).astype(np.float32) / BOARD_N
    col_fill = grid.sum(axis=0, keepdims=True).astype(np.float32) / BOARD_N
    row_fill = np.broadcast_to(row_fill, (BOARD_N, BOARD_N))
    col_fill = np.broadcast_to(col_fill, (BOARD_N, BOARD_N))
    combo_channel = np.full((BOARD_N, BOARD_N), min(combo, 20) / 20.0, dtype=np.float32)
    pwc_channel = np.full((BOARD_N, BOARD_N), min(pwc, 3) / 3.0, dtype=np.float32)
    stacked = np.stack([occupancy, row_fill, col_fill, combo_channel, pwc_channel], axis=0)
    return torch.tensor(stacked, dtype=torch.float32)


def encode_pieces(piece_ids):
    out = np.zeros((3, PIECE_N, PIECE_N), dtype=np.float32)
    for i in range(min(3, len(piece_ids))):
        pid = piece_ids[i]
        if pid is None:
            continue
        shape = PIECES[pid]
        ph, pw = shape.shape
        out[i, :ph, :pw] = shape
    return torch.tensor(out, dtype=torch.float32)


def encode_state(grid, piece_ids, combo, pwc):
    return encode_board(grid, combo, pwc), encode_pieces(piece_ids)


def enumerate_placements(grid, piece_id):
    piece = PIECES[piece_id]
    ph, pw = piece.shape
    windows = np.lib.stride_tricks.sliding_window_view(grid, (ph, pw))
    free = (windows * piece).sum(axis=(2, 3)) == 0
    return [(int(r), int(c)) for r, c in np.argwhere(free)]


def simulate_placement(grid, piece_id, row, col, combo, pwc):
    piece = PIECES[piece_id]
    ph, pw = piece.shape
    ng = grid.copy()
    ng[row:row + ph, col:col + pw] |= piece
    cells = int(np.sum(piece))

    rows_done = np.where(np.all(ng == 1, axis=1))[0]
    cols_done = np.where(np.all(ng == 1, axis=0))[0]
    ng[rows_done, :] = 0
    ng[:, cols_done] = 0
    lines = len(rows_done) + len(cols_done)

    if lines > 0:
        bonus = lines * 10 * (combo + 1)
        if lines > 2:
            bonus *= (lines - 1)
        if not np.any(ng):
            bonus += 360
        new_combo = combo + lines
        new_pwc = 0
    else:
        bonus = 0
        new_pwc = pwc + 1
        new_combo = 0 if new_pwc >= 3 else combo

    return ng, cells + bonus, new_combo, new_pwc


def best_placement(net, device, grid, piece_ids, combo, pwc):
    pid = piece_ids[0]
    candidates = enumerate_placements(grid, pid)
    if not candidates:
        return None

    queue_after = piece_ids[1:3]
    next_boards, next_pieces, scores, results = [], [], [], []
    for row, col in candidates:
        next_grid, score_gained, new_combo, new_pwc = simulate_placement(grid, pid, row, col, combo, pwc)
        nb, np_ = encode_state(next_grid, queue_after, new_combo, new_pwc)
        next_boards.append(nb)
        next_pieces.append(np_)
        scores.append(score_gained)
        results.append((next_grid, new_combo, new_pwc))

    with torch.no_grad():
        nb_batch = torch.stack(next_boards).to(device)
        np_batch = torch.stack(next_pieces).to(device)
        v_vals = net(nb_batch, np_batch).squeeze(1).cpu().numpy()

    combined = np.array(scores, dtype=np.float32) + GAMMA * v_vals
    best_idx = int(np.argmax(combined))
    row, col = candidates[best_idx]
    next_grid, new_combo, new_pwc = results[best_idx]
    return row, col, next_grid, scores[best_idx], new_combo, new_pwc, float(combined[best_idx])


def state_value(net, device, grid, combo, pwc):
    board_t, pieces_t = encode_state(grid, [], combo, pwc)
    with torch.no_grad():
        v = net(board_t.unsqueeze(0).to(device), pieces_t.unsqueeze(0).to(device))
    return float(v.item())


def best_order_placement(net, device, grid, piece_ids, combo, pwc):
    best_quality = float('-inf')
    best_moves = None

    for order in set(itertools.permutations(piece_ids)):
        g, cb, pc = grid, combo, pwc
        moves = []
        total_score = 0.0
        valid = True

        for i, pid in enumerate(order):
            result = best_placement(net, device, g, [pid] + list(order[i + 1:]), cb, pc)
            if result is None:
                valid = False
                break
            row, col, g, gained, cb, pc, _ = result
            moves.append((pid, row, col))
            total_score += gained

        if not valid:
            continue

        quality = total_score + GAMMA * state_value(net, device, g, cb, pc)
        if quality > best_quality:
            best_quality = quality
            best_moves = moves

    return best_moves or []


def best_order_placement_train(net, device, grid, piece_ids, combo, pwc):
    best_quality = float('-inf')
    best_moves = None

    for order in dict.fromkeys(itertools.permutations(piece_ids)):
        g, cb, pc = grid, combo, pwc
        moves = []
        total_score = 0.0
        valid = True

        for i, pid in enumerate(order):
            result = best_placement(net, device, g, [pid] + list(order[i + 1:]), cb, pc)
            if result is None:
                valid = False
                break
            row, col, g, gained, cb, pc, combined_val = result
            moves.append((pid, row, col, combined_val))
            total_score += gained

        if not valid:
            continue

        quality = total_score + GAMMA * state_value(net, device, g, cb, pc)
        if quality > best_quality:
            best_quality = quality
            best_moves = moves

    return (best_moves or []), (best_quality if best_moves else 0.0)


class DQNAgent:
    def __init__(self, checkpoint_path=None, device=None, search_orderings=False):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = DQNNet().to(self.device)
        if checkpoint_path:
            self.net.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.net.eval()
        self.search_orderings = search_orderings

    def choose_moves(self, state, engine):
        grid = state.board.grid.copy()
        piece_ids = list(state.pieces)
        combo = state.combo_count
        pwc = state.placements_without_clear

        if self.search_orderings:
            return best_order_placement(self.net, self.device, grid, piece_ids, combo, pwc)

        moves = []
        for i, pid in enumerate(piece_ids):
            result = best_placement(self.net, self.device, grid, piece_ids[i:], combo, pwc)
            if result is None:
                break
            row, col, grid, _, combo, pwc, _ = result
            moves.append((pid, row, col))

        return moves
