import asyncio
import random
from typing import Optional

from fastapi import WebSocket
from src.game.game_engine import GameEngine, GameState
from src.game.pieces import PIECES


def serialise_state(state: GameState, last_piece_id: Optional[str] = None) -> dict:
    d = {
        "board": state.board.grid.tolist(),
        "pieces": [
            {"id": pid, "grid": PIECES[pid].tolist()}
            for pid in state.pieces
        ],
        "score": state.score,
        "combo_count": state.combo_count,
        "game_over": state.game_over,
    }
    if last_piece_id is not None:
        d["last_piece_id"] = last_piece_id
    return d


def pick_random_move(state: GameState, piece_id: str, engine: GameEngine):
    placements = engine.get_valid_placements(state, piece_id)
    if not placements:
        return None
    return random.choice(placements)


async def run_ai_game(websocket: WebSocket, speed: float = 1.0, agent_type: str = "random", seed: Optional[int] = None):
    engine = GameEngine(seed=seed)
    state = engine.new_game()
    await websocket.send_json(serialise_state(state))

    delay = max(0.05, 1.0 / max(speed, 0.1))

    while not state.game_over:
        pieces_snapshot = list(state.pieces)
        placed_any = False

        for piece_id in pieces_snapshot:
            if piece_id not in state.pieces:
                continue

            move = pick_random_move(state, piece_id, engine)
            if move is None:
                continue

            row, col = move
            state = engine.apply_placement(state, piece_id, row, col)
            await websocket.send_json(serialise_state(state, piece_id))
            await asyncio.sleep(delay)
            placed_any = True

        if not placed_any:
            state.game_over = True
            break

        if len(state.pieces) == 0:
            state = engine.start_new_turn(state)
            if state.game_over:
                break
            await websocket.send_json(serialise_state(state))

    await websocket.send_json(serialise_state(state))
