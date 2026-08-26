import asyncio
import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from fastapi import WebSocket

from src.game.game_engine import GameEngine, GameState
from src.game.pieces import PIECES
from src.ai.greedy import GreedyAgent
from src.ai.beam import BeamAgent
from src.ai.mcts import MCTSAgent
from src.ai.dqn import DQNAgent
from src.ai.dqn_search import DQNSearchAgent

_EXECUTOR = ProcessPoolExecutor(max_workers=5, mp_context=multiprocessing.get_context('spawn'))


def serialise_state(state: GameState, last_piece_id: Optional[str] = None) -> dict:
    d = {
        "board": state.board.grid.tolist(),
        "pieces": [
            {"id": pid, "grid": PIECES[pid].tolist()}
            for pid in state.pieces
        ],
        "score": state.score,
        "combo_count": state.combo_count,
        "lines_cleared": state.lines_cleared_total,
        "game_over": state.game_over,
    }
    if last_piece_id is not None:
        d["last_piece_id"] = last_piece_id
    return d


def _pick_random_moves(state: GameState, engine: GameEngine):
    moves = []
    sim = state
    for pid in list(state.pieces):
        placements = engine.get_valid_placements(sim, pid)
        if not placements:
            break
        row, col = random.choice(placements)
        moves.append((pid, row, col))
        sim = engine.apply_placement(sim, pid, row, col)
    return moves


_AGENTS = {
    'greedy': GreedyAgent(search_orderings=True),
    'beam': BeamAgent(beam_width=16, lookahead_depth=1, samples=8, search_orderings=True),
    'mcts': MCTSAgent(n_simulations=500, time_limit=3.0),
    'dqn': DQNSearchAgent(checkpoint_path='models/dqn_vv24_ordersearch_diag_ep205000.pt', device='cpu', seed=42),
}


async def run_ai_game(websocket: WebSocket, speed: float = 1.0, agent_type: str = "random", seed: Optional[int] = None):
    engine = GameEngine(seed=seed)
    state = engine.new_game()
    await websocket.send_json(serialise_state(state))

    delay = max(0.02, 1.0 / max(speed, 0.1))

    while not state.game_over:
        if agent_type in _AGENTS:
            loop = asyncio.get_running_loop()
            moves = await loop.run_in_executor(_EXECUTOR, _AGENTS[agent_type].choose_moves, state, engine)
        else:
            moves = _pick_random_moves(state, engine)

        if not moves:
            state.game_over = True
            break

        for pid, row, col in moves:
            state = engine.apply_placement(state, pid, row, col)
            await websocket.send_json(serialise_state(state, pid))

            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=delay)
                if isinstance(msg, dict) and msg.get("type") == "set_speed":
                    new_speed = float(msg.get("speed", speed))
                    delay = max(0.02, 1.0 / max(new_speed, 0.1))
            except asyncio.TimeoutError:
                pass

        if len(state.pieces) == 0:
            state = engine.start_new_turn(state)
            if state.game_over:
                break
            await websocket.send_json(serialise_state(state))

    await websocket.send_json(serialise_state(state))
