from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from pathlib import Path
from typing import Optional

from src.game.game_engine import GameEngine, GameState
from src.game.pieces import PIECES

app = FastAPI()
engine = GameEngine()
game_state: Optional[GameState] = None

STATIC_DIR = Path(__file__).parent / "static"


class PlacementRequest(BaseModel):
    piece_id: str
    row: int
    col: int


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


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/new-game")
def new_game():
    global game_state
    game_state = engine.new_game()
    return serialise_state(game_state)


@app.get("/api/state")
def get_state():
    if game_state is None:
        return JSONResponse(status_code=400, content={"error": "No active game"})
    return serialise_state(game_state)


@app.post("/api/place")
def place(req: PlacementRequest):
    global game_state
    if game_state is None:
        return JSONResponse(status_code=400, content={"error": "No active game"})
    if req.piece_id not in game_state.pieces:
        return JSONResponse(status_code=400, content={"error": "Piece not in hand"})
    piece = PIECES.get(req.piece_id)
    if piece is None:
        return JSONResponse(status_code=400, content={"error": "Unknown piece ID"})
    if not game_state.board.can_place(piece, req.row, req.col):
        return JSONResponse(status_code=400, content={"error": "Invalid placement"})

    game_state = engine.apply_placement(game_state, req.piece_id, req.row, req.col)
    if len(game_state.pieces) == 0:
        game_state = engine.start_new_turn(game_state)
    elif engine.check_game_over(game_state):
        game_state.game_over = True

    return serialise_state(game_state, req.piece_id)


@app.websocket("/ws/ai")
async def ai_watch(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        speed = float(data.get("speed", 1.0))
        agent_type = data.get("agent", "random")
        seed = data.get("seed", None)
        if seed is not None:
            seed = int(seed)
        from src.web.websockets import run_ai_game
        await run_ai_game(websocket, speed, agent_type, seed)
    except WebSocketDisconnect:
        pass


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
