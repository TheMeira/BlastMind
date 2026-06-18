# BlastMind — Project Progress

## Phase 1 — Project Setup ✅
**Completed:** ~8 June 2026

- GitHub repository created: `TheMeira/BlastMind` (private)
- Project folder structure created under `BlockBlast - Project/`
- `requirements.txt`, `.gitignore`, `src/` package skeleton committed
- Piece shape catalogue built: 37 unique fixed-orientation pieces (P01–P37) documented in `Documents/General Documentation.md`
- Reference repository catalogue created at `Documents/Reference Repositories.md`

**Key decisions:**
- 37 pieces (not 19) — Block Blast uses fixed orientations, so each rotation is a separate piece
- Tech stack: Python/FastAPI backend + Phaser.js frontend + WebSockets for AI watch mode

---

## Phase 2 — Core Game Engine ✅
**Completed:** 11 June 2026

**Built:**
- `src/game/pieces.py` — 37 piece shapes as NumPy arrays in `PIECES` dict
- `src/game/board.py` — `Board` class: `can_place`, `place`, `clear_lines`, `copy`, `get_valid_placements`
- `src/game/piece_generator.py` — `PieceGenerator` with pure random selection
- `src/game/game_engine.py` — `GameEngine` + `GameState` dataclass: placement, scoring, combos, game-over detection
- `tests/test_board.py` — 18 unit tests
- `tests/test_game_engine.py` — 16 unit tests

**Key decisions:**
- `GameState` stores piece IDs (strings), not NumPy arrays — keeps state serialisable and memory-efficient for AI search
- Scoring: `bonus = lines × 10 × (combo+1)`, multiply by `(lines-1)` for >2 lines; +1 pt per block placed; +360 board clear bonus
- Combo resets after 3 placements without a line clear
- Game-over: all 3 current pieces simultaneously unplaceable
- 34/34 tests passing

---

## Phase 3 — Web Interface ✅
**Completed:** 12 June 2026

**Built:**
- `src/web/main.py` — FastAPI app: `GET /`, `POST /api/new-game`, `GET /api/state`, `POST /api/place`, `WebSocket /ws/ai`
- `src/web/websockets.py` — WebSocket handler running a random-agent game loop
- `src/web/static/index.html` — Phaser.js entry point (CDN-loaded)
- `src/web/static/js/main.js` — Phaser game config (800×680, FIT scale)
- `src/web/static/js/MenuScene.js` — title screen with Play and Watch AI buttons
- `src/web/static/js/GameScene.js` — main game: board rendering, piece panel, click-to-select, hover ghost, click-to-place, colour-coded pieces, AI watch mode via WebSocket
- `src/web/static/js/GameOverScene.js` — final score display, Play Again / Main Menu

**Key decisions:**
- Colour grid tracked client-side: diffs old/new board on each move to assign per-piece colours persistently
- Ghost piece uses piece colour at 45% alpha for valid placements, red for invalid
- AI watch mode connects WebSocket on scene start; random agent placeholder fills in until Phase 4 agents are ready
- Static files served at `/static/*`; `GET /` returns `index.html`; API routes defined before the static mount

---

---

## Phase 3b — AI Watch Mode Overhaul ✅
**Completed:** 18 June 2026

**Built:**
- `src/web/static/js/AgentSelectScene.js` — agent picker with toggleable buttons (1–5 agents), speed preset selector (default 1×), "Watch" launch button
- `src/web/static/js/MultiAgentScene.js` — unified watch scene handling 1–5 simultaneous agents:
  - 1 agent: full-size board, live stats panel (Pieces, Lines, Best Combo, Avg/Piece) in right margin
  - 2–5 agents: dynamic grid layout (2×1, 3×1, 2×2, 3×2) with CS computed to fill available area
  - Speed controls (0.5×–10×) present in-game, sends `set_speed` to all active WebSockets simultaneously
  - All agents share the same seed for fair piece-sequence comparison
  - Finished boards fade with a 55% dark overlay; transition to ComparisonScene after all are done
- `src/web/static/js/ComparisonScene.js` — results table sorted by score: Agent, Score, Pieces, Lines, Best Combo, Avg/Piece; winner highlighted in gold; "Watch Again" / "Main Menu" buttons
- `MenuScene.js` "Watch AI (Random)" button replaced with "Watch AI" → AgentSelectScene

**Key decisions:**
- Only Random agent is selectable; Greedy/Beam/DQN/MCTS shown as "coming soon" and are non-interactive until implemented
- On resize during watch, MultiAgentScene restarts with same agents/speed but a fresh seed (mid-stream reconnection is not feasible over WebSocket)
- Lines-cleared counted client-side by diffing rows/cols that were full in prevBoard but have gaps in newBoard
- High-score update on ComparisonScene only applies in single-agent mode (multi-agent comparison scores are not saved as personal bests)

---

## Up Next

### Phase 4 — Greedy Heuristic Agent
**Planned:** 15–21 June 2026

- `src/ai/greedy.py` — heuristic evaluation (holes, bumpiness, heights, line clears)
- Wire into WebSocket endpoint as `agent=greedy`, unlock in `AGENT_ROSTER` in `AgentSelectScene.js`
- Benchmark: games played, avg score, avg game length
