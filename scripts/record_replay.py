import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.game.game_engine import GameEngine
from src.game.pieces import PIECES
from src.ai.random_agent import RandomAgent

AGENTS = {
    'random': RandomAgent,
}


def record(agent_id='random', seed=None):
    if seed is None:
        seed = random.randint(0, 999999)

    AgentClass = AGENTS.get(agent_id)
    if AgentClass is None:
        raise ValueError(f"Unknown agent: {agent_id}. Available: {list(AGENTS)}")

    agent = AgentClass()
    engine = GameEngine(seed=seed)
    state = engine.new_game()

    def serialise(s, last_piece_id=None):
        return {
            'board': s.board.grid.tolist(),
            'score': s.score,
            'combo_count': s.combo_count,
            'pieces': [{'id': pid, 'grid': PIECES[pid].tolist()} for pid in s.pieces],
            'last_piece_id': last_piece_id,
            'game_over': s.game_over,
        }

    frames = [serialise(state)]

    while not state.game_over:
        pieces_snapshot = list(state.pieces)
        placed_any = False

        for piece_id in pieces_snapshot:
            if piece_id not in state.pieces:
                continue
            placements = engine.get_valid_placements(state, piece_id)
            if not placements:
                continue

            row, col = agent.choose(state, piece_id, placements)
            state = engine.apply_placement(state, piece_id, row, col)
            frames.append(serialise(state, piece_id))
            placed_any = True

        if not placed_any:
            break

        if len(state.pieces) == 0:
            state = engine.start_new_turn(state)
            if state.game_over:
                break
            frames.append(serialise(state))

    frames.append(serialise(state))

    return {
        'agent': agent_id,
        'seed': seed,
        'final_score': state.score,
        'frames': frames,
    }


if __name__ == '__main__':
    agent_id = sys.argv[1] if len(sys.argv) > 1 else 'random'
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"Recording {agent_id} agent (seed={seed or 'random'})...")
    data = record(agent_id, seed)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'replays')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{agent_id}.json')

    with open(out_path, 'w') as f:
        json.dump(data, f)

    print(f"Recorded {len(data['frames'])} frames  |  final score: {data['final_score']}")
    print(f"Saved to {out_path}")
