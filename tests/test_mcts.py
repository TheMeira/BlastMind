import pytest
from src.game.board import Board
from src.game.game_engine import GameEngine, GameState
from src.ai.mcts import MCTSAgent
from src.ai.greedy import GreedyAgent


@pytest.fixture
def engine():
    return GameEngine()


def test_single_legal_placement(engine):
    b = Board()
    b.grid[:, :] = 1
    b.grid[0, 0] = 0
    state = GameState(board=b, pieces=["P01"], score=0, combo_count=0, placements_without_clear=0)
    agent = MCTSAgent(n_simulations=50, time_limit=2.0, seed=1)
    moves = agent.choose_moves(state, engine)
    assert moves == [("P01", 0, 0)]


def test_matches_greedy_on_unambiguous_line_clear(engine):
    b = Board()
    b.grid[0, 1:] = 1
    state = GameState(board=b, pieces=["P01"], score=0, combo_count=0, placements_without_clear=0)
    mcts = MCTSAgent(n_simulations=500, time_limit=5.0, seed=2)
    greedy = GreedyAgent()
    mcts_moves = mcts.choose_moves(state, engine)
    greedy_moves = greedy.choose_moves(state, engine)
    assert mcts_moves == greedy_moves == [("P01", 0, 0)]


def test_single_simulation_does_not_crash(engine):
    state = engine.new_game()
    agent = MCTSAgent(n_simulations=1, time_limit=2.0, seed=3)
    moves = agent.choose_moves(state, engine)
    assert isinstance(moves, list)
    assert len(moves) >= 1


def test_dead_board_returns_empty_moves(engine):
    b = Board()
    b.grid[:, :] = 1
    state = GameState(board=b, pieces=["P01", "P02", "P04"], score=0, combo_count=0, placements_without_clear=0)
    agent = MCTSAgent(n_simulations=50, time_limit=2.0, seed=4)
    moves = agent.choose_moves(state, engine)
    assert moves == []


def test_determinism_same_seed_same_state(engine):
    state = engine.new_game()
    agent_a = MCTSAgent(n_simulations=200, time_limit=5.0, seed=42)
    agent_b = MCTSAgent(n_simulations=200, time_limit=5.0, seed=42)
    moves_a = agent_a.choose_moves(state.copy(), engine)
    moves_b = agent_b.choose_moves(state.copy(), engine)
    assert moves_a == moves_b


def test_rollout_policy_random_does_not_crash(engine):
    state = engine.new_game()
    agent = MCTSAgent(n_simulations=100, time_limit=3.0, rollout_policy="random", seed=5)
    moves = agent.choose_moves(state, engine)
    assert isinstance(moves, list)
    assert len(moves) >= 1


def test_returns_full_hand_on_open_board(engine):
    state = engine.new_game()
    agent = MCTSAgent(n_simulations=300, time_limit=5.0, seed=6)
    moves = agent.choose_moves(state, engine)
    assert len(moves) == 3
    placed_ids = sorted(pid for pid, _, _ in moves)
    assert placed_ids == sorted(state.pieces)
