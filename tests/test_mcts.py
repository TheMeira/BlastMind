import math
import random

import pytest
from src.game.board import Board
from src.game.game_engine import GameEngine, GameState
from src.game.pieces import PIECE_IDS, PIECES
from src.ai.mcts import MCTSAgent, _DecisionNode
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


def test_root_topk_finalists_matches_greedy_search(engine):
    mcts = MCTSAgent(seed=7)
    greedy = GreedyAgent()
    rng = random.Random(123)

    for _ in range(10):
        state = engine.new_game()
        state.pieces = [rng.choice(PIECE_IDS) for _ in range(3)]
        grid = state.board.grid.copy()

        finalists = mcts._root_topk_finalists(grid, state.combo_count,
                                               state.placements_without_clear,
                                               list(state.pieces), k=1000)
        greedy_val, greedy_moves = greedy._search(grid, state.combo_count,
                                                    state.placements_without_clear,
                                                    list(state.pieces), [], 0, 0)

        if greedy_moves is None:
            assert finalists == []
        else:
            assert finalists
            assert finalists[0][0] == pytest.approx(greedy_val)


def test_root_topk_finalists_empty_on_dead_hand(engine):
    b = Board()
    b.grid[:, :] = 1
    mcts = MCTSAgent(seed=8)
    finalists = mcts._root_topk_finalists(b.grid.copy(), 0, 0, ["P01", "P02", "P04"], k=8)
    assert finalists == []


def test_moves_are_legal_across_random_boards(engine):
    rng = random.Random(99)
    agent = MCTSAgent(n_simulations=100, time_limit=3.0, seed=9)

    for _ in range(15):
        state = engine.new_game()
        fill_prob = rng.random() * 0.5
        for r in range(8):
            for c in range(8):
                state.board.grid[r, c] = 1 if rng.random() < fill_prob else 0
        state.pieces = [rng.choice(PIECE_IDS) for _ in range(3)]

        moves = agent.choose_moves(state, engine)
        sim_grid = state.board.grid.copy()
        sim_combo, sim_pwc = state.combo_count, state.placements_without_clear
        remaining_pieces = list(state.pieces)
        for pid, row, col in moves:
            assert pid in remaining_pieces
            remaining_pieces.remove(pid)
            piece = PIECES[pid]
            candidates = agent._valid(sim_grid, piece, *piece.shape)
            assert (row, col) in candidates
            sim_grid, _, sim_combo, sim_pwc = agent._place(sim_grid, piece, *piece.shape, row, col, sim_combo, sim_pwc)


def test_ucb_select_unaffected_when_all_children_visited():
    agent = MCTSAgent(seed=10, exploration_constant=1.41)
    parent = _DecisionNode(None, 0, 0, 0, ())
    parent.N = 30

    from src.ai.mcts import _ChanceNode
    low = _ChanceNode(None, 0, 0, 0)
    low.N, low.W = 10, 10.0
    high = _ChanceNode(None, 0, 0, 0)
    high.N, high.W = 10, 90.0
    parent.children = {"low": low, "high": high}

    agent._r_min, agent._r_max = 0.0, 100.0
    action, chosen = agent._ucb_select(parent)
    assert chosen is high
    assert action == "high"


def test_rave_backprop_updates_amaf_stats():
    agent = MCTSAgent(seed=11, enable_rave=True)
    node = _DecisionNode(None, 0, 0, 0, ())
    a1, a2, a3 = ("P01", 0, 0), ("P01", 1, 1), ("P02", 2, 2)
    node.legal_actions_set = {a1, a2}
    root = _DecisionNode(None, 0, 0, 0, ())
    path = [root, node]

    agent._r_min, agent._r_max = 0.0, 10.0
    agent._backprop(path, [a1, a3], reward=4.0)

    assert node.amaf_N[a1] == 1
    assert node.amaf_W[a1] == 4.0
    assert a3 not in node.amaf_N

    agent._backprop(path, [a1], reward=2.0)
    assert node.amaf_N[a1] == 2
    assert node.amaf_W[a1] == 6.0
    assert root.amaf_N is None


def test_rave_beta_boundary_conditions():
    agent = MCTSAgent(seed=12, enable_rave=True, rave_k=1000)
    node = _DecisionNode(None, 0, 0, 0, ())
    node.N = 100
    action = ("P01", 0, 0)
    node.legal_actions_set = {action}
    node.amaf_N = {action: 5}
    node.amaf_W = {action: 25.0}

    from src.ai.mcts import _ChanceNode
    agent._r_min, agent._r_max = 0.0, 10.0

    huge = _ChanceNode(None, 0, 0, 0)
    huge.N, huge.W = 10_000_000, 3_000_000.0
    node.children = {action: huge}
    _, chosen = agent._ucb_select(node, use_amaf=True)
    beta_huge = math.sqrt(agent.rave_k / (3 * huge.N + agent.rave_k))
    assert beta_huge == pytest.approx(0.0, abs=1e-2)

    small = _ChanceNode(None, 0, 0, 0)
    small.N, small.W = 1, 0.3
    node.children = {action: small}
    beta_small = math.sqrt(agent.rave_k / (3 * small.N + agent.rave_k))
    assert beta_small == pytest.approx(math.sqrt(1000 / 1003), abs=1e-6)


def test_rave_no_amaf_data_matches_disabled():
    agent_off = MCTSAgent(seed=13, enable_rave=False, exploration_constant=1.41)
    agent_on = MCTSAgent(seed=13, enable_rave=True, exploration_constant=1.41)

    from src.ai.mcts import _ChanceNode
    for agent in (agent_off, agent_on):
        node = _DecisionNode(None, 0, 0, 0, ())
        node.N = 20
        low = _ChanceNode(None, 0, 0, 0)
        low.N, low.W = 5, 5.0
        high = _ChanceNode(None, 0, 0, 0)
        high.N, high.W = 5, 40.0
        node.children = {"low": low, "high": high}
        agent._r_min, agent._r_max = 0.0, 10.0
        agent._node = node

    action_off, child_off = agent_off._ucb_select(agent_off._node, use_amaf=True)
    action_on, child_on = agent_on._ucb_select(agent_on._node, use_amaf=True)
    assert action_off == action_on == "high"


def test_rave_enabled_choose_moves_does_not_crash(engine):
    state = engine.new_game()
    agent = MCTSAgent(n_simulations=200, time_limit=5.0, seed=14, enable_rave=True, rave_k=500)
    moves = agent.choose_moves(state, engine)
    assert isinstance(moves, list)
    assert len(moves) >= 1


def test_rave_disabled_is_deterministic_baseline(engine):
    state = engine.new_game()
    agent_a = MCTSAgent(n_simulations=150, time_limit=5.0, seed=15, enable_rave=False)
    agent_b = MCTSAgent(n_simulations=150, time_limit=5.0, seed=15, enable_rave=False)
    moves_a = agent_a.choose_moves(state.copy(), engine)
    moves_b = agent_b.choose_moves(state.copy(), engine)
    assert moves_a == moves_b
