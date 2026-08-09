import math
import random
import time

from src.game.game_engine import GameEngine, GameState
from src.game.pieces import PIECES, PIECE_IDS
from src.ai.greedy import GreedyAgent

_DEAD_END_PENALTY = -500.0


class _DecisionNode:
    __slots__ = ("grid", "combo", "pwc", "score", "remaining", "children", "untried_actions", "N", "W")

    def __init__(self, grid, combo, pwc, score, remaining):
        self.grid = grid
        self.combo = combo
        self.pwc = pwc
        self.score = score
        self.remaining = remaining
        self.children = {}
        self.untried_actions = None
        self.N = 0
        self.W = 0.0


class _ChanceNode:
    __slots__ = ("grid", "combo", "pwc", "score", "children", "N", "W")

    def __init__(self, grid, combo, pwc, score):
        self.grid = grid
        self.combo = combo
        self.pwc = pwc
        self.score = score
        self.children = {}
        self.N = 0
        self.W = 0.0


class MCTSAgent(GreedyAgent):

    def __init__(self, n_simulations=500, time_limit=2.0, exploration_constant=math.sqrt(2),
                 rollout_policy="heuristic", rollout_hands=2, max_tree_hands=2,
                 max_nodes=50_000, seed=None):
        self.n_simulations = n_simulations
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        self.rollout_policy = rollout_policy
        self.rollout_hands = rollout_hands
        self.max_tree_hands = max_tree_hands
        self.max_nodes = max_nodes
        self._rng = random.Random(seed)

    def choose_moves(self, state: GameState, engine: GameEngine):
        initial_score = state.score
        root = _DecisionNode(state.board.grid.copy(), state.combo_count,
                              state.placements_without_clear, initial_score,
                              tuple(state.pieces))
        self._init_untried(root)

        if not root.untried_actions and not root.children:
            return []

        self._node_count = 1
        self._r_min = float("inf")
        self._r_max = float("-inf")

        start = time.monotonic()
        for _ in range(self.n_simulations):
            if time.monotonic() - start > self.time_limit:
                break
            path, rollout_state = self._select_and_expand(root)
            grid, combo, pwc, score, remaining = rollout_state
            reward = self._rollout(grid, combo, pwc, score, remaining, initial_score)
            self._backprop(path, reward)

        moves = self._extract_moves(root, initial_score)
        if moves:
            return moves

        _, fallback_moves = self._search(root.grid, root.combo, root.pwc,
                                          list(root.remaining), [], initial_score, initial_score)
        return fallback_moves or []

    def _init_untried(self, node):
        actions = []
        for pid in sorted(set(node.remaining)):
            piece = PIECES[pid]
            ph, pw = piece.shape
            for row, col in self._valid(node.grid, piece, ph, pw):
                actions.append((pid, row, col))
        self._rng.shuffle(actions)
        node.untried_actions = actions

    def _apply_action(self, node, action):
        pid, row, col = action
        piece = PIECES[pid]
        ph, pw = piece.shape
        ng, gained, ncb, npc = self._place(node.grid, piece, ph, pw, row, col, node.combo, node.pwc)
        new_score = node.score + gained
        remaining = list(node.remaining)
        remaining.remove(pid)
        if remaining:
            child = _DecisionNode(ng, ncb, npc, new_score, tuple(remaining))
            self._init_untried(child)
        else:
            child = _ChanceNode(ng, ncb, npc, new_score)
        return child

    def _select_and_expand(self, root):
        node = root
        path = [node]
        hands_expanded = 0

        while True:
            if isinstance(node, _DecisionNode):
                if node.untried_actions:
                    if self._node_count >= self.max_nodes:
                        return path, (node.grid, node.combo, node.pwc, node.score, node.remaining)
                    action = node.untried_actions.pop()
                    child = self._apply_action(node, action)
                    node.children[action] = child
                    self._node_count += 1
                    path.append(child)
                    remaining = child.remaining if isinstance(child, _DecisionNode) else ()
                    return path, (child.grid, child.combo, child.pwc, child.score, remaining)

                if node.children:
                    node = self._ucb_select(node)
                    path.append(node)
                    continue

                return path, (node.grid, node.combo, node.pwc, node.score, node.remaining)

            if hands_expanded >= self.max_tree_hands or self._node_count >= self.max_nodes:
                hand = tuple(self._rng.choice(PIECE_IDS) for _ in range(3))
                return path, (node.grid, node.combo, node.pwc, node.score, hand)

            hand = tuple(self._rng.choice(PIECE_IDS) for _ in range(3))
            key = tuple(sorted(hand))
            hands_expanded += 1
            if key in node.children:
                node = node.children[key]
                path.append(node)
                continue

            child = _DecisionNode(node.grid, node.combo, node.pwc, node.score, hand)
            self._init_untried(child)
            node.children[key] = child
            self._node_count += 1
            path.append(child)
            return path, (child.grid, child.combo, child.pwc, child.score, child.remaining)

    def _ucb_select(self, node):
        log_n = math.log(max(node.N, 1))
        best_score = float("-inf")
        best_child = None
        for child in node.children.values():
            mean_q = child.W / child.N
            if self._r_max > self._r_min:
                norm_q = (mean_q - self._r_min) / (self._r_max - self._r_min)
                norm_q = min(1.0, max(0.0, norm_q))
            else:
                norm_q = 0.5
            score = norm_q + self.exploration_constant * math.sqrt(log_n / child.N)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _rollout(self, grid, combo, pwc, score, remaining, initial_score):
        g, cb, pc, sc = grid, combo, pwc, score
        pending = list(remaining)
        hands_left = self.rollout_hands

        while True:
            if not pending:
                if hands_left <= 0:
                    break
                hands_left -= 1
                pending = [self._rng.choice(PIECE_IDS) for _ in range(3)]

            pid = pending.pop(0)
            piece = PIECES[pid]
            ph, pw = piece.shape
            candidates = self._valid(g, piece, ph, pw)
            if not candidates:
                return self._eval(g, sc, initial_score) + _DEAD_END_PENALTY

            if self.rollout_policy == "random":
                row, col = self._rng.choice(candidates)
                g, gained, cb, pc = self._place(g, piece, ph, pw, row, col, cb, pc)
                sc += gained
            else:
                best_val = float("-inf")
                best = None
                for row, col in candidates:
                    ng, gained, ncb, npc = self._place(g, piece, ph, pw, row, col, cb, pc)
                    val = self._eval(ng, sc + gained, initial_score)
                    if val > best_val:
                        best_val = val
                        best = (ng, gained, ncb, npc)
                g, gained, cb, pc = best
                sc += gained

        return self._eval(g, sc, initial_score)

    def _backprop(self, path, reward):
        for node in path:
            node.N += 1
            node.W += reward
        if reward < self._r_min:
            self._r_min = reward
        if reward > self._r_max:
            self._r_max = reward

    def _extract_moves(self, root, initial_score):
        moves = []
        node = root
        g, cb, pc, sc = root.grid, root.combo, root.pwc, root.score

        for _ in range(3):
            if isinstance(node, _DecisionNode) and node.children:
                action, child = max(node.children.items(), key=lambda kv: kv[1].N)
                moves.append(action)
                g, cb, pc, sc = child.grid, child.combo, child.pwc, child.score
                node = child
                continue

            if not isinstance(node, _DecisionNode):
                break

            remaining = list(node.remaining)
            if not remaining:
                break

            pid = remaining[0]
            piece = PIECES[pid]
            ph, pw = piece.shape
            candidates = self._valid(g, piece, ph, pw)
            if not candidates:
                break

            best_val = float("-inf")
            best = None
            for row, col in candidates:
                ng, gained, ncb, npc = self._place(g, piece, ph, pw, row, col, cb, pc)
                val = self._eval(ng, sc + gained, initial_score)
                if val > best_val:
                    best_val = val
                    best = (row, col, ng, gained, ncb, npc)
            row, col, ng, gained, ncb, npc = best
            moves.append((pid, row, col))
            remaining.remove(pid)
            g, cb, pc, sc = ng, ncb, npc, sc + gained
            node = _DecisionNode(g, cb, pc, sc, tuple(remaining))

        return moves
