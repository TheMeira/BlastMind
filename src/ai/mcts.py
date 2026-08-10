import heapq
import itertools
import math
import random
import time

from src.game.game_engine import GameEngine, GameState
from src.game.pieces import PIECES, PIECE_IDS
from src.ai.greedy import GreedyAgent

_DEAD_END_PENALTY = -500.0


class _DecisionNode:
    __slots__ = ("grid", "combo", "pwc", "score", "remaining", "children", "untried_actions", "N", "W",
                 "amaf_N", "amaf_W", "legal_actions_set")

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
        self.amaf_N = None
        self.amaf_W = None
        self.legal_actions_set = None


class _ChanceNode:
    __slots__ = ("grid", "combo", "pwc", "score", "children", "N", "W", "moves")

    def __init__(self, grid, combo, pwc, score, moves=None):
        self.grid = grid
        self.combo = combo
        self.pwc = pwc
        self.score = score
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.moves = moves


class MCTSAgent(GreedyAgent):

    def __init__(self, n_simulations=500, time_limit=2.0, exploration_constant=math.sqrt(2),
                 rollout_policy="heuristic", rollout_hands=2, max_tree_hands=2,
                 max_nodes=50_000, root_top_k=8, enable_rave=False, rave_k=1000, seed=None):
        self.n_simulations = n_simulations
        self.time_limit = time_limit
        self.exploration_constant = exploration_constant
        self.rollout_policy = rollout_policy
        self.rollout_hands = rollout_hands
        self.max_tree_hands = max_tree_hands
        self.max_nodes = max_nodes
        self.root_top_k = root_top_k
        self.enable_rave = enable_rave
        self.rave_k = rave_k
        self._rng = random.Random(seed)

    def choose_moves(self, state: GameState, engine: GameEngine):
        initial_score = state.score
        grid = state.board.grid.copy()
        combo = state.combo_count
        pwc = state.placements_without_clear
        piece_ids = list(state.pieces)

        finalists = self._root_topk_finalists(grid, combo, pwc, piece_ids, self.root_top_k)
        if not finalists:
            return []

        root = _DecisionNode(grid, combo, pwc, initial_score, tuple(piece_ids))
        root.untried_actions = []
        for _, gained, g, cb, pc, moves in finalists:
            root.children[moves] = _ChanceNode(g, cb, pc, initial_score + gained, moves=moves)

        self._node_count = 1 + len(finalists)
        self._r_min = float("inf")
        self._r_max = float("-inf")

        start = time.monotonic()
        for _ in range(self.n_simulations):
            if time.monotonic() - start > self.time_limit:
                break
            path, path_actions, rollout_state = self._select_and_expand(root)
            g, cb, pc, sc, remaining = rollout_state
            reward, rollout_actions = self._rollout(g, cb, pc, sc, remaining, initial_score)
            self._backprop(path, path_actions + rollout_actions, reward)

        best = max(root.children.values(), key=lambda c: c.N)
        return list(best.moves)

    def _root_topk_finalists(self, grid, combo, pwc, piece_ids, k):
        heap = []
        tiebreak = itertools.count()

        def expand(g, cb, pc, remaining, gained, moves):
            if not remaining:
                val = self._eval(g, gained, 0)
                entry = (val, next(tiebreak), gained, g, cb, pc, tuple(moves))
                if len(heap) < k:
                    heapq.heappush(heap, entry)
                elif val > heap[0][0]:
                    heapq.heapreplace(heap, entry)
                return

            pid = remaining[0]
            piece = PIECES[pid]
            ph, pw = piece.shape

            for row, col in self._valid(g, piece, ph, pw):
                ng, add, ncb, npc = self._place(g, piece, ph, pw, row, col, cb, pc)
                expand(ng, ncb, npc, remaining[1:], gained + add,
                       moves + [(pid, row, col)])

        expand(grid, combo, pwc, piece_ids, 0, [])

        finalists = [(val, gained, g, cb, pc, moves) for val, _, gained, g, cb, pc, moves in heap]
        finalists.sort(key=lambda t: t[0], reverse=True)
        return finalists

    def _init_untried(self, node):
        actions = []
        for pid in sorted(set(node.remaining)):
            piece = PIECES[pid]
            ph, pw = piece.shape
            for row, col in self._valid(node.grid, piece, ph, pw):
                actions.append((pid, row, col))
        self._rng.shuffle(actions)
        node.untried_actions = actions
        node.legal_actions_set = set(actions)

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
        path_actions = []
        hands_expanded = 0

        while True:
            if isinstance(node, _DecisionNode):
                if node.untried_actions:
                    if self._node_count >= self.max_nodes:
                        return path, path_actions, (node.grid, node.combo, node.pwc, node.score, node.remaining)
                    action = node.untried_actions.pop()
                    child = self._apply_action(node, action)
                    node.children[action] = child
                    self._node_count += 1
                    path.append(child)
                    if node is root:
                        path_actions.extend(action)
                    else:
                        path_actions.append(action)
                    remaining = child.remaining if isinstance(child, _DecisionNode) else ()
                    return path, path_actions, (child.grid, child.combo, child.pwc, child.score, remaining)

                if node.children:
                    action, child = self._ucb_select(node, use_amaf=(node is not root))
                    if node is root:
                        path_actions.extend(action)
                    else:
                        path_actions.append(action)
                    node = child
                    path.append(node)
                    continue

                return path, path_actions, (node.grid, node.combo, node.pwc, node.score, node.remaining)

            if hands_expanded >= self.max_tree_hands or self._node_count >= self.max_nodes:
                hand = tuple(self._rng.choice(PIECE_IDS) for _ in range(3))
                return path, path_actions, (node.grid, node.combo, node.pwc, node.score, hand)

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
            return path, path_actions, (child.grid, child.combo, child.pwc, child.score, child.remaining)

    def _ucb_select(self, node, use_amaf=True):
        use_amaf = use_amaf and self.enable_rave
        log_n = math.log(max(node.N, 1))
        best_score = float("-inf")
        best_action = None
        best_child = None
        for action, child in node.children.items():
            if child.N == 0:
                return action, child
            mean_q = child.W / child.N
            combined_q = mean_q
            if use_amaf and node.amaf_N is not None and node.amaf_N.get(action, 0) > 0:
                amaf_q = node.amaf_W[action] / node.amaf_N[action]
                beta = math.sqrt(self.rave_k / (3 * child.N + self.rave_k))
                combined_q = (1 - beta) * mean_q + beta * amaf_q
            if self._r_max > self._r_min:
                norm_q = (combined_q - self._r_min) / (self._r_max - self._r_min)
                norm_q = min(1.0, max(0.0, norm_q))
            else:
                norm_q = 0.5
            score = norm_q + self.exploration_constant * math.sqrt(log_n / child.N)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _rollout(self, grid, combo, pwc, score, remaining, initial_score):
        g, cb, pc, sc = grid, combo, pwc, score
        pending = list(remaining)
        hands_left = self.rollout_hands
        actions_taken = []

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
                return self._eval(g, sc, initial_score) + _DEAD_END_PENALTY, actions_taken

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
                        best = (row, col, ng, gained, ncb, npc)
                row, col, g, gained, cb, pc = best
                sc += gained

            actions_taken.append((pid, row, col))

        return self._eval(g, sc, initial_score), actions_taken

    def _backprop(self, path, all_actions, reward):
        for node in path:
            node.N += 1
            node.W += reward
        if reward < self._r_min:
            self._r_min = reward
        if reward > self._r_max:
            self._r_max = reward

        if not self.enable_rave:
            return

        for node in path[1:]:
            if not isinstance(node, _DecisionNode) or node.legal_actions_set is None:
                continue
            for a in all_actions:
                if a in node.legal_actions_set:
                    if node.amaf_N is None:
                        node.amaf_N = {}
                        node.amaf_W = {}
                    node.amaf_N[a] = node.amaf_N.get(a, 0) + 1
                    node.amaf_W[a] = node.amaf_W.get(a, 0.0) + reward
