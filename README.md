# BlastMind — Comparative AI Agents for Block Blast

Six AI agents for Block Blast, an 8×8 tile-placement puzzle, built on a shared
game engine and compared under controlled conditions: a random baseline, a
greedy heuristic, beam search, Monte Carlo Tree Search, and two agents driven by
a learned afterstate value function (DQN and DQNSearch).

This repository accompanies an MSc dissertation. It contains the game engine,
the agents, the browser interface, the experimental scripts, and the raw
benchmark results reported in that document.

---

## Platform support

| Platform | Supported |
|---|---|
| Windows 10/11 (x64) | Yes |
| Linux (x86_64 or ARM64) | Yes |
| macOS 14+ on Apple Silicon | Yes |
| **macOS on Intel (x86_64)** | **No** |
| **macOS 13 or older** | **No** |

The two learned agents require PyTorch. PyTorch 2.12.0 does not publish an
Intel-Mac wheel, so `pip install -r requirements.txt` will fail on those
machines. Everything else in the project — the engine, the random, greedy, beam
search and tree search agents, and human play — depends only on NumPy and
FastAPI, but the pinned requirements file installs PyTorch regardless.

Python 3.12, 3.13 or 3.14 is required. The pinned SciPy release publishes no
wheels below 3.12, so earlier versions cannot install the dependencies.
Development used 3.12; the environment has also been verified on 3.13.

---

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Installing PyTorch takes a few minutes and several hundred megabytes.

---

## Running the browser application

```bash
uvicorn src.web.main:app
```

Then open <http://127.0.0.1:8000>.

The interface offers three modes: human play, watching a single agent, and
several agents playing the same seeded game side by side.

The trained network the learned agents use is included in this repository at
`models/dqn_vv24_ordersearch_diag_ep205000.pt`, so no additional download is
needed. If that file is removed, agent-watching stops working entirely rather
than degrading, because all agents are constructed together at startup.

---

## Running the tests

```bash
pytest
```

116 tests across 12 files, covering the engine, the agents, and the
training-time interventions.

---

## Reproducing the benchmark

Every game is determined by its seed, so the published results can be
regenerated exactly:

```bash
python scripts/run_final_benchmark.py
```

This replays all six agents over 500 games each on seeds 5000–5499 and writes
one row per seed to `results/final_benchmark_<agent>_500games.csv`, then
recomputes the summary and significance tables. It takes many hours, and is
resumable: re-running it skips seeds already completed.

To report how far a run has got without starting one:

```bash
python scripts/run_final_benchmark.py --progress
```

**A partial run overwrites published data.** Per-game CSVs are named by game
count, so a short run writes `..._3games.csv` and leaves the 500-game files
untouched. The derived artefacts are not named that way: `final_benchmark_summary.csv`
and the per-agent density, heatmap and occupancy files are rewritten from
whatever was just run. Restore them afterwards with:

```bash
git checkout -- results/
```

The three agents that make no random choices — random, greedy and DQN —
reproduce their published per-game scores exactly. The three that sample
internally require their generator state to be restored on resumption; this is
handled automatically, and the qualification is set out in the dissertation.

Figures are regenerated with:

```bash
python scripts/generate_report_figures.py
```

---

## Repository layout

| Path | Contents |
|---|---|
| `src/game/` | Game engine: board, piece definitions, seeded generator, scoring and combo logic |
| `src/ai/` | The six agents, plus variants that were tested and rejected |
| `src/web/` | FastAPI server, websocket endpoint, and the Phaser client under `static/js` |
| `scripts/` | Benchmarking, training, diagnostics and figure generation |
| `tests/` | Test suite |
| `models/` | The deployed network; other checkpoints are not tracked (see below) |
| `results/` | Per-game benchmark output, summary and significance tables, charts |

---

## A note on `models/`

Training produced 865 checkpoints totalling roughly 11.6 GB, so they are excluded
from version control with a single exception: `dqn_vv24_ordersearch_diag_ep205000.pt`,
the deployed network loaded by both learned agents, is committed so that a fresh
clone runs without any further setup. It is the only checkpoint needed to
reproduce the published results.

Scripts and the web server resolve this path relative to the repository root,
so they can be launched from any working directory.
