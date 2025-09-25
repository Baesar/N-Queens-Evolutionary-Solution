# N‑Queens Evolutionary Solver — Architecture Overview

Audience: short overview with focused deep dives on complex parts. Language: English. For usage flow and implementation design decisions. Function references for advanced/complex pieces are included; simpler utilities are summarized in an appendix.

## Goals
- Solve N‑Queens efficiently across a range of board sizes `n`.
- Use a Genetic Algorithm (GA) with local improvements and adaptive hyperparameters.
- Respect a dynamic wall‑clock budget that adapts to progress and scales with `n`.
- Support repeated runs and persistent parameter training that improves defaults over time.

## High‑Level Design
- Representation: Each board is a permutation‑like vector of length `n` where index = column and value = row. Fitness derives from non‑attacking pairs; attacks are minimized to zero.
- GA Loop: Initialize population → selection (tournament) → crossover (one‑point/uniform) → mutation → local greedy repair → elitism/diversity → adapt hyperparameters → optional early stop or restart.
- Dynamic Time: Each run gets a base time window that can extend on improvements (and optionally shrink on stagnation). The window scales with `n` via `dyn_time_window_for`.
- Training: Batched evaluation compares median duration, success rate, and average residual attacks. Accepted improvements persist to `params/param_store.json`.

## Modules and Responsibilities
- `board.py` — Board state, fitness, attacks, mutation.
- `app.py` — GA operators and main loop, evaluation and training orchestration, CLI.
- `constants.py` — Defaults, schedules, per‑`n` run policies, dynamic time scaling, training knobs, and store helpers.
- `view.py` — Output helpers (pretty printing boards).
- `params/*` — Persistent artifacts: param store and training logs (ignored by VCS).

## Runtime Flows
1) CLI parse and mode selection (single solve, repeated runs, or training) — `app.py:1321`.
2) If training: repeated batches call `evaluate_settings_over_runs` and explore tweaks — `app.py:856`.
3) If solving: call `run_ga` with dynamic time enabled (by default) — `app.py:265`.
4) Inside GA: evolve and adapt until solved, timeout, or early stop — `app.py:265`.
5) Persist improvements and optionally print the final board — `app.py:1498`.

## Key Design Decisions (Why)
- Tournament selection with bandit‑weighted crossover encourages exploration of the two crossover styles without hard coding a single choice.
- Local greedy repair (“min‑conflicts” flavor) speeds convergence by reducing attacks after mutation and crossover, acting like a repair heuristic rather than a full local search.
- Adaptive mutation/selection pressure balances exploration early (higher mutation) and exploitation near the solution (lower mutation, higher tournament‑k) to reduce plateaus.
- Dynamic time windows protect wall‑clock budgets while giving promising runs a bit more room to converge.
- Persistent training builds a knowledge base of good defaults per `n`, improving out‑of‑the‑box performance over time.

## Complex Components (Detailed)

### GA Main Loop — `run_ga`
- Where: `app.py:265`
- Purpose: Execute the GA for a single run on size `n`, honoring dynamic time; return best board and run stats.
- Signature: `run_ga(settings, board_size, time_limit_s=None, verbose=True, dynamic_time=False, trace_path=None, phase_summary=None, schedule=None) -> GAResult`
- Inputs:
  - `GASettings`: population, generations, elitism, tournament‑k, rates.
  - `board_size`: `n`.
  - `time_limit_s`: optional hard cap; if `dynamic_time` is on, treated as base window.
  - `schedule`: adaptive knobs (mut step up/down, repair limits, immigrant frac…).
- Returns: `GAResult(best: Board, duration: float, generations: int)`.
- Side effects: Optional trace TSV; prints progress when `verbose`.
- How it works:
  - Initializes population and best.
  - If `dynamic_time`: compute `(base, max)` from `dyn_time_window_for(n)`; deadline extends by `DYN_TIME_EXTEND_S` on improvements.
  - Each generation: selection (tournament), crossover (weighted one‑point vs uniform), mutation, local repair, elitism cap, optional immigrant injection, adaptive hyperparameter tweaks on cadence.
  - Early stop on stagnation, optional soft restart of worst fraction.
- Why: Combines global search (GA) with light local search (repair), and uses dynamic/adaptive knobs to avoid long stagnations while controlling time.

### Dynamic Time Scaling — `dyn_time_window_for`
- Where: `constants.py:297`
- Purpose: Compute `(base_seconds, max_seconds)` per `n` using optional explicit overrides and a linear interpolation across anchor sizes.
- Signature: `dyn_time_window_for(n: int) -> (float, float)`
- Inputs: `n` board size. Checks `DYN_TIME_BASE_PER_BOARD` / `DYN_TIME_MAX_PER_BOARD` first.
- Returns: `(base, max)` seconds.
- Why: Larger instances often need more time; scaling avoids one‑size‑fits‑all timeouts and improves success rate for big `n`.

### Training Orchestrator — `train_parameters`
- Where: `app.py:856`
- Purpose: Run many short batches, accept parameter or schedule changes that improve median duration (or average attacks) without hurting success; persist the best.
- Signature: `train_parameters(board_size, kilo=1, batch_runs=34, explore_prob=..., time_limit_s=None, workers=None, dynamic_time=True, trace_every=0, mp_start=None)`
- Inputs: `board_size`, batch size, total budget (`kilo*1000` runs), pool size, dynamic time flag.
- Returns: None (persists to `params/param_store.json`).
- Side effects: Appends CSV/TSV/text‑column logs in `params/`; updates param store.
- How it works:
  - Start from recommended or stored best; evaluate baseline with `evaluate_settings_over_runs`.
  - Propose candidates: systematic deltas, midpoint sweeps between last two acceptances, occasional perturbations, and aggressive probes on long plateaus.
  - Accept if median time improves (or avg attacks drops) with non‑worse success.
- Why: Empirically tune both GA hyperparameters and adaptive schedule per `n`, capturing improvements for future runs.

### Batch Evaluation — `evaluate_settings_over_runs`
- Where: `app.py:602`
- Purpose: Run `run_ga` multiple times (optionally parallel) and summarize metrics.
- Signature: `evaluate_settings_over_runs(settings, board_size, runs, time_limit_s, workers=None, dynamic_time=False, schedule=None) -> dict`
- Outputs: median duration, success rate, average generations, fitness, attacks.
- Side effects: Uses process pool (configurable start method) when `workers>1`.
- Why: Robustly compare parameter candidates with enough samples for stable medians.

### Selection/Crossover/Repair
- Tournament selection — `app.py:130`:
  - Picks best among `k` random candidates. Returns clone to avoid aliasing.
  - Why: Simple, strong selection pressure, tunable via `k`.
- Crossover — `app.py:136` and `app.py:144` with wrapper at `app.py:157`:
  - One‑point and uniform operators; bandit‑style weights updated based on child vs parent fitness.
  - Why: Allow the system to favor the operator that currently helps more.
- Local greedy repair — `app.py:175` and finish `app.py:221`:
  - Small per‑child repair to reduce attacks; optional final greedy finish when guaranteeing a solution.
  - Why: Hybridizes GA with local search for faster convergence.

### Board Representation — `Board`
- Where: `board.py:4`
- Purpose: Encapsulate state, fitness, attacks, mutation, cloning.
- Used by: GA operators and evaluation.
- Why: Keeps core problem logic separate from GA orchestration.

## CLI and Execution Modes
- Parser — `app.py:1321`.
- Single solve (default): compute and print best board; if not solved and guarantee is on, attempt greedy finish and limited restarts.
- Repeated runs: `--repeats N` prints summary and persists improved results.
- Training: `--train-params` / `--train-kilo` orchestrates batches and persistence.
- Dynamic time: Enabled by default; `--time-limit` sets the base cap, still extendable up to computed max.

## Persistence and Logs
- Parameter store: `params/param_store.json` — best settings/schedule/metrics per `n`.
- Logs: `params/training_log_n{n}.csv|.tsv|.text_column.txt` — human‑readable summaries and quick scanning of progress.
- VCS: `params/*` is ignored; only `.gitkeep` remains to keep the folder in the repo.

## Appendix: Utilities and Helpers (Brief)
- `constants.py:35 recommend_params(n)` — Baseline guesses for GA settings, optionally seeded from nearest stored `n`.
- `constants.py:366 default_schedule()` / `386 recommend_schedule(n)` — Adaptive knobs; per‑`n` recommendations from store or nearest neighbor.
- `app.py:718 _perturb(settings)` / `729 _perturb_schedule(sched)` — Random tweaks for exploration.
- `app.py:756 _systematic_candidates(...)` / `805 _midpoint_candidates(...)` — Candidate generation strategies.
- Worker wrappers and store I/O — `app.py:527 _load_store`, `537 _save_store`, `544 _worker_run_once`, `563 _worker_run_many`.

## Extending the System
- Add a new crossover or mutation operator: implement and register in `app.py`, include in the bandit weighting, and optionally expose via schedule.
- Adjust dynamic time scaling: tune anchors in `constants.py:297 dyn_time_window_for` or add explicit per‑`n` overrides.
- Add new acceptance criteria for training: extend the check in `train_parameters` to combine additional metrics.

