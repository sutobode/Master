# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase for **"Zero-shot Transfer of Single-Crane Deep RL Policies for Multi-Crane Container Retrieval"** (M-CRP), targeting Transportation Research Part C. It has two layers:

1. **Original single-crane CRP model** (`model/`, `env/`, `trainer.py`, `main.py`) — a POMO-style attention encoder/decoder trained via RL to solve the single-crane Container Retrieval Problem (Shin et al. 2026). This is treated as a frozen, pretrained artifact (`baselines/models/proposed/epoch(100).pt`), not something actively retrained.
2. **This project's contribution**: M-CRP (multi-crane) extension that extracts the frozen encoder+scorer from that model (`policy/zero_shot.py`) and drives it through a new multi-crane environment (`mcenv/`) using pluggable crane-assignment strategies (`strategies/`), without any retraining ("zero-shot transfer").

Read `README.md` for exact commands to reproduce every result, and `docs/superpowers/plans/2026-07-10-script-consolidation-handoff.md` (Vietnamese) for experiment status — this is the most recent and authoritative account of what changed and why, including real-run verification evidence and current dataset state. `docs/superpowers/plans/2026-07-10-revision-handoff.md` explains the earlier cost-model/lower-bound rewrite (still accurate for that part, but its own run commands are superseded). `SESSION_HANDOFF.md`'s pre-2026-07-10 content is historical only (numbers there predate the lower-bound/cost-model rewrite and are no longer valid).

## Commands

```bash
python run_all.py --dry-run                  # master pipeline: prints every step below in order, runs nothing
python run_all.py --skip-shin --skip-multi-large   # small-scale-only pass (fast, verify pipeline works)
python run_all.py --regenerate-mc            # FULL pipeline: every dataset, every method, no instance caps (see --help)

pip install -r requirements.txt              # note: file has duplicate/pinned versions; install what's needed for your platform (CPU vs CUDA torch)

python -m pytest tests/ -v                    # full suite
python -m pytest tests/test_mcenv.py -v        # single file
python -m pytest tests/test_mcenv.py::test_mcenv_c1_episode_cost_identical_to_env -v   # single test

python -m benchmarks.generate_mc_instances     # (re)generate M-CRP layouts -> benchmarks/mc_instances/lee_mc/
                                                # ONE FILE PER UNIQUE LAYOUT (crane count is an experiment
                                                # parameter, not a dataset dimension) — DELETES existing
                                                # files in that directory first; back up before running
                                                # if you have hand-curated instances there.

python -m analysis.run_single_crane_full --dataset lee              # Setting A (C=1), small scale: OriginalModel+ZeroShot+4 heuristics, 70 instances
python -m analysis.run_single_crane_full --dataset shin --max_per_scale 3   # Setting A, large scale (Shin_instances): SAME 6 methods
python experiment.py --quick                                        # smoke test only: 3 instances x 2 cranes x 2 strategies (~5-10s), ZeroShot only
python -m analysis.run_multi_crane_full --dataset small --max_instances 3   # smoke test, Setting B (C=2,3): ZeroShot+3 multi-crane heuristics x 4 strategies
python -m analysis.run_multi_crane_full --dataset small              # full run: 70 layouts x 2 crane-counts x 4 strategies x 4 methods = 2,240 runs
python -m analysis.run_multi_crane_full --dataset large --max_instances 3  # Setting B, large scale (lee_mc_large/, generate first — see below)

python -m analysis.analyze results/multi_crane_small.csv   # statistical report -> results/analysis_report.txt
python -m analysis.visualize_v2                                  # figures -> results/figures_v2/
```

`compare_mc_baselines.py`, `analysis/run_comprehensive.py`, `analysis/supplementary_analysis.py`, `analysis/fix_critical_issues.py`, `analysis/run_mc_baselines_extra.py`, `analysis/visualize.py`, `run_full_experiment.py`, `analysis/run_single_crane_v2.py`, `analysis/run_single_crane_large.py`, `analysis/run_mc_baselines_v2.py`, and `analysis/run_mc_large.py` are **deprecated** (the last four superseded by `run_single_crane_full.py`/`run_multi_crane_full.py`, which unify what those scripts split across incompatible method sets and separate un-joined CSVs — see README.md §4 for why); running any of them now prints a message and exits. `compare_all.py` still works (it doesn't touch the changed `compute_lb_mc`/`parse_instance_file` APIs) but only exercises 2 small configs; prefer `analysis/run_single_crane_full.py` for the full 70-instance comparison. `experiment.py` itself is still imported (not deprecated) for `parse_instance_file`/`load_instance_tensor`/`verify_backward_compatibility`, and its `--quick` CLI remains valid for smoke-testing — but its full-sweep output (ZeroShot only, no baselines) is superseded by `run_multi_crane_full.py` for actual paper numbers. See `README.md` for the full per-dataset, per-baseline experiment matrix.

Everything under `results/` is generated/gitignored output — never hand-edit it, only regenerate via the scripts above.

## Architecture

### Data representation
A yard instance is a 4D tensor `x: (batch, n_bays, n_rows, n_tiers)`, float, where each stack's containers are stored bottom-up and a value of `0` means empty. Almost all environment code reshapes this to `(batch, max_stacks, max_tiers)` internally (`max_stacks = n_bays * n_rows`), where stack index = `bay_idx * n_rows + row_idx` (0-indexed) — this indexing convention (`dest_bay = dest_idx // n_rows + 1`) recurs throughout `env/`, `mcenv/`, `strategies/`, `bounds/`. Container values double as retrieval priority (lower = retrieved sooner); target stack = the stack containing the globally-next container to retrieve.

### Single-crane path (pretrained, frozen)
`env/env.py` (`Env`) implements the CRP simulator: `clear()` auto-retrieves any stack whose top container is next-due, `step(dest_index)` relocates the target stack's top container to `dest_index` and accrues travel+handling cost (`t_acc`, `t_bay`, `t_row`, `t_pd`). `model/encoder.py` + `model/decoder.py` (wired together in `model/model.py`) form the trained attention policy that picks `dest_index` each step; `trainer.py`/`main.py` is the (now largely inactive) RL training loop against this `Env`.

### Zero-shot multi-crane path (this project)
```
policy/zero_shot.py     ZeroShotPolicy: loads epoch(100).pt, extracts decoder's encoder + attention/scoring
                         weights (W_target, W_global, W_K1/K2, W_Q, W_V, MHA) but NOT its env-loop logic,
                         exposing get_scores()/get_action() so an external loop can call it per step.
        |
strategies/*.py          CraneAssignmentStrategy subclasses decide WHICH crane executes the move the
                         policy chose. All implement assign(env, target_stack, dest_stack) -> crane_id,
                         reading env.crane_time/env.crane_bays/env._a7_compatible/env._bay_ready:
                           S1 RoundRobin      - alternate cranes regardless of geometry
                           S2 ZoneSplit       - partition bays into contiguous zones, one per crane (best perf.)
                           S3 LoadBalance     - least accumulated busy time (env.crane_time), ties rotate
                                                (not lowest-index) so it doesn't degenerate to S1/crane-0-only
                           S4 GreedyOptimal   - earliest estimated finish time (release + A6 wait + travel)
        |
mcenv/mcenv.py            MCEnv wraps a single base_env: Env and adds a PER-CRANE TIMING MODEL:
                           crane_time[c] (when crane c is next free), bay_free_time[b] (when bay b is next
                           free). A6 interference is enforced as a WAIT (crane_time is bumped to bay_free_time,
                           counted in interference_events/interference_wait), never by silently swapping
                           which crane executes the strategy's chosen move. A7 non-crossing is enforced by
                           _enforce_a7(): reassigns to the nearest order-compatible crane when the requested
                           one would cross a neighbour (counted in a7_reassignments; the degenerate case where
                           NO crane is compatible, e.g. C > B, is counted separately in a7_violations and the
                           original crane_id is kept). step() ALWAYS sets env.last_crane_id to the crane that
                           actually executed the move — callers must read this, not the crane_id they passed
                           in, since it may differ after A7 enforcement. env.makespan is max(crane_time) —
                           the PRIMARY objective (total working time, `total_cost`, is secondary: adding
                           cranes cannot reduce total work, only makespan).
        |
engine/mcrp_inference.py  run_mcrp_episode(policy, env, strategy, ...) is the driver loop: get policy
                         action -> strategy picks crane -> env.step() -> read env.last_crane_id for
                         attribution -> repeat until terminated. Returns total_cost, makespan, and
                         interference/a7 counters.
        |
bounds/lowerbound_mc.py   compute_lb_mc() returns {'work': tensor, 'makespan': tensor} — NOT a single
                         tensor. Both are provably valid lower bounds (never exceed a feasible schedule's
                         cost, unlike an earlier revision that divided a single-crane traversal bound by C
                         and could go negative). Raises ValueError if t_pd < (n_rows-2)*t_row, the condition
                         the "fixed" per-container term's validity proof depends on. At C=1 both bounds
                         collapse to baselines/lowerbound.py's single-crane Theorem-2 bound.
        |
analysis/run_multi_crane_full.py   Orchestrates the full sweep over instances x n_cranes x strategies x
                         {ZeroShot, M-Lin2015, M-Kim2016, M-Leveling} for a chosen `--dataset {small,large}`,
                         verifies backward-compatibility (C=1 zero-shot must match the original model's cost
                         EXACTLY, tolerance 0.01%, via experiment.py's verify_backward_compatibility) before
                         each run, asserts gap_work/gap_makespan >= 0 for every run (a violation means the
                         lower bound itself is wrong — investigate, don't silence), and writes one row per
                         (instance, n_cranes, strategy, method) to a single CSV with columns total_cost/
                         makespan/gap_work/gap_makespan/interference_wait/a7_reassignments/a7_violations.
                         analysis/run_single_crane_full.py is the C=1-only counterpart (--dataset {lee,shin}).
        |
analysis/analyze.py,        Consume the experiment CSV(s) into statistical tables (analyze.py, primary
analysis/visualize_v2.py    metric gap_makespan) and figures (visualize_v2.py, 300 DPI).
```

Key invariant exploited throughout the tests: **`MCEnv` with `n_cranes=1` must be numerically equivalent to the original `Env`** — this is the backward-compatibility contract that makes "zero-shot" claims valid. It holds to **0.00%** (not just "within a few %"): the extracted policy's action sequence and the simulator's cost accounting are both exact reproductions of the original at C=1. Checked in `tests/test_mcenv.py::test_mcenv_c1_episode_cost_identical_to_env` and at the start of every `analysis/run_multi_crane_full.py` run (`verify_backward_compatibility`, imported from `experiment.py`).

### Baselines
`baselines/*.py` (lin2015, kim2016, leveling, durasevic2025, simple/advanced heuristics, lowerbound) are single-crane heuristics compared against the DRL policy. `baselines/multi_crane/multi_crane_baseline.py` re-implements each heuristic's *destination rule* (`LinDest`/`KimDest`/`LevelingDest`, the "restricted" variant matching assumption A3) and drives it through the identical `MCEnv` + strategy + timing-model protocol as ZeroShot via `run_mc_heuristic_episode()`, so `analysis/run_multi_crane_full.py` is a move-for-move fair comparison — it does NOT run the original single-crane episode and rescale its cost (an earlier, invalid version of this file did that). These filenames correspond to methods compared in the original Shin et al. paper (`1-s2.0-S0968090X25005005-main.pdf`): `lin2015.py`↔Lin, `kim2016.py`↔Kim, `leveling.py`↔the Zehendner et al. online-CRP heuristic, `durasevic2025.py`↔GP; the paper's own TS/GRASP/DRL1/DRL2 baselines are not reimplemented here.

### Benchmark datasets (`benchmarks/`)
Three distinct instance sources, not interchangeable:
- `Lee_instances/` — the original Lee & Lee (2010) benchmark (70–720 containers, `Individual, random/` and `Individual, upside down/` subfolders), used for single-crane comparisons.
- `Shin_instances/` — the extremely-large-scale benchmark (20/30 bays, 1440–2880 containers) generated by `generate_benchmarks.py` following Shin et al.'s procedure, same `random`/`upside down` split.
- `mc_instances/lee_mc/` — this project's own M-CRP layouts, produced by `python -m benchmarks.generate_mc_instances` from the Lee layouts. **One file per unique layout** (`mc_<type><bays><rows><tiers>_<idx>.txt`, no crane-count suffix); the file's header carries deterministic crane start bays for both `C=2` and `C=3` (`# crane_start_bays_c2 = [...]`, `# crane_start_bays_c3 = [...]`), and `experiment.py`'s `parse_instance_file()` returns them as a `{2: [...], 3: [...]}` dict. An earlier revision emitted a separate `_c2.txt`/`_c3.txt` file pair per layout with byte-identical yard content, double-counting every layout — do not reintroduce that format.
Instance filenames encode `(type)(bays)(rows)(tiers)_(n_containers)_(idx)`, e.g. `R021606_0140_001` = random, 2 bays, 16 rows, 6 tiers, 140 containers; `benchmarks/benchmarks.py:layout_to_n_containers` maps `(bays, rows, tiers) -> n_containers` for the standard Lee scales. `find_and_process_file()` in `benchmarks/benchmarks.py` is the shared loader for both `Lee_instances` and `Shin_instances`.

### Paper / docs
`docs/latex/crp_rl_paper_Q1.tex` is the current paper source (Elsevier `elsarticle`, targeting TR-C); it contains `%%FILL_...` placeholders where numbers must be filled in from a fresh experiment run (see README.md) before it will read as a finished draft. Compile with `pdflatex -interaction=nonstopmode file.tex` run twice. `docs/superpowers/plans/2026-07-10-revision-handoff.md` is the most detailed account of the lower-bound/cost-model/dataset rewrite; `FIX_PLAN.md` and `SESSION_HANDOFF.md` predate it and describe the *previous* (since-superseded) state. `docs/superpowers/` holds planning docs for this project (not shared tooling).
