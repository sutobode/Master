# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research codebase for **"Zero-shot Transfer of Single-Crane Deep RL Policies for Multi-Crane Container Retrieval"** (M-CRP), targeting Transportation Research Part C. It has two layers:

1. **Original single-crane CRP model** (`model/`, `env/`, `trainer.py`, `main.py`) — a POMO-style attention encoder/decoder trained via RL to solve the single-crane Container Retrieval Problem (Shin et al. 2026). This is treated as a frozen, pretrained artifact (`baselines/models/proposed/epoch(100).pt`), not something actively retrained.
2. **This project's contribution**: M-CRP (multi-crane) extension that extracts the frozen encoder+scorer from that model (`policy/zero_shot.py`) and drives it through a new multi-crane environment (`mcenv/`) using pluggable crane-assignment strategies (`strategies/`), without any retraining ("zero-shot transfer").

Read `README.md` and `SESSION_HANDOFF.md` (Vietnamese) for the current experimental results and paper status — `SESSION_HANDOFF.md` is the authoritative "where things left off" doc and is updated at the end of each work session.

## Commands

```bash
pip install -r requirements.txt              # note: file has duplicate/pinned versions; install what's needed for your platform (CPU vs CUDA torch)

python -m pytest tests/ -v                    # full suite, expect 36/36 passed
python -m pytest tests/test_mcenv.py -v        # single file
python -m pytest tests/test_mcenv.py::test_mcenv_interference_detection -v   # single test

python benchmarks/generate_mc_instances.py     # (re)generate 140 M-CRP instances -> benchmarks/mc_instances/lee_mc/

python compare_all.py                          # single-crane: ZeroShot vs all baselines (fast, few instances)
python experiment.py --quick                   # smoke test: 3 instances x 2 cranes x 2 strategies (~5-10s)
python experiment.py --cranes 2 3 --strategies S1 S2 S3 S4   # full M-CRP sweep, in-memory
python run_full_experiment.py --batch_size 15  # same sweep but batched + resumable, writes results/mcrp_experiment_<timestamp>.csv
python compare_mc_baselines.py                 # multi-crane heuristic baselines (M-Lin2015 etc.)
python analysis/run_comprehensive.py            # every baseline x every instance (~48 min)

python -m analysis.analyze                      # statistical report -> results/analysis_report.txt
python -m analysis.visualize                    # figures -> results/figures/
```

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
                         policy chose. All implement assign(env, target_stack, dest_stack) -> crane_id:
                           S1 RoundRobin   - alternate cranes regardless of geometry
                           S2 ZoneSplit    - partition bays into contiguous zones, one per crane (best perf.)
                           S3 LoadBalance  - assign to least-loaded crane
                           S4 GreedyOptimal - 1-step lookahead cost simulation per crane
        |
mcenv/mcenv.py            MCEnv wraps a single base_env: Env and adds crane bookkeeping (crane_bays,
                         crane_rows, crane_busy) plus constraint enforcement:
                           A6 interference: no two cranes in the same bay at once
                           A7 non-crossing: crane left-right order along the bay axis is preserved
                         step(dest_stack, crane_id) re-points base_env.curr_bay/curr_row to the assigned
                         crane's position before delegating to base_env.step(), so travel cost is computed
                         from that crane's actual location.
        |
engine/mcrp_inference.py  run_mcrp_episode(policy, env, strategy, ...) is the driver loop: get policy
                         action -> strategy picks crane -> env.step() -> repeat until terminated.
        |
bounds/lowerbound_mc.py   compute_lb_mc(): Theorem 3 lower bound (LB_retrieval + LB_relocation/n_cranes +
                         LB_interference), used to compute optimality gap % for every run. Built on top of
                         baselines/lowerbound.py's single-crane LB.
        |
experiment.py /            Orchestrate the full sweep over instances x n_cranes x strategies, verify
run_full_experiment.py     backward-compatibility (C=1 zero-shot must match the original model's cost within
                         2%) before each run, and write one row per (instance, n_cranes, strategy) to a CSV.
        |
analysis/analyze.py,        Consume the experiment CSV(s) into statistical tables (analyze.py) and figures
analysis/visualize.py       (visualize.py); analysis/run_comprehensive.py and supplementary_analysis.py
                         produce the full-benchmark and cost-decomposition/case-study numbers used in the paper.
```

Key invariant exploited throughout the tests: **`MCEnv` with `n_cranes=1` must be numerically equivalent to the original `Env`** — this is the backward-compatibility contract that makes "zero-shot" claims valid, and it's actively checked both in `tests/test_mcenv.py` and at the start of every `experiment.py` run (`verify_backward_compatibility`).

### Baselines
`baselines/*.py` (lin2015, kim2016, leveling, durasevic2025, simple/advanced heuristics, lowerbound) are single-crane heuristics compared against the DRL policy. `baselines/multi_crane/multi_crane_baseline.py` wraps a single-crane heuristic with a crane-assignment strategy the same way `policy/zero_shot.py` + `strategies/` do, to produce a fair non-DRL multi-crane baseline (e.g. M-Lin2015).

### Paper / docs
`docs/latex/crp_rl_paper_Q1.tex` is the current paper source (Elsevier `elsarticle`, targeting TR-C). Compile with `pdflatex -interaction=nonstopmode file.tex` run twice. `FIX_PLAN.md` tracks known code issues mapped to proposal claims C1-C5; check it before assuming a component (e.g. GreedyOptimal's lookahead, A7 non-crossing) is fully correct versus a documented gap. `docs/superpowers/` holds planning docs for this project (not shared tooling).
