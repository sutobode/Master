"""Master pipeline: runs every step needed to reproduce all paper numbers,
in order, in ONE command. Mirrors README.md section 5 exactly -- this script
does not add any new experiment, it just sequences the existing ones so you
do not have to run 9 commands by hand.

RESUME: steps 3-6 (analysis.run_single_crane_full / run_multi_crane_full)
write their output CSV incrementally (flush+fsync after every completed
instance/layout), so if the machine loses power, crashes, or you Ctrl+C
mid-run, just re-run the SAME command -- each script detects which
instances/layouts are already in its output CSV and only processes what's
left. Verified via real interrupt-and-resume tests (not just reasoning about
the code): killing a run partway through and resuming it 2-3 times in a row
reproduces bit-identical ZeroShot results to an uninterrupted run. Pass
--fresh to either script directly (not exposed as a run_all.py flag, since
starting over should be a deliberate per-step choice) to discard existing
partial output and start that step over.

DEFAULT = FULL DATA, EVERY DATASET, NO SUBSAMPLING. Every step below always
runs the proposed method (ZeroShot / S1-S4) against every applicable
comparator on the FULL benchmark, by default:
  - Setting A (single-crane, C=1): proposed method (ZeroShot) vs SOTA
    (Original Model, Shin et al. 2026) vs 4 published heuristic baselines
    (Lin2015/Kim2016/Leveling/Durasevic2025) -- on ALL 70 Lee_instances AND
    ALL 160 Shin_instances (large scale). No "Original Model" exists for
    C>1 in prior work -- that gap is exactly what this project fills.
  - Setting B (multi-crane, C in {2,3}): proposed method (ZeroShot x S1-S4)
    vs 3 published heuristics extended to multi-crane (M-Lin2015/M-Kim2016/
    M-Leveling, x S1-S4) -- on ALL 70 small M-CRP layouts AND ALL (up to
    160) large M-CRP layouts, both crane counts.
`--max-per-scale`/`--max-instances-large` exist only to SHORTEN a test run;
leaving them unset (the default) means full data on every dataset.

Steps (each can be skipped/tuned via flags -- see --help):
  0. pytest tests/ -v                                   (correctness gate)
  1. [conditional] regenerate benchmarks/mc_instances/lee_mc/ to the new
     1-file-per-layout format -- DESTRUCTIVE (deletes old files in that dir),
     so this step only RUNS if --regenerate-mc is passed; otherwise it just
     detects the old format and tells you to pass the flag.
  2. [conditional] generate benchmarks/mc_instances/lee_mc_large/ if missing
     -- NOT destructive (writes to its own fresh directory), so this runs
     automatically whenever the directory is absent.
  3. analysis.run_single_crane_full --dataset lee                      (Setting A, small, ALL 70 instances)
  4. analysis.run_single_crane_full --dataset shin                     (Setting A, large, ALL 160 instances)
  5. analysis.run_multi_crane_full  --dataset small                    (Setting B, small, ALL 70 layouts)
  6. analysis.run_multi_crane_full  --dataset large                   (Setting B, large, ALL available layouts)
  7. analysis.analyze results/multi_crane_small.csv
  8. analysis.visualize_v2

Steps 4 and 6 (large scale) are the slowest by far -- each instance has
1440-2880 containers vs 70-720 for small scale, and step 6 alone is
4 methods x 4 strategies x 2 crane-counts x up to 160 layouts.

TWO independent speedups now applied (both in analysis/run_single_crane_full.py
and/or analysis/run_multi_crane_full.py):
  1. Redundant-DRL-call elimination (Setting B only): ZeroShot's decision
     sequence is invariant to n_cranes/strategy (record_zeroshot_trajectory()
     + replay_zeroshot_episode() in engine/mcrp_inference.py) -- computed
     once per instance instead of once per (n_cranes x strategy) combo.
     Measured ~3.9-3.95x speedup, bit-identical results (tests/test_engine.py).
  2. Multiprocessing across instances (both settings, --workers, default
     cpu_count()-2): each instance/layout is fully independent, so they run
     in parallel worker processes (each pinned to 1 thread via
     torch.set_num_threads(1)). Measured ~2-2.2x on small test batches
     (4-16 tasks) where per-worker startup overhead is proportionally large;
     expected closer to the worker count on the real 70-160 instance/layout
     runs where that overhead is negligible -- NOT measured at full scale
     (would take hours), watch the live ETA the scripts print.

ESTIMATES below combine both, extrapolated from small measured samples --
treat as optimistic, not a promise; real time may be higher (MCEnv/DRL cost
can grow faster than linearly with yard size; sustained multi-core load on a
laptop can throttle). See docs/superpowers/plans/2026-07-10-script-consolidation-handoff.md
for the raw per-method numbers behind these:
  Step 3 (Setting A, small, 70 instances):    ~3-6 min
  Step 4 (Setting A, large, 160 instances):   ~1.5-3 hours
  Step 5 (Setting B, small, 70 layouts):      ~5-15 min
  Step 6 (Setting B, large, 160 layouts):     ~40 min-1.5 hours
  Full pipeline, nothing skipped:             ~2.5-5 hours (down from ~1.5-2 days originally)
Use --skip-shin/--skip-multi-large for a fast small-scale-only pass first to
confirm everything works, then run the full thing. If you see the live ETA
climbing over time instead of holding steady (thermal throttling), pass a
lower --workers value (e.g. 6-8) for long unattended runs.

A step that fails (non-zero exit, e.g. an assertion like "gap < 0 -- bound
invalid" or "backward-compatibility FAILED") STOPS the whole pipeline by
default -- these assertions exist specifically to catch a broken lower bound
or a broken zero-shot fidelity claim, so silently continuing past one would
produce numbers that must not go in the paper. Pass --continue-on-error to
override (not recommended).

Usage:
  python run_all.py --dry-run                              # just print what would run
  python run_all.py --skip-shin --skip-multi-large           # small-scale only (fast), verify pipeline works
  python run_all.py --regenerate-mc                          # FULL run: every dataset, every method, no caps
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
LEE_MC_DIR = os.path.join(ROOT, 'benchmarks', 'mc_instances', 'lee_mc')
LEE_MC_LARGE_DIR = os.path.join(ROOT, 'benchmarks', 'mc_instances', 'lee_mc_large')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true', help='Print the steps that would run, without running them')
    p.add_argument('--skip-tests', action='store_true', help='Skip step 0 (pytest)')
    p.add_argument('--regenerate-mc', action='store_true',
                    help='Actually regenerate benchmarks/mc_instances/lee_mc/ if it is in the old '
                         'format (DESTRUCTIVE: deletes existing files there first). Without this '
                         'flag, an old-format directory only prints instructions and the pipeline stops.')
    p.add_argument('--skip-lee', action='store_true', help='Skip step 3 (Setting A, small scale)')
    p.add_argument('--skip-shin', action='store_true', help='Skip step 4 (Setting A, large scale)')
    p.add_argument('--skip-multi-small', action='store_true', help='Skip step 5 (Setting B, small scale)')
    p.add_argument('--skip-multi-large', action='store_true', help='Skip step 6 (Setting B, large scale)')
    p.add_argument('--skip-analysis', action='store_true', help='Skip steps 7-8 (analyze + visualize)')
    p.add_argument('--workers', type=int, default=None,
                    help='Passed through to steps 3-6 (analysis.run_single_crane_full / '
                         'run_multi_crane_full): number of parallel worker processes. Default '
                         '(unset): each script uses its own default of cpu_count()-2.')
    p.add_argument('--max-per-scale', type=int, default=20,
                    help='analysis.run_single_crane_full --dataset shin: instances per (bay,tier,type) group. '
                         'Default 20 = ALL available (Shin_instances has exactly 20 per group) -- i.e. full '
                         'data by default. Pass a smaller number only to shorten a test run.')
    p.add_argument('--max-instances-large', type=int, default=None,
                    help='analysis.run_multi_crane_full --dataset large: number of large layouts. Default '
                         '(unset) = ALL available (up to 160) -- i.e. full data by default, matching '
                         '--dataset small which always uses all 70 layouts. Pass a number only to shorten '
                         'a test run (this can take many hours at the full 160).')
    p.add_argument('--continue-on-error', action='store_true',
                    help='Keep running later steps even if one step fails (NOT recommended -- a failed '
                         'assertion usually means the lower bound or backward-compat contract is broken)')
    return p.parse_args()


def lee_mc_state():
    """Return 'missing', 'old', or 'new' for benchmarks/mc_instances/lee_mc/."""
    if not os.path.isdir(LEE_MC_DIR):
        return 'missing'
    files = sorted(f for f in os.listdir(LEE_MC_DIR) if f.endswith('.txt'))
    if not files:
        return 'missing'
    with open(os.path.join(LEE_MC_DIR, files[0]), 'r') as f:
        header = f.read(200)
    return 'new' if 'crane_start_bays_c2' in header else 'old'


def run_step(name, cmd, dry_run, continue_on_error):
    print(f'\n{"=" * 70}\n[STEP] {name}\n  $ {" ".join(cmd)}\n{"=" * 70}', flush=True)
    if dry_run:
        return True
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f'\n[FAILED] {name} (exit {result.returncode}, {elapsed:.0f}s)')
        if not continue_on_error:
            print('Stopping pipeline (pass --continue-on-error to override).')
            sys.exit(result.returncode)
        return False
    print(f'\n[OK] {name} ({elapsed:.0f}s)')
    return True


def main():
    args = parse_args()
    py = sys.executable
    ok = True

    if not args.skip_tests:
        ok &= run_step('0. Unit tests', [py, '-m', 'pytest', 'tests/', '-v'], args.dry_run, args.continue_on_error)

    # Only prepare multi-crane datasets if a multi-crane step will actually
    # run this invocation -- a Setting-A-only (Lee/Shin) run has no reason to
    # need lee_mc/ regenerated or lee_mc_large/ generated, and should not be
    # blocked by their state.
    need_mc_small = not args.skip_multi_small
    need_mc_large = not args.skip_multi_large

    if need_mc_small:
        state = lee_mc_state()
        if state == 'missing':
            print(f'\n[STOP] {LEE_MC_DIR} is empty/missing. Run: python -m benchmarks.generate_mc_instances')
            if not args.dry_run:
                sys.exit(1)
        elif state == 'old':
            if args.regenerate_mc:
                run_step('1. Regenerate lee_mc/ (old format -> new)',
                          [py, '-m', 'benchmarks.generate_mc_instances'], args.dry_run, args.continue_on_error)
            else:
                print(f'\n[STOP] {LEE_MC_DIR} is in the OLD format (missing crane_start_bays_c2 header). '
                      f'Re-run with --regenerate-mc to fix it (this DELETES the old files there first), '
                      f'or run `python -m benchmarks.generate_mc_instances` yourself.')
                if not args.dry_run:
                    sys.exit(1)
        else:
            print(f'\n[SKIP] {LEE_MC_DIR} already in the new format ({len(os.listdir(LEE_MC_DIR))} files).')
    else:
        print('\n[SKIP] step 1 (lee_mc/ prep) -- --skip-multi-small, no multi-crane small-scale step this run.')

    if need_mc_large:
        if not os.path.isdir(LEE_MC_LARGE_DIR) or not any(f.endswith('.txt') for f in os.listdir(LEE_MC_LARGE_DIR)):
            run_step('2. Generate lee_mc_large/ (new directory, does not touch lee_mc/)',
                      [py, '-c', 'from benchmarks.generate_mc_instances import generate_large; generate_large()'],
                      args.dry_run, args.continue_on_error)
        else:
            print(f'\n[SKIP] {LEE_MC_LARGE_DIR} already exists.')
    else:
        print('\n[SKIP] step 2 (lee_mc_large/ prep) -- --skip-multi-large, no multi-crane large-scale step this run.')

    workers_flag = ['--workers', str(args.workers)] if args.workers is not None else []

    if not args.skip_lee:
        run_step('3. Setting A, small scale (Lee_instances, 6 methods)',
                  [py, '-m', 'analysis.run_single_crane_full', '--dataset', 'lee'] + workers_flag,
                  args.dry_run, args.continue_on_error)

    if not args.skip_shin:
        run_step('4. Setting A, large scale (Shin_instances, 6 methods)',
                  [py, '-m', 'analysis.run_single_crane_full', '--dataset', 'shin',
                   '--max_per_scale', str(args.max_per_scale)] + workers_flag,
                  args.dry_run, args.continue_on_error)

    if not args.skip_multi_small:
        run_step('5. Setting B, small scale (ZeroShot + 3 heuristics x 4 strategies)',
                  [py, '-m', 'analysis.run_multi_crane_full', '--dataset', 'small'] + workers_flag,
                  args.dry_run, args.continue_on_error)

    if not args.skip_multi_large:
        cmd = [py, '-m', 'analysis.run_multi_crane_full', '--dataset', 'large'] + workers_flag
        if args.max_instances_large is not None:
            cmd += ['--max_instances', str(args.max_instances_large)]
        run_step('6. Setting B, large scale (SAME methods/strategies, ALL available layouts by default)',
                  cmd, args.dry_run, args.continue_on_error)

    if not args.skip_analysis:
        results_csv = os.path.join(ROOT, 'results', 'multi_crane_small.csv')
        if args.dry_run or os.path.exists(results_csv):
            run_step('7. Statistical report', [py, '-m', 'analysis.analyze', 'results/multi_crane_small.csv'],
                      args.dry_run, args.continue_on_error)
            run_step('8. Figures (300 DPI)', [py, '-m', 'analysis.visualize_v2'],
                      args.dry_run, args.continue_on_error)
        else:
            print(f'\n[SKIP] steps 7-8 -- {results_csv} does not exist yet '
                  f'(run with --skip-multi-small unset at least once first, or pass --skip-analysis to silence this).')

    print('\n' + '=' * 70)
    print('[DRY RUN COMPLETE]' if args.dry_run else '[PIPELINE COMPLETE]')
    print('=' * 70)


if __name__ == '__main__':
    main()
