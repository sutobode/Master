"""Generate M-CRP benchmark layouts from the Lee & Lee (2010) and Shin et al.
(2026) large-scale instances.

One file per unique yard layout. The crane count C is an EXPERIMENT
parameter, not a dataset dimension: earlier revisions emitted `_c2`/`_c3`
twins with byte-identical yard content, which double-counted every layout
(140 files for 70 layouts) and invalidated instance-count statistics.
Each file carries the deterministic crane start bays for C=2 and C=3 as
header metadata.

'small'/'medium' pull from Lee_instances (max 10 bays); 'large' pulls from
Shin_instances (20/30 bays, the extremely-large-scale benchmark) -- an
earlier revision pointed ALL scales at Lee_instances, so 'large' (bays
20/30, which Lee_instances does not contain) silently produced zero files
via the FileNotFoundError/continue below. Verify with
`python -m benchmarks.generate_mc_instances --report` before regenerating:
each scale's actual yielded count should match len(bays)*len(tiers)*id_range.
"""

import os
from benchmarks.benchmarks import find_and_process_file

LEE_DIR = 'benchmarks/Lee_instances'
SHIN_DIR = 'benchmarks/Shin_instances'

SCALES = {
    'small': {'bays': [1, 2, 4], 'rows': [16], 'tiers': [6, 8], 'id_range': (1, 5), 'dir': LEE_DIR},
    'medium': {'bays': [6, 8, 10], 'rows': [16], 'tiers': [6, 8], 'id_range': (1, 5), 'dir': LEE_DIR},
    'large': {'bays': [20, 30], 'rows': [16], 'tiers': [6, 8], 'id_range': (1, 20), 'dir': SHIN_DIR},
}

# The default dataset (small+medium, 70 layouts) excludes 'large': its
# instances have 1440-2880 containers vs. 70-720 for small/medium, so
# merging it in would silently multiply every downstream experiment's
# runtime. Generate it separately (generate_large()) into its own directory
# for an explicit scale-generalization experiment.
DEFAULT_SCALES = {k: v for k, v in SCALES.items() if k != 'large'}

OUTPUT_DIR = 'benchmarks/mc_instances'


def crane_start_bays(n_bays, n_cranes):
    """Deterministic evenly-spread start bays, strictly increasing when the
    yard has enough bays (respects the A7 left-to-right crane order)."""
    starts = []
    for c in range(n_cranes):
        starts.append(max(1, min(n_bays, c * n_bays // n_cranes + 1)))
    return starts


def _iter_layouts(scales=None):
    """Yield (inst_type, scale_name, bay, row, tier, idx, source_dir) for every
    layout that actually has a source file, without touching disk output.
    Missing files (e.g. Lee_instances has only 2 upside-down ids per
    bay/tier, not `id_range`'s full span) are skipped via the same
    FileNotFoundError check `generate_all` uses, so this is the authoritative
    "what will actually be generated" preview."""
    for inst_type in ['random', 'upsidedown']:
        for scale_name, spec in (scales or DEFAULT_SCALES).items():
            for bay in spec['bays']:
                for row in spec['rows']:
                    for tier in spec['tiers']:
                        if tier == 8 and bay in [8, 10]:
                            continue
                        for idx in range(spec['id_range'][0], spec['id_range'][1] + 1):
                            try:
                                x, name = find_and_process_file(
                                    spec['dir'], inst_type, bay, row, tier, idx, no_print=True
                                )
                            except (FileNotFoundError, ValueError):
                                continue
                            yield inst_type, scale_name, bay, row, tier, idx, spec['dir'], name


def report(scales=None):
    """Non-destructive dry run: count how many layouts each scale/type would
    actually yield, without deleting or writing anything. Run this before
    `generate_all()` to sanity-check a SCALES change (e.g. a wrong `dir`
    silently yields 0, as the old Lee-only 'large' scale did)."""
    scales = scales or SCALES
    counts = {}
    for inst_type, scale_name, *_ in _iter_layouts(scales):
        counts[(scale_name, inst_type)] = counts.get((scale_name, inst_type), 0) + 1
    for (scale_name, inst_type), n in sorted(counts.items()):
        print(f'{scale_name:8s} {inst_type:12s} {n:4d} files')
    total = sum(counts.values())
    print(f'Total available: {total} files across {len(scales)} scales')
    for scale_name in scales:
        n = sum(v for (s, _), v in counts.items() if s == scale_name)
        if n == 0:
            print(f'WARNING: scale "{scale_name}" would yield 0 files — check its "dir" and bay/tier/id_range')
    return counts


def generate_all(output_dir=None, scales=None):
    """Regenerate M-CRP layouts. DESTRUCTIVE: deletes every .txt file under
    `output_dir` (default benchmarks/mc_instances/lee_mc, the real dataset)
    before writing. Tests must pass a temp `output_dir` — never call this
    against the real directory from a test, it will destroy hand-curated or
    previously-generated instances with no way back except `git checkout`.

    `scales` defaults to DEFAULT_SCALES (small+medium, 70 layouts, unchanged
    from before this fix). Pass `scales=SCALES` or use `generate_large()` to
    also/instead generate the 'large' (Shin_instances, 20/30-bay) scale --
    kept out of the default on purpose, since its 1440-2880-container
    instances are much slower per run than the 70-720-container default set."""
    out_dir = output_dir or f'{OUTPUT_DIR}/lee_mc'
    os.makedirs(out_dir, exist_ok=True)

    for old in os.listdir(out_dir):
        if old.endswith('.txt'):
            os.remove(os.path.join(out_dir, old))

    count = 0
    type_counts = {}
    for inst_type, scale_name, bay, row, tier, idx, src_dir, name in _iter_layouts(scales):
        type_prefix = 'R' if inst_type == 'random' else 'U'
        out_name = f'mc_{type_prefix}{bay:02d}{row:02d}{tier:02d}_{idx:03d}.txt'
        out_path = os.path.join(out_dir, out_name)

        subdir = 'individual, random' if inst_type == 'random' else 'individual, upside down'
        orig_path = os.path.join(src_dir, subdir, name)
        with open(orig_path, 'r') as f_in:
            content = f_in.read()
        with open(out_path, 'w') as f_out:
            f_out.write(f'# crane_start_bays_c2 = {crane_start_bays(bay, 2)}\n')
            f_out.write(f'# crane_start_bays_c3 = {crane_start_bays(bay, 3)}\n')
            f_out.write(content)
        count += 1
        type_counts[inst_type] = type_counts.get(inst_type, 0) + 1

    for inst_type, n in type_counts.items():
        print(f'Generated {n} layouts for type={inst_type}')
    print(f'Total: {count} unique M-CRP layouts in {out_dir}/')
    return count


def generate_large(output_dir=None):
    """Generate the 'large' (Shin_instances, 20/30-bay, 1440-2880-container)
    M-CRP layouts into their own directory (default
    benchmarks/mc_instances/lee_mc_large/), for a dedicated scale-
    generalization experiment. Does NOT touch benchmarks/mc_instances/lee_mc/
    (the default 70-layout dataset)."""
    out_dir = output_dir or f'{OUTPUT_DIR}/lee_mc_large'
    return generate_all(output_dir=out_dir, scales={'large': SCALES['large']})


if __name__ == '__main__':
    import sys
    if '--report' in sys.argv:
        report()
    elif '--report-all' in sys.argv:
        report(scales=SCALES)
    elif '--large' in sys.argv:
        generate_large()
    else:
        generate_all()
