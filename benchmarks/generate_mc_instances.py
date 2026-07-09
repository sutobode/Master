"""Generate M-CRP benchmark layouts from the Lee & Lee (2010) instances.

One file per unique yard layout. The crane count C is an EXPERIMENT
parameter, not a dataset dimension: earlier revisions emitted `_c2`/`_c3`
twins with byte-identical yard content, which double-counted every layout
(140 files for 70 layouts) and invalidated instance-count statistics.
Each file carries the deterministic crane start bays for C=2 and C=3 as
header metadata.
"""

import os
from benchmarks.benchmarks import find_and_process_file

SCALES = {
    'small': {'bays': [1, 2, 4], 'rows': [16], 'tiers': [6, 8], 'id_range': (1, 5)},
    'medium': {'bays': [6, 8, 10], 'rows': [16], 'tiers': [6, 8], 'id_range': (1, 5)},
    'large': {'bays': [20, 30], 'rows': [16], 'tiers': [6, 8], 'id_range': (1, 20)},
}

ORIGINAL_DIR = 'benchmarks/Lee_instances'
OUTPUT_DIR = 'benchmarks/mc_instances'


def crane_start_bays(n_bays, n_cranes):
    """Deterministic evenly-spread start bays, strictly increasing when the
    yard has enough bays (respects the A7 left-to-right crane order)."""
    starts = []
    for c in range(n_cranes):
        starts.append(max(1, min(n_bays, c * n_bays // n_cranes + 1)))
    return starts


def generate_all(output_dir=None):
    """Regenerate M-CRP layouts. DESTRUCTIVE: deletes every .txt file under
    `output_dir` (default benchmarks/mc_instances/lee_mc, the real dataset)
    before writing. Tests must pass a temp `output_dir` — never call this
    against the real directory from a test, it will destroy hand-curated or
    previously-generated instances with no way back except `git checkout`."""
    out_dir = output_dir or f'{OUTPUT_DIR}/lee_mc'
    os.makedirs(out_dir, exist_ok=True)

    for old in os.listdir(out_dir):
        if old.endswith('.txt'):
            os.remove(os.path.join(out_dir, old))

    count = 0
    for inst_type in ['random', 'upsidedown']:
        type_count = 0
        for scale_name, spec in SCALES.items():
            for bay in spec['bays']:
                for row in spec['rows']:
                    for tier in spec['tiers']:
                        if tier == 8 and bay in [8, 10]:
                            continue
                        for idx in range(spec['id_range'][0], spec['id_range'][1] + 1):
                            if inst_type == 'upsidedown' and idx > 2:
                                continue
                            try:
                                x, name = find_and_process_file(
                                    ORIGINAL_DIR, inst_type, bay, row, tier, idx, no_print=True
                                )
                            except (FileNotFoundError, ValueError):
                                continue

                            type_prefix = 'R' if inst_type == 'random' else 'U'
                            out_name = f'mc_{type_prefix}{bay:02d}{row:02d}{tier:02d}_{idx:03d}.txt'
                            out_path = os.path.join(out_dir, out_name)

                            subdir = (
                                'individual, random'
                                if inst_type == 'random'
                                else 'individual, upside down'
                            )
                            orig_path = os.path.join(ORIGINAL_DIR, subdir, name)
                            with open(orig_path, 'r') as f_in:
                                content = f_in.read()
                            with open(out_path, 'w') as f_out:
                                f_out.write(f'# crane_start_bays_c2 = {crane_start_bays(bay, 2)}\n')
                                f_out.write(f'# crane_start_bays_c3 = {crane_start_bays(bay, 3)}\n')
                                f_out.write(content)
                            count += 1
                            type_count += 1
        print(f'Generated {type_count} layouts for type={inst_type}')

    print(f'Total: {count} unique M-CRP layouts in {out_dir}/')
    return count


if __name__ == '__main__':
    generate_all()
