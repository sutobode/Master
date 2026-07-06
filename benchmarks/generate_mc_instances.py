"""Generate M-CRP benchmark instances from Lee & Lee and Shin et al. instances."""

import os
from benchmarks.benchmarks import find_and_process_file

SCALES = {
    'small': {'bays': [1, 2, 4], 'rows': [16], 'tiers': [6, 8], 'id_range': (1, 5)},
    'medium': {'bays': [6, 8, 10], 'rows': [16], 'tiers': [6, 8], 'id_range': (1, 5)},
    'large': {'bays': [20, 30], 'rows': [16], 'tiers': [6, 8], 'id_range': (1, 20)},
}

ORIGINAL_DIR = 'benchmarks/Lee_instances'
OUTPUT_DIR = 'benchmarks/mc_instances'


def generate_all():
    os.makedirs(f'{OUTPUT_DIR}/lee_mc', exist_ok=True)
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

                            for n_cranes in [2, 3]:
                                if n_cranes == 2:
                                    starts = [1, max(1, bay // 2 + 1)]
                                else:
                                    starts = [1, max(1, bay // 3 + 1), max(1, 2 * bay // 3 + 1)]

                                type_prefix = 'R' if inst_type == 'random' else 'U'
                                out_name = (
                                    f'mc_{type_prefix}{bay:02d}{row:02d}{tier:02d}'
                                    f'_{idx:03d}_c{n_cranes}.txt'
                                )
                                out_path = f'{OUTPUT_DIR}/lee_mc/{out_name}'

                                subdir = (
                                    'individual, random'
                                    if inst_type == 'random'
                                    else 'individual, upside down'
                                )
                                orig_path = os.path.join(ORIGINAL_DIR, subdir, name)
                                with open(orig_path, 'r') as f_in:
                                    content = f_in.read()
                                with open(out_path, 'w') as f_out:
                                    f_out.write(f'# n_cranes = {n_cranes}\n')
                                    f_out.write(f'# crane_start_bays = {starts}\n')
                                    f_out.write(content)
                                count += 1
                                type_count += 1
        print(f'Generated {type_count} instances for type={inst_type}')

    print(f'Total: {count} M-CRP instances in {OUTPUT_DIR}/lee_mc/')
    return count


if __name__ == '__main__':
    generate_all()
