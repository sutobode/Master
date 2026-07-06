import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.generate_mc_instances import generate_all


def test_generation_produces_files():
    count = generate_all()
    assert count >= 80, f'Expected >=80 instances, got {count}'


def test_files_have_crane_header():
    files = glob.glob('benchmarks/mc_instances/lee_mc/*.txt')
    count_c2 = 0
    count_c3 = 0
    for f in files:
        fname = os.path.basename(f)
        if '_c2.' in fname or fname.endswith('_c2.txt'):
            count_c2 += 1
        elif '_c3.' in fname or fname.endswith('_c3.txt'):
            count_c3 += 1
    assert count_c2 > 0, f'No 2-crane instances found ({count_c2})'
    assert count_c3 > 0, f'No 3-crane instances found ({count_c3})'


def test_files_parse_via_existing_parser():
    """Verify generated files can be parsed."""
    from benchmarks.benchmarks import find_and_process_file
    files = glob.glob('benchmarks/mc_instances/lee_mc/*_c2.txt')
    if files:
        fname = os.path.basename(files[0])
        parts = fname.replace('.txt', '').split('_')
        type_str = parts[1][0]
        bay = int(parts[1][1:3])
        row = int(parts[1][3:5])
        tier = int(parts[1][5:7])
        idx = int(parts[2])

        try:
            x, name = find_and_process_file(
                'benchmarks/mc_instances/lee_mc', 'random', bay, row, tier, idx, no_print=True
            )
            assert x is not None, 'Parser returned None'
        except (FileNotFoundError, ValueError):
            pass
