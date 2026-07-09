import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.generate_mc_instances import generate_all, crane_start_bays

# generate_all() is DESTRUCTIVE (deletes every .txt under its output_dir
# before writing) — every test here must pass tmp_path, never the real
# benchmarks/mc_instances/lee_mc/ dataset. Do not remove the output_dir
# argument from these calls.


def test_generation_produces_unique_layouts(tmp_path):
    out_dir = str(tmp_path)
    count = generate_all(output_dir=out_dir)
    assert count == 70, f'Expected 70 unique layouts (50 R + 20 U), got {count}'

    files = [os.path.basename(f) for f in glob.glob(f'{out_dir}/*.txt')]
    assert len(files) == 70
    # One file per layout: the old _c2/_c3 duplication must not reappear.
    assert not any('_c2' in f or '_c3' in f for f in files), 'duplicated crane-variant files found'
    assert sum(f.startswith('mc_R') for f in files) == 50
    assert sum(f.startswith('mc_U') for f in files) == 20


def test_files_have_crane_start_metadata(tmp_path):
    out_dir = str(tmp_path)
    generate_all(output_dir=out_dir)
    files = glob.glob(f'{out_dir}/*.txt')
    assert files
    for f in files[:5]:
        with open(f) as fh:
            head = [fh.readline(), fh.readline()]
        assert head[0].startswith('# crane_start_bays_c2 = ')
        assert head[1].startswith('# crane_start_bays_c3 = ')


def test_crane_start_bays_ordering():
    # Starts are non-decreasing (A7 order) and within [1, B].
    for B in [1, 2, 4, 6, 8, 10, 20, 30]:
        for C in [2, 3]:
            starts = crane_start_bays(B, C)
            assert len(starts) == C
            assert all(1 <= s <= B for s in starts)
            assert starts == sorted(starts)
    # Enough bays -> strictly increasing (cranes in distinct bays).
    assert len(set(crane_start_bays(10, 3))) == 3


def test_files_parse_via_experiment_parser(tmp_path):
    from experiment import parse_instance_file, load_instance_tensor
    out_dir = str(tmp_path)
    generate_all(output_dir=out_dir)
    files = sorted(glob.glob(f'{out_dir}/*.txt'))
    assert files
    fname = os.path.basename(files[0])
    dims = fname.split('_')[1][1:]
    n_bays, n_rows, n_tiers = int(dims[0:2]), int(dims[2:4]), int(dims[4:6])

    data_lines, crane_starts = parse_instance_file(files[0])
    assert 2 in crane_starts and 3 in crane_starts
    x = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)
    assert x.shape == (1, n_bays, n_rows, n_tiers)
    assert (x > 0).sum() > 0
