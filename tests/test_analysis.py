import sys, os, tempfile, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.analyze import MCRPAnalyzer


def _make_dummy_csv():
    data = {
        'instance': ['mc_R011606_001', 'mc_R011606_002', 'mc_R101606_001', 'mc_U201606_001'] * 2,
        'n_cranes': [2, 2, 2, 2, 3, 3, 3, 3],
        'strategy': ['S1', 'S1', 'S2', 'S2'] * 2,
        'total_cost': [1000, 1100, 950, 1050, 990, 1080, 940, 1040],
        'makespan': [700, 750, 600, 640, 560, 590, 470, 500],
        'lb_work': [800, 900, 800, 900, 790, 880, 790, 880],
        'lb_makespan': [500, 550, 450, 500, 400, 430, 360, 390],
        'gap_work': [25.0, 22.2, 18.8, 16.7, 25.3, 22.7, 19.0, 18.2],
        'gap_makespan': [40.0, 36.4, 33.3, 28.0, 40.0, 37.2, 30.6, 28.2],
        'n_steps': [50, 55, 48, 52, 50, 55, 48, 52],
        'interference': [5, 7, 2, 3, 6, 8, 2, 3],
        'interference_wait': [120.0, 150.0, 10.0, 20.0, 130.0, 160.0, 12.0, 25.0],
        'a7_reassignments': [0, 1, 0, 0, 1, 2, 0, 0],
        'time_s': [0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8],
    }
    df = pd.DataFrame(data)
    path = os.path.join(tempfile.gettempdir(), 'test_mcrp_v2.csv')
    df.to_csv(path, index=False)
    return path


def test_analyzer_loads():
    a = MCRPAnalyzer(_make_dummy_csv())
    assert len(a.df) == 8
    assert a.primary == 'gap_makespan'


def test_table1_gap_comparison():
    a = MCRPAnalyzer(_make_dummy_csv())
    t1 = a.table1_gap_comparison()
    assert 'mean' in t1.columns
    assert 'std' in t1.columns
    assert 'S1' in t1.index.get_level_values('strategy').values
    assert 'S2' in t1.index.get_level_values('strategy').values


def test_table2_pairwise_wilcoxon():
    a = MCRPAnalyzer(_make_dummy_csv())
    t2 = a.table2_pairwise_wilcoxon(2)
    assert 's1' in t2.columns
    assert 'p_value' in t2.columns
    assert 'sigma_d' in t2.columns  # paired-difference std reported


def test_table3_interference():
    a = MCRPAnalyzer(_make_dummy_csv())
    t3 = a.table3_interference_summary()
    assert ('interference', 'mean') in t3.columns
    assert ('interference_wait', 'mean') in t3.columns


def test_speedup_uses_makespan():
    a = MCRPAnalyzer(_make_dummy_csv())
    sp = a.compute_speedup()
    assert len(sp) > 0
    # C=3 makespans are smaller in the dummy data -> speedup > 1.
    assert (sp[('speedup_makespan', 'mean')] > 1).all()


def test_failure_modes():
    a = MCRPAnalyzer(_make_dummy_csv())
    fail = a.identify_failure_modes(threshold_gap=20.0)
    assert len(fail) >= 0


def test_summary_text():
    a = MCRPAnalyzer(_make_dummy_csv())
    text = a.summary_text()
    assert 'Table 1' in text
    assert 'Table 2' in text
    assert 'Table 3' in text
    assert 'Makespan Speedup' in text
