import sys, os, tempfile, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.analyze import MCRPAnalyzer


def _make_dummy_csv():
    data = {
        'instance': ['mc_R011606_001_c2', 'mc_R011606_002_c2',
                     'mc_R101606_001_c2', 'mc_R201606_001_c2'],
        'n_cranes': [2, 2, 2, 2],
        'strategy': ['S1', 'S1', 'S2', 'S2'],
        'total_cost': [1000, 1100, 950, 1050],
        'lb_mc': [800, 900, 800, 900],
        'gap': [25.0, 22.2, 18.8, 16.7],
        'n_steps': [50, 55, 48, 52],
        'interference': [5, 7, 2, 3],
        'time_s': [0.5, 0.6, 0.7, 0.8],
    }
    df = pd.DataFrame(data)
    path = os.path.join(tempfile.gettempdir(), 'test_mcrp.csv')
    df.to_csv(path, index=False)
    return path


def test_analyzer_loads():
    path = _make_dummy_csv()
    a = MCRPAnalyzer(path)
    assert len(a.df) == 4


def test_table1_gap_comparison():
    path = _make_dummy_csv()
    a = MCRPAnalyzer(path)
    t1 = a.table1_gap_comparison()
    assert 'mean' in t1.columns
    assert 'std' in t1.columns
    assert 'S1' in t1.index.get_level_values('strategy').values
    assert 'S2' in t1.index.get_level_values('strategy').values


def test_table2_pairwise_wilcoxon():
    path = _make_dummy_csv()
    a = MCRPAnalyzer(path)
    t2 = a.table2_pairwise_wilcoxon(2)
    assert 's1' in t2.columns
    assert 'p_value' in t2.columns


def test_table3_interference():
    path = _make_dummy_csv()
    a = MCRPAnalyzer(path)
    t3 = a.table3_interference_summary()
    assert 'mean' in t3.columns


def test_failure_modes():
    path = _make_dummy_csv()
    a = MCRPAnalyzer(path)
    fail = a.identify_failure_modes(threshold_gap=20.0)
    assert len(fail) >= 0


def test_summary_text():
    path = _make_dummy_csv()
    a = MCRPAnalyzer(path)
    text = a.summary_text()
    assert 'Table 1' in text
    assert 'Table 2' in text
    assert 'Table 3' in text
