import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import Model
from baselines.lowerbound import get_wt_lb
from benchmarks.benchmarks import find_and_process_file

MODEL_ARGS = argparse.Namespace(
    device=torch.device('cpu'), embed_dim=128, n_encode_layers=3, n_heads=8,
    ff_hidden=512, tanh_c=10, lstm=True, bay_embedding=True,
    online=False, online_known_num=None
)

MODEL_PATH = 'baselines/models/proposed/epoch(100).pt'


def get_model():
    m = Model(MODEL_ARGS)
    m.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    m.eval()
    m.decoder.set_sampler('greedy')
    return m


def test_model_loads():
    m = get_model()
    assert m is not None


def test_model_inference_single_instance():
    m = get_model()
    x, _ = find_and_process_file('benchmarks/Lee_instances', 'random', 1, 16, 6, 1, no_print=True)
    with torch.no_grad():
        wt, ll = m(x, None)
    assert wt.shape == (x.shape[0],)
    assert wt[0].item() > 0
    assert ll.shape == (x.shape[0],)


def test_model_gap_matches_paper():
    """Verify gap on R-type Lee benchmark is ~7.8% as reported in paper Table 1."""
    m = get_model()
    gaps = []
    configs = [(1,6),(2,6),(4,6),(6,6),(8,6),(10,6),(1,8),(2,8),(4,8),(6,8)]

    for bay, tier in configs:
        if tier == 8 and bay in [8, 10]:
            continue
        inputs, _ = zip(*[find_and_process_file(
            'benchmarks/Lee_instances', 'random', bay, 16, tier, i, no_print=True
        ) for i in range(1, 6)])
        x = torch.cat(inputs)
        with torch.no_grad():
            wt, _ = m(x, None)
        lbs = torch.tensor([get_wt_lb(x[i:i+1]) for i in range(x.shape[0])])
        instance_gaps = 100 * (wt - lbs) / lbs
        gaps.append(instance_gaps.mean().item())

    avg_gap = sum(gaps) / len(gaps)
    print(f'Average gap on R-type Lee benchmark: {avg_gap:.2f}%')
    assert 5.0 <= avg_gap <= 12.0, f'Gap {avg_gap:.2f}% outside expected [5%, 12%]'
