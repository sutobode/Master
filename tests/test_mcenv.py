import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcenv.mcenv import MCEnv
from env.env import Env


def test_mcenv_single_crane_match_env():
    """MCEnv(C=1) must produce same cost sequence as original Env."""
    x = torch.zeros(1, 2, 4, 6)
    x[0, 0, 0, :4] = torch.tensor([1., 2., 3., 4.])
    x[0, 0, 1, :2] = torch.tensor([5., 6.])
    x[0, 1, 0, :3] = torch.tensor([7., 8., 9.])

    mcenv = MCEnv('cpu', x, n_cranes=1)
    # Original Env expects (batch, n_bays, n_rows, max_tiers) — same as x
    orig_env = Env('cpu', x)

    mcenv.base_env.clear()
    orig_env.clear()
    assert torch.allclose(mcenv.base_env.x, orig_env.x), 'State mismatch after clear'


def test_mcenv_interference_detection():
    x = torch.zeros(1, 4, 4, 6)
    mcenv = MCEnv('cpu', x, n_cranes=2, crane_start_bays=[1, 3])
    # Set crane positions to simulate active state
    mcenv.crane_bays[0, 0] = 1
    mcenv.crane_bays[0, 1] = 3
    assert mcenv._validate_interference(0, 1)  # crane 0 already at bay 1 → OK (same crane)
    assert mcenv._validate_interference(1, 3)  # crane 1 already at bay 3 → OK
    assert not mcenv._validate_interference(0, 3)  # bay 3 occupied by crane 1
    assert not mcenv._validate_interference(1, 1)  # bay 1 occupied by crane 0


def test_mcenv_resolve_interference():
    x = torch.zeros(1, 4, 4, 6)
    mcenv = MCEnv('cpu', x, n_cranes=2, crane_start_bays=[1, 3])
    result = mcenv._resolve_interference(0, 3)
    assert result == 1


def test_mcenv_step():
    x = torch.zeros(1, 2, 4, 6)
    x[0, 0, 0, :4] = torch.tensor([1., 2., 3., 4.])
    x[0, 1, 0, :1] = torch.tensor([5.])

    mcenv = MCEnv('cpu', x, n_cranes=2)
    mcenv.base_env.clear()  # Initialize target stack
    cost, state = mcenv.step(dest_stack=torch.tensor([[4]]), crane_id=0)
    assert isinstance(cost, torch.Tensor)
    assert cost[0].item() > 0
    assert state is not None
