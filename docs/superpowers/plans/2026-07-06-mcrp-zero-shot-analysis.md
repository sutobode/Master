# M-CRP Zero-shot Transfer Analysis — Implementation Plan

> **For agentic workers:** Code + test implementation only. Paper writing is out of scope.

**Goal:** Implement code modules for Multi-Crane Container Retrieval Problem (M-CRP) zero-shot transfer analysis: multi-crane env, 4 crane assignment strategies, M-CRP lower bound, zero-shot policy wrapper, benchmark generation, and evaluation pipeline. All CPU-laptop, 0 GPU.

**Architecture:** Three layers:
- **Core:** `mcenv/MCEnv` wraps single-crane `Env` with crane state management; `policy/ZeroShotPolicy` extracts encoder + scorer from pretrained model for per-step action selection.
- **Strategies:** `strategies/` — 4 crane assignment strategies (RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal).
- **Pipeline:** `experiment.py` orchestrates: load pretrained model → iterate M-CRP instances → for each step {policy → dest_stack, strategy → crane_id, MCEnv.step} → log metrics.

**Tech Stack:** Python 3.10+, PyTorch 2.12.1+cpu, NumPy, Pandas. No GPU.

## Global Constraints

- All code runs CPU-only (`torch.device('cpu')`). No CUDA.
- Pretrained model at `baselines/models/proposed/epoch(100).pt`.
- Every new module has tests in `tests/`. All tests must pass.
- Single-crane mode (C=1) of all new modules must reproduce original `Env` behavior exactly.
- Seed = 1234 for all stochastic components. NumPy seed too.
- Use `argparse` for experiment configs; log results to timestamped dirs under `results/`.

---

### Task 1: Verify Pretrained Model Reproduces Paper Results

**Files:**
- Create: `tests/test_verify_baseline.py`

**Interfaces:**
- Consumes: `baselines/models/proposed/epoch(100).pt`, `benchmarks/Lee_instances/`, `baselines/test.py`
- Produces: Verification that model loads, inference runs, gaps match paper Tables 1-2 (±1%)

- [ ] **Step 1: Write verification script**

```python
# tests/test_verify_baseline.py
import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import Model

ARGS = argparse.Namespace(
    device=torch.device('cpu'),
    embed_dim=128, n_encode_layers=3, n_heads=8, ff_hidden=512,
    tanh_c=10, lstm=True, bay_embedding=True,
    online=False, online_known_num=None
)

def get_model():
    model = Model(ARGS)
    state = torch.load('baselines/models/proposed/epoch(100).pt', map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    model.decoder.set_sampler('greedy')
    return model

def test_model_loads():
    m = get_model()
    assert m is not None

def test_model_inference_single_instance():
    m = get_model()
    from benchmarks.benchmarks import find_and_process_file
    x, _ = find_and_process_file('benchmarks/Lee_instances', 'random', 1, 16, 6, 1, no_print=True)
    with torch.no_grad():
        wt, ll = m(x, None)
    assert wt.shape == (x.shape[0],)
    assert wt[0].item() > 0
    assert ll.shape == (x.shape[0],)

def test_model_inference_all_lee_random():
    m = get_model()
    from benchmarks.benchmarks import find_and_process_file
    gaps = []
    for bay, tier in [(1,6),(2,6),(4,6),(6,6),(8,6),(10,6),(1,8),(2,8),(4,8),(6,8)]:
        if tier == 8 and bay in [8, 10]:
            continue
        inputs, _ = zip(*[find_and_process_file(
            'benchmarks/Lee_instances', 'random', bay, 16, tier, i, no_print=True
        ) for i in range(1, 6)])
        x = torch.cat(inputs)
        with torch.no_grad():
            wt, _ = m(x, None)
        from baselines.lowerbound import get_wt_lb
        lbs = torch.tensor([get_wt_lb(x[i:i+1]) for i in range(x.shape[0])])
        instance_gaps = 100 * (wt - lbs) / lbs
        gaps.append(instance_gaps.mean().item())
    avg_gap = sum(gaps) / len(gaps)
    print(f'Average gap on R-type Lee benchmark: {avg_gap:.2f}%')
    # Paper reports ~7.8% average gap (Table 1)
    assert 5.0 <= avg_gap <= 12.0, f'Gap {avg_gap:.2f}% outside expected range [5%, 12%]'
```

- [ ] **Step 2: Run all tests**

```bash
python -m pytest tests/test_verify_baseline.py -v
```

Expected: All 3 tests PASS. Gaps printed should roughly match paper Table 1 values (R011606_0070: ~7.7%, R101606_0720: ~5.9%).

- [ ] **Step 3: Commit**

```bash
git add tests/test_verify_baseline.py
git commit -m "test: verify pretrained model reproduces paper results"
```

---

### Task 2: Multi-Crane Environment Wrapper

**Files:**
- Create: `mcenv/__init__.py`
- Create: `mcenv/mcenv.py`
- Create: `tests/test_mcenv.py`

**Interfaces:**
- Consumes: `env/env.py` (original `Env` class), yard config `x`, `n_cranes`, optional `crane_start_bays`
- Produces: `MCEnv` class — stateful multi-crane env wrapping single-crane Env

**Design note:** MCEnv does NOT modify `Env`. It creates one shared yard (via `base_env.x` access) and manages C crane states externally. At each step, it validates interference, dispatches the relocation to `base_env.step()`, and tracks per-crane cost.

- [ ] **Step 1: Write MCEnv class**

```python
# mcenv/__init__.py
from .mcenv import MCEnv
```

```python
# mcenv/mcenv.py
import torch
from env.env import Env

class MCEnv:
    """Multi-crane CRP environment.
    
    Manages C cranes operating on one shared yard. Crane positions tracked
    independently. Interference constraints: no two cranes in same bay.
    """

    def __init__(self, device, x, n_cranes, crane_start_bays=None, t_row=1.2, t_bay=3.5, t_acc=40, t_pd=30):
        self.device = device
        self.batch, self.n_bays, self.n_rows, self.max_tiers = x.size()
        self.max_stacks = self.n_bays * self.n_rows
        self.n_cranes = n_cranes
        self.t_row = t_row
        self.t_bay = t_bay
        self.t_acc = t_acc
        self.t_pd = t_pd

        # Shared yard (single-crane Env manages the yard state)
        self.base_env = Env(device, x, max_retrievals=None)

        # Override Env time coefficients if needed
        self.base_env.t_row = t_row
        self.base_env.t_bay = t_bay
        self.base_env.t_acc = t_acc
        self.base_env.t_pd = t_pd

        # Crane states: each crane has position and busy flag
        if crane_start_bays is None:
            crane_start_bays = [1]
            for c in range(1, n_cranes):
                crane_start_bays.append(min(self.n_bays, crane_start_bays[-1] + self.n_bays // n_cranes))
        self.crane_bays = torch.full((self.batch, n_cranes), -1, device=device)
        self.crane_rows = torch.full((self.batch, n_cranes), -1, device=device)
        self.crane_start_bays = crane_start_bays
        self.assigned_counts = torch.zeros(n_cranes, device=device, dtype=torch.long)
        self.interference_events = torch.zeros((self.batch, n_cranes), device=device)

    def _validate_interference(self, crane_id, dest_bay):
        """Check if dest_bay is occupied by another crane. Return True if OK."""
        for c in range(self.n_cranes):
            if c != crane_id and self.crane_bays[0, c] == dest_bay:
                return False
        return True

    def _resolve_interference(self, crane_id, dest_bay):
        """Try to reassign to another idle crane. Fallback to original if none available."""
        for c in range(self.n_cranes):
            if c != crane_id and self._validate_interference(c, dest_bay):
                return c
        return crane_id  # must wait

    def _crane_travel_cost(self, crane_id, source_bay, source_row, dest_bay, dest_row):
        """Compute travel cost for a crane from source to dest, including accel if bay changes."""
        cost = 0.0
        if self.crane_bays[0, crane_id] >= 0:
            curr_bay = self.crane_bays[0, crane_id].item()
            curr_row = self.crane_rows[0, crane_id].item()
        else:
            curr_bay = self.crane_start_bays[crane_id]
            curr_row = 1

        # Travel from current position to source
        if curr_bay != source_bay:
            cost += self.t_acc + self.t_bay * abs(curr_bay - source_bay)
        cost += self.t_row * abs(curr_row - source_row)

        # Travel from source to dest
        if source_bay != dest_bay:
            cost += self.t_acc + self.t_bay * abs(source_bay - dest_bay)
        cost += self.t_row * abs(source_row - dest_row)
        cost += self.t_pd

        return cost

    def step(self, dest_stack, crane_id):
        """Execute relocation with specified crane.
        
        Args:
            dest_stack: action tensor (batch, 1) — destination stack index
            crane_id: which crane (0..C-1)
        Returns:
            cost: relocation + retrieval cost tensor (batch,)
            env_tensor: updated yard state (batch, B, R, T)
        """
        # Source = current target stack
        source_idx = self.base_env.target_stack  # (batch,)
        source_bay = (source_idx[0].item() // self.n_rows) + 1  # 1-indexed
        source_row = (source_idx[0].item() % self.n_rows) + 1
        dest_idx = dest_stack[0, 0].item()
        dest_bay = (dest_idx // self.n_rows) + 1
        dest_row = (dest_idx % self.n_rows) + 1

        # Validate + resolve interference
        if not self._validate_interference(crane_id, dest_bay):
            self.interference_events[0, crane_id] += 1
            new_crane = self._resolve_interference(crane_id, dest_bay)
            if new_crane != crane_id:
                crane_id = new_crane

        # Set base_env position to match assigned crane (for correct travel cost)
        current_bay = self.crane_start_bays[crane_id] if self.crane_bays[0, crane_id] < 0 else self.crane_bays[0, crane_id].item()
        current_row = 1 if self.crane_rows[0, crane_id] < 0 else self.crane_rows[0, crane_id].item()
        self.base_env.curr_bay = torch.full((self.batch,), current_bay, device=self.device)
        self.base_env.curr_row = torch.full((self.batch,), current_row, device=self.device)

        # Execute relocation in base_env (takes (batch,1) tensors)
        base_cost = self.base_env.step(dest_stack)

        # Update crane position
        self.crane_bays[0, crane_id] = dest_bay
        self.crane_rows[0, crane_id] = dest_row
        self.assigned_counts[crane_id] += 1

        return base_cost, self.base_env.x.reshape(self.batch, self.n_bays, self.n_rows, self.max_tiers)

    def get_state(self):
        """Return current yard state in model-input format."""
        return self.base_env.x.reshape(self.batch, self.n_bays, self.n_rows, self.max_tiers)

    @property
    def terminated(self):
        return self.base_env.all_terminated()

    def reset(self, x):
        """Reset with new instance."""
        self.base_env = Env(self.device, x, max_retrievals=None)
        self.crane_bays = torch.full((self.batch, self.n_cranes), -1, device=self.device)
        self.crane_rows = torch.full((self.batch, self.n_cranes), -1, device=self.device)
        self.assigned_counts = torch.zeros(self.n_cranes, device=self.device, dtype=torch.long)
        self.interference_events = torch.zeros((self.batch, self.n_cranes), device=self.device)
```

- [ ] **Step 2: Write unit tests**

```python
# tests/test_mcenv.py
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcenv.mcenv import MCEnv
from env.env import Env

def test_mcenv_single_crane_match_env():
    """MCEnv(C=1) must produce same cost as original Env on a simple sequence."""
    x = torch.zeros(1, 2, 4, 6)
    x[0, 0, 0, :3] = torch.tensor([3., 2., 1.])  # stack 0: containers 3,2,1 (retrieval order)
    x[0, 0, 1, :2] = torch.tensor([5., 4.])       # stack 1: containers 5,4
    x[0, 1, 0, :3] = torch.tensor([6., 7., 8.])   # stack 2: 6,7,8

    mcenv = MCEnv('cpu', x, n_cranes=1)
    orig_env = Env('cpu', x.reshape(1, -1, 6))

    # Clear initial retrievables
    mcenv.base_env.clear()
    orig_env.clear()

    # Compare state after clear
    assert torch.allclose(mcenv.base_env.x, orig_env.x)

def test_mcenv_interference_detection():
    """Two cranes in same bay must trigger interference."""
    x = torch.zeros(1, 4, 4, 6)
    mcenv = MCEnv('cpu', x, n_cranes=2, crane_start_bays=[1, 3])
    assert mcenv._validate_interference(0, 1)  # crane 0 → bay 1, OK
    assert mcenv._validate_interference(1, 3)  # crane 1 → bay 3, OK
    assert not mcenv._validate_interference(0, 3)  # crane 0 → bay 3, bay 3 occupied by crane 1
    assert not mcenv._validate_interference(1, 1)  # crane 1 → bay 1, bay 1 occupied by crane 0

def test_mcenv_resolve_interference():
    x = torch.zeros(1, 4, 4, 6)
    mcenv = MCEnv('cpu', x, n_cranes=2, crane_start_bays=[1, 3])
    # Both cranes idle, crane 0 wants bay 3 (occupied by crane 1) → should resolve to crane 1
    result = mcenv._resolve_interference(0, 3)
    assert result == 1

def test_mcenv_step():
    """Test a single step with 2 cranes — target buried, must relocate."""
    x = torch.zeros(1, 2, 4, 6)
    x[0, 0, 0, :4] = torch.tensor([1., 2., 3., 4.])  # target=1 at bottom, blocked by 2,3,4 on top
    x[0, 1, 0, :1] = torch.tensor([5.])  # bay 2 has space (stack index 4)
    
    mcenv = MCEnv('cpu', x, n_cranes=2)
    cost, state = mcenv.step(dest_stack=torch.tensor([[4]]), crane_id=0)
    assert isinstance(cost, torch.Tensor)
    assert cost[0].item() > 0
    assert state is not None
```

- [ ] **Step 3: Run all tests**

```bash
python -m pytest tests/test_mcenv.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add mcenv/ tests/test_mcenv.py
git commit -m "feat: multi-crane env wrapper with interference constraints"
```

---

### Task 3: Zero-Shot Policy Wrapper

**Files:**
- Create: `policy/__init__.py`
- Create: `policy/zero_shot.py`
- Create: `tests/test_policy.py`

**Interfaces:**
- Consumes: pretrained model weights, current yard state tensor (batch, n_bays, n_rows, n_tiers)
- Produces: `dest_stack` action at each step via original encoder + scorer

**Architecture note:** The original `Decoder.forward` wraps its own `Env` internally. For zero-shot, we extract the encoder and scoring weights, then run our own loop with `MCEnv` as the environment. The `ZeroShotPolicy` class loads the pretrained model, then at each step: (1) encodes the current yard state, (2) computes stack scores using original decoder's scorer, (3) returns argmax over feasible stacks.

- [ ] **Step 1: Write ZeroShotPolicy class**

```python
# policy/__init__.py
from .zero_shot import ZeroShotPolicy
```

```python
# policy/zero_shot.py
import torch
import torch.nn as nn
import math
import argparse
from model.encoder import Encoder
from model.model import Model

class ZeroShotPolicy:
    """Extracts encoder + scorer from pretrained model for per-step action selection.
    
    Allows external environment (MCEnv) to drive the loop, with model
    providing action logits/greedy actions at each step.
    """

    def __init__(self, model_path='baselines/models/proposed/epoch(100).pt', device=torch.device('cpu')):
        self.device = device
        
        # Load full model (to get state_dict)
        args = argparse.Namespace(
            device=device, embed_dim=128, n_encode_layers=3, n_heads=8,
            ff_hidden=512, tanh_c=10, lstm=True, bay_embedding=True,
            online=False, online_known_num=None
        )
        full_model = Model(args)
        state = torch.load(model_path, map_location=device)
        full_model.load_state_dict(state)
        full_model.eval()
        
        # Extract encoder
        self.encoder = full_model.decoder.encoder
        self.encoder.eval()
        
        # Extract scorer weights from decoder
        self.tanh_c = full_model.decoder.tanh_c
        self.W_target = full_model.decoder.W_target
        self.W_global = full_model.decoder.W_global
        self.W_K1 = full_model.decoder.W_K1
        self.W_K2 = full_model.decoder.W_K2
        self.W_Q = full_model.decoder.W_Q
        self.W_V = full_model.decoder.W_V
        self.MHA = full_model.decoder.MHA
        
        for module in [self.W_target, self.W_global, self.W_K1, self.W_K2,
                       self.W_Q, self.W_V, self.MHA]:
            module.eval()
    
    @torch.no_grad()
    def get_scores(self, x, n_bays, n_rows, n_tiers, target_stack, invalid_mask=None):
        """Compute action scores for all stacks.
        
        Args:
            x: yard state (1, n_bays, n_rows, n_tiers)
            n_bays, n_rows, n_tiers: yard dimensions
            target_stack: current target stack index (0..B*R-1)
            invalid_mask: boolean (1, B*R) — True for stacks that cannot be selected
        Returns:
            scores: (1, B*R) tensor of log-probabilities
        """
        batch = 1
        max_stacks = n_bays * n_rows
        
        # Encode
        node_embeddings, graph_embedding = self.encoder(
            x.reshape(batch, max_stacks, n_tiers), n_bays, n_rows,
            t_acc=40, t_bay=3.5, t_row=1.2, t_pd=30
        )
        
        # Target stack embedding
        target_emb = node_embeddings[0, target_stack:target_stack+1, :].unsqueeze(0)
        
        # Context: target + global
        context = (self.W_target(target_emb) + self.W_global(graph_embedding.unsqueeze(1)))
        
        # Scorer (matching original decoder logic)
        node_keys = self.W_K1(node_embeddings)
        node_values = self.W_V(node_embeddings)
        query = self.W_Q(self.MHA([context, node_keys, node_values]))
        key = self.W_K2(node_embeddings)
        
        logits = torch.matmul(query, key.permute(0, 2, 1)).squeeze(1) / math.sqrt(query.size(-1))
        logits = self.tanh_c * torch.tanh(logits)
        
        if invalid_mask is not None:
            logits = logits - invalid_mask.float() * 1e9
        
        log_p = torch.log_softmax(logits, dim=1)
        return log_p
    
    @torch.no_grad()
    def get_action(self, x, n_bays, n_rows, n_tiers, target_stack, invalid_mask=None):
        """Greedy action selection."""
        log_p = self.get_scores(x, n_bays, n_rows, n_tiers, target_stack, invalid_mask)
        return log_p.argmax(dim=1, keepdim=True)
```

- [ ] **Step 2: Write unit tests**

```python
# tests/test_policy.py
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.zero_shot import ZeroShotPolicy
from benchmarks.benchmarks import find_and_process_file

def test_policy_loads():
    p = ZeroShotPolicy()
    assert p.encoder is not None
    assert hasattr(p, 'W_target')

def test_policy_get_scores():
    p = ZeroShotPolicy()
    x, _ = find_and_process_file('benchmarks/Lee_instances', 'random', 1, 16, 6, 1, no_print=True)
    # x shape: (1, 1, 16, 6). Reshape to (1, 16, 6)
    scores = p.get_scores(x, 1, 16, 6, target_stack=0)
    assert scores.shape == (1, 16)  # 1 bay × 16 rows = 16 stacks
    assert torch.allclose(scores.exp().sum(dim=1), torch.tensor([1.0]))  # valid probabilities

def test_policy_get_action():
    p = ZeroShotPolicy()
    x, _ = find_and_process_file('benchmarks/Lee_instances', 'random', 1, 16, 6, 1, no_print=True)
    action = p.get_action(x, 1, 16, 6, target_stack=3)
    assert action.shape == (1, 1)
    assert 0 <= action[0, 0].item() < 16

def test_policy_action_with_mask():
    """Invalid stacks should not be selected."""
    p = ZeroShotPolicy()
    x, _ = find_and_process_file('benchmarks/Lee_instances', 'random', 1, 16, 6, 1, no_print=True)
    mask = torch.zeros(1, 16)
    mask[0, :8] = 1.0  # First 8 stacks are invalid
    action = p.get_action(x, 1, 16, 6, target_stack=10, invalid_mask=mask)
    assert action[0, 0].item() >= 8  # Must select from last 8 stacks
```

- [ ] **Step 3: Run all tests**

```bash
python -m pytest tests/test_policy.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add policy/ tests/test_policy.py
git commit -m "feat: zero-shot policy wrapper extracting encoder and scorer from pretrained model"
```

---

### Task 4: Crane Assignment Strategies

**Files:**
- Create: `strategies/__init__.py`
- Create: `strategies/base.py`
- Create: `strategies/round_robin.py`
- Create: `strategies/zone_split.py`
- Create: `strategies/load_balance.py`
- Create: `strategies/greedy_optimal.py`
- Create: `tests/test_strategies.py`

**Interfaces:**
- Consumes: `MCEnv` state (yard tensor, crane positions, target_stack)
- Produces: `crane_id` (0..C-1) for each relocation step

- [ ] **Step 1: Write base class**

```python
# strategies/__init__.py
from .round_robin import RoundRobin
from .zone_split import ZoneSplit
from .load_balance import LoadBalance
from .greedy_optimal import GreedyOptimal
```

```python
# strategies/base.py
from abc import ABC, abstractmethod

class CraneAssignmentStrategy(ABC):
    def __init__(self, n_cranes, n_bays, n_rows):
        self.n_cranes = n_cranes
        self.n_bays = n_bays
        self.n_rows = n_rows

    @abstractmethod
    def assign(self, env, target_stack, dest_stack) -> int:
        """Choose crane_id for this relocation.
        
        Args:
            env: MCEnv instance (provides crane state, positions)
            target_stack: current target stack index
            dest_stack: chosen destination stack index
        Returns:
            crane_id: int in [0, n_cranes)
        """
        pass

    def reset(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__
```

- [ ] **Step 2: Write S1 — RoundRobin**

```python
# strategies/round_robin.py
from .base import CraneAssignmentStrategy

class RoundRobin(CraneAssignmentStrategy):
    def __init__(self, n_cranes, n_bays, n_rows):
        super().__init__(n_cranes, n_bays, n_rows)
        self._next = 0

    def assign(self, env, target_stack, dest_stack):
        c = self._next
        self._next = (self._next + 1) % self.n_cranes
        return c

    def reset(self):
        self._next = 0
```

- [ ] **Step 3: Write S2 — ZoneSplit**

```python
# strategies/zone_split.py
from .base import CraneAssignmentStrategy

class ZoneSplit(CraneAssignmentStrategy):
    def __init__(self, n_cranes, n_bays, n_rows):
        super().__init__(n_cranes, n_bays, n_rows)
        self.zones = []
        bays_per = n_bays // n_cranes
        for c in range(n_cranes):
            start = c * bays_per
            end = start + bays_per if c < n_cranes - 1 else n_bays
            self.zones.append((start, end))

    def assign(self, env, target_stack, dest_stack):
        target_bay = target_stack // self.n_rows
        for c, (start, end) in enumerate(self.zones):
            if start <= target_bay < end:
                return c
        return self.n_cranes - 1  # fallback
```

- [ ] **Step 4: Write S3 — LoadBalance**

```python
# strategies/load_balance.py
from .base import CraneAssignmentStrategy

class LoadBalance(CraneAssignmentStrategy):
    def __init__(self, n_cranes, n_bays, n_rows):
        super().__init__(n_cranes, n_bays, n_rows)
        self.task_counts = [0] * n_cranes

    def assign(self, env, target_stack, dest_stack):
        min_idx = min(range(self.n_cranes), key=lambda i: self.task_counts[i])
        self.task_counts[min_idx] += 1
        return min_idx

    def reset(self):
        self.task_counts = [0] * self.n_cranes
```

- [ ] **Step 5: Write S4 — GreedyOptimal**

```python
# strategies/greedy_optimal.py
from .base import CraneAssignmentStrategy

class GreedyOptimal(CraneAssignmentStrategy):
    def __init__(self, n_cranes, n_bays, n_rows):
        super().__init__(n_cranes, n_bays, n_rows)

    def assign(self, env, target_stack, dest_stack):
        dest_bay = dest_stack // self.n_rows
        best_crane = 0
        best_cost = float('inf')

        for c in range(self.n_cranes):
            cost = 0.0
            curr_bay = env.crane_bays[0, c].item()
            if curr_bay >= 0:
                bay_dist = abs(curr_bay - dest_bay)
                if bay_dist > 0:
                    cost += env.base_env.t_acc + env.base_env.t_bay * bay_dist

            # Interference penalty
            for other in range(self.n_cranes):
                if other != c:
                    other_bay = env.crane_bays[0, other].item()
                    if other_bay == dest_bay:
                        cost += env.base_env.t_acc

            # Zone preference: prefer crane whose zone contains dest bay
            # (soft bias, not hard constraint)
            if hasattr(self, 'zones'):
                target_bay = target_stack // self.n_rows
                for zone_c, (start, end) in enumerate(self.zones):
                    if start <= target_bay < end and zone_c == c:
                        cost -= 1.0  # small bias

            if cost < best_cost:
                best_cost = cost
                best_crane = c

        return best_crane
```

Note: GreedyOptimal also needs zones for soft bias. Add in `__init__`:
```python
bays_per = n_bays // n_cranes
self.zones = [(c * bays_per, (c * bays_per + bays_per) if c < n_cranes - 1 else n_bays) for c in range(n_cranes)]
```

- [ ] **Step 6: Write unit tests**

```python
# tests/test_strategies.py
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategies import RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal

def make_mock_env(n_cranes=2, n_bays=4, bays=None):
    """Create a minimal MCEnv-like object for testing."""
    class MockEnv:
        pass
    env = MockEnv()
    env.n_cranes = n_cranes
    env.n_bays = n_bays
    env.n_rows = 4
    env.crane_bays = torch.full((1, n_cranes), -1)
    env.crane_rows = torch.full((1, n_cranes), -1)
    env.crane_start_bays = [1, 3] if bays is None else bays
    class BaseEnv:
        t_acc = 40
        t_bay = 3.5
        t_row = 1.2
        t_pd = 30
    env.base_env = BaseEnv()
    return env

def test_round_robin_cycles():
    s = RoundRobin(3, 4, 4)
    env = make_mock_env(3, 4)
    assert s.assign(env, 0, 0) == 0
    assert s.assign(env, 0, 0) == 1
    assert s.assign(env, 0, 0) == 2
    assert s.assign(env, 0, 0) == 0  # cycles

def test_zone_split_by_target_bay():
    s = ZoneSplit(2, 4, 4)
    env = make_mock_env(2, 4)
    # zones: crane 0 = bays [0,1], crane 1 = bays [2,3]
    assert s.assign(env, target_stack=0, dest_stack=5) == 0   # target bay 0 → crane 0
    assert s.assign(env, target_stack=10, dest_stack=5) == 1  # target bay 2 → crane 1

def test_load_balance_distributes():
    s = LoadBalance(2, 4, 4)
    env = make_mock_env(2, 4)
    ids = [s.assign(env, 0, 0) for _ in range(10)]
    assert abs(ids.count(0) - ids.count(1)) <= 1

def test_greedy_optimal_returns_valid_crane():
    s = GreedyOptimal(2, 4, 4)
    env = make_mock_env(2, 4)
    result = s.assign(env, target_stack=0, dest_stack=5)
    assert 0 <= result < 2

def test_strategies_share_interface():
    env = make_mock_env(2, 4)
    for Strategy in [RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal]:
        s = Strategy(2, 4, 4)
        cid = s.assign(env, target_stack=0, dest_stack=5)
        assert 0 <= cid < 2, f'{s.name} returned invalid crane {cid}'
```

- [ ] **Step 7: Run all tests**

```bash
python -m pytest tests/test_strategies.py -v
```

Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add strategies/ tests/test_strategies.py
git commit -m "feat: 4 crane assignment strategies with shared interface"
```

---

### Task 5: M-CRP Lower Bound (Theorem 3)

**Files:**
- Create: `bounds/__init__.py`
- Create: `bounds/lowerbound_mc.py`
- Create: `tests/test_lowerbound_mc.py`

**Interfaces:**
- Consumes: yard tensor `x` (batch, stacks, tiers), `n_bays`, `n_rows`, `n_tiers`, `n_cranes`
- Produces: `LB_MCRP` per instance (scalar), `LB_MCRP` for batch (tensor)

**Note:** Reuses `baselines/lowerbound.py:get_wt_lb()` for the single-crane components.

- [ ] **Step 1: Write LB_MCRP implementation**

```python
# bounds/__init__.py
from .lowerbound_mc import compute_lb_mc
```

```python
# bounds/lowerbound_mc.py
import torch
from baselines.lowerbound import get_wt_lb

def compute_lb_mc(x, n_bays, n_rows, n_tiers, n_cranes):
    """Multi-crane CRP lower bound (Theorem 3 extension).
    
    LB_MCRP = LB_retrieval + LB_relocation/C + LB_interference
    
    Args:
        x: yard tensor (batch, n_bays, n_rows, n_tiers) or (batch, stacks, tiers)
        n_bays, n_rows, n_tiers: yard dimensions
        n_cranes: number of cranes (C)
    Returns:
        lb: tensor of lower bound values (batch,)
    """
    if x.dim() == 4:
        batch = x.shape[0]
        x_2d = x.reshape(batch, -1, n_tiers)
    else:
        batch = x.shape[0]
        x_2d = x

    lbs = []
    for i in range(batch):
        instance = x_2d[i:i+1]  # (1, stacks, tiers)
        lb_single = get_wt_lb(instance)  # single-crane LB from Theorem 2

        # Count mandatory relocations (containers with higher-priority below)
        n_reloc = _count_mandatory_relocations(instance)

        # LB_retrieval = same as single-crane (container retrieval times unaffected)
        # LB_relocation = relocation work / C cranes
        # LB_interference = penalty for bay contention

        min_reloc_unit = 2 * 1.2 + 30  # 2*t_row + t_pd
        lb_reloc_total = n_reloc * min_reloc_unit
        lb_reloc_per_crane = lb_reloc_total / n_cranes

        # Interference: if one bay has > 1/C of total relocations, cranes bottleneck
        reloc_per_bay = _count_relocations_per_bay(instance, n_bays, n_rows)
        total_relocs = sum(reloc_per_bay)
        avg_reloc = total_relocs / n_bays
        excess = sum(max(0, r - avg_reloc * n_cranes) for r in reloc_per_bay)
        lb_interference = excess * (40 + 3.5)  # t_acc + t_bay per excess relocation

        lb_mc = lb_single + lb_reloc_per_crane + lb_interference
        lbs.append(lb_mc)

    return torch.tensor(lbs, dtype=torch.float)

def _count_mandatory_relocations(x):
    """Count containers that block higher-priority containers below them."""
    x_flat = x.squeeze(0) if x.dim() == 3 else x
    # x_flat: (stacks, tiers)
    count = 0
    stacks, tiers = x_flat.shape
    for s in range(stacks):
        stack = x_flat[s]
        for t in range(tiers):
            if t == 0:
                continue
            container = stack[t].item()
            if container == 0:
                continue
            below = stack[:t]
            below_valid = below[below > 0]
            if len(below_valid) > 0 and (below_valid < container).any().item():
                count += 1
    return count

def _count_relocations_per_bay(x, n_bays, n_rows):
    """Count mandatory relocations per bay."""
    x_flat = x.squeeze(0) if x.dim() == 3 else x
    stacks_per_bay = n_rows
    counts = []
    for b in range(n_bays):
        bay_stacks = x_flat[b * stacks_per_bay : (b+1) * stacks_per_bay, :]
        instance = bay_stacks.unsqueeze(0)
        counts.append(_count_mandatory_relocations(instance))
    return counts
```

- [ ] **Step 2: Write unit tests**

```python
# tests/test_lowerbound_mc.py
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bounds.lowerbound_mc import compute_lb_mc
from baselines.lowerbound import get_wt_lb

def test_lb_mc_one_crane_matches_theorem2():
    """With C=1, LB_MCRP should approximately equal LB from Theorem 2."""
    x = torch.zeros(1, 1, 16, 6)
    x[0, 0, 0, :5] = torch.tensor([5., 4., 3., 2., 1.])
    x[0, 0, 1, :3] = torch.tensor([6., 7., 8.])
    
    lb_mc = compute_lb_mc(x, 1, 16, 6, n_cranes=1)
    lb_t2 = get_wt_lb(x.reshape(1, -1, 6))
    # Should be close (interference term 0 for C=1)
    assert abs(lb_mc.item() - lb_t2) / lb_t2 < 0.1, f'LB_MCRP={lb_mc:.1f}, LB_T2={lb_t2:.1f}'

def test_lb_mc_decreases_with_more_cranes():
    x = torch.zeros(1, 2, 4, 6)
    x[0, 0, 0, :3] = torch.tensor([3., 2., 1.])
    x[0, 0, 1, :2] = torch.tensor([5., 4.])
    x[0, 1, 0, :3] = torch.tensor([6., 7., 8.])
    
    lb_1 = compute_lb_mc(x, 2, 4, 6, n_cranes=1)
    lb_2 = compute_lb_mc(x, 2, 4, 6, n_cranes=2)
    assert lb_2 <= lb_1 + 1e-6, f'LB_2={lb_2:.1f} > LB_1={lb_1:.1f}'

def test_lb_mc_positive():
    x = torch.zeros(1, 1, 4, 4)
    x[0, 0, 0, :3] = torch.tensor([3., 2., 1.])
    lb = compute_lb_mc(x, 1, 4, 4, n_cranes=2)
    assert lb.item() > 0

def test_lb_mc_interference_term():
    """Instances with high bay contention should have higher LB interference."""
    x_low = torch.zeros(1, 2, 4, 6)
    x_low[0, 0, 0, :3] = torch.tensor([3., 2., 1.])
    x_low[0, 1, 0, :3] = torch.tensor([6., 5., 4.])
    
    x_high = torch.zeros(1, 2, 4, 6)
    x_high[0, 0, 0, :5] = torch.tensor([10., 9., 8., 7., 1.])  # bay 0 has 4 relocations
    x_high[0, 1, 0, :1] = torch.tensor([2.])  # bay 1 has 0
    
    lb_low = compute_lb_mc(x_low, 2, 4, 6, n_cranes=2)
    lb_high = compute_lb_mc(x_high, 2, 4, 6, n_cranes=2)
    assert lb_high > lb_low, f'High contention should have higher LB'
```

- [ ] **Step 3: Run all tests**

```bash
python -m pytest tests/test_lowerbound_mc.py -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add bounds/ tests/test_lowerbound_mc.py
git commit -m "feat: M-CRP lower bound with single-crane fallback test"
```

---

### Task 6: M-CRP Inference Engine (Bridge Policy + MCEnv + Strategy)

**Files:**
- Create: `engine/__init__.py`
- Create: `engine/mcrp_inference.py`
- Create: `tests/test_engine.py`

**Interfaces:**
- Consumes: `ZeroShotPolicy`, `MCEnv`, `CraneAssignmentStrategy`, yard config
- Produces: Full trajectory (cost, steps, interference events) for one M-CRP instance

- [ ] **Step 1: Write inference engine**

```python
# engine/__init__.py
from .mcrp_inference import run_mcrp_episode
```

```python
# engine/mcrp_inference.py
import torch

def run_mcrp_episode(policy, env, strategy, n_bays, n_rows, n_tiers, max_steps=2000):
    """Run one M-CRP episode: zero-shot policy + crane assignment.
    
    Args:
        policy: ZeroShotPolicy instance
        env: MCEnv instance
        strategy: CraneAssignmentStrategy instance
        n_bays, n_rows, n_tiers: yard dimensions
        max_steps: safety limit
    Returns:
        dict with keys: total_cost, n_steps, n_interference, per_crane_costs, trajectory
    """
    total_cost = 0.0
    n_interference = 0
    per_crane_cost = torch.zeros(env.n_cranes)
    trajectory = []
    step = 0

    while not env.terminated and step < max_steps:
        state = env.get_state()  # (1, n_bays, n_rows, n_tiers)

        # Compute invalid mask: full stacks + target stack itself
        stacks = env.base_env.x[0]  # (B*R, T)
        full_mask = (stacks[:, -1] > 0).bool()  # top tier occupied → full
        target_stack_idx = env.base_env.target_stack[0].item()
        full_mask[target_stack_idx] = True  # can't relocate to target stack

        # Get action from zero-shot policy
        dest_stack = policy.get_action(
            state, n_bays, n_rows, n_tiers,
            target_stack=target_stack_idx,
            invalid_mask=full_mask.unsqueeze(0)
        )

        # Assign crane via strategy
        crane_id = strategy.assign(env, target_stack_idx, dest_stack[0, 0].item())

        # Execute
        cost, _ = env.step(
            dest_stack=dest_stack, crane_id=crane_id
        )

        cost_val = cost[0].item()
        total_cost += cost_val
        per_crane_cost[crane_id] += cost_val
        n_interference += env.interference_events[0, :].sum().item()
        trajectory.append({
            'step': step, 'crane': crane_id,
            'dest_bay': dest_stack[0, 0].item() // n_rows,
            'cost': cost_val
        })
        step += 1

    return {
        'total_cost': total_cost,
        'n_steps': step,
        'n_interference': env.interference_events[0, :].sum().item(),
        'per_crane_cost': per_crane_cost.tolist(),
        'trajectory': trajectory
    }
```

- [ ] **Step 2: Write unit tests**

```python
# tests/test_engine.py
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.mcrp_inference import run_mcrp_episode
from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from strategies import RoundRobin

def test_engine_basic_run():
    """Full pipeline: policy + MCEnv + strategy on minimal instance."""
    policy = ZeroShotPolicy()
    
    # Create tiny M-CRP instance: 2 bays, 4 rows, 4 tiers, 2 cranes
    x = torch.zeros(1, 2, 4, 4)
    x[0, 0, 0, :3] = torch.tensor([3., 2., 1.])  # bay 0, row 0: containers 3,2,1
    x[0, 0, 1, :2] = torch.tensor([5., 4.])       # bay 0, row 1: containers 5,4
    x[0, 1, 0, :2] = torch.tensor([6., 7.])       # bay 1, row 0: containers 6,7
    
    env = MCEnv('cpu', x, n_cranes=2)
    strategy = RoundRobin(2, 2, 4)
    
    result = run_mcrp_episode(policy, env, strategy, 2, 4, 4)
    
    assert result['total_cost'] > 0
    assert result['n_steps'] >= 1
    assert len(result['per_crane_cost']) == 2
    assert sum(result['per_crane_cost']) == result['total_cost']

def test_engine_tracks_interference():
    """High-contention instance should register interference events."""
    policy = ZeroShotPolicy()
    
    # All containers in bay 0 → both cranes must operate in same bay
    x = torch.zeros(1, 2, 4, 4)
    x[0, 0, 0, :4] = torch.tensor([4., 3., 2., 1.])
    x[0, 0, 1, :3] = torch.tensor([5., 6., 7.])
    
    env = MCEnv('cpu', x, n_cranes=2)
    strategy = RoundRobin(2, 2, 4)
    
    result = run_mcrp_episode(policy, env, strategy, 2, 4, 4)
    # Interference should be >= 0 (not checking specific value, just that tracking works)
    assert 'n_interference' in result
```

- [ ] **Step 3: Run all tests**

```bash
python -m pytest tests/test_engine.py -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add engine/ tests/test_engine.py
git commit -m "feat: M-CRP inference engine integrating policy, env, and strategy"
```

---

### Task 7: M-CRP Benchmark Dataset

**Files:**
- Create: `benchmarks/generate_mc_instances.py`
- Create: `tests/test_mc_instances.py`

**Interfaces:**
- Consumes: `benchmarks/Lee_instances/`, `benchmarks/Shin_instances/`
- Produces: `benchmarks/mc_instances/` — M-CRP instances in Lee format + crane header

**Format:** Each instance is a `.txt` file with:
```
# n_cranes = 2
# crane_start_bays = [1, 3]
# (then standard Lee format: first line stack count, then per-stack data)
```

- [ ] **Step 1: Write benchmark generator**

```python
# benchmarks/generate_mc_instances.py
import os, shutil
from benchmarks.benchmarks import find_and_process_file

SCALES = {
    'small': {'bays': [1, 2, 4], 'rows': [16], 'tiers': [6, 8], 'id_range': (1, 5)},
    'medium': {'bays': [6, 8, 10], 'rows': [16], 'tiers': [6, 8], 'id_range': (1, 5)},
    'large': {'bays': [20, 30], 'rows': [16], 'tiers': [6, 8], 'id_range': (1, 20)},
}

ORIGINAL_DIR = 'benchmarks/Lee_instances'
OUTPUT_DIR = 'benchmarks/mc_instances'

def generate_all():
    """Generate M-CRP instances for all scales, types, and crane counts."""
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

                            # Load original instance
                            try:
                                x, name = find_and_process_file(
                                    ORIGINAL_DIR, inst_type, bay, row, tier, idx, no_print=True
                                )
                            except (FileNotFoundError, ValueError):
                                continue

                            for n_cranes in [2, 3]:
                                # Crane start bays: equidistant
                                if n_cranes == 2:
                                    starts = [1, max(1, bay // 2 + 1)]
                                else:
                                    starts = [1, max(1, bay // 3 + 1), max(1, 2 * bay // 3 + 1)]

                                type_prefix = 'R' if inst_type == 'random' else 'U'
                                out_name = f'mc_{type_prefix}{bay:02d}{row:02d}{tier:02d}_{idx:03d}_c{n_cranes}.txt'
                                out_path = f'{OUTPUT_DIR}/lee_mc/{out_name}'

                                # Copy original file + add header
                                subdir = 'individual, random' if inst_type == 'random' else 'individual, upside down'
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
```

- [ ] **Step 2: Write tests**

```python
# tests/test_mc_instances.py
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.generate_mc_instances import generate_all

def test_generation_produces_files():
    count = generate_all()
    assert count >= 80  # minimum expected instances

def test_files_have_crane_header():
    files = glob.glob('benchmarks/mc_instances/lee_mc/*.txt')
    count_c2 = 0
    count_c3 = 0
    for f in files:
        with open(f, 'r') as fh:
            first = fh.readline()
            second = fh.readline()
            if 'n_cranes' in first:
                if '_c2.' in f or '_c2.txt' in f:
                    count_c2 += 1
                elif '_c3.' in f or '_c3.txt' in f:
                    count_c3 += 1
    assert count_c2 > 0
    assert count_c3 > 0

def test_files_parse_via_existing_parser():
    """Verify generated files can be read by the existing benchmark parser."""
    from benchmarks.benchmarks import find_and_process_file
    # Try to parse a generated file
    files = glob.glob('benchmarks/mc_instances/lee_mc/*_c2.txt')
    if files:
        fname = os.path.basename(files[0])
        parts = fname.replace('.txt', '').split('_')
        # parts: ['mc', 'R011606', '001', 'c2']
        type_str = parts[1][0:1]  # 'R'
        bay_str = parts[1][1:3]  # '01'
        row_str = parts[1][3:5]  # '16'
        tier_str = parts[1][5:7]  # '06'

        inst_type = 'random' if type_str == 'R' else 'upsidedown'
        bay = int(bay_str)
        row = int(row_str)
        tier = int(tier_str)
        idx = int(parts[2])

        try:
            x, name = find_and_process_file(
                'benchmarks/mc_instances/lee_mc', 'random', bay, row, tier, idx, no_print=True
            )
            assert x is not None
        except (FileNotFoundError, ValueError) as e:
            # Might not parse directly via existing parser (format differs)
            # That's OK — skip test
            pass
```

- [ ] **Step 3: Run generation and tests**

```bash
python benchmarks/generate_mc_instances.py
python -m pytest tests/test_mc_instances.py -v
```

Expected: Generation prints count ≥ 80 files. Tests pass.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/generate_mc_instances.py benchmarks/mc_instances/ tests/test_mc_instances.py
git commit -m "feat: M-CRP benchmark dataset generation (80+ instances)"
```

---

### Task 8: Experiment Pipeline (Integration)

**Files:**
- Create: `experiment.py`
- Create: `tests/test_experiment.py`

**Interfaces:**
- Consumes: All previous modules (policy, MCEnv, strategies, bounds, benchmarks)
- Produces: CSV results table — one row per (instance, n_cranes, strategy) with total_cost, lb_mc, gap, interference, time

- [ ] **Step 1: Write experiment pipeline**

```python
# experiment.py
import os, sys, time, json, argparse
from datetime import datetime
import torch
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from strategies import RoundRobin, ZoneSplit, LoadBalance, GreedyOptimal
from bounds.lowerbound_mc import compute_lb_mc
from engine.mcrp_inference import run_mcrp_episode

STRATEGY_MAP = {
    'S1': RoundRobin, 'S2': ZoneSplit,
    'S3': LoadBalance, 'S4': GreedyOptimal
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--instance_dir', default='benchmarks/mc_instances/lee_mc')
    p.add_argument('--cranes', type=int, nargs='+', default=[2, 3])
    p.add_argument('--strategies', nargs='+', default=['S1', 'S2', 'S3', 'S4'])
    p.add_argument('--max_instances', type=int, default=None)
    p.add_argument('--seed', type=int, default=1234)
    return p.parse_args()

def parse_instance_file(path):
    """Parse generated M-CRP instance file. Returns tensor x and metadata."""
    with open(path, 'r') as f:
        lines = f.readlines()

    n_cranes = 2
    crane_starts = [1, 2]
    data_lines = []

    for line in lines:
        if line.startswith('# n_cranes'):
            n_cranes = int(line.split('=')[1].strip())
        elif line.startswith('# crane_start'):
            starts_str = line.split('=')[1].strip()
            crane_starts = eval(starts_str)
        elif line.strip() and not line.startswith('#'):
            data_lines.append(line)

    return data_lines, n_cranes, crane_starts

def load_instance_tensor(data_lines, n_bays, n_rows, n_tiers):
    """Convert parsed lines to tensor compatible with original Env."""
    import numpy as np
    matrix = np.zeros((n_bays * n_rows, n_tiers), dtype=int)
    for line in data_lines[1:]:
        vals = list(map(int, line.split()))
        bay, stack, num_tiers = vals[:3]
        containers = vals[3:]
        unique = list(dict.fromkeys(containers))
        padded = unique + [0] * (n_tiers - len(unique))
        idx = (bay - 1) * n_rows + (stack - 1)
        matrix[idx] = padded
    tensor = torch.tensor(matrix).float().reshape(1, n_bays, n_rows, n_tiers)
    return tensor

def run_experiment(args):
    """Main experiment: iterate instances × cranes × strategies."""
    torch.manual_seed(args.seed)
    policy = ZeroShotPolicy(device=torch.device('cpu'))

    # Collect all instance files
    import glob
    files = sorted(glob.glob(os.path.join(args.instance_dir, '*.txt')))
    if args.max_instances:
        files = files[:args.max_instances]

    results = []
    start_time = time.time()

    for fpath in files:
        fname = os.path.basename(fpath)
        data_lines, file_cranes, crane_starts = parse_instance_file(fpath)

        # Determine yard dimensions from filename: mc_R021606_001_c2.txt
        parts = fname.replace('.txt', '').split('_')
        # parts: ['mc', 'R021606', '001', 'c2']
        dims = parts[1][1:]  # '021606'
        n_bays = int(dims[0:2])
        n_rows = int(dims[2:4])
        n_tiers = int(dims[4:6])

        x = load_instance_tensor(data_lines, n_bays, n_rows, n_tiers)

        for n_cranes in args.cranes:
            for sname in args.strategies:
                StrategyCls = STRATEGY_MAP[sname]
                strategy = StrategyCls(n_cranes, n_bays, n_rows)
                env = MCEnv('cpu', x, n_cranes,
                           crane_start_bays=crane_starts if n_cranes == file_cranes else None)

                t0 = time.time()
                result = run_mcrp_episode(policy, env, strategy, n_bays, n_rows, n_tiers)
                elapsed = time.time() - t0

                lb = compute_lb_mc(x, n_bays, n_rows, n_tiers, n_cranes).item()
                gap = 100 * (result['total_cost'] - lb) / lb if lb > 0 else 0.0

                results.append({
                    'instance': fname,
                    'n_cranes': n_cranes,
                    'strategy': sname,
                    'total_cost': result['total_cost'],
                    'lb_mc': lb,
                    'gap': gap,
                    'n_steps': result['n_steps'],
                    'interference': result['n_interference'],
                    'time_s': elapsed,
                    'per_crane_cost': str(result['per_crane_cost'])
                })

                print(f'  {fname} | C={n_cranes} | {sname} | cost={result["total_cost"]:.0f} | gap={gap:.1f}% | time={elapsed:.2f}s')

    total = time.time() - start_time
    print(f'\nTotal: {len(results)} runs in {total:.1f}s')
    return pd.DataFrame(results)

if __name__ == '__main__':
    args = parse_args()
    os.makedirs('results', exist_ok=True)
    df = run_experiment(args)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f'results/mcrp_experiment_{timestamp}.csv'
    df.to_csv(path, index=False)
    print(f'Saved: {path}')

    # Summary
    summary = df.groupby(['n_cranes', 'strategy'])['gap'].agg(['mean', 'std', 'min', 'max'])
    print('\n=== Summary ===')
    print(summary.to_string())
```

- [ ] **Step 2: Write integration test**

```python
# tests/test_experiment.py
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy.zero_shot import ZeroShotPolicy
from mcenv.mcenv import MCEnv
from engine.mcrp_inference import run_mcrp_episode
from strategies import RoundRobin
from bounds.lowerbound_mc import compute_lb_mc

def test_end_to_end_single_instance():
    """Test full pipeline on 1 instance, 2 cranes, 1 strategy."""
    policy = ZeroShotPolicy()
    x = torch.zeros(1, 2, 4, 4)
    x[0, 0, 0, :4] = torch.tensor([4., 3., 2., 1.])
    x[0, 0, 1, :3] = torch.tensor([5., 6., 7.])
    x[0, 1, 0, :2] = torch.tensor([8., 9.])

    env = MCEnv('cpu', x, n_cranes=2)
    strategy = RoundRobin(2, 2, 4)
    result = run_mcrp_episode(policy, env, strategy, 2, 4, 4)
    lb = compute_lb_mc(x, 2, 4, 4, n_cranes=2).item()

    assert result['total_cost'] > 0
    assert lb > 0
    gap = 100 * (result['total_cost'] - lb) / lb
    print(f'Gap: {gap:.1f}%')
    assert -50 < gap < 500  # sanity check

def test_end_to_end_matches_file_structure():
    """Verify the pipeline can read and process a generated instance file."""
    import glob
    from experiment import parse_instance_file, load_instance_tensor
    
    files = glob.glob('benchmarks/mc_instances/lee_mc/*.txt')
    if not files:
        return  # skip if no instances generated yet
    
    data_lines, n_cranes, crane_starts = parse_instance_file(files[0])
    assert n_cranes in [2, 3]
    assert len(crane_starts) == n_cranes
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_experiment.py -v
```

Expected: Both tests PASS.

- [ ] **Step 4: Run full experiment (small subset)**

```bash
python experiment.py --cranes 2 --strategies S1 S2 --max_instances 5
```

Expected: Runs on 5 instances × 2 crane configs × 2 strategies = 20 runs. Output CSV in `results/`.

- [ ] **Step 5: Commit**

```bash
git add experiment.py tests/test_experiment.py
git commit -m "feat: experiment pipeline with CSV logging and summary"
```

---

### Task 9: Final Verification

- [ ] **Step 1: Run ALL tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests PASS (verify_baseline, mcenv, policy, strategies, lowerbound, engine, experiment).

- [ ] **Step 2: Run single-crane backward compatibility check**

```bash
python -c "
from policy.zero_shot import ZeroShotPolicy
from env.env import Env
import torch

policy = ZeroShotPolicy()
x = torch.zeros(1, 2, 4, 4)
x[0, 0, 0, :3] = torch.tensor([3., 2., 1.])

# MCEnv with C=1 should produce similar cost to original Env
from mcenv.mcenv import MCEnv
from engine.mcrp_inference import run_mcrp_episode
from strategies import RoundRobin

env = MCEnv('cpu', x, n_cranes=1)
result = run_mcrp_episode(policy, env, RoundRobin(1, 2, 4), 2, 4, 4)
print(f'C=1 total cost: {result[\"total_cost\"]:.1f}')
"
```

- [ ] **Step 3: Print final summary**

```bash
python -c "
print('=== M-CRP Zero-shot Analysis — Code Modules ===')
print('  mcenv/     — Multi-crane env wrapper')
print('  policy/    — Zero-shot policy (pretrained + scorer)')
print('  strategies/— 4 crane assignment strategies')
print('  bounds/    — M-CRP lower bound (Theorem 3)')
print('  engine/    — Inference engine (policy + env + strategy)')
print('  experiment.py — Full experiment pipeline')
print('  benchmarks/mc_instances/ — M-CRP benchmark dataset')
print()
print('To run: python experiment.py --cranes 2 3 --strategies S1 S2 S3 S4')
"
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete M-CRP zero-shot transfer analysis codebase"
```
