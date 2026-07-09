"""
M-CRP: Multi-Crane Container Retrieval Problem — valid lower bounds
===================================================================

Formal setting (extension of CRP from Shin et al. 2026, TR-C):

Input:
  - Yard: B bays x R rows x T tiers. C cranes.
  - N containers, each with retrieval order r in {1..N}.

Constraints:
  (A1-A5) Same as single-crane CRP (Shin et al. 2026 Assumptions 1-5)
  (A6) Interference: no two cranes may occupy the same bay at the same time
  (A7) Non-crossing: crane order along the bay axis is preserved over time

Objectives:
  - total work  W  = sum over cranes of (travel + handling) time
  - makespan    M* = max over cranes of completion time

Theorem 3 (this file). Let M be the set of containers that must be
relocated at least once (a container is in M iff some container beneath it
in its initial stack has a smaller retrieval order); let r_i be the initial
row of container i and B_occ the number of bays holding >= 1 container.
Assume t_pd >= (R - 2) * t_row (holds for the standard parameters
t_pd = 30, t_row = 1.2, R = 16). Then for every feasible M-CRP schedule:

  reloc_unit(C) = (2*t_row if C == 1 else t_row) + t_pd
  LB_work(C)    =   sum_{i not in M} (t_row * r_i + t_pd)
                  + sum_{i in M}     (reloc_unit(C) + t_row + t_pd)
                  + max(0, B_occ - C) * (t_acc + t_bay)
  LB_makespan(C) = max( LB_work(C) / C ,  max_b W_b )
  where W_b = sum_{i not in M, bay(i)=b} (t_row * r_i + t_pd).

Validity arguments (paper Section 3.3):
  * i not in M: retrieving i from its initial position costs
    t_row * r_i + t_pd; voluntarily relocating i first costs at least
    reloc_unit(C) + t_row + t_pd >= t_row * r_i + t_pd whenever
    t_pd >= (r_i - 2) * t_row, so the initial-position term is a valid
    per-container lower bound under the stated parameter condition.
  * i in M: at least one relocation (pickup + >= 1 row of travel +
    set-down; with C = 1 the crane must also return to the target stack,
    adding a second t_row as in Shin et al. Theorem 2) plus a final
    retrieval from row >= 1.
  * Bay-entry: a crane must be physically present in every occupied bay
    (pickups happen at the container's stack). C cranes cover at most C
    bays with their free initial placement; each additional occupied bay
    costs at least one inter-bay move, t_acc + t_bay.
  * Makespan: total work split over C cranes gives LB_work / C; and by A6
    all operations inside one bay are time-sequential, so the fixed
    (never-relocated) retrieval workload of any single bay is a makespan
    lower bound.
  * C = 1: both bounds reduce to a quantity dominated by Shin et al.'s
    Theorem 2 bound, so we return that (tighter, published) bound instead.

Unlike the previous revision of this file, neither bound can exceed the
cost of a feasible schedule (the old formula subtracted-and-divided the
single-crane traversal bound, which over-estimated multi-crane retrieval
cost and produced negative "gaps").
"""

import torch
from baselines.lowerbound import get_wt_lb


def _mandatory_mask(stack_2d):
    """Boolean mask (stacks, tiers): True where the container must be
    relocated at least once (some smaller retrieval order lies below it)."""
    s, t = stack_2d.shape
    below_min = torch.full((s, t), float('inf'))
    running = torch.full((s,), float('inf'))
    for tier in range(t):
        below_min[:, tier] = running
        vals = stack_2d[:, tier]
        occupied = vals > 0
        running = torch.where(occupied, torch.minimum(running, vals), running)
    occupied = stack_2d > 0
    return occupied & (below_min < stack_2d)


def compute_lb_mc(x, n_bays, n_rows, n_tiers, n_cranes,
                  t_row=1.2, t_bay=3.5, t_acc=40.0, t_pd=30.0):
    """Valid lower bounds for the M-CRP.

    Args:
        x: (batch, n_bays, n_rows, n_tiers) yard tensor, 0 = empty.
        n_cranes: C >= 1.

    Returns:
        dict with 'work' and 'makespan' tensors of shape (batch,).
        At C=1 both equal Shin et al.'s Theorem 2 bound (tighter and valid).
    """
    if x.dim() != 4:
        raise ValueError(f'Expected 4D input (batch, n_bays, n_rows, n_tiers), got {x.dim()}D')
    if t_pd < (n_rows - 2) * t_row:
        raise ValueError(
            f'compute_lb_mc requires t_pd >= (n_rows - 2) * t_row for the '
            f'per-container "fixed" term to be a valid lower bound '
            f'(Theorem, proof sketch); got t_pd={t_pd}, n_rows={n_rows}, '
            f't_row={t_row} -> threshold={(n_rows - 2) * t_row}. Using '
            f'out-of-range parameters here can silently reintroduce invalid '
            f'(negative-gap) bounds.'
        )
    batch = x.shape[0]

    lb_work, lb_makespan = [], []
    for i in range(batch):
        if n_cranes == 1:
            lb = float(get_wt_lb(x[i:i + 1]))
            lb_work.append(lb)
            lb_makespan.append(lb)
            continue

        stacks = x[i].reshape(n_bays * n_rows, n_tiers)
        mandatory = _mandatory_mask(stacks)
        occupied = stacks > 0

        stack_idx = torch.arange(n_bays * n_rows).unsqueeze(1).expand_as(stacks)
        rows = (stack_idx % n_rows) + 1          # initial row of each slot
        bays = stack_idx // n_rows               # 0-indexed bay of each slot

        fixed = occupied & ~mandatory            # never necessarily relocated
        reloc_unit = t_row + t_pd                # C >= 2: no forced return leg

        work_fixed = (t_row * rows[fixed].float()).sum().item() + t_pd * int(fixed.sum())
        n_mand = int(mandatory.sum())
        work_mand = n_mand * (reloc_unit + t_row + t_pd)

        n_bays_occ = int(torch.unique(bays[occupied]).numel())
        work_entry = max(0, n_bays_occ - n_cranes) * (t_acc + t_bay)

        work = work_fixed + work_mand + work_entry

        per_bay = torch.zeros(n_bays)
        if fixed.any():
            fb = bays[fixed]
            fw = t_row * rows[fixed].float() + t_pd
            per_bay.scatter_add_(0, fb.reshape(-1), fw.reshape(-1))
        makespan = max(work / n_cranes, float(per_bay.max()))
        if n_bays == 1:
            # Every operation touches the single bay; A6 serializes them all,
            # so the makespan is bounded below by the total work.
            makespan = work

        lb_work.append(work)
        lb_makespan.append(makespan)

    return {
        'work': torch.tensor(lb_work, dtype=torch.float),
        'makespan': torch.tensor(lb_makespan, dtype=torch.float),
    }


def _count_mandatory_relocations(x):
    """Number of containers that must be relocated at least once.

    Accepts (1, stacks, tiers) or (stacks, tiers)."""
    x_flat = x.squeeze(0) if x.dim() == 3 else x
    return int(_mandatory_mask(x_flat).sum())
