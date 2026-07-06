"""
M-CRP: Multi-Crane Container Retrieval Problem
==============================================

Formal definition (extension of CRP from Shin et al. 2026, TR-C):

Input:
  - Yard: B bays x R rows x T tiers. C cranes.
  - N containers, each with retrieval order r in {1..N}.
  - Crane start positions: {pos_c} for c in {1..C}.

Constraints:
  (A1-A5) Same as single-crane CRP (Shin et al. 2026 Assumptions 1-5)
  (A6) Interference: |bay_c - bay_{c'}| >= 1 for all c != c'
  (A7) Non-crossing: order of cranes along bay axis preserved over time

Decision:
  At each step, select (dest_stack, crane_id) for blocking container.

Objective:
  Minimize total working time sum_c (travel_c + handling_c) + penalty

NP-hardness:
  Reduction from BRP (Caserta et al. 2012). CRP with C=1 is NP-hard
  (Shin et al. 2026 Theorem 1), thus M-CRP with C>=1 is NP-hard.

Lower bound (Theorem 3):
  LB_MCRP = LB_retrieval + LB_relocation / C + LB_interference
"""

import torch
from baselines.lowerbound import get_wt_lb


def compute_lb_mc(x, n_bays, n_rows, n_tiers, n_cranes):
    if x.dim() == 4:
        batch = x.shape[0]
        x_2d = x.reshape(batch, -1, n_tiers)
    else:
        batch = x.shape[0]
        x_2d = x

    lbs = []
    for i in range(batch):
        instance = x_2d[i:i+1]
        lb_single = get_wt_lb(instance)

        n_reloc = _count_mandatory_relocations(instance)
        min_reloc_unit = 2 * 1.2 + 30
        lb_reloc_total = n_reloc * min_reloc_unit
        lb_reloc_per_crane = lb_reloc_total / n_cranes

        reloc_per_bay = _count_relocations_per_bay(instance, n_bays, n_rows)
        total_relocs = sum(reloc_per_bay)
        max_per_bay = max(reloc_per_bay) if reloc_per_bay else 0
        ideal_per_crane = total_relocs / max(n_cranes, 1)
        excess = max(0, max_per_bay - ideal_per_crane)
        lb_interference = excess * (40 + 3.5)

        lb_mc = lb_single + lb_reloc_per_crane + lb_interference
        lbs.append(lb_mc)

    return torch.tensor(lbs, dtype=torch.float)


def _count_mandatory_relocations(x):
    x_flat = x.squeeze(0) if x.dim() == 3 else x
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
    x_flat = x.squeeze(0) if x.dim() == 3 else x
    stacks_per_bay = n_rows
    counts = []
    for b in range(n_bays):
        bay_stacks = x_flat[b * stacks_per_bay: (b + 1) * stacks_per_bay, :]
        counts.append(_count_mandatory_relocations(bay_stacks.unsqueeze(0)))
    return counts
