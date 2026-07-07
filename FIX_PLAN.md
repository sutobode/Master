# Fix Plan: M-CRP Zero-shot Transfer

Mapping proposal claims (in docs/superpowers/proposal/) to code issues.

## C1: M-CRP Definition (MCEnv)

| # | Issue | Fix | Priority |
|---|-------|-----|----------|
| 1 | `_resolve_interference` fallback returns interfering crane → violates A6 | Raise RuntimeError instead | CRITICAL |
| 2 | `_validate_interference` checks position, not busy state | Add `crane_busy` tracking | HIGH |
| 3 | A7 non-crossing constraint not implemented | Add `_validate_non_crossing()` | MEDIUM |

## C2: Lower Bound (bounds/lowerbound_mc.py)

| # | Issue | Fix | Priority |
|---|-------|-----|----------|
| 4 | LB backward compat test tolerance = 10% | Reduce to 2% | HIGH |
| 5 | `_count_mandatory_relocations` duplicates `count_disorder_per_row` | Refactor to share code | LOW |

## C3: Strategies (strategies/)

| # | Issue | Fix | Priority |
|---|-------|-----|----------|
| 6 | S4 GreedyOptimal isn't 1-step lookahead, is heuristic O(C²) | Implement true lookahead: for each crane, simulate cost | HIGH |
| 7 | S3 LoadBalance complexity claimed O(C·log C) but is O(C) | Fix to use heapq for O(C·log C) | LOW |
| 8 | Duplicate zone init code in ZoneSplit + GreedyOptimal | Factor into base class | LOW |

## C4: Zero-shot Pipeline (policy/, engine/)

| # | Issue | Fix | Priority |
|---|-------|-----|----------|
| 9 | `max_steps=2000` hardcoded → too low for large instances | Dynamic: `n_slots * 2` | CRITICAL |
| 10 | `get_scores` hardcodes cost params 40,3.5,1.2,30 | Pass from env | MEDIUM |
| 11 | `compare_all.py` only tests 2 instances | Extend to all configs | HIGH |
| 12 | `experiment.py` no C=1 sanity check at start | Add backward compat verification | HIGH |

## C5: Benchmark (benchmarks/)

| # | Issue | Fix | Priority |
|---|-------|-----|----------|
| 13 | Generate only 16 rows | Add n_rows parameter | LOW |
