"""D: whole-row ownership alternatives at a fixed four-thread budget."""
from orchestration import Pipeline

def softmax_cases(g,native,threads):
    assert threads==4
    # EXPERIMENTAL CHANGE: only softmax's thread-to-destination assignment differs.
    # Fixed CSC, EH, NHF and per-row/head/neighbor order; aggregation stays static.
    return {v:Pipeline(g,native=native,softmax_schedule=v,threads=threads) for v in ['static','degree_balanced']}

def aggregation_cases(g,native,threads):
    assert threads==4
    # EXPERIMENTAL CHANGE: only aggregation ownership changes, never its arithmetic
    # or layout. Softmax stays static. Rows are not split: no atomics or races.
    return {v:Pipeline(g,native=native,aggregation_schedule=v,threads=threads) for v in ['static','degree_balanced']}
