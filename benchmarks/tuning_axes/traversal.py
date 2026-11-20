"""B: actual stage-specific sole-format DGL primitive views, not whole-layer flags."""
from orchestration import Pipeline

def score_cases(g,native,threads):
    # EXPERIMENTAL CHANGE: only gsddmm(add) traversal is COO versus CSR.
    # Destination softmax and aggregation always use their CSC views.
    return {v:Pipeline(g,score=v,aggregation='csc',threads=threads) for v in ['coo','csr']}

def aggregation_cases(g,native,threads):
    # EXPERIMENTAL CHANGE: only gspmm(mul,sum) is destination CSC or edge COO.
    # Score traversal remains COO; destination normalization remains CSC.
    return {v:Pipeline(g,score='coo',aggregation=v,threads=threads) for v in ['csc','coo']}
