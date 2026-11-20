"""Frozen one-factor measurement and comparison design."""

DATASETS = ("cora", "citeseer", "pubmed")
STAGES = ("projection", "terms", "score", "softmax", "aggregation", "output")
SWEEPS = dict(terms=("A1","A2"), score=("B1","B2"), softmax=("C1","C2","C3"),
              aggregation=("D1","D2","D3","D4"))
DEFAULT = {stage: orders[0] for stage, orders in SWEEPS.items()}
ORDERS = dict(A1="N → H → F", A2="H → N → F", B1="E → H", B2="H → E",
              C1="V → H → Eᵢₙ", C2="H → V → Eᵢₙ", C3="V → Eᵢₙ → H",
              D1="V → Eᵢₙ → H → F", D2="V → H → Eᵢₙ → F",
              D3="V → H → F → Eᵢₙ", D4="H → V → Eᵢₙ → F")
SYMBOLS = {code: "loop_"+code.lower() for code in ORDERS}
SEED = 2026112200
FLOOR = 1.01


def design(sweep, code):
    if code not in SWEEPS[sweep]:
        raise ValueError(f"{code} does not belong to the {sweep} sweep")
    return dict(DEFAULT, **{sweep: code})


def columns(sweep):
    if sweep=="stock":
        return ["plain/DGL", "profile/DGL", "timer/empty", "timer/six"]
    orders = SWEEPS[sweep]
    return [*(f"{mode}/{code}" for mode in ("plain","profile","isolated") for code in orders),
            "allocation", "timer/empty", "timer/six"]


def memory(n, edges):
    """Distinct live operand/output payload, not cache traffic or allocator usage."""
    z, term, edge, weights = n*4*8*4, n*4*4, edges*4*4, 4*8*4
    ptr, idx = (n+1)*8, edges*8
    return {
        "terms": dict(tensor_bytes=z+2*term+2*weights, index_bytes=0, output_bytes=2*term, scratch_bytes=0),
        "score": dict(tensor_bytes=2*term+edge, index_bytes=2*idx, output_bytes=edge, scratch_bytes=0),
        "softmax": dict(tensor_bytes=2*edge, index_bytes=ptr+idx, output_bytes=edge, scratch_bytes=2*4*4),
        "aggregation": dict(tensor_bytes=2*z+edge, index_bytes=ptr+2*idx, output_bytes=z, scratch_bytes=0),
    }
