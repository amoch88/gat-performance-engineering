"""Additional ordering policies, with unchanged graph construction and feature parity."""
import numpy as np

from policies import AXES as HISTORICAL_AXES, case_specs as historical_specs, permutation as historical_permutation

AXES = {
    "node": (*HISTORICAL_AXES["node"], "metis"),
    "edge": (*HISTORICAL_AXES["edge"], "src_dst", "dst_src", "morton"),
}
LABELS = {
    "metis": "METIS (32 partitions)", "src_dst": "Source then destination",
    "dst_src": "Destination then source", "morton": "Morton (Z-order)",
}
ADDED_EDGE = ("src_dst", "dst_src", "morton")
PARTITIONS = 32
MORTON_CONVENTION = "uint64: src bit i at 2*i; dst bit i at 2*i+1; ties by original EID"


def case_specs(dataset, axis):
    old = historical_specs(dataset, axis)
    return [(p, p, None) for p in AXES[axis] if p != "random"] + [
        case for case in old if case[1] == "random"
    ]


def morton_keys(src, dst):
    """Interleave up to 32 bits per coordinate without truncation."""
    def spread(ids):
        ids = np.asarray(ids)
        if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids > 0xFFFFFFFF):
            raise ValueError("Morton coordinates must be unsigned 32-bit integer values")
        x = ids.astype(np.uint64)
        for shift, mask in ((16, 0x0000FFFF0000FFFF), (8, 0x00FF00FF00FF00FF),
                            (4, 0x0F0F0F0F0F0F0F0F), (2, 0x3333333333333333),
                            (1, 0x5555555555555555)):
            x = (x | (x << np.uint64(shift))) & np.uint64(mask)
        return x
    if np.shape(src) != np.shape(dst):
        raise ValueError("Morton coordinates must have the same shape")
    return spread(src) | (spread(dst) << np.uint64(1))


def nodes_from_assignment(assignment, num_nodes, partitions):
    assignment = np.asarray(assignment)
    if (assignment.shape != (num_nodes,) or assignment.dtype.kind not in "iu"
            or np.any(assignment < 0) or np.any(assignment >= partitions)):
        raise ValueError("Invalid METIS assignments")
    return np.argsort(assignment, kind="stable").astype(np.int64)


def permutation(num_nodes, src, dst, axis, policy, seed=None, metis=None):
    if policy in HISTORICAL_AXES[axis]:
        return historical_permutation(num_nodes, src, dst, axis, policy, seed)
    if axis == "node" and policy == "metis":
        if metis is None:
            raise ValueError("A frozen METIS permutation is required")
        return metis, np.arange(len(src), dtype=np.int64)
    if axis == "edge":
        eids = np.arange(len(src), dtype=np.int64)
        if policy == "src_dst":
            sort_keys = (eids, dst, src)
        elif policy == "dst_src":
            sort_keys = (eids, src, dst)
        elif policy == "morton":
            sort_keys = (eids, morton_keys(src, dst))
        else:
            raise ValueError(f"Unknown edge policy {policy}")
        return np.arange(num_nodes, dtype=np.int64), np.lexsort(sort_keys)
    raise ValueError(f"Unknown {axis} policy {policy}")
