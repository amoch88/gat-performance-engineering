"""One-factor ordering policies and exact DGL physical graph audits."""
from collections import deque
import hashlib
from time import perf_counter_ns

import dgl
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

NODE = ("original", "rcmk", "degree_asc", "degree_desc", "bfs", "random")
EDGE = ("original", "src", "dst", "random")
DATASETS = ("cora", "citeseer", "pubmed")
AXES = {"node": NODE, "edge": EDGE}
FORMATS = ("coo", "csr", "csc")
SEED = 2026112000


def case_specs(dataset, axis):
    # Use the same three independent permutations in all eight process rounds.
    base = SEED + 10000 + 100 * DATASETS.index(dataset) + 10 * list(AXES).index(axis)
    return [(p, p, None) for p in AXES[axis] if p != "random"] + [
        (f"random_{i}", "random", base+i) for i in range(3)
    ]


def undirected_neighbors(n, src, dst):
    # Structural OR of both directions. Unique coordinates make the BFS
    # neighbor order deterministic and keep parallel edges from weighting RCMK.
    row = np.r_[src, dst]
    col = np.r_[dst, src]
    graph = coo_matrix((np.ones(len(row), dtype=np.int64), (row,col)),shape=(n,n)).tocsr()
    graph.data.fill(1)
    graph.sort_indices()
    return graph


def bfs_order(graph):
    """Weak components, ascending root ID and ascending unseen neighbor ID."""
    n = graph.shape[0]
    seen = np.zeros(n,dtype=bool)
    order = []
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        queue = deque([root])
        while queue:
            v = queue.popleft()
            order.append(v)
            for w in graph.indices[graph.indptr[v]:graph.indptr[v+1]]:
                if not seen[w]:
                    seen[w] = True
                    queue.append(int(w))
    return np.array(order,dtype=np.int64)


def permutation(n, src, dst, axis, policy, seed=None):
    """Return new-to-original node and edge maps."""
    if policy not in AXES[axis]:
        raise ValueError(f"Unknown {axis} ordering {policy}")
    nodes = np.arange(n,dtype=np.int64)
    edges = np.arange(len(src),dtype=np.int64)
    if policy=="random":
        if seed is None:
            raise ValueError("Random ordering requires a seed")
        rng = np.random.default_rng(seed)
        return (rng.permutation(nodes),edges) if axis=="node" else (nodes,rng.permutation(edges))
    if axis=="edge":
        if policy in ("src","dst"):
            edges = np.argsort(src if policy=="src" else dst,kind="stable")
    elif policy.startswith("degree"):
        total = np.bincount(src,minlength=n)+np.bincount(dst,minlength=n)
        nodes = np.argsort(total if policy=="degree_asc" else -total,kind="stable")
    elif policy in ("rcmk","bfs"):
        adjacency = undirected_neighbors(n,src,dst)
        nodes = (reverse_cuthill_mckee(adjacency,symmetric_mode=True).astype(np.int64)
                 if policy=="rcmk" else bfs_order(adjacency))
    return nodes,edges


def build(base, src, dst, axis, policy, seed):
    """Time policy calculation, remapping/rebuild, then native materialization."""
    begin = perf_counter_ns()
    nodes,edges = permutation(base.num_nodes(),src,dst,axis,policy,seed)
    planned = perf_counter_ns()
    inverse = np.empty_like(nodes)
    inverse[nodes] = np.arange(len(nodes))
    # DGL assigns EIDs in endpoint-array order. No DGL reorder_graph defaults
    # can regroup the node sweep. Rebuild the original with this same path.
    graph = dgl.graph((inverse[src[edges]],inverse[dst[edges]]),
                      num_nodes=len(nodes),idtype=base.idtype,device="cpu")
    node_ids,edge_ids = torch.from_numpy(nodes),torch.from_numpy(edges)
    for key,value in base.ndata.items():
        # Match node-feature allocation in the identity control and each
        # candidate. Every edge-order case shares the same unchanged input.
        graph.ndata[key] = value[node_ids] if axis=="node" else value
    for key,value in base.edata.items():
        graph.edata[key] = value[edge_ids] if axis=="edge" and policy!="original" else value
    before = graph.formats()
    rebuilt = perf_counter_ns()
    graph.create_formats_()
    complete = perf_counter_ns()
    return graph,nodes,edges,[planned-begin,rebuilt-planned,complete-rebuilt],before


def digest(*arrays):
    return [hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for a in arrays]


def sequence_stats(a):
    step = np.abs(np.diff(a))
    return dict(mean_step=float(step.mean()) if len(step) else 0.,
                repeat_fraction=float((step==0).mean()) if len(step) else 0.)


def sparse_diagnostics(graph, nodes, edges, src, dst):
    """Decode every exposed sparse representation, including original EIDs."""
    n,e = len(nodes),len(edges)
    raw_src,raw_dst = (a.long().numpy() for a in graph.edges(order="eid"))
    result = {}
    for fmt in FORMATS:
        tensors = tuple(a.long().numpy() for a in graph.adj_tensors(fmt))
        if fmt=="coo":
            u,v = tensors
            stored_ids = np.arange(e)
            index_arrays = tensors
        else:
            ptr,minor,stored_ids = tensors
            if len(stored_ids)==0:
                stored_ids = np.arange(e)
            major = np.repeat(np.arange(n),np.diff(ptr))
            u,v = (major,minor) if fmt=="csr" else (minor,major)
            index_arrays = (ptr,minor)
        np.testing.assert_array_equal(np.sort(stored_ids),np.arange(e))
        np.testing.assert_array_equal(u,raw_src[stored_ids])
        np.testing.assert_array_equal(v,raw_dst[stored_ids])
        logical = edges[stored_ids]
        np.testing.assert_array_equal(nodes[u],src[logical])
        np.testing.assert_array_equal(nodes[v],dst[logical])
        result[fmt] = dict(index_sha256=digest(*index_arrays),
                           stored_eid_sha256=digest(stored_ids)[0],
                           original_eid_traversal_sha256=digest(logical)[0],
                           src=sequence_stats(u),dst=sequence_stats(v),
                           original_eid=sequence_stats(logical),
                           exposed_array_lengths=[len(a) for a in tensors],
                           exposed_eid_array=fmt!="coo" and bool(len(tensors[-1])))
    return result


def validate_graph(base, graph, nodes, edges, src, dst, axis):
    n,e = len(nodes),len(edges)
    np.testing.assert_array_equal(np.sort(nodes),np.arange(n))
    np.testing.assert_array_equal(np.sort(edges),np.arange(e))
    if axis=="node":
        np.testing.assert_array_equal(edges,np.arange(e))
    else:
        np.testing.assert_array_equal(nodes,np.arange(n))
    assert graph.num_nodes()==n and graph.num_edges()==e
    for original,changed,ids in ((base.ndata,graph.ndata,nodes),(base.edata,graph.edata,edges)):
        assert set(original)==set(changed)
        for key in original:
            assert torch.equal(original[key][torch.from_numpy(ids)],changed[key])
    features, measured = base.ndata["feat"], graph.ndata["feat"]
    assert measured.stride()==features.stride() and measured.storage_offset()==features.storage_offset()
    assert (measured.data_ptr()!=features.data_ptr())==(axis=="node")
    assert set(graph.formats()["created"])==set(FORMATS) and not graph.formats()["not created"]
    return sparse_diagnostics(graph,nodes,edges,src,dst)


def graph_diagnostics(graph):
    u,v = (a.long().numpy() for a in graph.edges(order="eid"))
    n = graph.num_nodes()
    span = np.abs(u-v)
    counts = (np.bincount(v,minlength=n),np.bincount(u,minlength=n))
    def distribution(values):
        return dict(min=int(values.min()),max=int(values.max()),mean=float(values.mean()),
                    median=float(np.median(values)),p95=float(np.quantile(values,.95)))
    return dict(bandwidth=int(span.max(initial=0)),mean_edge_span=float(span.mean()) if len(span) else 0.,
                indegree=distribution(counts[0]),outdegree=distribution(counts[1]),
                total_degree=distribution(counts[0]+counts[1]),
                self_loops=int((u==v).sum()),parallel_edges=len(u)-len(np.unique(u*n+v)))
