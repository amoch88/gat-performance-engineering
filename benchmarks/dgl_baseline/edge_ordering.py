"""Reorder edges while retaining node IDs and real features."""
import torch
from common import dgl, run_experiment


def cases(graph, features, defaults):
    # EXPERIMENTAL CHANGE: original is the DGL loader's edge-ID order after
    # remove/add self-loops. Official reorder_graph groups edges by src or dst,
    # without node permutation. Investigate locality in ATTENTION edge stages,
    # EDGE SOFTMAX and AGGREGATION; projection/output shapes stay fixed.
    # This uses an official API, not a kernel edit. All formats stay enabled;
    # their traversal indices and EID maps may or may not actually differ.
    yield 'original', {}, graph, features, None
    for policy in ['src', 'dst']:
        changed = dgl.reorder_graph(graph, node_permute_algo=None, edge_permute_algo=policy)
        assert torch.equal(changed.ndata[dgl.NID], torch.arange(graph.num_nodes()))
        assert torch.equal(changed.ndata['feat'], features)
        changed.create_formats_()
        yield policy, {}, changed, changed.ndata['feat'], None


if __name__ == '__main__':
    run_experiment('edge_ordering', cases)
