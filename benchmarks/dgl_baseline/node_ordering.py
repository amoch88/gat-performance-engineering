"""Official node permutations, with a fixed source-grouped edge policy."""
import torch
from common import dgl, run_experiment


def cases(graph, features, defaults):
    # EXPERIMENTAL CHANGE: baseline has original node IDs; compare official RCM
    # and a deterministic ascending total-degree order (ties: original node ID).
    # Use official reorder_graph, with src-grouped edge policy in EVERY case.
    # This investigates node/neighbor locality in LINEAR PROJECTION, ATTENTION,
    # EDGE SOFTMAX, AGGREGATION and OUTPUT, not modified kernel implementations.
    # DGL permutes features along with nodes; NID maps new rows to original IDs.
    # The shared verifier restores that order before checking numerical output.
    for policy in ['original', 'rcmk', 'degree']:
        kwargs = {}
        algo = None
        if policy == 'rcmk':
            algo = 'rcmk'
        elif policy == 'degree':
            algo = 'custom'
            degree = graph.in_degrees() + graph.out_degrees()
            kwargs['permute_config'] = {'nodes_perm': torch.argsort(degree, stable=True)}
        changed = dgl.reorder_graph(graph, node_permute_algo=algo,
                                    edge_permute_algo='src', **kwargs)
        ids = changed.ndata[dgl.NID]
        assert torch.equal(changed.ndata['feat'], features[ids])
        changed.create_formats_()
        yield policy, {}, changed, changed.ndata['feat'], ids


if __name__ == '__main__':
    run_experiment('node_ordering', cases)
