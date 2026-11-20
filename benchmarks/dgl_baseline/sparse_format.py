"""Restrict graph storage using official DGL format APIs."""
from common import run_experiment


def cases(graph, features, defaults):
    # EXPERIMENTAL CHANGE: baseline allows/materializes COO+CSR+CSC. Restrict to
    # one format at a time to investigate ATTENTION edge operations, EDGE SOFTMAX
    # and AGGREGATION storage/dispatch. LINEAR PROJECTION and OUTPUT dimensions
    # are unchanged. These official graph APIs do NOT modify a compiled kernel.
    # Some operators may reject a restriction; record unsupported cases honestly.
    for fmt in ['coo', 'csr', 'csc']:
        changed = graph.formats(fmt)
        changed.create_formats_()  # Materialize outside timing, before checks.
        state = changed.formats()
        assert state['created'] == [fmt] and state['not created'] == []
        yield fmt, {}, changed, features, None


if __name__ == '__main__':
    run_experiment('sparse_format', cases)
