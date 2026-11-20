"""Sweep output features per head, holding four heads fixed."""
from common import run_experiment


def cases(graph, features, defaults):
    # EXPERIMENTAL CHANGE: official out_feats argument, baseline 8 per head ->
    # 8/16/32/64, holding heads=4 fixed. Investigate larger projected vectors.
    # LINEAR PROJECTION, ATTENTION SCORE COMPUTATION, AGGREGATION and OUTPUT
    # widths change; EDGE SOFTMAX still has four scalars per edge. No internal
    # kernel is modified. Different widths are different workloads, not speedups.
    for width in [8, 16, 32, 64]:
        yield str(width), {'out_features': width}, graph, features, None


if __name__ == '__main__':
    run_experiment('output_features', cases)
