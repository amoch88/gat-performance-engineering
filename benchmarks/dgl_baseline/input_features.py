"""Sweep real dataset input widths; this changes the computational workload."""
from common import resize_features, run_experiment


def cases(graph, features, defaults):
    # EXPERIMENTAL CHANGE: baseline is the dataset's original width. Investigate
    # 16/32/64/128/original via prefix truncation (or zero-padding if wider).
    # This deterministic transform preserves real node association and occurs
    # before timing. Only the official constructor's in_feats changes; it is NOT
    # a kernel modification. Primarily affects LINEAR PROJECTION; projection
    # values can also change ATTENTION, EDGE SOFTMAX, AGGREGATION and OUTPUT.
    for width in dict.fromkeys([16, 32, 64, 128, features.shape[1]]):
        yield str(width), {}, graph, resize_features(features, width), None


if __name__ == '__main__':
    run_experiment('input_features', cases)
