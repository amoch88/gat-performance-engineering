"""Sweep heads with eight output features per head, not fixed total width."""
from common import run_experiment


def cases(graph, features, defaults):
    # EXPERIMENTAL CHANGE: official num_heads argument, baseline 4 -> 1/2/4/8.
    # Keep out_feats=8 per head. Total output width is heads*8, so this changes
    # work in ALL FIVE stages (including edge softmax's number of scalars).
    # Investigate head scaling; no internal DGL kernel is modified. Do not call
    # changed output dimensions equivalent-computation speedups.
    for heads in [1, 2, 4, 8]:
        yield str(heads), {'heads': heads}, graph, features, None


if __name__ == '__main__':
    run_experiment('attention_heads', cases)
