"""C: physical layouts compared within ONE compiled consumer family."""
from orchestration import Pipeline

def projected_cases(g,native,threads):
    # EXPERIMENTAL CHANGE: NHF vs physically contiguous HNF. Same node/edge
    # arithmetic, CSC reductions, static scheduling, edge EH layout and threads.
    # HNF packing is per-forward and INCLUDED. Consumers honor actual offsets.
    return {name:Pipeline(g,native=native,zlayout=layout,threads=threads) for name,layout in [('nhf',0),('hnf',1)]}

def attention_cases(g,native,threads):
    # EXPERIMENTAL CHANGE: scores/attention physically EH or HE. All producers
    # and consumers use eoff(); there is no repack to EH before normalization or
    # aggregation. Node layout, loops, schedules and thread count are unchanged.
    return {name:Pipeline(g,native=native,elayout=layout,threads=threads) for name,layout in [('eh',0),('he',1)]}
