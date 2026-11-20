"""A: complete official GATConv, only graph index dtype changes."""
from time import perf_counter_ns
from common import *
from orchestration import Official

def cases(g,native,threads):
    bounds=check_bounds(g);result={}
    for name in ['int64','int32']:
        if name=='int32' and not bounds['int32_legal']:
            result[name]={'unsupported':'Graph/index/intermediate bounds exceed int32 range','bounds':bounds};continue
        # EXPERIMENTAL CHANGE: official graph cast only. Native features, weights,
        # ordering, all-format policy and complete official layer are unchanged.
        start=perf_counter_ns();view=g.long() if name=='int64' else g.int()
        cast_ms=(perf_counter_ns()-start)/1e6;start=perf_counter_ns();view.create_formats_()
        materialize_ms=(perf_counter_ns()-start)/1e6
        validate_view(view,g);case=Official(view)
        case.info.update(bounds=bounds,cast_ms=cast_ms,materialization_ms=materialize_ms)
        result[name]=case
    return result
