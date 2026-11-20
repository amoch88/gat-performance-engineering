"""Insert six stage timers in copies of the two executable forward bodies."""
import ast
import inspect
from textwrap import dedent
from time import perf_counter_ns
from types import MethodType

STAGES = ("projection","terms","score","softmax","aggregation","output")


def instrument(function, stock):
    """Reuse original statements and verify the stage boundaries before insertion."""
    tree = ast.parse(dedent(inspect.getsource(function)))
    definition, = tree.body
    if stock:
        scope = definition.body[-1]
        assert isinstance(scope,ast.With) and ast.unparse(scope.items[0].context_expr)=="graph.local_scope()"
        body = scope.body
        starts = [
            "if isinstance(feat, tuple):", "el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)",
            "graph.srcdata.update({'ft': feat_src, 'el': el})",
            "graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))",
            "graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))",
            "if self.res_fc is not None:", "if get_attention:",
        ]
    else:
        body = definition.body
        starts = [
            "z = p.project(layer.feat_drop(x), layer)", "left, right = p.node_terms(z, layer)",
            "scores = self.score(left, right, layer.leaky_relu.negative_slope)",
            "alpha = self.softmax(scores)", "aggregated = self.aggregation(z, alpha)",
            "result = p.bias(aggregated, layer)", "if details:",
        ]
    statements = [ast.unparse(statement).splitlines()[0] for statement in body]
    positions = [statements.index(label) for label in starts]
    assert positions == sorted(positions) and len(set(positions))==7
    if not stock:
        assert positions == list(range(2,9))
    for name,start,end in reversed(list(zip(STAGES,positions,positions[1:]))):
        body[end:end] = ast.parse(f"_m9_record[{name!r}] = _m9_clock() - _m9_start").body
        body[start:start] = ast.parse("_m9_start = _m9_clock()").body
    namespace = dict(function.__globals__)
    record = {}
    namespace.update(_m9_clock=perf_counter_ns,_m9_record=record)
    exec(compile(ast.fix_missing_locations(tree),"<milestone9-profile>","exec"),namespace)
    return namespace[definition.name],record


def stock_profile(layer):
    # This separate object shares the exact parameters. The official reference's
    # bound method remains untouched and is used for all plain stock measurements.
    import copy
    profiled = copy.copy(layer)
    func,record = instrument(type(layer).forward,stock=True)
    profiled.forward = MethodType(func,profiled)
    return profiled,record
