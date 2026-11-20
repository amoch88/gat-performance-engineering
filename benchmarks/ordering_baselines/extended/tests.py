"""Exact ordering, native METIS replay and stock GATConv correctness."""
import json
import unittest

from benchmark_extended import (build, dgl, metis_plan, saved_metis_nodes, stock_profile,
                                torch, validate_graph, validate_output)
from common import configure, model_for
from policies_ext import AXES, case_specs, morton_keys, nodes_from_assignment, np, permutation
from report_extended import decision, secondary_edge_contrasts


class ExtendedOrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(1)

    def test_grouping_and_lexicographic_order_with_parallel_edges(self):
        src = np.array([2, 0, 2, 0, 2, 0, 1, 0, 1])
        dst = np.array([2, 2, 0, 0, 0, 1, 1, 2, 0])
        expected = dict(src=[1,3,5,7,6,8,0,2,4], dst=[2,3,4,8,5,6,0,1,7],
                        src_dst=[3,5,1,7,8,6,2,4,0], dst_src=[3,8,2,4,5,6,1,7,0],
                        morton=[3,8,5,6,2,4,1,7,0])
        for policy, order in expected.items():
            with self.subTest(policy=policy):
                nodes, edges = permutation(3, src, dst, "edge", policy)
                np.testing.assert_array_equal(nodes, [0, 1, 2])
                np.testing.assert_array_equal(edges, order)

    def test_morton_width_and_convention(self):
        pairs = [(0,0), (1,0), (0,1), (0,2), (2,0), (2**31,0), (0,2**31), (2**32-1,2**32-1)]
        src, dst = np.array(pairs, dtype=np.uint64).T
        np.testing.assert_array_equal(morton_keys(src, dst),
            np.array([0,1,2,8,4,2**62,2**63,2**64-1], dtype=np.uint64))
        x = np.tile(np.arange(32760, 32776), 16)
        y = np.repeat(np.arange(65530, 65546), 16)
        self.assertEqual(len(np.unique(morton_keys(x, y))), 256)
        for invalid in ([-1], [2**32], [2.0]):
            with self.assertRaises(ValueError):
                morton_keys(invalid, [0])

    def test_partition_ties_and_invalid_assignment(self):
        np.testing.assert_array_equal(nodes_from_assignment([1,0,1,2,0,2], 6, 3), [1,4,0,2,3,5])
        for invalid in ([0,1], [0,1,3], [0,-1,2], [0.0,1.0,2.0]):
            with self.assertRaises(ValueError):
                nodes_from_assignment(invalid, 3, 3)

    def test_destination_comparison_and_practical_boundary(self):
        dst = [7., 8., 9., 10.]
        values = {"original": [1.]*4, "dst": dst, "src_dst": [v/1.2 for v in dst],
                  "dst_src": dst, "morton": [v/.9 for v in dst]}
        inputs = {"cora": {"edge": {"round_medians_ms": {
            p: {"forward": times} for p, times in values.items()}}}}
        contrasts = secondary_edge_contrasts(inputs)
        self.assertTrue(all(r["control"]=="dst" and r["holm_p"]>=r["p"] for r in contrasts))
        for row, expected in zip(contrasts, (1.2,1.,.9), strict=True):
            self.assertAlmostEqual(row["speedup"],expected)
        self.assertEqual([decision(r,1.01) for r in contrasts], ["Qualifies","Inconclusive","Loss"])
        self.assertEqual(decision(dict(holm_p=.03,speedup=1.02,ci_low=1.01),1.01),
                         "Gain below practical criterion")

    def test_native_metis_and_all_orderings_against_stock(self):
        n = 96
        u = np.arange(n)
        src = np.r_[u,u,u,7,7]
        dst = np.r_[u,(u+1)%n,(u+5)%n,14,14]
        base = dgl.graph((src,dst), num_nodes=n)
        x = torch.arange(n*7, dtype=torch.float32).reshape(n,7)/19
        base.ndata["feat"] = x
        base.ndata["token"] = torch.arange(n)
        base.edata["token"] = torch.arange(len(src))
        base.create_formats_()
        saved = json.loads(json.dumps(metis_plan(base)))
        nodes = saved_metis_nodes(saved, n)
        np.testing.assert_array_equal(np.sort(nodes), u)
        self.assertGreater(len(np.unique(saved["assignment"])), 1)
        model = model_for(7)
        profile,_ = stock_profile(model)
        with torch.inference_mode():
            output,alpha = model(base, x, get_attention=True)
            for axis in AXES:
                self.assertEqual(len(AXES[axis]), 7)
                for label, policy, seed in case_specs("cora", axis):
                    with self.subTest(axis=axis, policy=label):
                        g,reordered,edges,elapsed,_ = build(base, src, dst, axis, policy, seed, nodes)
                        validate_graph(base, g, reordered, edges, src, dst, axis)
                        self.assertEqual(g.ndata["feat"].data_ptr()==x.data_ptr(), axis=="edge")
                        self.assertTrue(torch.equal(g.edata["token"], torch.from_numpy(edges)))
                        result = validate_output(g,g.ndata["feat"],model,profile,reordered,edges,output,alpha)
                        self.assertTrue(result["bitwise_profile_match"])
                        self.assertTrue(all(t > 0 for t in elapsed))
                        if policy=="metis":
                            np.testing.assert_array_equal(reordered, saved["new_to_original"])


if __name__=="__main__":
    unittest.main()
