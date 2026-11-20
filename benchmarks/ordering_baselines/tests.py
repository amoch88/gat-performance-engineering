"""Ordering policy, graph identity, and stock profile regression tests."""
import unittest

from benchmark import validate_output, torch, dgl
from acquire import configure, model_for
from policies import AXES, build, case_specs, permutation, validate_graph, np
from stage_profile import STAGES, stock_profile


class OrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(1)

    def test_total_degree_ties_and_weak_component_bfs(self):
        src,dst = np.array([3,0,3,2,5,5]),np.array([1,3,1,2,4,5])
        expected = {"bfs":[0,3,1,2,4,5,6,7],
                    "degree_asc":[6,7,0,4,1,2,3,5],
                    "degree_desc":[3,5,1,2,0,4,6,7]}
        for policy,order in expected.items():
            with self.subTest(policy=policy):
                nodes,edges = permutation(8,src,dst,"node",policy)
                np.testing.assert_array_equal(nodes,order)
                np.testing.assert_array_equal(edges,np.arange(len(src)))
        rcmk,_ = permutation(8,src,dst,"node","rcmk")
        np.testing.assert_array_equal(np.sort(rcmk),np.arange(8))
        for policy,order in (("src",[1,3,0,2,4,5]),("dst",[0,2,3,1,4,5])):
            nodes,edges = permutation(8,src,dst,"edge",policy)
            np.testing.assert_array_equal(nodes,np.arange(8))
            np.testing.assert_array_equal(edges,order)
        with self.assertRaises(ValueError):
            permutation(8,src,dst,"edge","random")

    def test_graph_data_parallel_edges_and_stock_attention(self):
        n = 8
        src = np.r_[np.arange(n),3,0,3,2,5,5]
        dst = np.r_[np.arange(n),1,3,1,2,4,5]
        graph = dgl.graph((src,dst),num_nodes=n)
        x = torch.arange(n*7,dtype=torch.float32).reshape(n,7)/19
        graph.ndata["feat"] = x
        graph.ndata["tokens"] = torch.arange(n)
        graph.edata["tokens"] = torch.arange(len(src))
        graph.create_formats_()
        layer = model_for(7)
        profile,regions = stock_profile(layer)
        with torch.inference_mode():
            expected,attention = layer(graph,x,get_attention=True)
            for axis in AXES:
                for name,policy,seed in case_specs("cora",axis):
                    with self.subTest(axis=axis,policy=name):
                        ordered,nodes,edges,elapsed,_ = build(graph,src,dst,axis,policy,seed)
                        physical = validate_graph(graph,ordered,nodes,edges,src,dst,axis)
                        result = validate_output(ordered,ordered.ndata["feat"],layer,profile,nodes,edges,expected,attention)
                        self.assertEqual(tuple(regions),STAGES)
                        self.assertTrue(result["bitwise_profile_match"])
                        self.assertTrue(all(v>0 for v in elapsed))
                        self.assertEqual(set(physical),{"coo","csr","csc"})
                        same_feature_storage = ordered.ndata["feat"].data_ptr()==x.data_ptr()
                        self.assertEqual(same_feature_storage,axis=="edge")
                        self.assertTrue(torch.equal(ordered.edata["tokens"],torch.from_numpy(edges)))
                        # Replay seeded policies and check exact map stability.
                        repeat = permutation(n,src,dst,axis,policy,seed)
                        np.testing.assert_array_equal(nodes,repeat[0])
                        np.testing.assert_array_equal(edges,repeat[1])
        self.assertIs(layer.forward.__func__,dgl.nn.pytorch.GATConv.forward)


if __name__=="__main__":
    unittest.main()
