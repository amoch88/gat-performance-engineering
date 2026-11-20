"""Check that the timing copies execute the same GAT on a multigraph."""
import unittest
from runner import (dgl, np, torch, setup_cases, reference_stages, CONFIGS, MATCHED,
                    SUPPORTED, FALLBACK, OPS)
from stage_profile import STAGES
from acquire import configure, model_for


class TimingValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(1)

    def test_multigraph_with_partial_tiles(self):
        n = 67
        u = np.r_[np.arange(n), [0,0,0,32,64,66,66,2,4]]
        v = np.r_[np.arange(n), [66,66,66,66,0,66,66,2,2]]
        rng = np.random.default_rng(9)
        order = rng.permutation(len(u))
        u,v = u[order],v[order]
        graph = dgl.graph((u,v),num_nodes=n)
        graph.ndata["feat"] = x = torch.from_numpy(rng.normal(size=(n,7)).astype(np.float32))
        layer = model_for(x.shape[1])
        original = layer.forward.__func__
        with torch.inference_mode():
            ref = reference_stages(graph,x,layer)
            forwards,profiles,calls,context,storage,_,routes,checks,_ = setup_cases(graph,x,layer,u,v,ref)
            self.assertEqual(list(forwards),CONFIGS)
            self.assertEqual(set(checks),set(CONFIGS))
            self.assertIs(layer.forward.__func__,original)
            for native,key in zip(("coo","csr","csc"),MATCHED,strict=True):
                for op in OPS:
                    expected = native if native in SUPPORTED[op] else FALLBACK[op]
                    self.assertEqual(routes[native][op],expected)
                    self.assertEqual(routes[key][op],"custom_"+expected)
                    self.assertEqual(f"isolated/{op}/{native}" in calls,native in SUPPORTED[op])
            for f,(profile,record) in profiles.items():
                record.clear()
                torch.testing.assert_close(profile(),forwards[f](),atol=0,rtol=0)
                self.assertEqual(tuple(record),STAGES)
                self.assertTrue(all(value>0 for value in record.values()))
            self.assertTrue(all(run()>0 for run in context.values()))
            self.assertEqual(storage["custom_coo"]["bytes"],len(u)*2*4)


if __name__=="__main__":
    unittest.main()
