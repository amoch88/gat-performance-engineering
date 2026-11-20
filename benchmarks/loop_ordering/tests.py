"""Compiled routes, numerical invariants and one-factor timing analysis."""
import unittest

from report import comparisons, decision
from pipeline import Forward, State, baseline, bitwise, dgl, np, prepared, torch
from common import configure, model_for
from compiled import compile_library
from design import DEFAULT, STAGES, SWEEPS, SYMBOLS, design


class LoopOrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(1)
        cls.binary,cls.build = compile_library()

    def fixed_graph(self):
        n = 48
        # Most edges land at one destination, two have the same endpoints,
        # and some nodes (including the last two) are initially isolated.
        src = np.r_[np.arange(32),3,3,43,41,44]
        dst = np.r_[np.full(32,35),35,35,40,40,44]
        base = dgl.graph((src,dst),num_nodes=n)
        self.assertEqual(int(base.in_degrees()[-1]),0)
        g = dgl.add_self_loop(dgl.remove_self_loop(base))
        g.create_formats_()
        self.assertEqual(int((g.edges()[0]==g.edges()[1]).sum()),n)
        self.assertGreater(int(g.in_degrees(35)),30)
        x = torch.arange(n*7,dtype=torch.float32).reshape(n,7)/53-3
        return State(g,self.binary),x,model_for(x.shape[1])

    def test_all_compiled_orders_against_stock(self):
        state,x,layer = self.fixed_graph()
        with torch.inference_mode():
            ref,shared = baseline(state,x,layer)
            for stage,orders in SWEEPS.items():
                for code in orders:
                    with self.subTest(stage=stage,code=code):
                        plan = design(stage,code)
                        self.assertEqual(sum(plan[s]!=DEFAULT[s] for s in plan),int(code!=DEFAULT[stage]))
                        data = prepared(stage,code,state,x,layer,ref,shared)
                        self.assertEqual(data["routes"],{s:SYMBOLS[c] for s,c in plan.items()})
                        self.assertEqual(tuple(data["profile"]),STAGES)
                        self.assertTrue(data["check"]["custom_bitwise"])
            self.assertEqual({c:v["address"]-self.build["symbols"]["A1"]["address"]
                              for c,v in self.build["symbols"].items()},state.lib.offsets())
            for orders in SWEEPS.values():
                self.assertEqual(len({self.build["symbols"][code]["code_sha256"] for code in orders}),len(orders))

    def test_softmax_stable_destination_global_and_empty_rows(self):
        src = np.array([0,0,1,2,3,0,1])
        dst = np.array([2,2,2,2,2,4,4])
        g = dgl.graph((src,dst),num_nodes=6)
        g.create_formats_()
        state = State(g,self.binary)
        values = torch.tensor([[1000.,-1000.,1.,0.],[1000.,-1010.,3.,0.],
                               [999.,-1001.,-2000.,0.],[-2000.,-999.,5.,1.],
                               [997.,-2000.,6.,-1.],[1100.,-1400.,-5.,0.],
                               [1099.,-1401.,-6.,0.]],dtype=torch.float32)
        ref = torch.empty_like(values)
        for v in (2,4):
            ref[dst==v] = torch.softmax(values[dst==v].double(),dim=0).float()
        first = None
        with torch.inference_mode():
            for code in SWEEPS["softmax"]:
                forward = Forward(state,design("softmax",code))
                result = forward.softmax(values)
                torch.testing.assert_close(result,ref,rtol=1e-6,atol=1e-7)
                if first is not None:
                    bitwise(result,first)
                first = result
            z = torch.ones((6,4,8),dtype=torch.float32)
            for code in SWEEPS["aggregation"]:
                y = Forward(state,design("aggregation",code)).aggregation(z,first)
                torch.testing.assert_close(y[[2,4]],torch.ones((2,4,8),dtype=torch.float32))
                self.assertTrue(torch.equal(y[[0,1,3,5]],torch.zeros((4,4,8),dtype=torch.float32)))
            with self.assertRaises(dgl.DGLError):
                Forward(state,DEFAULT)(torch.ones((6,7)),model_for(7))

    def test_empty_coo_and_csc(self):
        g = dgl.graph(([],[]),num_nodes=3,idtype=torch.int64)
        g.create_formats_()
        state = State(g,self.binary)
        for code in SWEEPS["softmax"]:
            a = Forward(state,design("softmax",code)).softmax(torch.empty((0,4)))
            self.assertEqual(a.shape,(0,4))
        for code in SWEEPS["aggregation"]:
            y = Forward(state,design("aggregation",code)).aggregation(torch.ones((3,4,8)),a)
            self.assertTrue(torch.equal(y,torch.zeros_like(y)))

    def test_planned_families_and_threshold(self):
        data = {}
        for stage,orders in SWEEPS.items():
            data[stage] = dict(process_medians={
                c:{key:[1.+i*.2 for i in range(8)] if j==0 else
                   [(1.+i*.2)/1.1 for i in range(8)] for key in ("forward","isolated","in_forward")}
                for j,c in enumerate(orders)})
        rows = comparisons({"cora":data,"citeseer":data},{s:[] for s in SWEEPS})
        self.assertEqual(len(rows),42)
        for row in rows:
            self.assertAlmostEqual(row["speedup"],1.1)
            self.assertGreaterEqual(row["holm_p"],row["p"])
            self.assertEqual(decision(row),"Qualifies")
        self.assertEqual(decision(dict(alias=False,holm_p=.03,speedup=1.02,ci_low=1.01)),
                         "Faster, below practical bound")
        self.assertEqual(decision(dict(alias=True,holm_p=.03,speedup=1.02,ci_low=1.02)),
                         "Machine-code alias")


if __name__=="__main__":
    unittest.main()
