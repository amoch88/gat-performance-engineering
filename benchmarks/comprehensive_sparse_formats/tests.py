"""Regression tests for the padded traversal and full-format substitutions."""
import unittest

from benchmark import Forward, OPS, SIZES, dgl, np, reference_stages, torch, validate
from benchmark import FALLBACK, Native, Pipeline, SUPPORTED, view
from padded import PaddedKernel, StorageLimit, build_padded, library
from acquire import configure, model_for


class PaddedFormats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(1)
        cls.lib,_ = library()

    def test_all_padded_sizes_against_stock(self):
        rng = np.random.default_rng(80)
        n = 67
        src = np.r_[np.arange(n),rng.integers(n,size=180),[0,0,0,1,32,63,64]]
        dst = np.r_[np.arange(n),rng.integers(n,size=180),[66,66,66,66,66,66,66]]
        order = rng.permutation(len(src))
        src,dst = src[order],dst[order]
        graph = dgl.graph((src,dst),num_nodes=n)
        graph.ndata["feat"] = x = torch.from_numpy(rng.normal(size=(n,19)).astype(np.float32))
        layer = model_for(x.shape[1])
        fixed = Pipeline(graph)
        with torch.inference_mode():
            ref = reference_stages(graph,x,layer)
            for kind in ("dense","sell"):
                for size in SIZES:
                    with self.subTest(kind=kind,size=size):
                        store = build_padded(n,src,dst,kind,size,1<<30)
                        store.validate(src,dst)
                        if kind=="dense":
                            self.assertEqual(store.groups,len(set(zip(src//size,dst//size))))
                            self.assertEqual(len(store.arrays[4]),len(src)-len(set(zip(src,dst))))
                        else:
                            self.assertEqual(store.groups,(n+size-1)//size)
                            self.assertEqual(np.count_nonzero(store.arrays[2]>=0),len(src))
                        kernels = {op:PaddedKernel(self.lib,store,op) for op in OPS}
                        validate(Forward(fixed,kernels),x,layer,ref,dst)

    def test_extreme_logits_duplicates_and_empty_rows(self):
        src = np.array([0,0,31,32,64,66,66,3,4])
        dst = np.array([66,66,66,66,66,66,66,2,2])
        logits = torch.tensor([999.,1000.,990.,-1000.,-1002.,995.,998.,-1000.,-999.])
        logits = logits.view(-1,1,1).expand(-1,4,1).contiguous()
        expected = torch.empty_like(logits)
        for v in np.unique(dst):
            expected[dst==v] = torch.softmax(logits[dst==v],dim=0)
        for kind in ("dense","sell"):
            for size in SIZES:
                store = build_padded(67,src,dst,kind,size,1<<30)
                store.validate(src,dst)
                alpha = PaddedKernel(self.lib,store,"softmax")(logits)
                torch.testing.assert_close(alpha,expected,rtol=1e-6,atol=1e-7)
                out = PaddedKernel(self.lib,store,"aggregation")(torch.ones(67,4,8),alpha)
                torch.testing.assert_close(out[[2,66]],torch.ones(2,4,8))
                self.assertEqual(int(out[0].count_nonzero()),0)

    def test_guards_and_empty_graph(self):
        src,dst = np.array([63]),np.array([0])
        for kind in ("dense","sell"):
            for size in SIZES:
                with self.assertRaises(StorageLimit) as caught:
                    build_padded(67,src,dst,kind,size,8)
                self.assertGreater(caught.exception.required,caught.exception.limit)
                empty = np.empty(0,dtype=np.int64)
                store = build_padded(67,empty,empty,kind,size,1<<30)
                store.validate(empty,empty)
                alpha = PaddedKernel(self.lib,store,"softmax")(torch.empty(0,4,1))
                self.assertEqual(alpha.numel(),0)
                y = PaddedKernel(self.lib,store,"aggregation")(torch.ones(67,4,8),alpha)
                self.assertEqual(int(y.count_nonzero()),0)

    def test_native_hybrid_routes(self):
        src = np.array([0,1,2,2,2,1])
        dst = np.array([0,1,2,0,0,2])
        graph = dgl.graph((src,dst),num_nodes=3)
        graph.ndata["feat"] = x = torch.rand(3,7)
        layer = model_for(7)
        fixed = Pipeline(graph)
        views = {f:view(graph,f) for f in ("coo","csr","csc")}
        with torch.inference_mode():
            ref = reference_stages(graph,x,layer)
            for name in views:
                kernels = {}
                for op in OPS:
                    fmt = name if name in SUPPORTED[op] else FALLBACK[op]
                    kernels[op] = Native(views[fmt],op,fmt)
                validate(Forward(fixed,kernels),x,layer,ref,dst)


if __name__=="__main__":
    unittest.main()
