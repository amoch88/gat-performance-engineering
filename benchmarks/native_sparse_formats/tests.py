"""Milestone 7 edge identity, cross-tile softmax and native format tests."""
import ctypes
import unittest

from acquire import configure, model_for, dgl, torch
from experiment import Kernel, Variant, build_library
from common import reference_stages, validate_view
from orchestration import Pipeline
from sparse import BLOCKED, SUPPORTED, Native, build_blocked, view, np
from storage import build


class TestNativeSparse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(1)
        path,_ = build_library()
        cls.library = ctypes.CDLL(str(path.resolve()))

    def test_all_variants_against_stock(self):
        n = 67
        rng = np.random.default_rng(71)
        src = np.r_[np.arange(n), rng.integers(n,size=200), [0,0,1,33,66]]
        dst = np.r_[np.arange(n), rng.integers(n,size=200), [66,66,66,66,0]]
        order = rng.permutation(len(src))
        src,dst = src[order],dst[order]
        g = dgl.graph((src,dst),num_nodes=n)
        x = torch.from_numpy(rng.normal(size=(n,19)).astype(np.float32))
        g.ndata["feat"] = x
        layer = model_for(x.shape[1])
        fixed = Pipeline(g)
        views = {f:view(g,f) for f in ("coo","csr","csc")}
        for v in views.values():
            validate_view(v,g)
        with torch.inference_mode():
            ref = reference_stages(g,x,layer)
            for op,allowed in SUPPORTED.items():
                for f,v in views.items():
                    if f in allowed:
                        Variant(fixed,Native(v,op,f)).validate(x,layer,ref,dst)
                    else:
                        with self.assertRaises(ValueError):
                            Native(v,op,f)
            for name,b in BLOCKED.values():
                with self.subTest(format=name,block=b):
                    storage = build_blocked(n,src,dst,name,b)
                    storage.validate(src,dst)
                    self.assertEqual(storage.tiles,len(set(zip(src//b,dst//b))))
                    if b >= 16:
                        prior = build(n,src,dst,name,b)
                        for old,new in zip(prior.arrays,storage.arrays,strict=True):
                            np.testing.assert_array_equal(old,new)
                    for op in SUPPORTED:
                        Variant(fixed,Kernel(self.library,storage,op)).validate(x,layer,ref,dst)

    def test_empty_and_cross_tile_extreme_logits(self):
        empty = np.empty(0,dtype=np.int64)
        src,dst = np.array([0,2,4,8,16,32,64]),np.full(7,66)
        logits = torch.tensor([1000.,999.,997.,-1000.,998.,-999.,990.]).view(-1,1,1).repeat(1,4,1)
        for name,b in BLOCKED.values():
            storage = build_blocked(67,src,dst,name,b)
            storage.validate(src,dst)
            result = Kernel(self.library,storage,"softmax")(logits)
            torch.testing.assert_close(result,torch.softmax(logits,dim=0))
            storage = build_blocked(67,empty,empty,name,b)
            storage.validate(empty,empty)
            result = Kernel(self.library,storage,"aggregation")(torch.ones(67,4,8),torch.empty(0,4,1))
            self.assertEqual(result.count_nonzero(),0)


if __name__ == "__main__":
    unittest.main()
