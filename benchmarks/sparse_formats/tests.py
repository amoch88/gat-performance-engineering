"""Edge identity, tile traversal, and independent numerical oracles."""
import ctypes
import unittest

from experiment import Kernel, OPERATIONS, Variant, build_library
from common import H, F, configure, dgl, model_for, reference_stages, torch
from orchestration import Pipeline
from storage import FORMATS, build, np


class TestFormats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(1)
        path, _ = build_library()
        cls.lib = ctypes.CDLL(str(path.resolve()))

    def test_duplicate_edges_shuffled_eids_partial_tiles(self):
        n = 67
        rng = np.random.default_rng(12)
        u = np.r_[np.arange(n), rng.integers(n, size=211), [0, 0, 65, 65]]
        v = np.r_[np.arange(n), rng.integers(n, size=211), [66, 66, 1, 1]]
        order = rng.permutation(len(u))
        u, v = u[order], v[order]
        g = dgl.graph((u, v), num_nodes=n)
        g.ndata["feat"] = x = torch.from_numpy(rng.normal(size=(n, 17)).astype(np.float32))
        layer = model_for(17)
        fixed = Pipeline(g)
        with torch.inference_mode():
            ref = reference_stages(g, x, layer)
            for name, block in FORMATS:
                with self.subTest(name=name, block=block):
                    s = build(n, u, v, name, block)
                    s.validate(u, v)
                    if block:
                        self.assertEqual(s.tiles, len(set(zip(u//block, v//block))))
                    for op in OPERATIONS:
                        Variant(fixed, Kernel(self.lib, s, op)).validate(x, layer, ref, v)

    def test_softmax_cross_tile_extreme_logits_and_isolates(self):
        u = np.array([0, 32, 64, 66, 0, 65, 0])
        v = np.array([66, 66, 66, 66, 1, 1, 1])
        q = torch.tensor([1000., 999., -2000., 998., -1000., -1001., -1002.]).view(-1,1,1).repeat(1,H,1)
        expected = torch.empty_like(q)
        for dst in np.unique(v):
            expected[v==dst] = torch.softmax(q[v==dst], dim=0)
        for name, block in FORMATS:
            s = build(67, u, v, name, block)
            s.validate(u, v)
            torch.testing.assert_close(Kernel(self.lib, s, "softmax")(q), expected)
            z = torch.ones(67,H,F)
            got = Kernel(self.lib, s, "aggregation")(z, expected)
            torch.testing.assert_close(got[v[0]], z[v[0]])
            self.assertEqual(got[3].count_nonzero(), 0)

    def test_empty_edges_and_bad_indices(self):
        u = np.array([], dtype=np.int64)
        for name, block in FORMATS:
            s = build(17, u, u, name, block)
            s.validate(u, u)
            self.assertEqual(Kernel(self.lib, s, "softmax")(torch.empty(0,H,1)).numel(), 0)
            self.assertEqual(Kernel(self.lib, s, "aggregation")(torch.zeros(17,H,F), torch.empty(0,H,1)).count_nonzero(), 0)
        with self.assertRaises(ValueError):
            build(17, np.array([17]), np.array([0]), "csr", 0)


if __name__ == "__main__":
    unittest.main()
