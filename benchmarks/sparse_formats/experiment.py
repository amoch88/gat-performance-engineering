"""Sparse-only substitutions into the unchanged Milestone 5 fixed-stage pipeline."""
import ctypes as C
import hashlib
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tuning_axes"))
from common import H, F, RTOL, ATOL, torch
from orchestration import Pipeline
from native import FLAGS
from storage import KINDS
import numpy as np

OPERATIONS = ("score", "softmax", "aggregation")


def build_library():
    compiler = os.environ.get("CXX", "g++")
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    source = HERE / "kernels.cpp"
    digest = hashlib.sha256(source.read_bytes() + repr((compiler, version, FLAGS)).encode()).hexdigest()
    cache = Path("results/sparse_formats_build")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"lib-{digest[:20]}.so"
    if not target.exists():
        temp = target.with_suffix(".tmp")
        subprocess.run([compiler, *FLAGS, str(source), "-o", str(temp)], check=True)
        temp.replace(target)
    return target, {"compiler": version, "flags": FLAGS,
                    "binary_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


class Kernel:
    def __init__(self, library, storage, operation):
        self.lib = library
        self.fn = library.sparse_run
        self.fn.argtypes = [C.c_int] * 6 + [C.c_void_p] * 8 + [C.c_float] + [C.c_void_p] * 3
        self.fn.restype = None
        self.op = operation
        self.storage = storage  # Keeps every pointer's allocation alive.
        self.args = (KINDS.index(storage.name), OPERATIONS.index(operation),
                     storage.n, storage.e, storage.block, storage.tiles,
                     *(a.ctypes.data for a in storage.arrays))

    def __call__(self, x, y=None, slope=0.2):
        n, e = self.storage.n, self.storage.e
        shape = (n, H, F) if self.op == "aggregation" else (e, H, 1)
        out = torch.empty(shape, dtype=torch.float32)
        maxima = sums = None
        if self.op == "softmax":
            maxima = torch.empty((n, H), dtype=torch.float32)
            sums = torch.empty_like(maxima)
        self.fn(*self.args, x.data_ptr(), y.data_ptr() if y is not None else None,
                slope, out.data_ptr(), maxima.data_ptr() if maxima is not None else None,
                sums.data_ptr() if sums is not None else None)
        return out


class Variant:
    def __init__(self, fixed: Pipeline, kernel: Kernel):
        self.fixed, self.kernel = fixed, kernel

    def forward(self, x, layer, details=False):
        p, k = self.fixed, self.kernel
        p.guard()
        z = p.project(x, layer)
        l, r = p.node_terms(z, layer)
        s = (k(l, r, layer.leaky_relu.negative_slope) if k.op == "score"
             else p.activate(p.add(l, r), layer))
        a = k(s) if k.op == "softmax" else p.normalize(s)
        agg = k(z, a) if k.op == "aggregation" else p.reduce(z, a)
        out = p.bias(agg, layer)
        if details:
            return out, dict(projected=z, source_contribution=l, destination_contribution=r,
                             logits=s, attention=a, aggregated=agg, output=out)
        return out

    def isolated(self, ref, layer):
        if self.kernel.op == "score":
            return self.kernel(ref["source_contribution"], ref["destination_contribution"],
                               layer.leaky_relu.negative_slope)
        if self.kernel.op == "softmax":
            return self.kernel(ref["logits"])
        return self.kernel(ref["projected"], ref["attention"])

    def validate(self, x, layer, ref, dst):
        target = dict(score="logits", softmax="attention", aggregation="aggregated")[self.kernel.op]
        isolated = self.isolated(ref, layer)
        torch.testing.assert_close(isolated, ref[target], rtol=RTOL, atol=ATOL)
        output, stages = self.forward(x, layer, details=True)
        errors = {}
        for name, value in stages.items():
            torch.testing.assert_close(value, ref[name], rtol=RTOL, atol=ATOL)
            errors[name] = float((value - ref[name]).abs().max())
        sums = np.zeros((len(x), H), dtype=np.float64)
        np.add.at(sums, dst, stages["attention"].numpy().reshape(-1, H))
        active = np.bincount(dst, minlength=len(x)) > 0
        np.testing.assert_allclose(sums[active], 1, atol=ATOL, rtol=0)
        torch.testing.assert_close(self.forward(x, layer), output, rtol=0, atol=0)
        return {"stage_max_abs_error": errors,
                "isolated_max_abs_error": float((isolated - ref[target]).abs().max()),
                "destination_normalization_max_abs_error": float(np.max(np.abs(sums[active]-1), initial=0)),
                "bitwise_repeatable": True}
