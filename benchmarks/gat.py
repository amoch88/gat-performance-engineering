"""Stock DGL CPU GAT baseline; run from the repository root."""
import argparse
import json
import os
import platform
import random
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns

os.environ["DGLBACKEND"] = "pytorch"

import dgl
import numpy as np
import torch
from dgl.nn.pytorch import GATConv


@dataclass(frozen=True)
class Workload:
    nodes: int = 1000
    edges: int = 8000  # Random directed edges, before adding exactly one loop/node.
    features: int = 32
    out_features: int = 8  # Per head; output shape is (nodes, heads, out_features).
    heads: int = 4
    seed: int = 0


def prepare(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    dgl.seed(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    start = perf_counter_ns()
    src = torch.randint(config.nodes, (config.edges,), generator=generator)
    dst = torch.randint(config.nodes, (config.edges,), generator=generator)
    # Sample with replacement: parallel non-loop edges are intentional.
    graph = dgl.graph((src, dst), num_nodes=config.nodes, device="cpu")
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
    graph.create_formats_()  # Avoid lazy sparse-format construction during timing.
    construction_ms = (perf_counter_ns() - start) / 1e6
    features = torch.randn(config.nodes, config.features, generator=generator)
    layer = GATConv(config.features, config.out_features, config.heads,
                    feat_drop=0.0, attn_drop=0.0, allow_zero_in_degree=False)
    layer = layer.to(device="cpu", dtype=torch.float32).eval()
    return graph, features, layer, construction_ms


def smoke_test():
    """Known-answer one-node graph plus shape, repeatability and finiteness checks."""
    graph = dgl.graph(([0], [0]), num_nodes=1)
    layer = GATConv(2, 2, 1, feat_drop=0.0, attn_drop=0.0, bias=False).eval()
    with torch.no_grad():
        layer.fc.weight.copy_(torch.eye(2))
    with torch.inference_mode():
        actual = layer(graph, torch.tensor([[2.0, -3.0]]))
        torch.testing.assert_close(actual, torch.tensor([[[2.0, -3.0]]]))
        # Zero attention logits yield an exact mean over incoming neighbors.
        graph2 = dgl.graph(([0, 1, 0], [0, 0, 1]), num_nodes=2)
        with torch.no_grad():
            layer.attn_l.zero_()
            layer.attn_r.zero_()
        torch.testing.assert_close(
            layer(graph2, torch.tensor([[2.0, 4.0], [6.0, 8.0]])),
            torch.tensor([[[4.0, 6.0]], [[2.0, 4.0]]]))
        config = Workload(nodes=7, edges=12, features=3, heads=2, out_features=4)
        g, x, model, _ = prepare(config)
        first = model(g, x)
        torch.testing.assert_close(first, model(g, x), rtol=0, atol=0)
        g2, x2, model2, _ = prepare(config)
        torch.testing.assert_close(first, model2(g2, x2), rtol=0, atol=0)
        assert first.shape == (7, 2, 4)
        assert first.dtype == torch.float32 and first.device.type == "cpu"
        assert torch.isfinite(first).all().item()
    assert torch.version.cuda is None, "Install the CPU-only PyTorch wheel"


def metadata():
    cpu_model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    def git(*args):
        try:
            process = subprocess.run(["git", *args], capture_output=True, text=True,
                                     cwd=Path(__file__).resolve().parent)
        except FileNotFoundError:
            return None
        return process.stdout.strip() if process.returncode == 0 else None

    commit = git("rev-parse", "--verify", "HEAD")
    dirty = git("status", "--porcelain")
    return {
        "cpu_model": cpu_model, "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "os": platform.platform(), "architecture": platform.machine(),
        "python": platform.python_version(), "dgl": dgl.__version__,
        "torch": torch.__version__, "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {k: os.environ.get(k) for k in
                               ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES",
                                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "torch_build": torch.__config__.show(),
        "docker_image": os.environ.get("BENCHMARK_IMAGE"),
        "git_commit": commit or os.environ.get("BENCHMARK_GIT_COMMIT"),
        "git_dirty": bool(dirty) if dirty is not None else None,
    }


def run(config, warmup, repetitions):
    graph, features, layer, construction_ms = prepare(config)
    with torch.inference_mode():
        for _ in range(warmup):
            layer(graph, features)
        times = []
        for _ in range(repetitions):
            start = perf_counter_ns()
            output = layer(graph, features)
            elapsed = perf_counter_ns() - start
            times.append(elapsed / 1e6)
        assert output.shape == (config.nodes, config.heads, config.out_features)
        assert torch.isfinite(output).all().item()
    return {
        "schema_version": 1, "variant": "dgl-gatconv-stock", "device": "cpu", "dtype": "float32",
        "workload": asdict(config), "actual_edges": graph.num_edges(),
        "warmup": warmup, "repetitions": repetitions,
        "graph_construction_ms": construction_ms,
        "runtime_ms": {"samples": times, "median": statistics.median(times),
                       "mean": statistics.mean(times), "stddev": statistics.stdev(times),
                       "min": min(times)},
        "environment": metadata(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=1000)
    edges = parser.add_mutually_exclusive_group()
    edges.add_argument("--edges", type=int)
    edges.add_argument("--avg-degree", type=int, help="Requested random edges per node, before loops")
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--out-features", type=int, default=8, help="Output features per head")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/gat.json"))
    parser.add_argument("--smoke-test", action="store_true", help="Check correctness and exit")
    args = parser.parse_args()
    for name in ("nodes", "features", "out_features", "heads", "threads", "warmup"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    if not 0 <= args.seed < 2**31:
        parser.error("--seed must be in [0, 2**31)")
    for name in ("edges", "avg_degree"):
        if getattr(args, name) is not None and getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    smoke_test()  # Always required before measurements; entirely outside timing.
    if args.smoke_test:
        print("CPU GAT correctness smoke test passed")
        return
    edge_count = args.edges if args.edges is not None else args.nodes * (
        args.avg_degree if args.avg_degree is not None else 8)
    config = Workload(args.nodes, edge_count, args.features, args.out_features, args.heads, args.seed)
    result = run(config, args.warmup, args.repetitions)
    result["rerun_argv"] = ["uv", "run", "--frozen", "python", "benchmarks/gat.py",
        "--nodes", str(config.nodes), "--edges", str(config.edges),
        "--features", str(config.features), "--out-features", str(config.out_features),
        "--heads", str(config.heads), "--seed", str(config.seed), "--threads", str(args.threads),
        "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
        "--output", str(args.output)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result["runtime_ms"].items() if k != "samples"}))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
