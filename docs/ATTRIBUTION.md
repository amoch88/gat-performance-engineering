# Source, data, and attribution

- Research source:
  [amoch88/dgl-performance-model](https://github.com/amoch88/dgl-performance-model),
  committed source snapshot
  [9659684473a0531aa06e40f614e1fbfbbcfc5b76](https://github.com/amoch88/dgl-performance-model/tree/9659684473a0531aa06e40f614e1fbfbbcfc5b76).
  The benchmark implementations and recorded data retain that provenance.
  The [archived baseline notes](../RESEARCH_README.md) preserve the earlier
  synthetic runner, dataset-access guidance, and stock DGL parameter
  experiments. This project does not change the upstream project's
  distribution permissions.
- The annotated DGL layer preserves its original attribution and
  [DGL license](../benchmarks/dgl_baseline/DGL-LICENSE) (Apache License
  2.0). The tests compare its executable class AST against the installed
  DGL 1.1.3 import. Stock timings use the installed import.
- [DGL](https://github.com/dmlc/dgl),
  [PyTorch](https://github.com/pytorch/pytorch),
  [NumPy](https://numpy.org/), and [SciPy](https://scipy.org/) are installed
  from the pinned dependency sources, not vendored into the Git checkout.
  Their own licenses apply. GNU C++/binutils are installed from Debian in
  the compiler image; Valgrind is optional for the isolated native QA.
- Cora, CiteSeer and PubMed are acquired through the public DGL citation
  loaders at runtime. Dataset archives, intermediate caches, and model
  binaries are not redistributed here.
- Algorithm and paper citations with publication details appear in
  [the technical analysis](TECHNICAL_ANALYSIS.md#related-work-and-references).
