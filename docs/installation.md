# Installation

FlexCible is a Python package (3.10 or later) built on PyTorch. It needs cooler, pyBigWig and pyfaidx to read the input files. A graphics card is used when available; training and prediction also run on the CPU.

## With conda or mamba

```bash
mamba create -n flexcible -c conda-forge -c bioconda python=3.11 pytorch cooler pybigwig pyfaidx numpy scipy pyyaml tqdm matplotlib
mamba activate flexcible
git clone https://github.com/MHH-Bioinformatics-Hematology/FlexCible.git
cd FlexCible
pip install -e .
```

For a CUDA build of PyTorch follow the selector at pytorch.org and install `torch` first, then the remaining packages.

## Optional dependencies

| package | used for |
|---|---|
| `hicrep` | the stratum-adjusted correlation coefficient in `flexcible evaluate`; without it, a built-in implementation of the same measure is used |
| `hicexplorer` | TAD and loop calls in the whole-chromosome evaluation (`hicFindTADs`, `hicDetectLoops`) and plotting of predicted maps |
| `torch.compile` support (PyTorch 2.1 or later) | faster training; set `train.compile: false` to run without it |

## Check the installation

```bash
flexcible --help
python -m pytest tests -q        # about 30 s on a CPU
```

The test suite builds a tiny synthetic genome and runs the whole pipeline on it; all tests should pass.

## Two ways to call the tool

Every command exists both as a subcommand and as a Python module:

```bash
flexcible predict --config run.yaml --out pred.cool
python -m flexcible.predict --config run.yaml --out pred.cool
```

Multi-GPU training uses the module form with `torchrun` (see the [tutorial](tutorial.md#train)).
