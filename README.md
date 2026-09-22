# Randomized SVD for LLM Weight Compression

Computational kernel for low-rank factorization (Randomized SVD) applied to transformer weight matrix compression, benchmarked against deterministic SVD (Golub-Kahan).

## Setup

```bash
uv sync
```

This installs all dependencies, including the CUDA 12.6 build of PyTorch
from the pinned `pytorch-cu126` index in `pyproject.toml`.

Authenticate with Hugging Face before downloading gated model weights
(Llama-3-8B, Mistral-7B):

```bash
uv run huggingface-cli login
```

## Project Structure

```
src/randomized_svd/   Core kernel implementation
tests/                 Unit tests (pytest)
scripts/               Entry-point scripts (download weights, run benchmarks)
notebooks/             Exploratory analysis
data/                  Downloaded/cached weight matrices (gitignored)
```