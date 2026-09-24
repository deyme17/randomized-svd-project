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

> **Note:** `uv run huggingface-cli login` / `uv run hf auth login` may fail on some Windows setups with a `uv trampoline failed to canonicalize script path` error. This Python-based login command avoids this issue:
```bash
uv run python -c "from huggingface_hub import interpreter_login; interpreter_login()"
```


## Project Structure

```
src/                   Core kernels implementation
tests/                 Unit tests (pytest)
scripts/               Entry-point scripts (download weights, run benchmarks)
notebooks/             Exploratory analysis
data/                  Downloaded/cached weight matrices (gitignored)
```