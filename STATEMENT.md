# Computational Kernel for Low-Rank Factorization (Randomized SVD) for Weight Matrix Compression in LLMs

**AI Laboratory — Academic Year 2026/2027**
**Computational Projects**

## Background

Compression of large language models (LLMs) and low-rank adaptation (LoRA) rely on approximating dense weight matrices $W \in \mathbb{R}^{m \times n}$ as products of smaller matrices $A \in \mathbb{R}^{m \times k}$ and $B \in \mathbb{R}^{k \times n}$ (where $k \ll \min(m, n)$), reducing memory footprint and inference latency.

## Project Objective

Develop and optimize a computational kernel in Python/Numba for Randomized SVD (Stochastic Singular Value Decomposition) to decompose large-scale projection matrices, comparing it against standard deterministic approaches (Golub-Kahan).

## Tasks to Complete

### 1. Numerical Kernel Implementation
- Create a vectorized version of Randomized SVD using fast matrix multiplication (`scipy.linalg.blas.dgemm` or Numba JIT).
- Implement fast QR orthogonalization via Householder transformations.

### 2. Memory and Stride Management
- Ensure that the random projection matrix multiplications $Y = W\Omega$ respect memory contiguity (C-contiguous vs. F-contiguous) to minimize cache misses.

### 3. Practical Application
- Apply the factorization to the weight matrix of a transformer (e.g., attention layers of Llama-3-8B or Mistral-7B).

### 4. Benchmarking & Evaluation
- Measure execution time and peak RAM/VRAM usage as a function of $k$ (approximation rank).
- Quantify reconstruction error using the Frobenius norm:

$$
\text{Error} = \frac{\|W - AB\|_F}{\|W\|_F}
$$

- Measure the inference **speedup** of the compressed linear layer relative to the original layer.
