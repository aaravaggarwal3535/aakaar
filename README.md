<h1 align="center" style="display: flex; align-items: center; justify-content: center;">
  <img src="assets/aakaar_logo.svg" alt="Aakaar Logo" width="60" style="vertical-align: middle; margin-right: 15px;"/>
  <span style="vertical-align: middle;">Aakaar</span>
</h1>

<p align="center">
  <em>A high-performance, custom-built deep learning tensor library with a dynamic autograd engine and native C++/CUDA hardware acceleration.</em>
</p>

<br>

Built from the ground up, Aakaar bridges the gap between Python's ease of use and C++'s execution speed, providing a PyTorch-like API for tensor manipulation, automatic differentiation, and neural network construction.

---


## Core Architecture

Aakaar's `Tensor` is a custom C++ object that can live directly in GPU VRAM or in host memory, exposed to Python via pybind11. Tensors carry their own `shape` and `strides`, so operations like slicing, transposing, and reshaping return lightweight zero-copy views into the same underlying memory wherever possible — data only moves when you explicitly ask for it via `.to_numpy()` or `.to(device)`.

Every differentiable operation records itself into a dynamic computation graph (`grad_fn`), which `.backward()` walks in reverse topological order to compute gradients — the same fundamental design as PyTorch's autograd, built independently from scratch.

## Current capabilities

**Tensors**
- N-dimensional tensors on CPU or CUDA, with real shape/stride tracking
- Zero-copy views: slicing (`t[1:3, 2:4]`, negative indices, step slicing), `.transpose()`, `.T` (full axis reversal), `.view()` / `.reshape()`
- `.contiguous()` to materialize a view when an operation requires dense memory
- `from_numpy()` to load real data in; `.to_numpy()` to get it back out
- `.to(device)` / `.to_device()` to move tensors between CPU and CUDA

**Autograd**
- `requires_grad`, `.grad`, `.backward()` with correct gradient accumulation across branching (diamond) graphs
- `retain_graph` support for reusing a graph across multiple backward passes
- `no_grad()` context manager and `.detach()` for inference / parameter-update code that shouldn't be tracked
- Broadcasting-aware gradients for every elementwise and matmul operation, verified against numerical (finite-difference) gradients, not just symbolic derivation

**Operations**
- Elementwise: `+`, `-`, `*`, `/` (tensor-tensor and tensor-scalar, with full broadcasting), unary negation
- `matmul()` / `@`: N-dimensional, batched, with broadcasting batch dimensions on both forward and backward
- Reductions: `sum(dim=...)`, `sum()` (full reduction), `max(dim=...)` (with correct argmax-routed gradients)
- Activations: `relu`, `sigmoid`, `tanh`, `leaky_relu` — all with float4-vectorized CUDA kernels and an alignment-safe scalar fallback
- `exp()`, `log()`
- `softmax()` (numerically stable, max-subtraction based) and `cross_entropy_from_probs()`

**Neural network building blocks** (`aakaar.nn`, `aakaar.optim`)
- `nn.Linear` — a fully-connected layer with standard uniform initialization
- `optim.SGD` — gradient descent optimizer using in-place parameter updates (`copy_()`) so parameter objects keep their identity across training steps
- `zero_grad_all()` for clearing gradients across a parameter list

**Automatic CPU fallback**
- If no CUDA toolkit is available at install time, Aakaar builds a CPU-only extension automatically. `device="cpu"` works everywhere; `device="cuda"` raises a clear error on CPU-only builds instead of failing to install.

## Installation

```bash
pip install aakaar
```

Prebuilt wheels are available for Windows (Python 3.10–3.14, with CUDA support). On other platforms, `pip` builds Aakaar from source — this requires a C++ compiler (e.g. `g++`) for CPU-only support, and additionally the NVIDIA CUDA Toolkit (`nvcc`) for GPU acceleration. If no CUDA toolkit is found at install time, Aakaar automatically builds a CPU-only extension.

## Quick start: tensors and autograd

```python
import aakaar
import numpy as np

# Load real data
x = aakaar.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)

# Standard ops, all differentiable
y = (x * 2 + 1).sum()
y.backward()
print(x.grad.to_numpy())  # [[2. 2.] [2. 2.]]

# Zero-copy slicing and views
big = aakaar.rand((10, 10), device="cpu")
view = big[1:5, 1:5]
print(view.is_contiguous())  # False — it's a strided view, no data copied

# Move to GPU
gpu_tensor = x.to("cuda")
```

## Training a small neural network

```python
import aakaar
from aakaar.nn import Linear
from aakaar.optim import SGD
import numpy as np

# Synthetic data: y = sin(x)
N = 64
x_np = np.linspace(-3, 3, N).reshape(N, 1).astype(np.float32)
y_np = np.sin(x_np).astype(np.float32)
x = aakaar.from_numpy(x_np)
y = aakaar.from_numpy(y_np)

fc1 = Linear(1, 16)
fc2 = Linear(16, 1)

def forward(x):
    h = fc1(x).tanh()
    return fc2(h)

def mse_loss(pred, target):
    diff = pred - target
    return (diff * diff).sum() / pred.size

params = fc1.parameters() + fc2.parameters()
opt = SGD(params, lr=0.05)

for epoch in range(300):
    opt.zero_grad()
    loss = mse_loss(forward(x), y)
    loss.backward()
    opt.step()

print(f"final loss: {loss.item():.6f}")
```

## Notes and known limitations

* `matmul` requires contiguous tensors; call `.contiguous()` on sliced/transposed operands first.
* Elementwise CUDA kernels also require contiguous inputs for their fast vectorized path.
* `matmul` backward supports broadcasting batch dimensions, but not yet arbitrary mixed-rank batch shapes beyond standard right-aligned broadcasting rules.
* Only `float32` is currently supported. Support for additional dtypes (float16, float64, int types) is a planned future addition, not yet implemented.
* This is an actively developed project; APIs may change between minor versions.

## ☕ Support the Development

If Aakaar helped you learn how autograd architectures work under the hood, or if you want to support independent, dependency-free deep learning infrastructure, consider dropping a tip!

* **International Supporters:** You can buy me a coffee via [Ko-fi (Coming Soon)](#) using any standard international debit/credit card. 
* **India (UPI):** Since international gateways occasionally restrict domestic transfers, you can support directly via UPI:  
  aaravaggarwal3535@okicici