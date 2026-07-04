# Aakaar

Aakaar is a custom, standalone deep learning tensor library built from the ground up using Python, C++, and raw CUDA. It is designed to provide a lightweight, transparent architecture for high-performance GPU computations without relying on heavy external frameworks like PyTorch or TensorFlow.

## Core Architecture

Aakaar bypasses standard NumPy arrays by implementing a custom C++ `Tensor` object that can live directly in GPU VRAM. Python interacts with this data via pybind11, acting as a lightweight remote control. This prevents severe performance bottlenecks over the PCI-e bus, keeping data on the GPU until explicitly requested back to the host with `.to_numpy()`.

Tensors carry their own `shape` and `strides`, so operations like slicing return lightweight views into the same underlying memory rather than copies — no data movement happens until you actually ask for it.

## Current capabilities

* N-dimensional GPU-native and CPU `Tensor` objects with full lifecycle management
* CUDA-accelerated uniform random number generation via cuRAND
* Matrix multiplication via cuBLAS (GPU) with a CPU fallback path
* NumPy-style indexing and slicing (`t[i]`, `t[1:3]`, `t[1:3, 2:4]`, negative indices, step slicing) that returns zero-copy views on both CPU and GPU
* Automatic CPU fallback: if no CUDA toolkit is available at install time, Aakaar builds a CPU-only extension so `device="cpu"` still works everywhere
* Direct host-to-device and device-to-host memory mapping via `.to_numpy()`

## Installation

```bash
pip install aakaar
```

Prebuilt wheels are available for Windows (Python 3.10–3.14, with CUDA support). On other platforms, `pip` will build Aakaar from source — this requires a C++ compiler (e.g. `g++`) for CPU-only support, and additionally the NVIDIA CUDA Toolkit (`nvcc`) if you want GPU acceleration. If no CUDA toolkit is found at install time, Aakaar automatically builds a CPU-only extension and `device="cuda"` calls will raise a clear error instead of failing to install.

## Quick start

```python
import aakaar

# Create a tensor of random values, on CPU or CUDA
a = aakaar.rand((4, 5), device="cpu", seed=1337)
print(a)
# aakaar.Tensor([...], device='cpu', shape=(4, 5))

print(a.shape)      # [4, 5]
print(len(a))       # 4 (size of first dimension)
print(a[0, 2])      # single element, as a Python float

# NumPy-style slicing returns zero-copy views
sub = a[1:3, 2:4]
print(sub.shape)          # [2, 2]
print(sub.is_contiguous())  # False (it's a strided view)
print(sub.to_numpy())       # materializes the view as a real NumPy array

# Matrix multiplication
b = aakaar.rand((5, 3), device="cpu", seed=42)
c = aakaar.matmul(a, b)
print(c.shape)  # [4, 3]

# On a CUDA-enabled build with a GPU available:
x = aakaar.rand((1024, 1024), device="cuda")
y = aakaar.rand((1024, 1024), device="cuda")
z = aakaar.matmul(x, y)   # runs via cuBLAS, stays on-device
result = z.to_numpy()     # only now does data move to host
```

## Notes

* `matmul` currently supports 2D tensors only, and both operands must be on the same device.
* Slicing with a non-unit step on CUDA tensors is correct but not yet memory-optimal for very large tensors — `to_numpy()` on such a view copies the full spanned region rather than just the selected elements.
* This is an actively developed project; APIs may change between minor versions.