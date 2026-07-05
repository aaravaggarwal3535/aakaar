# Aakaar

<p align="center">
  <b>A lightweight, high-performance C++/CUDA tensor engine for Python.</b><br>
  Built from scratch using Python, C++, raw CUDA, cuBLAS, cuRAND, and pybind11.
</p>

---

## Overview

**Aakaar** is a standalone tensor computation library designed to provide transparent, high-performance numerical computing without relying on heavyweight deep learning frameworks.

Unlike traditional libraries that abstract away the underlying implementation, Aakaar exposes a clean Python API backed by a custom C++ tensor engine capable of running on both CPU and NVIDIA GPUs.

Its goal is to serve as both a practical tensor library and an educational framework for understanding how modern AI frameworks work internally.

---

# Features

### 🚀 High Performance

- Native C++ tensor engine
- Raw CUDA kernels for GPU execution
- Optimized matrix multiplication using **cuBLAS**
- Random number generation using **cuRAND**

---

### 🧠 Custom Tensor Implementation

- N-dimensional tensors
- Shape and stride aware
- Supports arbitrary dimensions
- Automatic memory management
- CPU and GPU tensor storage

Example:

```python
import aakaar

x = aakaar.rand((3,4))
print(x.shape)
```

---

### ⚡ Dual Device Support

Create tensors directly on either CPU or GPU.

```python
cpu_tensor = aakaar.rand((4,4), device="cpu")

gpu_tensor = aakaar.rand((4,4), device="cuda")
```

If CUDA is unavailable, Aakaar automatically falls back to CPU mode.

---

### 📐 Zero-Copy Tensor Views

Tensor slicing never copies memory.

Instead, Aakaar creates lightweight tensor views by modifying only:

- shape
- strides
- storage offset

Example:

```python
x = aakaar.rand((5,5))

view = x[1:4, 2:5]

print(view.shape)
```

The returned tensor references the original memory.

No additional allocation occurs.

Supported slicing includes:

- Integer indexing
- Negative indexing
- Range slicing
- Step slicing
- Multi-dimensional slicing

Examples:

```python
x[2]

x[-1]

x[1:4]

x[:,2]

x[::2]

x[1:4,2:5]
```

---

### 🔄 NumPy Interoperability

Transfer tensors back to Python using

```python
tensor.to_numpy()
```

This explicitly copies data from the tensor engine into NumPy memory.

GPU tensors remain in VRAM until this function is called.

---

### ⚙ Automatic CPU Fallback

Installation automatically detects CUDA.

If CUDA is unavailable:

- CUDA files are skipped
- CPU backend is compiled
- API remains identical

No code changes are required.

---

### 🖥 GPU Memory Residency

GPU tensors remain entirely inside GPU memory.

Operations execute without repeatedly transferring data across PCIe.

Data moves back to host memory only when:

```python
tensor.to_numpy()
```

is called.

---

# Installation

Install directly from PyPI:

```bash
pip install aakaar
```

---

# Requirements

## Windows

Prebuilt wheels are available for

- Python 3.10
- Python 3.11
- Python 3.12
- Python 3.13
- Python 3.14

CUDA support is included in compatible builds.

---

## Linux / macOS

Aakaar builds from source.

Requirements:

- C++ compiler (g++, clang++)
- Python development headers

Optional:

- NVIDIA CUDA Toolkit
- nvcc compiler

Without CUDA, installation automatically produces a CPU-only build.

---

# Quick Start

## Creating Tensors

```python
import aakaar

x = aakaar.rand((4,5), seed=42)

print(x)
```

---

## CPU Tensor

```python
cpu = aakaar.rand((4,5), device="cpu")
```

---

## CUDA Tensor

```python
gpu = aakaar.rand((4,5), device="cuda")
```

---

## Tensor Properties

```python
print(x.shape)

print(len(x))

print(x.device)

print(x.ndim)

print(x.size)
```

---

## Accessing Elements

```python
value = x[0,2]
```

Returns a Python float.

---

## Tensor Slicing

```python
sub = x[1:3,2:4]

print(sub.shape)
```

No memory copy occurs.

---

## Checking Contiguity

```python
print(sub.is_contiguous())
```

---

## Convert to NumPy

```python
numpy_array = sub.to_numpy()
```

---

# Matrix Multiplication

Aakaar performs hardware-accelerated matrix multiplication through **cuBLAS**.

```python
a = aakaar.rand((1024,1024), device="cuda")

b = aakaar.rand((1024,1024), device="cuda")

c = aakaar.matmul(a,b)
```

The computation stays entirely on the GPU.

Move results back to Python only when needed:

```python
result = c.to_numpy()
```

---

# Random Number Generation

Random tensors are generated using **cuRAND** on CUDA builds.

```python
x = aakaar.rand((512,512), device="cuda")
```

CPU builds use the native C++ backend.

---

# Architecture

Aakaar follows a layered architecture designed for minimal overhead.

```
Python API
      │
      ▼
pybind11 Bindings
      │
      ▼
Custom C++ Tensor Engine
      │
 ┌────┴─────┐
 │          │
CPU Backend CUDA Backend
 │          │
 │       cuBLAS
 │       cuRAND
 │
Memory Manager
```

Python serves only as the interface.

Tensor metadata, indexing, slicing, memory management, and mathematical operations are executed entirely in compiled C++ or CUDA.

---

# Memory Model

Every tensor stores:

- Pointer to data
- Shape
- Strides
- Storage offset
- Device information
- Data type

Tensor views reuse the same underlying storage.

Only metadata changes during slicing.

---

# Performance Philosophy

Aakaar minimizes unnecessary memory movement.

Typical workflow:

```
Python

↓

Create Tensor

↓

GPU Memory

↓

Multiple CUDA Operations

↓

Matrix Multiplication

↓

More CUDA Operations

↓

to_numpy()

↓

Host Memory
```

The expensive PCIe transfer occurs only when explicitly requested.

---

### CUDA Views

Step slicing on CUDA tensors is mathematically correct.

However,

```python
tensor.to_numpy()
```

may currently copy the full spanned memory region instead of only the selected elements for highly fragmented strided views.

This optimization is under active development.

---

### API Stability

Aakaar is under active development.

Internal APIs may change between minor releases as additional functionality is introduced, including:

- Automatic differentiation (Autograd)
- Tensor broadcasting
- Additional mathematical operators
- Neural network primitives
- Optimized CUDA kernels

---

# Roadmap

Planned features include:

- Automatic differentiation
- Broadcasting
- Tensor arithmetic operators
- Convolution kernels
- Reduction operations
- Activation functions
- Optimizers
- Neural network layers
- Mixed precision support
- CUDA graph execution
- Multi-GPU support

---

# Why Aakaar?

Aakaar was created to demonstrate how modern tensor libraries work internally while remaining lightweight enough to study, modify, and extend.

Instead of hiding the implementation behind millions of lines of code, Aakaar focuses on providing a clean architecture where developers can understand:

- Tensor memory layout
- GPU execution
- CUDA programming
- Shape and stride mechanics
- pybind11 integration
- High-performance numerical computing

It is both a usable tensor engine and a learning resource for developers interested in building AI infrastructure from the ground up.

---

# License

This project is released under the MIT License.