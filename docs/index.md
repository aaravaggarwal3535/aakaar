<h1 align="center" style="display: flex; align-items: center; justify-content: center; gap: 15px;">
  <img src="assets/aakaar_logo.svg" alt="Aakaar Logo" width="60" style="vertical-align: middle;"/>
  <span style="vertical-align: middle;">Aakaar</span>
  <img src="https://img.shields.io/badge/version-v0.1.13-indigo?style=for-the-badge" alt="Version 0.1.13" style="vertical-align: middle; height: 28px;">
</h1>

<p align="center">
  <em>A high-performance, custom-built deep learning tensor library with a dynamic autograd engine and native C++/CUDA hardware acceleration.</em>
</p>

<br>

Built from the ground up, Aakaar bridges the gap between Python's ease of use and C++'s execution speed, providing a PyTorch-like API for tensor manipulation, automatic differentiation, and neural network construction.

## Key Features

* **Dynamic Autograd Engine:** Automatically constructs computational graphs on the fly and computes exact gradients via reverse-mode automatic differentiation.
* **Hardware Acceleration:** Seamlessly dispatches mathematical operations to optimized OpenBLAS (CPU) or cuBLAS (GPU) backends.
* **Standalone CUDA Support:** GPU acceleration works out-of-the-box. Pre-compiled wheels bundle the necessary CUDA runtime libraries, meaning **no CUDA Toolkit installation is required** for end-users.
* **Familiar API:** Designed to be intuitive for users familiar with modern deep learning frameworks, featuring `aa.Tensor`, `aa.nn.Linear`, and `aa.optim.SGD`.

## Installation

Aakaar provides pre-compiled wheels for Python 3.10 through 3.14 on both Windows and Linux.

```bash
pip install aakaar
```