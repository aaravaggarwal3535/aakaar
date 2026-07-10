# Installation

Aakaar is built as a lightweight Python extension using C++ and CUDA. It can be built from source or installed via pre-built binaries depending on your system environment.

## Prerequisites

Before installing, ensure your host system matches the following software baselines:

* **Operating System:** Windows 10/11 (64-bit) or Ubuntu 22.04+ LTS
* **Python Version:** Python 3.10, 3.11, 3.12, or 3.13

### GPU Acceleration Requirements (Optional)

To leverage local hardware acceleration and execute vectorized CUDA kernels:
* **NVIDIA CUDA Toolkit:** No need to install explicitly as it comes with the package
* **Hardware:** NVIDIA GPU
* **NVIDIA Driver**

If the installation script does not detect a valid CUDA installation via the `CUDA_PATH` or `PATH` environment variables, it will automatically fallback to a CPU-only compilation using standard C++.

## Standard Installation

Install the latest stable version directly from PyPI:

```bash
pip install aakaar
```