# Aakaar

Aakaar is a custom, standalone deep learning tensor library built from the ground up using Python, C++, and raw CUDA. It is designed to provide a lightweight, transparent architecture for high-performance GPU computations without relying on heavy external frameworks like PyTorch or TensorFlow.

## Core Architecture

Aakaar bypasses standard NumPy arrays by implementing a custom C++ Tensor object that resides directly in GPU VRAM. Python interacts with this data via Pybind11, acting as a lightweight remote control. This prevents severe performance bottlenecks over the PCI-e bus, keeping data on the GPU until explicitly requested back to the host CPU.

Current capabilities include:
* Custom GPU-native Tensor class lifecycle management.
* CUDA-accelerated uniform random number generation via cuRAND.
* Direct host-to-device and device-to-host memory mapping.

## Installation

To build Aakaar from source, you must have the NVIDIA CUDA Toolkit (nvcc) and a compatible C++ compiler (e.g., g++) installed.

1. Clone the repository:
```bash
git clone [https://github.com/YOUR_USERNAME/aakaar.git](https://github.com/YOUR_USERNAME/aakaar.git)
cd aakaar
pip install setuptools wheel pybind11
pip install -e .
```

quick start
```bash
import aakaar

# Initialize the CUDA engine and allocate a GPU Tensor
print("Generating 100,000 random numbers on the GPU...")
data = aakaar.rand(100000, seed=1337)

# The data remains on the GPU as an Aakaar Tensor
print(type(data)) 
# <class 'aakaar._C.Tensor'>

# Bring the data across the PCI-e bus to the CPU for inspection
cpu_data = data.cpu()
print(cpu_data[:5])
```