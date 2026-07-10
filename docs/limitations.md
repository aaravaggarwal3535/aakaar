# Limitations & Known Issues

This document outlines the current technical constraints of the Aakaar library. 

### 1. Memory Contiguity Requirement
Many performance-critical operations, specifically `aakaar.matmul` and custom CUDA kernels, strictly require input tensors to be **contiguous** in memory.
* **The Constraint:** If you perform slicing or transposing, the underlying data layout becomes non-contiguous. Aakaar actively guards against this and will raise a `ValueError: matmul requires contiguous tensors` rather than computing incorrect math.
* **The Workaround:** Always call `.contiguous()` on your tensors before passing them into linear layers or matrix multiplication functions if they have been subjected to slicing or reshuffling operations.

### 2. Autograd Graph Scope
* **Operations:** The autograd engine only tracks operations explicitly routed through the `aakaar` API. If you extract a tensor's data to NumPy (using `.to_numpy()`), perform operations using external libraries like `scipy`, and ingest it back into an `aa.Tensor`, the computation graph will break, and gradients will not flow past that point.

### 3. Hardware Acceleration & Device Assignment
* **Device Validation:** Currently, the `.to(device)` method does not strictly validate the requested GPU index against available hardware. For example, moving a tensor to `cuda:1` on a single-GPU machine will silently succeed in Python but may cause undefined behavior or driver errors during execution. Always use `aa.device_count()` to verify available hardware.
* **Multi-GPU:** Distributed training (data or model parallelism) across multiple GPUs is a planned feature and not yet supported natively.

### 4. Floating Point Precision
* **Implicit Casting:** The framework is explicitly optimized for `float32` (single precision). If you ingest higher-precision data types (such as NumPy `float64`), Aakaar will silently cast the data down to `float32` upon tensor creation to maintain compatibility with the underlying C++ and cuBLAS routines.

---

*If you encounter a bug or a missing implementation, please submit an issue on the repository with the code snippet that triggered the `RuntimeError` or `ValueError`.*