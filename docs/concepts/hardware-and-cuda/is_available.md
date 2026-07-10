# aakaar.is_available

Checks if the CUDA runtime is accessible and if the Aakaar backend has successfully bound to at least one compatible NVIDIA GPU.

### Signature

```python
aa.is_available()
```

### Returns

- bool: Returns True if a CUDA-enabled device is detected and the library's custom CUDA kernels are ready for execution; returns False otherwise.

### Details

This function performs an internal handshake with the CUDA driver. It is the recommended way to branch your code between hardware-accelerated paths (GPU) and high-compatibility paths (CPU).

If this returns False, Aakaar will automatically route all mathematical operations through standard C++ CPU routines, ensuring your code remains portable across systems without needing to manually rewrite your loops for different environments.

Example:

```python
import aakaar as aa

# Check if we can leverage local hardware acceleration
if aa.is_available():
    print("GPU acceleration enabled: Dispatching kernels to CUDA.")
    device = "cuda"
else:
    print("GPU not detected: Defaulting to CPU fallback.")
    device = "cpu"

# Usage in a model
data = aa.rand((10, 10)).to(device)
```