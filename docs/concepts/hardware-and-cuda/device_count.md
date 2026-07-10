# aakaar.device_count

Returns the total number of CUDA-capable NVIDIA GPU devices currently accessible by the Aakaar runtime environment.

### Signature

```python
aa.device_count()
```

### Returns

- int: The number of detected physical GPU devices.

### Details

This function queries the system driver to enumerate available hardware. It is particularly useful when developing multi-GPU distribution strategies or when you need to select a specific device index for computation.

If aa.is_available() returns False, this function will typically return 0.

Example:

```python
import aakaar as aa

# Check available hardware
count = aa.device_count()

if count > 0:
    print(f"Found {count} GPU(s) available for acceleration.")
    # Example: Select the first available GPU
    device = "cuda:0"
else:
    print("No GPUs detected. Running on CPU.")
    device = "cpu"
```