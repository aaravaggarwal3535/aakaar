# aakaar.rand

Generates a tensor of a specified shape, filled with random values drawn from a uniform distribution on the interval `[0, 1)`.

### Signature

```python
aakaar.rand(shape, requires_grad=False)
```

### Parameters

- shape (tuple or list of int) – The dimensions of the output tensor. For example, (3, 4) creates a 2D matrix with 3 rows and 4 columns.
- requires_grad (bool, optional) – If set to True, the autograd engine will begin tracking operations on the returned tensor to build a computation graph. Default: False.

### Details

Calling aakaar.rand triggers a fresh memory allocation via the C++ backend. The underlying PRNG (Pseudo-Random Number Generator) executes on the host CPU, meaning the initial tensor is always allocated in standard system RAM.

If you require these random values for hardware-accelerated operations, you must explicitly migrate the returned tensor using the .to("cuda") method.

This function is highly useful for initializing synthetic data, validating matrix dimensions, or manually constructing learnable parameter weights when bypassing the higher-level aakaar.nn structures.

Example:

```python
import aakaar as aa

# Generate a 3x3 matrix of random values with tracking enabled
weights = aa.rand((3, 3), requires_grad=True)

print("Shape:", weights.shape)
print("Requires Grad:", weights.requires_grad)

# Convert to NumPy array to view the generated values
print(weights.to_numpy())
```