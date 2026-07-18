# Tensor Overview

The `aakaar.Tensor` is the fundamental data structure in the Aakaar library. It is a multi-dimensional matrix containing elements of a single data type. Tensors are extremely similar to NumPy arrays, but with two critical superpowers: they can execute massive parallel computations on CUDA hardware, and they can track their own execution history for automatic differentiation (autograd).

---

## Tensor Creation

You can create tensors from scratch using built-in distribution functions, or convert existing NumPy arrays directly into Aakaar tensors without unnecessary memory copying.

```python
import aakaar
import numpy as np

# 1. From random distributions
x_float = aakaar.rand((3, 4), device='cpu')
x_int = aakaar.randint((3, 4), low=1, high=10, dtype='int32')

# 2. From existing NumPy arrays
np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
x_from_np = aakaar.from_numpy(np_array, device='cuda')
```

## Device Management

Aakaar is built for high-performance hardware acceleration. You can easily transition tensors between system RAM (CPU) and VRAM (GPU).

```python
x = aakaar.rand((100, 100), device='cpu')

# Move to GPU for accelerated computation
x_gpu = x.to_device('cuda') 
# Alternatively, use the shorthand:
x_gpu = x.to('cuda')

# Move back to CPU to interface with standard Python libraries
x_cpu = x_gpu.to('cpu')
```

## Shape Manipulation

Aakaar provides zero-copy shape manipulation where possible, allowing you to view the same underlying memory block in different dimensional structures.

```python
x = aakaar.rand((4, 6))

# Flatten the tensor (requires matching total elements)
y = x.view([24])

# Reshape into a different grid
z = x.reshape([2, 12])

# Transpose dimensions (e.g., for matrix multiplication alignment)
x_t = x.transpose(0, 1)

# Shorthand for full matrix transpose
x_T = x.T
```

## Indexing, Slicing & Memory Contiguity

Tensors support standard Python slicing. However, slicing a tensor creates a "view" that may no longer be contiguous in physical memory. Aakaar handles this automatically, but provides tools to force contiguous memory alignment when necessary for custom CUDA kernels.

```python
big = aakaar.rand((5, 5))

# Extract a 3x3 sub-grid
sliced = big[1:4, 1:4]

# Check memory layout
print(sliced.is_contiguous()) # Returns False

# Force a physical memory copy to align the data linearly
aligned = sliced.contiguous()
print(aligned.is_contiguous()) # Returns True
```

You can also perform in-place memory updates using .copy_():

```python
x = aakaar.rand((10,))
y = aakaar.rand((10,))
x.copy_(y) # Overwrites x's memory with y's values
```

## Mathematical Operations

Aakaar supports a full suite of element-wise operations, reductions, and hardware-accelerated linear algebra.

### Element-wise & Scalar Math
```python
a = aakaar.rand((5,))
b = aakaar.rand((5,))

c = a + b    # Element-wise addition
d = a * 5.0  # Scalar multiplication
e = a.abs()  # Absolute value
f = a.sqrt() # Square root
```

### Reductions
```python
x = aakaar.rand((3, 4))

# Sum across columns (dim=1)
row_sums = x.sum(dim=1)

# Global maximum
global_max = x.max()
```

## Matrix Multiplication
Matrix multiplication uses the standard @ operator or the explicit .matmul() call. When executed on a CUDA device, this heavily leverages custom hardware acceleration (and Tensor Cores if TF32 is enabled).

```python
a = aakaar.rand((128, 256), device='cuda')
b = aakaar.rand((256, 64), device='cuda')

# Hardware-accelerated dot product
c = a @ b
```