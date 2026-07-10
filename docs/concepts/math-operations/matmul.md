# aakaar.matmul

Computes the matrix product of two tensors. The behavior depends on the dimensionality of the tensors, but it primarily targets 2D matrix multiplication.

### Signature

```python
aakaar.matmul(input, other)
```

### Parameters

- input (Tensor) – The first tensor to be multiplied.
- other (Tensor) – The second tensor to be multiplied.

### Dimensional Requirements

For standard 2D matrices, the inner dimensions must match. If input is of shape (N, M) and other is of shape (M, P), the resulting tensor will have the shape (N, P).

### Details and Hardware Dispatch

Matrix multiplication is handled differently depending on the active device:
- CPU: Aakaar leverages optimized BLAS libraries (like OpenBLAS) when available in the host environment to compute matrix products using highly parallelized host-side routines.
- CUDA: When executed on the GPU, Aakaar wraps and dispatches the operation directly to NVIDIA's official cuBLAS library. This ensures maximum compute utilization, taking full advantage of the GPU's streaming multiprocessors and architecture-specific hardware optimizations.

**Important Limitation:** Currently, aa.matmul requires both operands to be contiguous in memory. If you have sliced or transposed a tensor (which changes its strides without altering its underlying memory allocation layout), you must call .contiguous() on it before passing it to matmul.

### Autograd Engine(in matmul)
During the backward pass, the autograd engine applies the standard matrix calculus chain rule.
If $C = AB$, the gradients are computed as:
- $\nabla_A = \nabla_C B^T$
- $\nabla_B = A^T \nabla_C$

Example:

```python
import aakaar as aa
import numpy as np

# Create a 2x3 matrix and a 3x2 matrix
matrix_a = aa.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32), requires_grad=True)
matrix_b = aa.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32), requires_grad=True)

# Compute the matrix product (accelerated via cuBLAS if moved to CUDA)
result = aa.matmul(matrix_a, matrix_b)

print("Result Shape:", result.shape) # Expected: (2, 2)
print(result.to_numpy())

# Compute gradients
loss = result.sum()
loss.backward()

# The gradients are automatically computed using the transposed matrices
print("Gradient w.r.t matrix_a:")
print(matrix_a.grad.to_numpy())
```