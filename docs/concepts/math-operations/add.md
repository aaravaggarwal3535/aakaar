# aakaar.add

Performs elementwise addition of two tensors. This operation supports implicit broadcasting when the shapes of the input tensors are complementarily aligned.

### Signature

```python
aakaar.add(input, other)
```

### Parameters

- input (Tensor) – The first source tensor.
- other (Tensor or float) – The second tensor to add to input. If a scalar float is provided, it will be added to every element of the input tensor.

### Details

The aakaar.add function dispatches the computation based on the device placement of the input tensors:
- CPU: Executes vectorized loops across the flat memory using optimized C++ host routines.
- CUDA: Dispatches to a custom float4-vectorized CUDA kernel on the GPU, maximizing VRAM memory bandwidth by loading 128-bit chunks per memory transaction.

### Broadcasting Rules

If the two input tensors do not have identical shapes, Aakaar automatically attempts to broadcast them by prepending dimensions of size 1 or stretching dimensions of size 1 to match the larger tensor.

During backpropagation, the autograd engine is fully broadcasting-aware; it automatically sums out the accumulated gradients along the broadcasted dimensions to ensure the gradient shapes exactly match the original inputs.

Example:

```python
import aakaar as aa
import numpy as np

# Create two tensors
matrix = aa.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), requires_grad=True)
bias = aa.from_numpy(np.array([0.5, 1.5], dtype=np.float32), requires_grad=True)

# Elementwise addition with implicit broadcasting row-by-row
result = aa.add(matrix, bias)

# Compute gradients
loss = result.sum()
loss.backward()

print("Result:")
print(result.to_numpy())

print("Gradient w.r.t bias (automatically summed across rows):")
print(bias.grad.to_numpy())
```