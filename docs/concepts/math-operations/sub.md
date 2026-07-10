# aakaar.sub

Performs elementwise subtraction of the second tensor from the first tensor. Like addition, this operation fully supports implicit broadcasting when the shapes of the input tensors are aligned.

### Signature

```python
aakaar.sub(input, other)
```

### Parameters

- input (Tensor) – The first source tensor (the minuend).
- other (Tensor or float) – The second tensor (the subtrahend) or a scalar value to subtract from every element of the input tensor.

### Details

The aakaar.sub function utilizes the same dispatch architecture as the addition operation:
- CPU: Executes standard C++ loops across the linear memory buffer.
- CUDA: Dispatches to a custom float4-vectorized CUDA kernel on the GPU for maximum throughput on contiguous data.
During the backward pass, the autograd engine correctly applies the negative derivative chain rule to the other tensor. If y = a - b, the gradient flowing back to a is multiplied by 1, while the gradient flowing back to b is multiplied by -1.

### Broadcasting Rules

Standard broadcasting semantics apply. If other is smaller than input, its dimensions are virtually expanded to match. The backward pass automatically sums the negative gradients across these broadcasted dimensions to ensure dimensional consistency.

```python
import aakaar as aa
import numpy as np

# Create two tensors
matrix = aa.from_numpy(np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32), requires_grad=True)
penalty = aa.from_numpy(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)

# Elementwise subtraction with implicit broadcasting row-by-row
result = aa.sub(matrix, penalty)

# Compute gradients
loss = result.sum()
loss.backward()

print("Result:")
print(result.to_numpy())
# [[ 4.  8.]
#  [14. 18.]]

# The gradient w.r.t penalty will be negative and automatically summed across the rows
print("Gradient w.r.t penalty:")
print(penalty.grad.to_numpy())
# [-2. -2.]
```