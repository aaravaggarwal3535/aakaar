# aakaar.zero_grad_all

A global utility function that traverses the internal computation graph to clear accumulated gradients from all tracked tensors in the current environment.

### Signature

```python
aa.zero_grad_all()
```

### Details

In Aakaar, gradients are accumulated (summed) in the .grad attribute of tensors every time .backward() is called. This is intentional, as it allows for training loops with gradient accumulation over multiple mini-batches.

However, you must manually reset these buffers before starting a new optimization step; otherwise, the gradients from the previous iteration will pollute the new calculation.

### When to use it

- Standard Training Loops: Call this at the very beginning of every training iteration before running the forward pass and calling loss.backward().
- Global Cleanup: If you are working in an interactive environment (like a Jupyter notebook) and want to completely reset all state without re-initializing your model parameters.

Example:

```python
import aakaar as aa
import numpy as np

# Initialize a parameter with gradient tracking
param = aa.from_numpy(np.array([2.0], dtype=np.float32), requires_grad=True)

# Simulate a training step
y = param * 3.0
y.backward()
print("Grad after step 1:", param.grad.to_numpy())

# If we don't zero the grad, the next backward pass will add to this value
y.backward()
print("Grad after step 2 (without zero_grad):", param.grad.to_numpy())

# Reset all gradients in the global graph
aa.zero_grad_all()
print("Grad after zero_grad_all:", param.grad.to_numpy())
```