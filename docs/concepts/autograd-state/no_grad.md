# aakaar.no_grad

A context manager that disables the autograd engine's graph construction. This is primarily used for inference, validation, or evaluation loops where gradient tracking is unnecessary.

### Signature

```python
aa.no_grad()
```

### Details

When execution occurs within the aa.no_grad() block, the library skips the registration of operation nodes in the dynamic computation graph.
**why use it?**
- Reduced Memory Footprint: By not caching intermediate values or graph nodes, your memory usage drops significantly.
- Increased Speed: The overhead of graph construction and memory allocation for tracking is eliminated, leading to faster execution.
- Safety: It prevents unintended modifications to the execution graph during evaluation steps.

Example:

```python
import aakaar as aa
import numpy as np

# Initialize parameters with tracking enabled
w = aa.from_numpy(np.array([2.0], dtype=np.float32), requires_grad=True)

# 1. Training mode: Graph is built
y = w * 3.0
y.backward()
print("Gradient:", w.grad.to_numpy())

# 2. Evaluation mode: No graph is built
with aa.no_grad():
    y_eval = w * 3.0
    # y_eval.backward() would raise an error here as no graph exists
    print("Inference Result:", y_eval.to_numpy())
```