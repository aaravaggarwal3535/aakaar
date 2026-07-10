# aakaar.is_grad_enabled

Returns a boolean value indicating whether the global autograd engine is currently tracking operations to construct a dynamic computation graph.

### Signature

```python
aa.is_grad_enabled()
```

### Returns

- bool: Returns True if gradient tracking is active (default), and False if the engine is currently inside a aa.no_grad() context manager.

### Details

This is a utility function used primarily for debugging or for creating custom operations that behave differently depending on whether they are being called during a training phase (graph building) or an inference phase (evaluation).

If you are developing complex custom modules and need to check if the current execution context requires gradient tracking for conditional branching or performance optimization, this is the canonical way to query that state.

```python
import aakaar as aa

# Check initial state
print(f"Tracking enabled: {aa.is_grad_enabled()}") 

# Check state inside no_grad context
with aa.no_grad():
    print(f"Tracking enabled inside block: {aa.is_grad_enabled()}")

# Check state after exiting context
print(f"Tracking enabled after block: {aa.is_grad_enabled()}")
```