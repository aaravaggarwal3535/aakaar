# aakaar.from_numpy

Creates an `aakaar.Tensor` directly from a `numpy.ndarray`. This is the primary bridge between standard Python data workflows and the custom C++ execution backend.

### Signature

```python
aakaar.from_numpy(array, requires_grad=False)
```

### Parameters

- array (numpy.ndarray) – The input array containing the data. For optimal performance, the data type should match the backend precision (typically np.float32).
- requires_grad (bool, optional) – If set to True, the autograd engine will begin tracking operations on the returned tensor to build a computation graph. Default: False.

### Details

The returned tensor does not share memory with the original NumPy array. A completely new C++ backend memory allocation is triggered. By default, the memory is allocated on the host CPU in a contiguous, row-major layout.

Once instantiated, the tensor manages its own shape and strides internally, allowing for zero-copy views during slicing and transposing.

Example:

```python
import aakaar as aa
import numpy as np

# Prepare standard NumPy data
data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

# Bridge the data into the Aakaar runtime with gradient tracking enabled
tensor = aa.from_numpy(data, requires_grad=True)

print("Shape:", tensor.shape)
print("Strides:", tensor.strides)
```