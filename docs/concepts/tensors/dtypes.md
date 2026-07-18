# Data Types (Dtypes)

Aakaar provides native support for multiple numerical data types across both CPU and CUDA devices. While the library defaults to `float32` for standard neural network operations, you can now explicitly define data types for increased precision or integer-based arithmetic.

## Supported Data Types

| Data Type | String Alias | Hardware Support | Description |
| :--- | :--- | :--- | :--- |
| **Float32** | `'float32'` | CPU, CUDA | 32-bit float. **(Default)** |
| **Float64** | `'float64'` | CPU, CUDA | 64-bit float (Double precision). |
| **Int32** | `'int32'` | CPU, CUDA | 32-bit signed integer. |
| **Int64** | `'int64'` | CPU, CUDA | 64-bit signed integer. |

## Tensor Creation

You can specify the `dtype` during initialization. Aakaar handles the internal memory layout and device movement automatically.

```python
import aakaar

# Float tensors
x_f64 = aakaar.rand((3, 4), dtype='float64', device='cuda')

# Integer tensors (requires bounds)
x_i32 = aakaar.randint((10,), low=-50, high=50, dtype='int32')
```

## Type Safety & Constraints

Aakaar enforces strict policies to ensure stability, particularly with autograd and mathematical operations.

### Autograd Restrictions
**warning "Autograd is Float32 Only"**
The dynamic autograd engine is currently optimized for float32. To ensure performance and numerical stability, autograd is blocked for float64, int32, and int64. Attempting to set requires_grad=True on these types will raise a RuntimeError.

### Overflow-Safe Accumulation

Aakaar handles integer arithmetic with safety mechanisms:

- **Summation**: sum() on integer tensors automatically promotes the result to int64 to prevent overflow.
- **Matrix Multiplication (matmul)**: Integer matrix multiplications use internal int64 accumulation before casting back to the original type, allowing you to perform large-scale integer operations without overflow risk.

### Continuous Function Guards

Functions that require a continuous domain (e.g., sigmoid, tanh, exp, log, sqrt) will strictly reject integer tensors. Unlike some frameworks that silently cast to float, Aakaar requires you to perform the cast explicitly if you intend to apply these operations to integer data:

```python
x = aakaar.randint((5,), 1, 10, dtype='int32')

# Raises RuntimeError
y = x.sigmoid() 

# Correct approach
y = x.to(dtype='float32').sigmoid()
```

## Memory Management

All supported dtypes are fully compatible with .to_device('cuda') and .contiguous() operations, ensuring consistent round-trip behavior between CPU and GPU memory.