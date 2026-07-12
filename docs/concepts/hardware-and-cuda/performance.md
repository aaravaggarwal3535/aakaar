# Performance & Tuning

Aakaar provides low-level control over hardware execution to help you maximize throughput and accurately benchmark your models.

---

## CUDA Synchronization

Aakaar's CUDA operations execute asynchronously by default, which matches the behavior of industry-standard frameworks like PyTorch. 

**Important:** You must call `aakaar.synchronize()` before timing GPU code or reading wall-clock benchmarks. If you omit this step, your timer will only measure the kernel-launch dispatch time on the CPU, rather than the actual execution time on the GPU.

```python
import aakaar
import time

# ... setup tensors ...

aakaar.synchronize() # Wait for previous operations to finish
start = time.perf_counter()

c = aakaar.matmul(a, b)

aakaar.synchronize() # Wait for matmul to finish before stopping the clock
end = time.perf_counter()
```

## Tensor Core Acceleration (TF32)

On Ampere-generation GPUs and newer (e.g., RTX 30-series, 40-series, 50-series, A100, H100), you can enable tensor-core-accelerated matrix multiplications using the TF32 internal format.

Calling aakaar.set_tf32(True) trades a microscopically small amount of numerical precision for a massive speedup — typically 3-5x for large matmuls.

- **Default Behavior**: Off by default.
- **Memory Impact**: None. Tensors remain standard float32 in memory regardless of this setting; the conversion only happens internally inside the tensor cores during the math operation.

```python
# Enable Tensor Core acceleration
aakaar.set_tf32(True)

# This operation is now heavily accelerated 
c = aakaar.matmul(a, b)
```

## Benchmarking Sctipt

```python
import aakaar
import numpy as np
import time

# Initialize large matrices on the GPU
a = aakaar.rand((2048, 2048), device='cuda', seed=1)
b = aakaar.rand((2048, 2048), device='cuda', seed=2)
expected = a.to_numpy() @ b.to_numpy()

# 1. Benchmark Standard FP32
aakaar.set_tf32(False)
for _ in range(3): aakaar.matmul(a, b) # Warmup
aakaar.synchronize()

start = time.perf_counter()
for _ in range(20): c1 = aakaar.matmul(a, b)
aakaar.synchronize()
t_fp32 = (time.perf_counter() - start) / 20

# 2. Benchmark Accelerated TF32
aakaar.set_tf32(True)
for _ in range(3): aakaar.matmul(a, b) # Warmup
aakaar.synchronize()

start = time.perf_counter()
for _ in range(20): c2 = aakaar.matmul(a, b)
aakaar.synchronize()
t_tf32 = (time.perf_counter() - start) / 20

# 3. Calculate Results
flops = 2 * 2048**3
print(f"FP32: {t_fp32*1000:.4f} ms  ({(flops/t_fp32)/1e12:.2f} TFLOPS)")
print(f"TF32: {t_tf32*1000:.4f} ms  ({(flops/t_tf32)/1e12:.2f} TFLOPS)")
print(f"Speedup: {t_fp32/t_tf32:.2f}x")

# Verify the reduced precision remains highly accurate
is_correct = np.allclose(c2.to_numpy(), expected, atol=1e-2, rtol=1e-2)
print(f"TF32 correctness: {is_correct}")
```
*Note:*
> This script can give diffrent results on diffrent system for our achived results there is a speed up of about <b>1.87x</b> faster results tested on RTX4060 Consumer GPU, Commersial GPU can give more faster results.