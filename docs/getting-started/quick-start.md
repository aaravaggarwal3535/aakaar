# Quick Start

This guide will walk you through the core workflow of Aakaar. You will learn how to initialize tensors, manage device memory, compute gradients using the autograd engine, and train a basic neural network.

## 1. Verifying Install

To check wether the libarary is installed correctly and working correctly use the code snippet below.
```python
import aakaar as aa
print(f"Current Version: {aa.__version__}")
```

## 1. Tensors and Basic Operations

The fundamental data structure in Aakaar is the `Tensor`. You can easily create tensors by bridging data directly from NumPy arrays. By default, new tensors are allocated in standard CPU memory.

```python
import aakaar as aa
import numpy as np

# Initialize a tensor from a NumPy array
data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
x = aa.from_numpy(data)

print("Shape:", x.shape)
print("Strides:", x.strides)

# Standard mathematical operations are natively supported
y = (x * 2.0) + 1.5
print(y.to_numpy())
```

## 2. Hardware Acceleration

Aakaar gives you explicit control over memory placement. If your system has a compatible NVIDIA GPU and the CUDA runtime is available, you can migrate tensors to GPU VRAM for accelerated computation.

```python
# Check for CUDA availability
if aa.cuda.is_available():
    
    # Migrate tensor to GPU memory
    x_gpu = x.to("cuda")
    
    # Execute a vectorized CUDA kernel
    result_gpu = x_gpu.exp()
    
    # Bring the result back to host memory to interface with NumPy
    result_cpu = result_gpu.to("cpu")
```

## 3. The Autograd Engine

To optimize a model, you need to compute the gradients of a loss function with respect to your inputs. Aakaar handles this automatically via a reverse-mode autograd engine.

By setting requires_grad=True, the engine will begin tracking operations and building a dynamic computation graph.

```python
# Initialize variables with gradient tracking enabled
w = aa.from_numpy(np.array([2.0], dtype=np.float32), requires_grad=True)
x = aa.from_numpy(np.array([3.0], dtype=np.float32))
b = aa.from_numpy(np.array([1.0], dtype=np.float32), requires_grad=True)

# Forward pass: build the graph
# f(w, x, b) = w * x + b
y = (w * x) + b 

# Compute gradients using reverse accumulation
y.backward()

# Access the computed gradients
print("Gradient w.r.t w:", w.grad.to_numpy())  # Expected: [3.0]
print("Gradient w.r.t b:", b.grad.to_numpy())  # Expected: [1.0]
```

Context Management:
For inference or evaluation loops where gradients are not needed, use the no_grad context manager. This prevents graph construction, significantly reducing memory consumption and improving execution speed.

```python
with aa.no_grad():
    predictions = w * x + b
```

## 4. Training a Neural Network

Aakaar includes high-level abstractions in the aakaar.nn and aakaar.optim modules to streamline model building. Here is a complete example of training a small multi-layer perceptron to fit a synthetic dataset.

```python
import aakaar as aa
from aa.nn import Linear
from aakaar.optim import SGD
import numpy as np

# 1. Prepare synthetic data (y = x^2)
N = 100
x_data = np.linspace(-2, 2, N).reshape(N, 1).astype(np.float32)
y_data = (x_data ** 2).astype(np.float32)

x = aa.from_numpy(x_data)
y = aa.from_numpy(y_data)

# Move data to GPU if available
if aakaar.cuda.is_available():
    x = x.to("cuda")
    y = y.to("cuda")

# 2. Define a simple 2-layer network
layer1 = Linear(1, 16)
layer2 = Linear(16, 1)

def forward(inputs):
    hidden = layer1(inputs).relu()
    return layer2(hidden)

# 3. Setup the optimizer
parameters = layer1.parameters() + layer2.parameters()
optimizer = SGD(parameters, lr=0.01)

# 4. Training Loop
epochs = 200
for epoch in range(epochs):
    # Clear previous gradients
    optimizer.zero_grad()
    
    # Forward pass
    predictions = forward(x)
    
    # Compute Mean Squared Error loss
    diff = predictions - y
    loss = (diff * diff).sum() / predictions.size
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
```
