# aakaar.optim.SGD

Implements Stochastic Gradient Descent (SGD), an optimization algorithm that updates parameter weights based on the negative gradient of the loss function.

### Signature

```python
aa.optim.SGD(parameters, lr=0.01)
```

### Parameters

- parameters (list of Tensor) – An iterable list of Tensor objects (the weights and biases of your neural network layers) that should be updated.
- lr (float, optional) – The learning rate, which scales the magnitude of the weight updates. Default: 0.01.

### Details

SGD updates parameters using the following update rule:
$$\theta = \theta - \eta \cdot \nabla_\theta L$$
Where $\theta$ represents the parameters, $\eta$ is the learning rate, and $\nabla_\theta L$ is the gradient of the loss with respect to the parameters.

### Methods

#### zero_grad()
Clears the gradients of all parameters tracked by the optimizer. This must be called at the start of every training iteration to prevent the accumulation of gradients from previous steps.

#### step()
Performs a single optimization step. It iterates through the tracked parameters, applies the gradient update rule, and modifies the parameter values in-place.

Example:

```python
import aakaar as aa
import aakaar.nn as nn
import aakaar.optim as optim

# 1. Setup model parameters
layer = nn.Linear(10, 1)
params = [layer.weight, layer.bias]

# 2. Configure optimizer
optimizer = optim.SGD(params, lr=0.1)

# 3. Training Loop
for i in range(100):
    optimizer.zero_grad()
    
    # Forward pass
    output = layer(aa.rand((5, 10)))
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
```