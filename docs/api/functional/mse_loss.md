# aakaar.functional.mse_loss

Computes the Mean Squared Error (MSE) between the input prediction tensor and the target tensor.

### Signature

```python
aa.mse_loss(input, target)
```

### Parameters

- input (Tensor) – The predicted values from the model.
- target (Tensor) – The ground truth values. Must have the same shape as input.

### Details

The Mean Squared Error measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.The formula is defined as:
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
The mse_loss function is fully differentiable. During the backward pass, the autograd engine computes the gradient of the loss with respect to the inputs as:
$$\nabla_{\hat{y}} = \frac{2}{n} (\hat{y} - y)$$
This makes it the standard choice for regression tasks where the objective is to minimize the distance between continuous output values and target values.

Example:

```python
import aakaar as aa
import numpy as np

# Predictions and ground truth
preds = aa.from_numpy(np.array([2.5, 0.0, 2.0], dtype=np.float32), requires_grad=True)
targets = aa.from_numpy(np.array([3.0, -0.5, 2.0], dtype=np.float32))

# Compute Mean Squared Error
loss = aa.mse_loss(preds, targets)

# Backward pass to compute gradients
loss.backward()

print("Loss:", loss.item())
print("Gradients:", preds.grad.to_numpy())
```