# aakaar.functional.softmax

Applies the softmax function to an input tensor. This operation rescales the elements such that they lie in the range [0, 1] and sum to 1.0, effectively turning raw model outputs into probability distributions.

### Signature

```python
aa.softmax(input, dim=-1)
```

### Parameters

- input (Tensor) – The input tensor.
- dim (int, optional) – The dimension along which the softmax will be computed. Default: -1 (the last dimension).

### Details

The softmax function is defined as:
$$\text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j} \exp(x_j)}$$

Aakaar implements this numerically using the "max-subtraction" trick—subtracting the maximum value from the input vector before exponentiation—to ensure numerical stability and prevent overflow errors when dealing with large input values.

Example:

```python
import aakaar as aa
import numpy as np

# Create raw logits (e.g., from the final layer of a classifier)
logits = aa.from_numpy(np.array([[2.0, 1.0, 0.1]], dtype=np.float32))

# Compute probabilities
probs = aa.softmax(logits)

print("Probabilities:")
print(probs.to_numpy())
# Expected output: [[0.659, 0.242, 0.099]]
```