# aakaar.functional.cross_entropy_from_probs

Computes the cross-entropy loss between predicted probabilities and target class labels. This is the standard objective function for multi-class classification tasks.

### Signature

```python
aa.cross_entropy_from_probs(input, target)
```

### Parameters

- input (Tensor) – A tensor of shape (N, C) containing predicted probabilities for each class, where N is the batch size and C is the number of classes.
- target (Tensor) – A tensor of shape (N) containing the ground truth class indices (integers).

### Details

Cross-entropy measures the divergence between the predicted probability distribution and the actual label distribution (represented as a one-hot vector).The formula used is:
$$L = -\sum_{i=1}^{N} \log(\hat{y}_{i, \text{target}_i})$$
Where $\hat{y}_{i, \text{target}_i}$ is the predicted probability for the correct class for sample $i$.

Example:

```python
import aakaar as aa
import numpy as np

probs = aa.from_numpy(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32), requires_grad=True)

# Create one-hot targets manually
# Row 0 -> class 0, Row 1 -> class 1
one_hot_targets = aa.from_numpy(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))

loss = aa.cross_entropy_from_probs(probs, one_hot_targets)
loss.backward()

print(loss)
```