# HingeEmbeddingLoss

`HingeEmbeddingLoss` measures the similarity between input values based on
binary labels. It is commonly used in **metric learning**, **embedding
learning**, and **Siamese neural networks**, where the objective is to make
similar samples close together while keeping dissimilar samples sufficiently
far apart.

The target tensor must contain only **1** or **-1**.

---

## Signature

```python
HingeEmbeddingLoss(
    margin=1.0,
    reduction="mean"
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `margin` | `float` | Margin used for dissimilar pairs (`y = -1`). Default is `1.0`. |
| `reduction` | `str` | Specifies the reduction to apply: `"mean"`, `"sum"`, or `"none"`. Default is `"mean"`. |

---

## Inputs

| Input | Description |
|-------|-------------|
| `x` | Input tensor containing distances or similarity scores. |
| `y` | Target tensor containing only `1` or `-1`. `1` indicates similar samples, while `-1` indicates dissimilar samples. |

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-element losses when using `"none"`.

---

## Formula

For each element:

```text
if y = 1:
    Loss = x

if y = -1:
    Loss = max(0, margin - x)
```

Positive pairs are encouraged to have small distances, while negative pairs
are penalized only when their distance is less than the specified margin.

---

## Example

```python
import aakaar
from aakaar.losses import HingeEmbeddingLoss

distance = aakaar.tensor([
    0.2,
    1.5,
    0.8,
    2.3
])

target = aakaar.tensor([
    1,
    -1,
    1,
    -1
])

criterion = HingeEmbeddingLoss()

loss = criterion(
    distance,
    target
)
```

---

## Using a Custom Margin

```python
criterion = HingeEmbeddingLoss(
    margin=2.0
)
```

Increasing the margin requires negative pairs to be farther apart before
their loss becomes zero.

---

## Using Different Reductions

### Mean (default)

```python
criterion = HingeEmbeddingLoss(
    reduction="mean"
)
```

---

### Sum

```python
criterion = HingeEmbeddingLoss(
    reduction="sum"
)
```

---

### None

```python
criterion = HingeEmbeddingLoss(
    reduction="none"
)
```

Returns the loss for each element individually.

---

## Typical Training Loop

```python
criterion = HingeEmbeddingLoss()

loss = criterion(
    distances,
    labels
)

loss.backward()
optimizer.step()
```

---

## Typical Applications

- Siamese neural networks
- Face verification
- Signature verification
- Metric learning
- Similarity learning
- Image retrieval

---

## Notes

- Expects the target tensor to contain only **1** or **-1**.
- Input values typically represent distances or similarity scores between pairs of samples.
- Negative pairs are penalized only when their distance is smaller than the specified margin.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Commonly used for learning embedding spaces where similar samples are close together and dissimilar samples are separated by at least the margin.