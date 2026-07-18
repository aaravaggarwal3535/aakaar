# CosineEmbeddingLoss

`CosineEmbeddingLoss` measures the similarity between two input tensors using
**cosine similarity**. It is commonly used in **metric learning**, **face
recognition**, **sentence embeddings**, and **representation learning**, where
the goal is to learn embeddings that are either similar or dissimilar.

The target tensor specifies whether a pair of embeddings should be considered
similar (`1`) or dissimilar (`-1`).

---

## Signature

```python
CosineEmbeddingLoss(
    margin=0.0,
    reduction="mean"
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `margin` | `float` | Margin used for dissimilar pairs (`y = -1`). Default is `0.0`. |
| `reduction` | `str` | Specifies the reduction to apply: `"mean"`, `"sum"`, or `"none"`. Default is `"mean"`. |

---

## Inputs

| Input | Description |
|-------|-------------|
| `x1` | First embedding tensor. |
| `x2` | Second embedding tensor. |
| `y` | Target tensor containing only `1` or `-1`. `1` indicates similar pairs, while `-1` indicates dissimilar pairs. |

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-sample losses when using `"none"`.

---

## Formula

First, cosine similarity is computed as:

```text
cos(x1, x2) = (x1 · x2) / (||x1|| ||x2||)
```

The loss is then defined as:

```text
if y = 1:
    Loss = 1 - cos(x1, x2)

if y = -1:
    Loss = max(0, cos(x1, x2) - margin)
```

---

## Example

```python
import aakaar
from aakaar.losses import CosineEmbeddingLoss

x1 = aakaar.rand((4, 128))
x2 = aakaar.rand((4, 128))

target = aakaar.tensor([
    1,
    1,
    -1,
    -1
])

criterion = CosineEmbeddingLoss()

loss = criterion(x1, x2, target)
```

---

## Using a Margin

```python
criterion = CosineEmbeddingLoss(
    margin=0.5
)
```

Increasing the margin requires dissimilar pairs to be farther apart before
their loss becomes zero.

---

## Using Different Reductions

### Mean (default)

```python
criterion = CosineEmbeddingLoss(
    reduction="mean"
)
```

---

### Sum

```python
criterion = CosineEmbeddingLoss(
    reduction="sum"
)
```

---

### None

```python
criterion = CosineEmbeddingLoss(
    reduction="none"
)
```

Returns the loss for each embedding pair individually.

---

## Typical Training Loop

```python
criterion = CosineEmbeddingLoss()

loss = criterion(
    embedding1,
    embedding2,
    targets
)

loss.backward()
optimizer.step()
```

---

## Typical Applications

- Face recognition
- Person re-identification
- Sentence embedding models
- Siamese neural networks
- Contrastive representation learning
- Image retrieval

---

## Notes

- Expects two tensors with identical shapes.
- The target tensor should contain only **1** or **-1**.
- Uses cosine similarity rather than Euclidean distance.
- Computes Euclidean norms internally using `sqrt()`.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Widely used for learning embedding spaces where similar samples are pulled together and dissimilar samples are pushed apart.