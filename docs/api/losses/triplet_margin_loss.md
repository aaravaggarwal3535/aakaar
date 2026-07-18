# TripletMarginLoss

`TripletMarginLoss` computes a margin-based loss using three embeddings:
an **anchor**, a **positive** sample that should be similar to the anchor,
and a **negative** sample that should be dissimilar.

The objective is to make the anchor closer to the positive sample than to the
negative sample by at least a specified margin. This loss is widely used in
**metric learning**, **face recognition**, **person re-identification**, and
**embedding learning**.

The current implementation supports only the **Euclidean distance (p=2)**.

---

## Signature

```python
TripletMarginLoss(
    margin=1.0,
    p=2,
    reduction="mean"
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `margin` | `float` | Minimum desired separation between positive and negative distances. Default is `1.0`. |
| `p` | `int` | Distance metric. Currently only `2` (Euclidean distance) is supported. |
| `reduction` | `str` | Specifies the reduction to apply: `"mean"`, `"sum"`, or `"none"`. Default is `"mean"`. |

---

## Inputs

| Input | Description |
|-------|-------------|
| `anchor` | Anchor embedding tensor. |
| `positive` | Embedding of a sample similar to the anchor. |
| `negative` | Embedding of a sample dissimilar to the anchor. |

All three tensors must have the same shape.

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-sample losses when using `"none"`.

---

## Formula

First, compute the Euclidean distances:

```text
d_pos = ||anchor − positive||

d_neg = ||anchor − negative||
```

The loss is then

```text
Loss = max(0, d_pos − d_neg + margin)
```

The loss becomes zero once the negative sample is at least `margin` farther
from the anchor than the positive sample.

---

## Example

```python
import aakaar
from aakaar.losses import TripletMarginLoss

anchor = aakaar.rand((8, 128))
positive = aakaar.rand((8, 128))
negative = aakaar.rand((8, 128))

criterion = TripletMarginLoss(
    margin=1.0
)

loss = criterion(
    anchor,
    positive,
    negative
)
```

---

## Using a Larger Margin

```python
criterion = TripletMarginLoss(
    margin=2.0
)
```

A larger margin requires the negative embeddings to be farther from the anchor
before the loss becomes zero.

---

## Using Different Reductions

### Mean (default)

```python
criterion = TripletMarginLoss(
    reduction="mean"
)
```

---

### Sum

```python
criterion = TripletMarginLoss(
    reduction="sum"
)
```

---

### None

```python
criterion = TripletMarginLoss(
    reduction="none"
)
```

Returns the loss for each triplet individually.

---

## Typical Training Loop

```python
criterion = TripletMarginLoss()

loss = criterion(
    anchor_embeddings,
    positive_embeddings,
    negative_embeddings
)

loss.backward()
optimizer.step()
```

---

## Typical Applications

- Face recognition
- Person re-identification
- Image retrieval
- Similarity search
- Metric learning
- Siamese and triplet neural networks

---

## Notes

- Requires three input tensors: **anchor**, **positive**, and **negative**.
- All input tensors must have identical shapes.
- Currently supports only **Euclidean distance (`p=2`)**.
- Uses `sqrt()` internally when computing Euclidean distances.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Raises a `NotImplementedError` if a value other than `p=2` is specified.