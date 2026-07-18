# MultiMarginLoss

`MultiMarginLoss` computes the **multi-class hinge loss** for classification
tasks. It encourages the score of the correct class to be greater than the
scores of all other classes by at least a specified margin.

It is commonly used in **multi-class classification** and **margin-based
learning**.

In Aakaar, this loss expects **one-hot encoded targets** rather than integer
class indices.

---

## Signature

```python
MultiMarginLoss(
    p=1,
    margin=1.0,
    reduction="mean"
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `p` | `int` | Power used in the loss. Supported values are `1` and `2`. Default is `1`. |
| `margin` | `float` | Desired minimum margin between the correct class score and the remaining class scores. Default is `1.0`. |
| `reduction` | `str` | Specifies the reduction to apply: `"mean"`, `"sum"`, or `"none"`. Default is `"mean"`. |

---

## Inputs

| Input | Description |
|-------|-------------|
| `logits` | Predicted class scores. Shape is typically `(batch_size, num_classes)`. |
| `target_onehot` | One-hot encoded target tensor with the same shape as `logits`. |

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-sample losses when using `"none"`.

---

## Formula

For each sample:

```text
Loss = mean(max(0, margin - score(correct) + score(other))ᵖ)
```

where:

- `score(correct)` is the score of the correct class.
- `score(other)` represents every incorrect class.
- The correct class itself is excluded from the computation.

The parameter `p` determines whether the margin penalty is linear (`p = 1`)
or quadratic (`p = 2`).

---

## Example

```python
import aakaar
from aakaar.losses import MultiMarginLoss

logits = aakaar.tensor([
    [3.2, 0.5, 1.1],
    [0.3, 2.8, 1.0]
])

targets = aakaar.tensor([
    [1, 0, 0],
    [0, 1, 0]
])

criterion = MultiMarginLoss()

loss = criterion(
    logits,
    targets
)
```

---

## Using Quadratic Margins

```python
criterion = MultiMarginLoss(
    p=2
)
```

Quadratic margins penalize violations more heavily than linear margins.

---

## Using a Larger Margin

```python
criterion = MultiMarginLoss(
    margin=2.0
)
```

Increasing the margin requires greater separation between the correct class
and the incorrect classes.

---

## Using Different Reductions

### Mean (default)

```python
criterion = MultiMarginLoss(
    reduction="mean"
)
```

---

### Sum

```python
criterion = MultiMarginLoss(
    reduction="sum"
)
```

---

### None

```python
criterion = MultiMarginLoss(
    reduction="none"
)
```

Returns the loss for each sample individually.

---

## Typical Training Loop

```python
criterion = MultiMarginLoss()

loss = criterion(
    logits,
    targets
)

loss.backward()
optimizer.step()
```

---

## Typical Applications

- Multi-class classification
- Margin-based classifiers
- Image classification
- Pattern recognition
- Learning-to-rank variants
- Representation learning

---

## Notes

- Expects **one-hot encoded targets** in the current version of Aakaar.
- Supports only `p=1` and `p=2`.
- The correct class is excluded from the margin computation.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Encourages the correct class score to exceed all incorrect class scores by at least the specified margin.