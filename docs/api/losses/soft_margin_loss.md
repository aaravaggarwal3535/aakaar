# SoftMarginLoss

`SoftMarginLoss` computes a smooth margin-based loss for **binary
classification**. Instead of enforcing a hard margin, it uses a differentiable
objective that encourages positive samples to receive large positive scores and
negative samples to receive large negative scores.

The target tensor must contain only **1** or **-1**.

---

## Signature

```python
SoftMarginLoss(reduction="mean")
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `reduction` | `str` | Specifies the reduction to apply: `"mean"`, `"sum"`, or `"none"`. Default is `"mean"`. |

---

## Inputs

| Input | Description |
|-------|-------------|
| `x` | Predicted scores (logits). |
| `y` | Target tensor containing only `1` or `-1`. |

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-element losses when using `"none"`.

---

## Formula

For each prediction:

```text
Loss = log(1 + exp(-y × x))
```

where:

- `x` is the predicted score.
- `y` is the target label (`1` or `-1`).

Internally, Aakaar uses a numerically stable formulation equivalent to this
expression.

---

## Example

```python
import aakaar
from aakaar.losses import SoftMarginLoss

scores = aakaar.tensor([
    2.1,
    -1.5,
    0.8,
    -0.4
])

target = aakaar.tensor([
    1,
    -1,
    1,
    -1
])

criterion = SoftMarginLoss()

loss = criterion(
    scores,
    target
)
```

---

## Using Different Reductions

### Mean (default)

```python
criterion = SoftMarginLoss(
    reduction="mean"
)
```

Returns the average loss.

---

### Sum

```python
criterion = SoftMarginLoss(
    reduction="sum"
)
```

Returns the sum of all losses.

---

### None

```python
criterion = SoftMarginLoss(
    reduction="none"
)
```

Returns the loss for each prediction individually.

---

## Typical Training Loop

```python
criterion = SoftMarginLoss()

loss = criterion(
    predictions,
    labels
)

loss.backward()
optimizer.step()
```

---

## Typical Applications

- Binary classification
- Linear classifiers
- Margin-based learning
- Metric learning
- Representation learning
- Support vector machine-inspired models

---

## SoftMarginLoss vs HingeEmbeddingLoss

| SoftMarginLoss | HingeEmbeddingLoss |
|----------------|--------------------|
| Uses a smooth, differentiable margin | Uses a hard hinge margin |
| Expects prediction scores | Expects distances or similarity values |
| Penalizes all mistakes smoothly | Penalizes only margin violations |
| Suitable for binary classifiers | Suitable for embedding and similarity learning |

---

## Notes

- Expects the target tensor to contain only **1** or **-1**.
- Accepts raw prediction scores (logits).
- Uses a numerically stable implementation internally.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Suitable for binary classification tasks that benefit from a smooth margin-based objective.