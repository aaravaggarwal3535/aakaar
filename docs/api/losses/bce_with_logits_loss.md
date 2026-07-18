# BCEWithLogitsLoss

`BCEWithLogitsLoss` computes the **Binary Cross-Entropy (BCE)** loss directly
from **raw logits**. It combines a sigmoid activation and binary
cross-entropy into a single numerically stable operation, making it the
recommended loss for binary and multi-label classification tasks.

Unlike `BCELoss`, this loss **does not require** applying a sigmoid activation
before computing the loss.

---

## Signature

```python
BCEWithLogitsLoss(reduction='mean')
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `reduction` | `str` | Specifies the reduction to apply: `"mean"`, `"sum"`, or `"none"`. Default is `"mean"`. |

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-element losses when using `"none"`.

---

## Formula

For each prediction:

```text
Loss = max(x, 0) - x · y + log(1 + exp(-|x|))
```

where:

- **x** is the predicted logit.
- **y** is the target value (0 or 1).

This numerically stable formulation avoids overflow and underflow that can
occur when applying a sigmoid separately.

---

## Example

```python
import aakaar
from aakaar.losses import BCEWithLogitsLoss

logits = aakaar.tensor([
    [2.8],
    [-1.4],
    [0.9]
])

target = aakaar.tensor([
    [1.0],
    [0.0],
    [1.0]
])

criterion = BCEWithLogitsLoss()

loss = criterion(logits, target)
```

---

## Using Different Reductions

### Mean (default)

```python
criterion = BCEWithLogitsLoss(
    reduction="mean"
)
```

Returns the average loss across all elements.

---

### Sum

```python
criterion = BCEWithLogitsLoss(
    reduction="sum"
)
```

Returns the sum of all element-wise losses.

---

### None

```python
criterion = BCEWithLogitsLoss(
    reduction="none"
)
```

Returns the individual loss for every prediction.

---

## Typical Training Loop

```python
criterion = BCEWithLogitsLoss()

logits = model(inputs)

loss = criterion(logits, targets)

loss.backward()
optimizer.step()
```

Notice that **no sigmoid activation** is applied before computing the loss.

---

## BCEWithLogitsLoss vs BCELoss

| BCEWithLogitsLoss | BCELoss |
|-------------------|----------|
| Expects raw logits | Expects probabilities |
| Applies sigmoid internally | Requires sigmoid before the loss |
| Numerically stable | Less numerically stable |
| Recommended for most training tasks | Useful when probabilities are already available |

---

## Typical Applications

- Binary classification
- Multi-label image classification
- Medical image classification
- Recommendation systems
- Any task where each output is an independent binary prediction

---

## Notes

- Expects **raw logits**, not probabilities.
- Targets should typically contain values of **0** or **1**.
- Uses a numerically stable formulation that combines sigmoid and binary cross-entropy into a single computation.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Recommended over `BCELoss` for most training pipelines due to improved numerical stability.