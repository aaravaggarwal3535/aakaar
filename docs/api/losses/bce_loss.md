# BCELoss

`BCELoss` computes the **Binary Cross-Entropy (BCE)** loss between predicted
probabilities and target labels. It is commonly used for **binary
classification** and **multi-label classification** tasks where the model
outputs probabilities in the range **[0, 1]**.

Unlike `BCEWithLogitsLoss`, this loss expects the input to already be
probabilities (typically produced by a sigmoid activation).

---

## Signature

```python
BCELoss(reduction='mean')
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
Loss = -(y · log(p) + (1 - y) · log(1 - p))
```

where:

- **p** is the predicted probability.
- **y** is the target value (0 or 1).

---

## Example

```python
import aakaar
from aakaar.losses import BCELoss

pred = aakaar.tensor([
    [0.95],
    [0.12],
    [0.81]
])

target = aakaar.tensor([
    [1.0],
    [0.0],
    [1.0]
])

criterion = BCELoss()

loss = criterion(pred, target)
```

---

## Using Different Reductions

### Mean (default)

```python
criterion = BCELoss(reduction="mean")
```

Returns the average loss across all elements.

---

### Sum

```python
criterion = BCELoss(reduction="sum")
```

Returns the sum of all element-wise losses.

---

### None

```python
criterion = BCELoss(reduction="none")
```

Returns the individual loss for every element.

---

## Typical Training Loop

```python
criterion = BCELoss()

predictions = model(inputs)

loss = criterion(predictions, targets)

loss.backward()
optimizer.step()
```

The model should output **probabilities**, typically by applying a sigmoid
activation before computing the loss.

---

## BCELoss vs BCEWithLogitsLoss

| BCELoss | BCEWithLogitsLoss |
|---------|-------------------|
| Expects probabilities | Expects raw logits |
| Requires sigmoid before the loss | Sigmoid is applied internally |
| Less numerically stable | More numerically stable |
| Simpler when probabilities are already available | Recommended for most training pipelines |

---

## Notes

- Expects predicted values in the range **[0, 1]**.
- Targets should typically contain values of **0** or **1**.
- Uses a small epsilon internally to improve numerical stability when computing logarithms.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- For models that output raw logits, prefer using `BCEWithLogitsLoss` instead.