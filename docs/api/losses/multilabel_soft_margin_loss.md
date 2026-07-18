# MultiLabelSoftMarginLoss

`MultiLabelSoftMarginLoss` computes the binary cross-entropy loss with logits
for **multi-label classification** tasks. Unlike multi-class classification,
where each sample belongs to exactly one class, multi-label classification
allows each sample to belong to **multiple classes simultaneously**.

Internally, this loss computes `BCEWithLogitsLoss` for each class and then
averages the loss across all classes for every sample.

---

## Signature

```python
MultiLabelSoftMarginLoss(reduction="mean")
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
| `logits` | Raw prediction scores (logits) for each class. |
| `target` | Target tensor containing binary labels (`0` or `1`) for each class. |

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-sample losses when using `"none"`.

---

## Formula

For each class, the loss is computed using binary cross-entropy with logits:

```text
BCE = max(x, 0) - x · y + log(1 + exp(-|x|))
```

where:

- `x` is the predicted logit.
- `y` is the target label (`0` or `1`).

The final loss for each sample is the average BCE across all classes.

---

## Example

```python
import aakaar
from aakaar.losses import MultiLabelSoftMarginLoss

logits = aakaar.tensor([
    [2.5, -1.3, 0.8],
    [-0.6, 1.9, 2.4]
])

targets = aakaar.tensor([
    [1, 0, 1],
    [0, 1, 1]
])

criterion = MultiLabelSoftMarginLoss()

loss = criterion(
    logits,
    targets
)
```

---

## Using Different Reductions

### Mean (default)

```python
criterion = MultiLabelSoftMarginLoss(
    reduction="mean"
)
```

Returns the average loss across the batch.

---

### Sum

```python
criterion = MultiLabelSoftMarginLoss(
    reduction="sum"
)
```

Returns the sum of all sample losses.

---

### None

```python
criterion = MultiLabelSoftMarginLoss(
    reduction="none"
)
```

Returns the average loss for each sample individually.

---

## Typical Training Loop

```python
criterion = MultiLabelSoftMarginLoss()

logits = model(inputs)

loss = criterion(
    logits,
    targets
)

loss.backward()
optimizer.step()
```

---

## Typical Applications

- Multi-label image classification
- Image tagging
- Text classification with multiple labels
- Medical diagnosis
- Recommendation systems
- Audio event detection

---

## MultiLabelSoftMarginLoss vs BCEWithLogitsLoss

| MultiLabelSoftMarginLoss | BCEWithLogitsLoss |
|--------------------------|-------------------|
| Designed for multi-label classification | General binary classification loss |
| Computes BCE for every class | Computes BCE element-wise |
| Averages the loss across classes | Returns element-wise loss before reduction |
| Uses raw logits | Uses raw logits |

---

## Notes

- Expects **raw logits**, not probabilities.
- Target values should typically be **0** or **1** for each class.
- Internally uses `BCEWithLogitsLoss` with `reduction="none"` before averaging across classes.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Suitable when each sample may belong to multiple classes simultaneously.