# CrossEntropyLoss

`CrossEntropyLoss` computes the cross-entropy loss between predicted logits and
target labels. It is the most commonly used loss function for **multi-class
classification** problems.

In Aakaar, this loss expects **one-hot encoded targets** rather than integer
class indices. Internally, it computes the log-softmax of the logits followed
by the negative log-likelihood loss.

For the common case of **2D `float32` tensors** with `"mean"` reduction,
Aakaar automatically uses an optimized fused implementation for improved
performance.

---

## Signature

```python
CrossEntropyLoss(reduction="mean")
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
| `logits` | Raw model outputs before softmax. Shape is typically `(batch_size, num_classes)`. |
| `target_onehot` | One-hot encoded target tensor with the same shape as `logits`. |

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-sample losses when using `"none"`.

---

## Formula

Cross entropy is computed as:

```text
Loss = -Σ(target × log_softmax(logits))
```

When using `"mean"` reduction, the average loss across the batch is returned.

---

## Example

```python
import aakaar
from aakaar.losses import CrossEntropyLoss

logits = aakaar.tensor([
    [2.4, 0.3, -1.2],
    [0.1, 1.8, 0.7]
])

targets = aakaar.tensor([
    [1, 0, 0],
    [0, 1, 0]
])

criterion = CrossEntropyLoss()

loss = criterion(logits, targets)
```

---

## Using Different Reductions

### Mean (default)

```python
criterion = CrossEntropyLoss(
    reduction="mean"
)
```

Returns the average loss across the batch.

---

### Sum

```python
criterion = CrossEntropyLoss(
    reduction="sum"
)
```

Returns the sum of all sample losses.

---

### None

```python
criterion = CrossEntropyLoss(
    reduction="none"
)
```

Returns the loss for each sample individually.

---

## Typical Training Loop

```python
criterion = CrossEntropyLoss()

logits = model(inputs)

loss = criterion(
    logits,
    targets
)

loss.backward()
optimizer.step()
```

---

## Performance Optimization

For the following configuration:

- `float32` logits
- `float32` one-hot targets
- 2D tensors
- `"mean"` reduction

Aakaar automatically dispatches to an optimized fused implementation:

```text
aakaar._C._cross_entropy_fused(...)
```

For other tensor shapes, data types, or reduction modes, the implementation
falls back to:

```text
log_softmax → NLLLoss
```

This behavior is automatic and requires no changes to user code.

---

## CrossEntropyLoss vs NLLLoss

| CrossEntropyLoss | NLLLoss |
|------------------|----------|
| Accepts raw logits | Accepts log-probabilities |
| Computes `log_softmax` internally | Requires `log_softmax` beforehand |
| Easier to use | Provides more control over preprocessing |
| Recommended for most classification tasks | Useful when log-probabilities are already available |

---

## Notes

- Expects **raw logits**, not probabilities.
- Targets must be **one-hot encoded** in the current version of Aakaar.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Automatically uses a fused implementation when possible for improved performance.
- Commonly used for image classification, text classification, and other multi-class learning tasks.