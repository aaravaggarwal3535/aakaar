# NLLLoss

`NLLLoss` computes the **Negative Log-Likelihood (NLL)** loss for
classification tasks. It expects the input to contain **log-probabilities**
rather than raw logits.

In Aakaar, this loss expects **one-hot encoded targets** instead of integer
class indices. It is commonly used together with `log_softmax` or as the
backend for `CrossEntropyLoss`.

---

## Signature

```python
NLLLoss(reduction="mean")
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
| `log_probs` | Input tensor containing log-probabilities (typically produced by `log_softmax`). |
| `target_onehot` | One-hot encoded target tensor with the same shape as `log_probs`. |

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-sample losses when using `"none"`.

---

## Formula

For each sample:

```text
Loss = -Σ(target × log_probs)
```

Since the target is one-hot encoded, this effectively selects the
log-probability of the correct class and negates it.

---

## Example

```python
import aakaar
from aakaar import log_softmax
from aakaar.losses import NLLLoss

logits = aakaar.tensor([
    [2.3, 0.5, -1.1],
    [0.2, 1.9, 0.8]
])

log_probs = log_softmax(
    logits,
    dim=-1
)

targets = aakaar.tensor([
    [1, 0, 0],
    [0, 1, 0]
])

criterion = NLLLoss()

loss = criterion(
    log_probs,
    targets
)
```

---

## Using Different Reductions

### Mean (default)

```python
criterion = NLLLoss(
    reduction="mean"
)
```

Returns the average loss across the batch.

---

### Sum

```python
criterion = NLLLoss(
    reduction="sum"
)
```

Returns the sum of all sample losses.

---

### None

```python
criterion = NLLLoss(
    reduction="none"
)
```

Returns the loss for each sample individually.

---

## Typical Training Loop

```python
criterion = NLLLoss()

log_probs = aakaar.log_softmax(
    logits,
    dim=-1
)

loss = criterion(
    log_probs,
    targets
)

loss.backward()
optimizer.step()
```

---

## NLLLoss vs CrossEntropyLoss

| NLLLoss | CrossEntropyLoss |
|----------|------------------|
| Expects log-probabilities | Expects raw logits |
| Requires `log_softmax` beforehand | Computes `log_softmax` internally |
| More flexible when log-probabilities are already available | Simpler for most classification tasks |
| Backend used by `CrossEntropyLoss` | Recommended for general classification |

---

## Notes

- Expects **log-probabilities**, not raw logits.
- Targets must be **one-hot encoded** in the current version of Aakaar.
- Commonly used after `log_softmax`.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Serves as the underlying implementation used by `CrossEntropyLoss` when the fused path is not applicable.