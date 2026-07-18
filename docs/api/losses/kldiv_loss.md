# KLDivLoss

`KLDivLoss` computes the **Kullback-Leibler (KL) Divergence** between two
probability distributions. It measures how much one probability distribution
differs from another and is commonly used in **knowledge distillation**,
**variational autoencoders (VAEs)**, **language modeling**, and other
probabilistic learning tasks.

The input is expected to contain **log-probabilities**, while the target
contains probabilities by default.

---

## Signature

```python
KLDivLoss(
    reduction="mean",
    log_target=False
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `reduction` | `str` | Specifies the reduction to apply: `"mean"`, `"sum"`, `"batchmean"`, or `"none"`. Default is `"mean"`. |
| `log_target` | `bool` | If `True`, the target is assumed to contain log-probabilities instead of probabilities. Default is `False`. |

---

## Inputs

| Input | Description |
|-------|-------------|
| `input_log_probs` | Input tensor containing log-probabilities (typically produced by `log_softmax`). |
| `target` | Target probability distribution, or log-probabilities if `log_target=True`. |

---

## Returns

A scalar loss when using `"mean"`, `"sum"`, or `"batchmean"`, or a tensor
containing the per-element losses when using `"none"`.

---

## Formula

When `log_target=False` (default):

```text
Loss = target × (log(target) − input_log_probs)
```

When `log_target=True`:

```text
Loss = exp(target) × (target − input_log_probs)
```

For `"batchmean"` reduction, the summed loss is divided by the batch size.

---

## Example

```python
import aakaar
from aakaar import log_softmax
from aakaar.losses import KLDivLoss

logits = aakaar.rand((4, 10))

input_log_probs = log_softmax(
    logits,
    dim=-1
)

target = aakaar.rand((4, 10))
target = target / target.sum(dim=-1, keepdim=True)

criterion = KLDivLoss()

loss = criterion(
    input_log_probs,
    target
)
```

---

## Using Log Targets

If the target already contains log-probabilities:

```python
criterion = KLDivLoss(
    log_target=True
)

loss = criterion(
    input_log_probs,
    target_log_probs
)
```

---

## Using Different Reductions

### Mean (default)

```python
criterion = KLDivLoss(
    reduction="mean"
)
```

---

### Batch Mean

```python
criterion = KLDivLoss(
    reduction="batchmean"
)
```

Returns the summed divergence divided by the batch size.

---

### Sum

```python
criterion = KLDivLoss(
    reduction="sum"
)
```

---

### None

```python
criterion = KLDivLoss(
    reduction="none"
)
```

Returns the divergence for every element.

---

## Typical Training Loop

```python
criterion = KLDivLoss()

loss = criterion(
    input_log_probs,
    target_distribution
)

loss.backward()
optimizer.step()
```

---

## Typical Applications

- Knowledge distillation
- Variational Autoencoders (VAEs)
- Language modeling
- Distribution matching
- Reinforcement learning
- Generative models

---

## Notes

- Expects the input tensor to contain **log-probabilities**.
- By default, the target tensor should contain **probabilities**.
- Set `log_target=True` if the target already contains log-probabilities.
- Supports `"mean"`, `"sum"`, `"batchmean"`, and `"none"` reductions.
- A small epsilon is used internally when computing `log(target)` for improved numerical stability.
- Often used together with `log_softmax` when training probabilistic models.