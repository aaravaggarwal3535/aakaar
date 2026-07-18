# PoissonNLLLoss

`PoissonNLLLoss` computes the **Negative Log-Likelihood (NLL)** assuming the
target values follow a **Poisson distribution**. It is commonly used for
predicting **count data**, where the target represents the number of events
occurring within a fixed interval.

Typical applications include event counting, object counting, medical
statistics, and other regression tasks involving non-negative integer counts.

---

## Signature

```python
PoissonNLLLoss(
    log_input=True,
    full=False,
    eps=1e-8,
    reduction="mean"
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `log_input` | `bool` | If `True`, the input is assumed to contain the logarithm of the predicted rate. If `False`, the input is treated as the predicted rate itself. Default is `True`. |
| `full` | `bool` | Whether to include Stirling's approximation for the factorial term. Default is `False`. |
| `eps` | `float` | Small constant added for numerical stability when `log_input=False`. Default is `1e-8`. |
| `reduction` | `str` | Specifies the reduction to apply: `"mean"`, `"sum"`, or `"none"`. Default is `"mean"`. |

---

## Inputs

| Input | Description |
|-------|-------------|
| `pred` | Predicted log-rate or predicted rate, depending on `log_input`. |
| `target` | Ground-truth count values. |

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-element losses when using `"none"`.

---

## Formula

When `log_input=True`:

```text
Loss = exp(pred) − target × pred
```

When `log_input=False`:

```text
Loss = pred − target × log(pred + eps)
```

If `full=True`, Stirling's approximation is added:

```text
target × log(target)
− target
+ 0.5 × log(2π × target)
```

This improves the approximation for larger count values.

---

## Example

```python
import aakaar
from aakaar.losses import PoissonNLLLoss

pred = aakaar.tensor([
    0.8,
    1.3,
    0.4
])

target = aakaar.tensor([
    2.0,
    4.0,
    1.0
])

criterion = PoissonNLLLoss()

loss = criterion(
    pred,
    target
)
```

---

## Using Predicted Rates Instead of Log-Rates

```python
criterion = PoissonNLLLoss(
    log_input=False
)
```

In this mode, the input tensor should contain predicted Poisson rates rather
than their logarithms.

---

## Including Stirling's Approximation

```python
criterion = PoissonNLLLoss(
    full=True
)
```

This produces a more complete approximation of the Poisson negative
log-likelihood, especially for larger target values.

---

## Using Different Reductions

### Mean (default)

```python
criterion = PoissonNLLLoss(
    reduction="mean"
)
```

---

### Sum

```python
criterion = PoissonNLLLoss(
    reduction="sum"
)
```

---

### None

```python
criterion = PoissonNLLLoss(
    reduction="none"
)
```

Returns the loss for every prediction individually.

---

## Typical Training Loop

```python
criterion = PoissonNLLLoss()

loss = criterion(
    predictions,
    targets
)

loss.backward()
optimizer.step()
```

---

## Typical Applications

- Object counting
- Crowd counting
- Traffic flow prediction
- Medical event prediction
- Insurance claim modeling
- Count-based regression

---

## Notes

- Designed for regression tasks where the target represents **count data**.
- Supports inputs as either **log-rates** (`log_input=True`) or **rates** (`log_input=False`).
- `full=True` adds Stirling's approximation to better approximate the full Poisson log-likelihood.
- Uses a small `eps` value internally for numerical stability when computing logarithms.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.