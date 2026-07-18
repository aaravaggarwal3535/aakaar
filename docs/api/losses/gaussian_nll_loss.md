# GaussianNLLLoss

`GaussianNLLLoss` computes the **Negative Log-Likelihood (NLL)** assuming the
target values follow a Gaussian (normal) distribution. Unlike losses such as
`MSELoss`, it allows the model to predict both the **mean** and the
**uncertainty (variance)** of each prediction.

This loss is commonly used in **probabilistic regression**, **uncertainty
estimation**, and Bayesian deep learning.

---

## Signature

```python
GaussianNLLLoss(
    full=False,
    eps=1e-6,
    reduction="mean"
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `full` | `bool` | Whether to include the constant term \( \frac{1}{2}\log(2\pi) \) in the loss. Default is `False`. |
| `eps` | `float` | Small value added to the predicted variance for numerical stability. Default is `1e-6`. |
| `reduction` | `str` | Specifies the reduction to apply: `"mean"`, `"sum"`, or `"none"`. Default is `"mean"`. |

---

## Inputs

| Input | Description |
|-------|-------------|
| `pred` | Predicted mean values. |
| `target` | Ground-truth values. |
| `var` | Predicted variance for each element. Must be positive. |

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-element losses when using `"none"`.

---

## Formula

For each prediction:

```text
Loss = 0.5 × (log(var) + (pred - target)² / var)
```

When `full=True`, the following constant term is also added:

```text
0.5 × log(2π)
```

Internally, Aakaar computes:

```text
var_clamped = var + eps
```

to improve numerical stability.

---

## Example

```python
import aakaar
from aakaar.losses import GaussianNLLLoss

pred = aakaar.tensor([2.5, 1.8, 4.1])

target = aakaar.tensor([2.8, 2.0, 3.9])

variance = aakaar.tensor([0.4, 0.2, 0.6])

criterion = GaussianNLLLoss()

loss = criterion(
    pred,
    target,
    variance
)
```

---

## Including the Constant Term

```python
criterion = GaussianNLLLoss(
    full=True
)
```

This produces the complete Gaussian negative log-likelihood.

---

## Using Different Reductions

### Mean (default)

```python
criterion = GaussianNLLLoss(
    reduction="mean"
)
```

---

### Sum

```python
criterion = GaussianNLLLoss(
    reduction="sum"
)
```

---

### None

```python
criterion = GaussianNLLLoss(
    reduction="none"
)
```

Returns the loss for every prediction individually.

---

## Typical Training Loop

```python
criterion = GaussianNLLLoss()

loss = criterion(
    predicted_mean,
    targets,
    predicted_variance
)

loss.backward()
optimizer.step()
```

---

## Typical Applications

- Probabilistic regression
- Uncertainty estimation
- Bayesian neural networks
- Time-series forecasting
- Scientific prediction
- Autonomous systems

---

## Notes

- Requires three inputs: predicted mean, target values, and predicted variance.
- The predicted variance should always be positive.
- A small `eps` value is added internally to the variance for numerical stability.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Setting `full=True` includes the constant Gaussian normalization term.
- Particularly useful when predicting both values and their associated uncertainty.