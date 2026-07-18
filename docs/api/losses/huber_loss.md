# HuberLoss

`HuberLoss` computes a regression loss that combines the advantages of
**Mean Squared Error (MSE)** and **Mean Absolute Error (L1)**. Small prediction
errors are treated quadratically, while larger errors are treated linearly,
making the loss less sensitive to outliers than `MSELoss`.

This makes `HuberLoss` a popular choice for robust regression problems.

For the common case of **float32 tensors** with `"mean"` reduction, Aakaar
automatically uses an optimized fused implementation for improved performance.

---

## Signature

```python
HuberLoss(
    delta=1.0,
    reduction="mean"
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `delta` | `float` | Threshold separating the quadratic and linear regions of the loss. Default is `1.0`. |
| `reduction` | `str` | Specifies the reduction to apply: `"mean"`, `"sum"`, or `"none"`. Default is `"mean"`. |

---

## Inputs

| Input | Description |
|-------|-------------|
| `pred` | Predicted values. |
| `target` | Ground-truth values. |

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-element losses when using `"none"`.

---

## Formula

Let

```text
d = |pred − target|
```

Then the Huber loss is

```text
           0.5 × d²                      if d ≤ delta

Loss =

           delta × (d − 0.5 × delta)     if d > delta
```

Small errors behave like **MSE**, while large errors behave like **L1** loss.

---

## Example

```python
import aakaar
from aakaar.losses import HuberLoss

prediction = aakaar.tensor([
    2.4,
    1.2,
    3.8
])

target = aakaar.tensor([
    2.0,
    1.5,
    4.1
])

criterion = HuberLoss()

loss = criterion(
    prediction,
    target
)
```

---

## Using a Custom Delta

```python
criterion = HuberLoss(
    delta=2.0
)
```

Larger values of `delta` make the loss behave more like `MSELoss`, while
smaller values make it behave more like `L1Loss`.

---

## Using Different Reductions

### Mean (default)

```python
criterion = HuberLoss(
    reduction="mean"
)
```

---

### Sum

```python
criterion = HuberLoss(
    reduction="sum"
)
```

---

### None

```python
criterion = HuberLoss(
    reduction="none"
)
```

Returns the loss for every prediction individually.

---

## Performance Optimization

For the following configuration:

- `float32` predictions
- `float32` targets
- `"mean"` reduction

Aakaar automatically dispatches to an optimized fused implementation:

```text
aakaar._C._huber_loss_fused(...)
```

For other data types or reduction modes, the standard implementation is used
automatically.

---

## HuberLoss vs MSELoss vs L1Loss

| HuberLoss | MSELoss | L1Loss |
|------------|---------|---------|
| Robust to outliers | Sensitive to outliers | Very robust to outliers |
| Quadratic for small errors | Quadratic everywhere | Linear everywhere |
| Linear for large errors | Large errors dominate | Constant gradient |
| Good balance of stability and robustness | Best for clean data | Best when many outliers exist |

---

## Typical Applications

- Regression
- Object detection (bounding box regression)
- Time-series forecasting
- Reinforcement learning
- Robust machine learning models

---

## Notes

- Combines the behavior of `MSELoss` and `L1Loss`.
- Uses the `delta` parameter to determine when to switch from quadratic to linear loss.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Automatically uses an optimized fused implementation for supported `float32` inputs with `"mean"` reduction.
- Recommended for regression tasks where occasional outliers are expected.