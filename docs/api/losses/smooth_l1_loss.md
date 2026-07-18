# SmoothL1Loss

`SmoothL1Loss` computes a regression loss that behaves like **Mean Squared
Error (MSE)** for small prediction errors and **Mean Absolute Error (L1)** for
large prediction errors. Compared to `HuberLoss`, the transition between the
quadratic and linear regions is controlled by the `beta` parameter.

This loss is widely used in **object detection**, **bounding box regression**,
and other robust regression tasks.

---

## Signature

```python
SmoothL1Loss(
    beta=1.0,
    reduction="mean"
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `beta` | `float` | Threshold separating the quadratic and linear regions of the loss. Default is `1.0`. |
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

Then the Smooth L1 loss is

```text
           0.5 × d² / beta          if d < beta

Loss =

           d − 0.5 × beta           if d ≥ beta
```

Small errors are penalized quadratically, while larger errors are penalized
linearly.

---

## Example

```python
import aakaar
from aakaar.losses import SmoothL1Loss

prediction = aakaar.tensor([
    2.3,
    1.7,
    4.2
])

target = aakaar.tensor([
    2.0,
    2.0,
    4.0
])

criterion = SmoothL1Loss()

loss = criterion(
    prediction,
    target
)
```

---

## Using a Custom Beta

```python
criterion = SmoothL1Loss(
    beta=0.5
)
```

Smaller values of `beta` make the loss behave more like `L1Loss`, while larger
values make it closer to `MSELoss`.

---

## Using Different Reductions

### Mean (default)

```python
criterion = SmoothL1Loss(
    reduction="mean"
)
```

---

### Sum

```python
criterion = SmoothL1Loss(
    reduction="sum"
)
```

---

### None

```python
criterion = SmoothL1Loss(
    reduction="none"
)
```

Returns the loss for every prediction individually.

---

## Typical Training Loop

```python
criterion = SmoothL1Loss()

loss = criterion(
    predictions,
    targets
)

loss.backward()
optimizer.step()
```

---

## SmoothL1Loss vs HuberLoss

| SmoothL1Loss | HuberLoss |
|---------------|-----------|
| Controlled by the `beta` parameter | Controlled by the `delta` parameter |
| Commonly used for bounding box regression | General-purpose robust regression |
| Quadratic for small errors | Quadratic for small errors |
| Linear for large errors | Linear for large errors |

---

## Typical Applications

- Bounding box regression
- Object detection
- Regression
- Pose estimation
- Time-series forecasting
- Robust machine learning

---

## Notes

- Combines the advantages of `MSELoss` and `L1Loss`.
- Uses the `beta` parameter to determine when to switch from quadratic to linear loss.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Frequently used in object detection models because it is less sensitive to outliers than `MSELoss`.