# MSELoss

`MSELoss` computes the **Mean Squared Error (MSE)** between predicted values
and target values. It is one of the most widely used loss functions for
**regression** tasks, penalizing larger prediction errors more heavily than
smaller ones.

Because the error is squared, large mistakes contribute significantly more to
the final loss.

---

## Signature

```python
MSELoss(reduction="mean")
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
| `pred` | Predicted values. |
| `target` | Ground-truth values. |

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-element losses when using `"none"`.

---

## Formula

For each prediction:

```text
Loss = (pred − target)²
```

The final loss is computed according to the selected reduction method.

---

## Example

```python
import aakaar
from aakaar.losses import MSELoss

prediction = aakaar.tensor([
    2.4,
    1.7,
    3.9
])

target = aakaar.tensor([
    2.0,
    2.1,
    4.2
])

criterion = MSELoss()

loss = criterion(
    prediction,
    target
)
```

---

## Using Different Reductions

### Mean (default)

```python
criterion = MSELoss(
    reduction="mean"
)
```

Returns the average squared error.

---

### Sum

```python
criterion = MSELoss(
    reduction="sum"
)
```

Returns the sum of all squared errors.

---

### None

```python
criterion = MSELoss(
    reduction="none"
)
```

Returns the squared error for every element individually.

---

## Typical Training Loop

```python
criterion = MSELoss()

predictions = model(inputs)

loss = criterion(
    predictions,
    targets
)

loss.backward()
optimizer.step()
```

---

## MSELoss vs L1Loss

| MSELoss | L1Loss |
|----------|---------|
| Uses squared error | Uses absolute error |
| Strongly penalizes large errors | Penalizes errors linearly |
| More sensitive to outliers | More robust to outliers |
| Smooth and differentiable everywhere | Less sensitive to extreme values |

---

## Typical Applications

- Linear regression
- Neural network regression
- Time-series forecasting
- Function approximation
- Image reconstruction
- Scientific computing

---

## Notes

- Computes the squared difference between predictions and targets.
- Large prediction errors contribute disproportionately to the final loss.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Commonly used as the default loss for regression problems.
- Suitable when large prediction errors should be penalized more heavily than small ones.