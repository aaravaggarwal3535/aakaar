# L1Loss

`L1Loss` computes the **Mean Absolute Error (MAE)** between predicted values
and target values. It measures the average absolute difference between
predictions and ground-truth values, making it less sensitive to outliers than
`MSELoss`.

Because the error grows linearly, large prediction errors do not dominate the
loss as strongly as they do with squared-error losses.

---

## Signature

```python
L1Loss(reduction="mean")
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
Loss = |pred − target|
```

The final loss is obtained according to the selected reduction method.

---

## Example

```python
import aakaar
from aakaar.losses import L1Loss

prediction = aakaar.tensor([
    2.5,
    1.8,
    4.1
])

target = aakaar.tensor([
    2.0,
    2.0,
    3.8
])

criterion = L1Loss()

loss = criterion(
    prediction,
    target
)
```

---

## Using Different Reductions

### Mean (default)

```python
criterion = L1Loss(
    reduction="mean"
)
```

Returns the average absolute error.

---

### Sum

```python
criterion = L1Loss(
    reduction="sum"
)
```

Returns the sum of all absolute errors.

---

### None

```python
criterion = L1Loss(
    reduction="none"
)
```

Returns the absolute error for every element individually.

---

## Typical Training Loop

```python
criterion = L1Loss()

predictions = model(inputs)

loss = criterion(
    predictions,
    targets
)

loss.backward()
optimizer.step()
```

---

## L1Loss vs MSELoss

| L1Loss | MSELoss |
|---------|----------|
| Uses absolute error | Uses squared error |
| More robust to outliers | More sensitive to outliers |
| Linear error growth | Quadratic error growth |
| Constant gradient (except at zero) | Larger gradients for larger errors |

---

## Typical Applications

- Regression
- Time-series forecasting
- Image restoration
- Signal processing
- Robust machine learning models

---

## Notes

- Computes the absolute difference between predictions and targets.
- Less sensitive to outliers than `MSELoss`.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Suitable for regression tasks where robustness to occasional large errors is desired.
- Uses the tensor `abs()` operation internally.