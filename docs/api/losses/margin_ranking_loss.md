# MarginRankingLoss

`MarginRankingLoss` measures whether one input should be ranked higher than
another by at least a specified margin. It is commonly used in **ranking
problems**, **recommendation systems**, **learning to rank**, and **metric
learning**.

The target tensor specifies which input should receive the higher score.

---

## Signature

```python
MarginRankingLoss(
    margin=0.0,
    reduction="mean"
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `margin` | `float` | Minimum desired difference between the two inputs. Default is `0.0`. |
| `reduction` | `str` | Specifies the reduction to apply: `"mean"`, `"sum"`, or `"none"`. Default is `"mean"`. |

---

## Inputs

| Input | Description |
|-------|-------------|
| `x1` | First prediction tensor. |
| `x2` | Second prediction tensor. |
| `y` | Target tensor containing only `1` or `-1`. If `y = 1`, `x1` should rank higher than `x2`. If `y = -1`, `x2` should rank higher than `x1`. |

---

## Returns

A scalar loss when using `"mean"` or `"sum"`, or a tensor containing the
per-element losses when using `"none"`.

---

## Formula

For each pair:

```text
Loss = max(0, -y × (x1 - x2) + margin)
```

where:

- `y = 1` means `x1` should be greater than `x2`.
- `y = -1` means `x2` should be greater than `x1`.

If the desired ranking already satisfies the specified margin, the loss is
zero.

---

## Example

```python
import aakaar
from aakaar.losses import MarginRankingLoss

x1 = aakaar.tensor([
    2.3,
    0.8,
    3.4
])

x2 = aakaar.tensor([
    1.5,
    1.2,
    2.8
])

target = aakaar.tensor([
    1,
    -1,
    1
])

criterion = MarginRankingLoss(
    margin=0.5
)

loss = criterion(
    x1,
    x2,
    target
)
```

---

## Using a Custom Margin

```python
criterion = MarginRankingLoss(
    margin=1.0
)
```

Larger margins require a greater separation between the two inputs before the
loss becomes zero.

---

## Using Different Reductions

### Mean (default)

```python
criterion = MarginRankingLoss(
    reduction="mean"
)
```

---

### Sum

```python
criterion = MarginRankingLoss(
    reduction="sum"
)
```

---

### None

```python
criterion = MarginRankingLoss(
    reduction="none"
)
```

Returns the loss for each input pair individually.

---

## Typical Training Loop

```python
criterion = MarginRankingLoss(
    margin=0.5
)

loss = criterion(
    scores_a,
    scores_b,
    labels
)

loss.backward()
optimizer.step()
```

---

## Typical Applications

- Learning to rank
- Recommendation systems
- Information retrieval
- Metric learning
- Preference learning
- Pairwise ranking models

---

## Notes

- Expects the target tensor to contain only **1** or **-1**.
- A target of `1` indicates that `x1` should rank higher than `x2`.
- A target of `-1` indicates that `x2` should rank higher than `x1`.
- Supports `"mean"`, `"sum"`, and `"none"` reductions.
- Returns zero loss when the desired ranking satisfies the specified margin.
- Commonly used for pairwise ranking and preference-learning tasks.