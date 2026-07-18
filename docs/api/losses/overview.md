# Loss Functions Overview

The `aakaar.losses` module provides a collection of commonly used loss
functions for training machine learning and deep learning models. These losses
follow a familiar **PyTorch-style API**, where each loss is implemented as a
callable class.

Most loss functions support the standard reduction modes:

- `"mean"` *(default)*
- `"sum"`
- `"none"`

Several losses also include optimized fused implementations for improved
performance on supported tensor types.

---

# Available Loss Functions

| Loss | Description |
|------|-------------|
| `L1Loss` | Computes the Mean Absolute Error (MAE) between predictions and targets. |
| `MSELoss` | Computes the Mean Squared Error (MSE) for regression tasks. |
| `HuberLoss` | Robust regression loss that combines the behavior of L1 and MSE losses. |
| `SmoothL1Loss` | Similar to Huber loss, commonly used for bounding box regression. |
| `BCELoss` | Binary Cross-Entropy loss for probability predictions. |
| `BCEWithLogitsLoss` | Numerically stable binary cross-entropy computed directly from logits. |
| `CrossEntropyLoss` | Multi-class classification loss that accepts raw logits and one-hot targets. |
| `NLLLoss` | Negative Log-Likelihood loss for log-probability inputs. |
| `KLDivLoss` | Measures divergence between two probability distributions. |
| `PoissonNLLLoss` | Negative log-likelihood loss for Poisson-distributed count data. |
| `GaussianNLLLoss` | Negative log-likelihood loss for Gaussian-distributed regression with uncertainty estimation. |
| `MarginRankingLoss` | Pairwise ranking loss for learning ordered predictions. |
| `HingeEmbeddingLoss` | Margin-based loss for similarity and embedding learning. |
| `SoftMarginLoss` | Smooth margin-based loss for binary classification. |
| `MultiLabelSoftMarginLoss` | Multi-label classification loss based on binary cross-entropy with logits. |
| `MultiMarginLoss` | Multi-class hinge loss that enforces margins between class scores. |
| `CosineEmbeddingLoss` | Learns embedding similarity using cosine distance. |
| `TripletMarginLoss` | Metric learning loss using anchor, positive, and negative samples. |

---

# Basic Usage

Most loss functions are used in the same way:

```python
import aakaar
from aakaar.losses import MSELoss

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

# Reduction Modes

Almost every loss function supports the following reduction methods:

```python
reduction="mean"
```

Returns the average loss.

```python
reduction="sum"
```

Returns the sum of all loss values.

```python
reduction="none"
```

Returns the unreduced loss for each element or sample.

---

# Classification Losses

These losses are primarily used for classification tasks.

| Loss | Typical Use |
|------|-------------|
| `CrossEntropyLoss` | Multi-class classification |
| `NLLLoss` | Classification with log-probabilities |
| `BCELoss` | Binary classification from probabilities |
| `BCEWithLogitsLoss` | Binary classification from logits |
| `MultiLabelSoftMarginLoss` | Multi-label classification |
| `SoftMarginLoss` | Binary margin-based classification |
| `MultiMarginLoss` | Multi-class margin-based classification |

---

# Regression Losses

These losses are designed for regression problems.

| Loss | Typical Use |
|------|-------------|
| `L1Loss` | Mean Absolute Error |
| `MSELoss` | Mean Squared Error |
| `HuberLoss` | Robust regression |
| `SmoothL1Loss` | Bounding box regression |
| `GaussianNLLLoss` | Probabilistic regression |
| `PoissonNLLLoss` | Count prediction |

---

# Metric Learning Losses

These losses learn relationships between embeddings rather than predicting
class labels directly.

| Loss | Typical Use |
|------|-------------|
| `CosineEmbeddingLoss` | Embedding similarity |
| `TripletMarginLoss` | Metric learning |
| `MarginRankingLoss` | Ranking problems |
| `HingeEmbeddingLoss` | Similarity learning |

---

# Important Notes

- Most loss functions support `"mean"`, `"sum"`, and `"none"` reductions.
- `CrossEntropyLoss` and `NLLLoss` currently expect **one-hot encoded targets**.
- `BCEWithLogitsLoss` expects **raw logits**, while `BCELoss` expects **probabilities**.
- `TripletMarginLoss` currently supports only **Euclidean distance (`p=2`)**.
- Some losses, such as `CrossEntropyLoss` and `HuberLoss`, automatically use optimized fused implementations when supported.
- All loss classes are stateless and can be reused throughout the training process.

---

# See Also

- `aakaar.functional`
- `aakaar.nn`
- `aakaar.optim`
- `aakaar.transforms`
- `aakaar.data`