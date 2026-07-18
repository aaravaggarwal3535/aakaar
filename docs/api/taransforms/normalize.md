# Normalize

`Normalize` standardizes each channel of an image using the supplied mean and
standard deviation.

Normalization is commonly used before feeding images into neural networks,
helping improve training stability and convergence. This transform expects the
input image to be in **CHW (Channels × Height × Width)** format, making it
typically used immediately after `ToTensor`.

---

## Signature

```python
Normalize(mean, std)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `mean` | list or tuple | Mean value for each channel. |
| `std` | list or tuple | Standard deviation for each channel. |

---

## Returns

A normalized NumPy array with the same shape as the input.

---

## Formula

Normalization is performed independently for every channel using:

```text
output = (input - mean) / std
```

---

## Example

```python
from aakaar.transforms import Normalize, ToTensor
from PIL import Image

image = Image.open("cat.jpg")

tensor = ToTensor()(image)

normalize = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

normalized = normalize(tensor)
```

---

## Using with Compose

```python
from aakaar.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
)

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

---

## Channel-wise Normalization

For an RGB image, each channel is normalized independently.

```python
Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

This is equivalent to:

```text
Red   = (Red   - 0.485) / 0.229
Green = (Green - 0.456) / 0.224
Blue  = (Blue  - 0.406) / 0.225
```

---

## Single-Channel Images

For grayscale images, provide a single value for both `mean` and `std`.

```python
Normalize(
    mean=[0.5],
    std=[0.5]
)
```

---

## Typical Training Pipeline

```python
from aakaar.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
)

train_transform = Compose([
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

---

## Notes

- Expects images in **CHW** format.
- Typically applied immediately after `ToTensor`.
- Performs channel-wise normalization.
- The output has the same shape and data type as the input.
- The lengths of `mean` and `std` should match the number of image channels.
```