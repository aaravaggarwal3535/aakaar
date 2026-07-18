# GaussianBlur

`GaussianBlur` applies a Gaussian blur to an image using a **separable Gaussian
convolution** implemented entirely with NumPy. Blurring reduces high-frequency
image details, making models more robust to image noise, slight focus changes,
and texture variations.

The transform randomly samples the Gaussian standard deviation (`sigma`) within
the specified range for each image.

---

## Signature

```python
GaussianBlur(
    kernel_size,
    sigma=(0.1, 2.0)
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `kernel_size` | `int` | Size of the Gaussian kernel. If an even value is provided, it is automatically increased to the next odd number. |
| `sigma` | `float` or `tuple` | Standard deviation of the Gaussian kernel. If a tuple is provided, a random value is sampled from the specified range. Default is `(0.1, 2.0)`. |

---

## Returns

A blurred NumPy array with the same shape as the input image.

- `uint8` inputs produce `uint8` outputs.
- Floating-point inputs produce floating-point outputs.

---

## Example

```python
from PIL import Image
from aakaar.transforms import GaussianBlur

image = Image.open("cat.jpg")

transform = GaussianBlur(
    kernel_size=5
)

blurred = transform(image)
```

---

## Fixed Blur Strength

Use a fixed sigma value for consistent blur.

```python
transform = GaussianBlur(
    kernel_size=7,
    sigma=1.5
)
```

---

## Random Blur Strength

Randomly sample the blur intensity for every image.

```python
transform = GaussianBlur(
    kernel_size=5,
    sigma=(0.5, 2.0)
)
```

---

## Using with Compose

```python
from aakaar.transforms import (
    Compose,
    RandomResizedCrop,
    GaussianBlur,
    ToTensor,
    Normalize,
)

transform = Compose([
    RandomResizedCrop(224),
    GaussianBlur(
        kernel_size=5,
        sigma=(0.1, 2.0)
    ),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

---

## How It Works

For each image:

1. A Gaussian standard deviation (`sigma`) is randomly sampled.
2. A one-dimensional Gaussian kernel is generated.
3. Horizontal convolution is performed.
4. Vertical convolution is performed.
5. The blurred image is clipped to the valid pixel range and returned.

Because the implementation uses **separable convolution**, it is significantly
more efficient than applying a full two-dimensional Gaussian kernel directly.

---

## Typical Use Cases

- Image classification
- Self-supervised learning
- Contrastive learning
- Noise reduction
- Data augmentation
- Preprocessing before feature extraction

---

## Notes

- Accepts both **PIL Images** and **NumPy arrays**.
- Returns a NumPy array.
- Implemented entirely using NumPy without external dependencies such as SciPy.
- Automatically converts even kernel sizes to the next odd number.
- Supports both grayscale and RGB images.
- Preserves the input data type (`uint8` or floating point).
- A different blur strength may be applied to each image when a sigma range is specified.
- Typically used during training and omitted during validation or inference.