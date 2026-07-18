# Resize

`Resize` resizes an input image to a specified height and width. It operates on
**PIL Images** and is typically used as one of the first steps in an image
preprocessing pipeline.

This transform is useful for ensuring that all images have a consistent size
before being converted into tensors or passed to a neural network.

---

## Signature

```python
Resize(size)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `size` | `int` or `tuple` | Target output size. If an integer is provided, the image is resized to `(size, size)`. If a tuple is provided, it should be `(width, height)`. |

---

## Returns

A resized **PIL Image**.

---

## Example

```python
from PIL import Image
from aakaar.transforms import Resize

image = Image.open("cat.jpg")

transform = Resize((224, 224))

resized = transform(image)
```

---

## Square Resize

Providing a single integer creates a square image.

```python
transform = Resize(224)
```

Equivalent to:

```python
transform = Resize((224, 224))
```

---

## Rectangular Resize

A tuple can be used to specify custom dimensions.

```python
transform = Resize((320, 256))
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

## Typical Workflow

```
Original Image
      │
      ▼
Resize
      │
      ▼
ToTensor
      │
      ▼
Normalize
      │
      ▼
Model Input
```

---

## Notes

- Operates on **PIL Images**.
- Uses Pillow's built-in `resize()` method.
- Returns a resized image with the specified dimensions.
- When an integer is provided, both width and height are set to the same value.
- Typically placed before `ToTensor` in a preprocessing pipeline.
```