# RandomCrop

`RandomCrop` extracts a random region from an image. If optional padding is
specified, the image is padded before cropping, allowing crops to include
pixels beyond the original image boundaries.

Random cropping is one of the most common data augmentation techniques for
computer vision models, helping improve robustness by exposing the model to
different portions of the same image during training.

---

## Signature

```python
RandomCrop(size, padding=0)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `size` | `int` or `tuple` | Output crop size. An integer creates a square crop `(size, size)`, while a tuple specifies `(height, width)`. |
| `padding` | `int` | Optional padding applied equally to all sides of the image before cropping. Default is `0`. |

---

## Returns

A randomly cropped NumPy array.

---

## Example

```python
from PIL import Image
from aakaar.transforms import RandomCrop

image = Image.open("cat.jpg")

transform = RandomCrop(224)

cropped = transform(image)
```

---

## Using Padding

Padding expands the image before selecting the crop.

```python
transform = RandomCrop(
    size=224,
    padding=16
)
```

This is useful when training on small images, allowing greater variation in
the sampled crops.

---

## Rectangular Crops

Custom crop dimensions can be specified using a tuple.

```python
transform = RandomCrop((224, 320))
```

This creates crops with:

- Height = **224**
- Width = **320**

---

## Using with Compose

```python
from aakaar.transforms import (
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
)

transform = Compose([
    RandomCrop(224, padding=8),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

---

## Crop Selection

For each image, a random top-left corner is selected.

```text
Original Image
+---------------------------+
|                           |
|      +-------------+      |
|      |             |      |
|      | Random Crop |      |
|      |             |      |
|      +-------------+      |
|                           |
+---------------------------+
```

Each call to the transform may produce a different crop.

---

## Errors

A `ValueError` is raised if the requested crop size is larger than the image
(after padding has been applied).

```python
RandomCrop((512, 512))
```

Example error:

```text
ValueError:
RandomCrop size (512, 512) larger than (padded) image (224, 224)
```

---

## Notes

- Accepts both **PIL Images** and **NumPy arrays**.
- Returns a NumPy array containing the cropped region.
- Padding uses NumPy's `"edge"` mode, which extends the border pixels.
- Each invocation generates a new random crop.
- Commonly used as a training-time augmentation and typically placed before `ToTensor`.