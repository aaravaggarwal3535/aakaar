# RandomResizedCrop

`RandomResizedCrop` randomly selects a region of an image, crops it, and then
resizes the cropped region to a fixed output size. The crop area and aspect
ratio are sampled randomly, making this transform one of the most widely used
data augmentation techniques for image classification models.

If a valid random crop cannot be found after several attempts, a centered
square crop is used as a fallback.

---

## Signature

```python
RandomResizedCrop(
    size,
    scale=(0.08, 1.0),
    ratio=(3/4, 4/3)
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `size` | `int` or `tuple` | Output image size. An integer creates a square output, while a tuple specifies `(width, height)`. |
| `scale` | `tuple` | Range for the proportion of the original image area to crop. Default is `(0.08, 1.0)`. |
| `ratio` | `tuple` | Range of aspect ratios used when sampling the crop. Default is `(3/4, 4/3)`. |

---

## Returns

A resized NumPy array.

---

## Example

```python
from PIL import Image
from aakaar.transforms import RandomResizedCrop

image = Image.open("cat.jpg")

transform = RandomResizedCrop(224)

cropped = transform(image)
```

---

## Custom Scale

The `scale` parameter controls how much of the original image is retained.

```python
transform = RandomResizedCrop(
    size=224,
    scale=(0.5, 1.0)
)
```

This randomly crops between **50%** and **100%** of the original image area.

---

## Custom Aspect Ratio

The `ratio` parameter controls the shape of the sampled crop.

```python
transform = RandomResizedCrop(
    size=224,
    ratio=(0.9, 1.1)
)
```

Smaller ranges produce crops that are closer to square, while larger ranges
allow more rectangular crops.

---

## Using with Compose

```python
from aakaar.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    ToTensor,
    Normalize,
)

transform = Compose([
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
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

1. A random crop area is selected according to `scale`.
2. A random aspect ratio is sampled from `ratio`.
3. A valid crop is extracted from the image.
4. The crop is resized to the requested output size.
5. If no valid crop is found after multiple attempts, a centered square crop is used instead.

---

## Typical Training Pipeline

```text
Input Image
      │
      ▼
Random Crop
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

- Accepts both **PIL Images** and **NumPy arrays**.
- Returns a resized NumPy array.
- Uses bilinear interpolation for resizing.
- Randomly varies both crop area and aspect ratio.
- Falls back to a centered square crop if a valid random crop cannot be generated.
- Commonly used as the first augmentation step in image classification training pipelines.