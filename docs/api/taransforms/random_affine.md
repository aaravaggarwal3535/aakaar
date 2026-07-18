# RandomAffine

`RandomAffine` applies a random affine transformation to an image. An affine
transformation can combine **rotation**, **translation**, **scaling**, and
**shearing** into a single operation, making it one of the most powerful data
augmentation techniques for computer vision.

The output image maintains the original dimensions, while any newly exposed
regions are filled using the specified fill value.

---

## Signature

```python
RandomAffine(
    degrees=0,
    translate=None,
    scale=None,
    shear=0,
    fill=0
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `degrees` | `int` or `tuple` | Rotation range. If an integer is provided, the angle is sampled from `(-degrees, degrees)`. |
| `translate` | `tuple` or `None` | Maximum horizontal and vertical translation as fractions of the image size, e.g. `(0.2, 0.2)`. |
| `scale` | `tuple` or `None` | Range of scaling factors, e.g. `(0.8, 1.2)`. |
| `shear` | `int` or `tuple` | Shear angle range in degrees. If an integer is provided, values are sampled from `(-shear, shear)`. |
| `fill` | `int`, `float`, or `tuple` | Pixel value used to fill regions introduced by the transformation. Default is `0`. |

---

## Returns

A transformed NumPy array.

---

## Example

```python
from PIL import Image
from aakaar.transforms import RandomAffine

image = Image.open("cat.jpg")

transform = RandomAffine(
    degrees=15,
    translate=(0.1, 0.1),
    scale=(0.9, 1.1),
    shear=10
)

output = transform(image)
```

---

## Rotation Only

```python
transform = RandomAffine(
    degrees=20
)
```

---

## Translation

Translate the image by up to **10%** of its width and height.

```python
transform = RandomAffine(
    degrees=0,
    translate=(0.1, 0.1)
)
```

---

## Scaling

Randomly zoom in or out.

```python
transform = RandomAffine(
    degrees=0,
    scale=(0.8, 1.2)
)
```

---

## Shearing

Apply random shear transformations.

```python
transform = RandomAffine(
    degrees=0,
    shear=15
)
```

---

## Using with Compose

```python
from aakaar.transforms import (
    Compose,
    RandomAffine,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
)

transform = Compose([
    RandomAffine(
        degrees=20,
        translate=(0.1, 0.1),
        scale=(0.8, 1.2),
        shear=10
    ),
    RandomHorizontalFlip(),
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

1. A random rotation angle is selected.
2. A random scaling factor is sampled.
3. A random translation is applied.
4. A random shear angle is generated.
5. These transformations are combined into a single affine transformation matrix.
6. The transformed image is returned as a NumPy array.

---

## Typical Use Cases

- Image classification
- Object recognition
- OCR and document analysis
- Medical imaging
- General-purpose data augmentation

---

## Notes

- Accepts both **PIL Images** and **NumPy arrays**.
- Returns a NumPy array.
- Uses bilinear interpolation during the affine transformation.
- Output dimensions remain the same as the input image.
- Empty regions created by the transformation are filled using the specified `fill` value.
- Multiple geometric transformations are combined into a single operation for efficient augmentation.
- Typically used only during training and not during validation or inference.