# RandomRotation

`RandomRotation` rotates an image by a randomly selected angle within a
specified range. Random rotations help models become more robust to variations
in image orientation and are commonly used in image classification, document
analysis, and medical imaging.

The output image retains the original dimensions, with newly exposed regions
filled using the specified fill value.

---

## Signature

```python
RandomRotation(degrees, fill=0)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `degrees` | `int` or `tuple` | Rotation angle range. If an integer is provided, the angle is sampled from `(-degrees, degrees)`. If a tuple is provided, it specifies `(min_degree, max_degree)`. |
| `fill` | `int`, `float`, or `tuple` | Pixel value used to fill areas introduced by rotation. Default is `0`. |

---

## Returns

A rotated NumPy array.

---

## Example

```python
from PIL import Image
from aakaar.transforms import RandomRotation

image = Image.open("cat.jpg")

transform = RandomRotation(30)

rotated = transform(image)
```

---

## Custom Rotation Range

Specify a custom range of rotation angles.

```python
transform = RandomRotation(
    degrees=(-15, 45)
)
```

A random angle between **−15°** and **45°** is selected for each image.

---

## Custom Fill Value

Choose a different fill value for pixels outside the original image.

```python
transform = RandomRotation(
    degrees=20,
    fill=255
)
```

For RGB images, a tuple may also be used.

```python
transform = RandomRotation(
    degrees=20,
    fill=(255, 255, 255)
)
```

---

## Using with Compose

```python
from aakaar.transforms import (
    Compose,
    RandomRotation,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
)

transform = Compose([
    RandomRotation(20),
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

1. A random angle is sampled from the specified range.
2. The image is rotated using bilinear interpolation.
3. Empty regions created by the rotation are filled with the specified fill value.
4. The rotated image is returned as a NumPy array.

---

## Typical Use Cases

- Image classification
- Handwritten digit recognition
- Optical character recognition (OCR)
- Medical imaging
- Document analysis

---

## Notes

- Accepts both **PIL Images** and **NumPy arrays**.
- Returns a NumPy array.
- Uses bilinear interpolation for smooth image rotation.
- Output image dimensions remain the same as the input.
- Newly exposed regions are filled using the specified `fill` value.
- Random rotations are typically used during training and omitted during validation or inference.