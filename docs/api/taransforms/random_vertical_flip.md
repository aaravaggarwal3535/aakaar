# RandomVerticalFlip

`RandomVerticalFlip` randomly flips an image vertically with a specified
probability. This augmentation helps models become more robust to upside-down
or vertically mirrored images, making it useful for certain computer vision
tasks such as aerial imagery, microscopy, and medical imaging.

---

## Signature

```python
RandomVerticalFlip(p=0.5)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `p` | `float` | Probability of flipping the image vertically. Must be between `0.0` and `1.0`. Default is `0.5`. |

---

## Returns

A NumPy array. If the random condition is met, the returned image is
vertically flipped; otherwise, the original image is returned unchanged.

---

## Example

```python
from PIL import Image
from aakaar.transforms import RandomVerticalFlip

image = Image.open("cat.jpg")

transform = RandomVerticalFlip(p=0.5)

flipped = transform(image)
```

---

## Always Flip

To flip every image, set the probability to `1.0`.

```python
transform = RandomVerticalFlip(p=1.0)
```

---

## Never Flip

To disable flipping while keeping the transform in the pipeline, use a
probability of `0.0`.

```python
transform = RandomVerticalFlip(p=0.0)
```

---

## Using with Compose

```python
from aakaar.transforms import (
    Compose,
    RandomResizedCrop,
    RandomVerticalFlip,
    ToTensor,
    Normalize,
)

transform = Compose([
    RandomResizedCrop(224),
    RandomVerticalFlip(p=0.5),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

---

## Example Flip

Original image:

```text
+------------------+
|        A         |
|                  |
|                  |
|        B         |
+------------------+
```

After vertical flip:

```text
+------------------+
|        B         |
|                  |
|                  |
|        A         |
+------------------+
```

---

## Typical Use Cases

- Aerial and satellite imagery
- Medical image analysis
- Microscopy
- Document orientation augmentation
- General computer vision tasks where vertical orientation is not significant

---

## Notes

- Accepts both **PIL Images** and **NumPy arrays**.
- Returns a NumPy array.
- Each image is flipped independently based on the specified probability.
- Uses a contiguous NumPy array after flipping for compatibility with downstream operations.
- Vertical flipping is generally less common than horizontal flipping and should be used only when it is appropriate for the dataset and task.