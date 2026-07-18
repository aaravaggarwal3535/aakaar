# RandomHorizontalFlip

`RandomHorizontalFlip` randomly flips an image horizontally with a specified
probability. Horizontal flipping is one of the simplest and most effective data
augmentation techniques, allowing models to learn features that are invariant
to left-right orientation.

---

## Signature

```python
RandomHorizontalFlip(p=0.5)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `p` | `float` | Probability of flipping the image horizontally. Must be between `0.0` and `1.0`. Default is `0.5`. |

---

## Returns

A NumPy array. If the random condition is met, the returned image is
horizontally flipped; otherwise, the original image is returned unchanged.

---

## Example

```python
from PIL import Image
from aakaar.transforms import RandomHorizontalFlip

image = Image.open("cat.jpg")

transform = RandomHorizontalFlip(p=0.5)

flipped = transform(image)
```

---

## Always Flip

To flip every image, set the probability to `1.0`.

```python
transform = RandomHorizontalFlip(p=1.0)
```

---

## Never Flip

To disable flipping while keeping the transform in the pipeline, use a
probability of `0.0`.

```python
transform = RandomHorizontalFlip(p=0.0)
```

---

## Using with Compose

```python
from aakaar.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
)

transform = Compose([
    RandomResizedCrop(224),
    RandomHorizontalFlip(p=0.5),
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
| A              B |
|                  |
|                  |
+------------------+
```

After horizontal flip:

```text
+------------------+
| B              A |
|                  |
|                  |
+------------------+
```

---

## Typical Use Cases

- Image classification
- Object recognition
- Face recognition (when left-right symmetry is acceptable)
- General-purpose computer vision training

---

## Notes

- Accepts both **PIL Images** and **NumPy arrays**.
- Returns a NumPy array.
- Each image is flipped independently based on the specified probability.
- Uses a contiguous NumPy array after flipping for compatibility with downstream operations.
- Typically used during training and omitted during validation or inference.