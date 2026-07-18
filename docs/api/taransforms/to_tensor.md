# ToTensor

`ToTensor` converts an input image into a **channel-first (CHW)** NumPy array
with the `float32` data type. Pixel values are automatically scaled to the
range **[0, 1]** when the input image uses 8-bit integer values.

This transform is typically placed near the end of a preprocessing pipeline,
just before normalization.

---

## Signature

```python
ToTensor()
```

---

## Supported Input Types

`ToTensor` accepts:

- **PIL Image**
- **NumPy array** in HWC (`Height × Width × Channels`) format

---

## Returns

A **NumPy array** with:

- Layout: **CHW** (`Channels × Height × Width`)
- Data type: `float32`
- Pixel values:
  - `[0, 255] → [0.0, 1.0]`
  - Existing floating-point arrays in `[0.0, 1.0]` are preserved

---

## Example

```python
from PIL import Image
from aakaar.transforms import ToTensor

image = Image.open("cat.jpg")

transform = ToTensor()

tensor = transform(image)

print(tensor.shape)
```

Example output:

```text
(3, 224, 224)
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

## Value Scaling

### uint8 Input

```python
# Input
[[255, 128, 0]]
```

After conversion:

```python
[[1.0, 0.5019, 0.0]]
```

---

### Floating-Point Input

If the input image already contains floating-point values in the range
`[0.0, 1.0]`, no additional scaling is applied.

```python
image = image.astype("float32")

tensor = ToTensor()(image)
```

---

## Channel Conversion

Images stored in HWC format are automatically converted to CHW.

Before:

```text
Height × Width × Channels
224 × 224 × 3
```

After:

```text
Channels × Height × Width
3 × 224 × 224
```

---

## Typical Pipeline

```python
transform = Compose([
    Resize((256, 256)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])
```

---

## Notes

- Converts images to `float32`.
- Converts HWC images to CHW layout.
- Automatically scales `uint8` images from `[0, 255]` to `[0.0, 1.0]`.
- Floating-point images already in `[0.0, 1.0]` are left unchanged.
- Returns a NumPy array. Conversion to an `aakaar.Tensor` is typically performed later by the data loading pipeline or by the user.