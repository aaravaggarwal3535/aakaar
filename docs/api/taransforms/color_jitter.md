# ColorJitter

`ColorJitter` randomly modifies the appearance of an image by adjusting its
**brightness**, **contrast**, **saturation**, and **hue**. These color
augmentations help models become more robust to varying lighting conditions,
camera settings, and environmental changes.

Each property is adjusted independently using randomly sampled factors, making
every transformed image slightly different.

---

## Signature

```python
ColorJitter(
    brightness=0,
    contrast=0,
    saturation=0,
    hue=0
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `brightness` | `float` | Maximum brightness adjustment. Values are sampled from `[max(0, 1-brightness), 1+brightness]`. |
| `contrast` | `float` | Maximum contrast adjustment. Values are sampled from `[max(0, 1-contrast), 1+contrast]`. |
| `saturation` | `float` | Maximum saturation adjustment for RGB images. |
| `hue` | `float` | Maximum hue shift. A random value is sampled from `[-hue, hue]`. |

---

## Returns

A NumPy array with randomized color properties.

- If the input image is `uint8`, the output is returned as `uint8`.
- If the input image is floating point, the output remains floating point.

---

## Example

```python
from PIL import Image
from aakaar.transforms import ColorJitter

image = Image.open("cat.jpg")

transform = ColorJitter(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.1
)

output = transform(image)
```

---

## Brightness Adjustment

```python
transform = ColorJitter(
    brightness=0.4
)
```

Each image receives a random brightness factor within the specified range.

---

## Contrast Adjustment

```python
transform = ColorJitter(
    contrast=0.3
)
```

Contrast is adjusted independently for every image.

---

## Saturation Adjustment

```python
transform = ColorJitter(
    saturation=0.5
)
```

Saturation changes are applied only to RGB images.

---

## Hue Adjustment

```python
transform = ColorJitter(
    hue=0.1
)
```

Hue values are shifted in HSV color space before converting back to RGB.

---

## Using with Compose

```python
from aakaar.transforms import (
    Compose,
    RandomResizedCrop,
    ColorJitter,
    ToTensor,
    Normalize,
)

transform = Compose([
    RandomResizedCrop(224),
    ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
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

1. Brightness is randomly adjusted.
2. Contrast is randomly adjusted.
3. Saturation is randomly adjusted (RGB images only).
4. Hue is randomly shifted in HSV color space (RGB images only).
5. Pixel values are clipped to a valid range before being returned.

---

## Typical Use Cases

- Image classification
- Object detection
- Semantic segmentation
- Self-supervised learning
- General-purpose computer vision training

---

## Notes

- Accepts both **PIL Images** and **NumPy arrays**.
- Returns a NumPy array.
- Internally operates on floating-point values, regardless of input type.
- Preserves the original output data type (`uint8` or floating point).
- Saturation and hue adjustments are applied only to RGB images.
- Brightness, contrast, saturation, and hue are sampled independently for each image.
- Typically used during training and omitted during validation or inference.