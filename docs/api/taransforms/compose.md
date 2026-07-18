# Compose

`Compose` combines multiple image transforms into a single preprocessing
pipeline. Each transform is applied sequentially, with the output of one
transform becoming the input to the next.

This is the recommended way to build reusable image preprocessing and data
augmentation pipelines in **Aakaar**.

---

## Signature

```python
Compose(transforms)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `transforms` | `list` | A list of transform objects to execute in sequence. |

---

## Returns

A callable transform pipeline.

---

## Example

```python
from PIL import Image
from aakaar.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
)

transform = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

image = Image.open("cat.jpg")

processed = transform(image)
```

---

## Execution Order

Transforms are executed **exactly in the order they are provided**.

```python
transform = Compose([
    Resize(256),
    RandomCrop(224),
    ToTensor(),
    Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
```

Execution flow:

```
Input Image
      │
      ▼
Resize
      │
      ▼
RandomCrop
      │
      ▼
ToTensor
      │
      ▼
Normalize
      │
      ▼
Output
```

---

## Multiple Augmentations

Several augmentation transforms can be chained together.

```python
from aakaar.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    ToTensor,
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
])
```

---

## Validation Pipeline

Validation pipelines usually avoid random augmentations.

```python
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

## Notes

- Transforms are executed sequentially from first to last.
- The output of each transform becomes the input to the next.
- `Compose` does not modify transform behavior; it only manages execution order.
- The order of transforms matters. For example, `Normalize` should typically be applied after `ToTensor`.
- Any callable object that accepts an image and returns an image can be included in the pipeline.