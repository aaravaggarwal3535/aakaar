# aakaar.transforms

`aakaar.transforms` provides a collection of composable image preprocessing and
data augmentation utilities for computer vision tasks. These transforms are
designed to work with images before they are converted into an `aakaar.Tensor`,
making them ideal for preparing datasets used in training and inference.

Unlike tensor operations, transforms operate on **PIL Images** or **NumPy
arrays**. They can be chained together using `Compose` to build flexible and
reusable preprocessing pipelines.

---

## Supported Input Types

Most transforms accept one or both of the following image formats:

- **PIL Image**
- **NumPy array**
  - HWC layout (`Height × Width × Channels`)
  - `uint8` images with values in `[0, 255]`
  - Floating-point images with values in `[0.0, 1.0]`

Some transforms, such as `RandomErasing`, are intended to be applied **after**
`ToTensor`, when the image has already been converted to CHW format.

---

## Typical Pipeline

A preprocessing pipeline is created by combining multiple transforms using
`Compose`.

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

## Available Transforms

| Transform | Description |
|------------|-------------|
| `Compose` | Chains multiple transforms together into a single pipeline. |
| `Resize` | Resizes an image to a specified size. |
| `ToTensor` | Converts an image into a CHW floating-point NumPy array. |
| `Normalize` | Normalizes each channel using the provided mean and standard deviation. |
| `RandomCrop` | Extracts a random crop from an image. |
| `RandomResizedCrop` | Randomly crops and resizes an image. |
| `RandomHorizontalFlip` | Randomly flips an image horizontally. |
| `RandomVerticalFlip` | Randomly flips an image vertically. |
| `RandomRotation` | Rotates an image by a random angle. |
| `RandomAffine` | Applies a random affine transformation. |
| `ColorJitter` | Randomly adjusts brightness, contrast, saturation, and hue. |
| `RandomErasing` | Randomly removes a rectangular region from a tensor image. |
| `GaussianBlur` | Applies Gaussian blur using a NumPy implementation. |

---

## Building Pipelines

Transforms are executed **in the order they are provided**.

```python
transform = Compose([
    Resize(256),
    RandomCrop(224),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
```

Changing the order may change the output significantly. For example,
`Normalize` should typically be applied **after** `ToTensor`, while
`Resize` is generally performed before conversion to tensor format.

---

## Common Training Pipeline

```python
train_transform = Compose([
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
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
    RandomErasing(p=0.25),
])
```

This pipeline performs data augmentation followed by normalization, helping
improve model generalization during training.

---

## Common Validation Pipeline

Validation data should typically avoid random augmentations.

```python
val_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

Using deterministic preprocessing ensures that evaluation remains consistent
across different runs.

---

## Notes

- Images are generally expected to be in **HWC** format before `ToTensor`.
- `ToTensor` converts images to **CHW** format with floating-point values.
- `Normalize` expects channel-first (CHW) images.
- `RandomErasing` is designed to operate on tensor-style images after
  `ToTensor`.
- Individual transform documentation contains implementation details,
  parameters, examples, and usage notes.