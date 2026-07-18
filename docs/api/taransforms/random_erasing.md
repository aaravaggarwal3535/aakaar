# RandomErasing

`RandomErasing` randomly removes a rectangular region from an image tensor and
replaces it with a constant value or random noise. This augmentation helps
models become more robust to occlusion, missing regions, and partial object
visibility.

Unlike most image transforms, `RandomErasing` is designed to operate on
**CHW tensor-style arrays**, so it is typically applied <Bold>after</Bold> `ToTensor`
(and often after `Normalize`).

---

## Signature

```python
RandomErasing(
    p=0.5,
    scale=(0.02, 0.33),
    ratio=(0.3, 3.3),
    value=0
)
```

---

## Parameters

| Parameter | Type | Description |
|------------|------|-------------|
| `p` | `float` | Probability of applying erasing. Default is `0.5`. |
| `scale` | `tuple` | Range for the erased area as a fraction of the image area. Default is `(0.02, 0.33)`. |
| `ratio` | `tuple` | Range of aspect ratios for the erased rectangle. Default is `(0.3, 3.3)`. |
| `value` | `int`, `float`, or `"random"` | Fill value for the erased region. Use `"random"` to fill with random noise. |

---

## Expected Input

`RandomErasing` expects an image in:

```text
Channels × Height × Width
(CHW)
```

This is the format produced by `ToTensor`.

---

## Returns

A NumPy array with a randomly erased region.

---

## Example

```python
from aakaar.transforms import RandomErasing

transform = RandomErasing(
    p=0.5,
    value=0
)

output = transform(tensor)
```

---

## Using Random Noise

Fill the erased region with random values instead of a constant.

```python
transform = RandomErasing(
    p=0.5,
    value="random"
)
```

---

## Larger Erased Regions

Increase the possible erased area.

```python
transform = RandomErasing(
    scale=(0.1, 0.4)
)
```

This allows between **10%** and **40%** of the image area to be erased.

---

## Using with Compose

```python
from aakaar.transforms import (
    Compose,
    RandomResizedCrop,
    RandomHorizontalFlip,
    ToTensor,
    Normalize,
    RandomErasing,
)

transform = Compose([
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    RandomErasing(
        p=0.25,
        value="random"
    ),
])
```

---

## How It Works

For each image:

<List>
<List.Item>A random rectangle area is sampled according to <Code>scale</Code>.</List.Item>
<List.Item>A random aspect ratio is sampled according to <Code>ratio</Code>.</List.Item>
<List.Item>A valid rectangle position is selected.</List.Item>
<List.Item>The region is replaced with either a constant value or random noise.</List.Item>
<List.Item>The modified image is returned.</List.Item>
</List>

---

## Example Erasing

Before:

```text
+----------------------+
|                      |
|      IMAGE DATA      |
|                      |
+----------------------+
```

After:

```text
+----------------------+
|                      |
|   ███████████        |
|   ███████████        |
|                      |
+----------------------+
```

The shaded region represents the erased area.

---

## Typical Use Cases

<List>
<List.Item>Image classification</List.Item>
<List.Item>Person re-identification</List.Item>
<List.Item>Self-supervised learning</List.Item>
<List.Item>Robustness training</List.Item>
<List.Item>General-purpose computer vision augmentation</List.Item>
</List>

---

## Notes

<List>
<List.Item>Expects <Bold>CHW</Bold> tensor-style arrays, not HWC images.</List.Item>
<List.Item>Typically applied after <Code>ToTensor</Code> and often after <Code>Normalize</Code>.</List.Item>
<List.Item>Returns a NumPy array with the same shape as the input.</List.Item>
<List.Item>Uses up to several attempts to find a valid erase region; if none is found, the input is returned unchanged.</List.Item>
<List.Item>When <Code>value="random"</Code>, the erased region is filled with random values matching the input data type.</List.Item>
<List.Item>Most commonly used during training and disabled during validation or inference.</List.Item>
</List>