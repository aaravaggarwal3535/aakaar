"""
aakaar.transforms — composable image transforms, operating on
numpy arrays (HWC uint8/float) since that's what ImageFolderDataset yields
before conversion to an aakaar Tensor.
"""

import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor:
    """PIL Image or HWC numpy array -> CHW float32 numpy array in [0,1].
    (Final conversion to an aakaar Tensor happens in DataLoader, which also just produces a tensor-ready array/
    tensor, with device placement handled by the caller.)"""

    def __call__(self, img):
        arr = np.array(img, dtype=np.float32) if not isinstance(img, np.ndarray) else img.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 3:
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return arr


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(-1, 1, 1)

    def __call__(self, img):
        return (img - self.mean) / self.std


class Resize:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, img):
        return img.resize(self.size)  # expects a PIL Image, applied before ToTensor

class RandomCrop:
    """Crops a random `size x size` (or (h,w)) region, padding first if the
    source is smaller than the crop size (torchvision default: reflect
    padding is NOT replicated here — this pads with zeros/edge instead,
    a real, stated simplification vs. torchvision's more flexible padding
    modes)."""

    def __init__(self, size, padding=0):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.padding = padding

    def __call__(self, img):
        arr = np.array(img) if not isinstance(img, np.ndarray) else img
        if self.padding > 0:
            pad_width = [(self.padding, self.padding), (self.padding, self.padding)]
            if arr.ndim == 3:
                pad_width.append((0, 0))
            arr = np.pad(arr, pad_width, mode="edge")

        h, w = arr.shape[:2]
        th, tw = self.size
        if h < th or w < tw:
            raise ValueError(f"RandomCrop size {self.size} larger than (padded) image {(h, w)}")

        top = np.random.randint(0, h - th + 1)
        left = np.random.randint(0, w - tw + 1)
        return arr[top:top+th, left:left+tw]


class RandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3/4, 4/3)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            pil_img = img
        else:
            pil_img = Image.fromarray(np.array(img).astype(np.uint8))
        w, h = pil_img.size
        area = w * h

        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            log_ratio = (np.log(self.ratio[0]), np.log(self.ratio[1]))
            aspect = np.exp(np.random.uniform(*log_ratio))

            crop_w = int(round(np.sqrt(target_area * aspect)))
            crop_h = int(round(np.sqrt(target_area / aspect)))

            if 0 < crop_w <= w and 0 < crop_h <= h:
                left = np.random.randint(0, w - crop_w + 1)
                top = np.random.randint(0, h - crop_h + 1)
                cropped = pil_img.crop((left, top, left + crop_w, top + crop_h))
                return np.array(cropped.resize(self.size, Image.BILINEAR))

        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        cropped = pil_img.crop((left, top, left + min_dim, top + min_dim))
        return np.array(cropped.resize(self.size, Image.BILINEAR))

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        arr = np.array(img) if not isinstance(img, np.ndarray) else img
        if np.random.random() < self.p:
            return np.ascontiguousarray(arr[:, ::-1])
        return arr


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        arr = np.array(img) if not isinstance(img, np.ndarray) else img
        if np.random.random() < self.p:
            return np.ascontiguousarray(arr[::-1, :])
        return arr


class RandomRotation:
    def __init__(self, degrees, fill=0):
        self.degrees = degrees if isinstance(degrees, tuple) else (-degrees, degrees)
        self.fill = fill

    def __call__(self, img):
        from PIL import Image
        pil_img = img if isinstance(img, Image.Image) else Image.fromarray(np.array(img).astype(np.uint8))
        angle = np.random.uniform(*self.degrees)
        rotated = pil_img.rotate(angle, resample=Image.BILINEAR, fillcolor=self.fill)
        return np.array(rotated)


class RandomAffine:
    def __init__(self, degrees=0, translate=None, scale=None, shear=0, fill=0):
        self.degrees = degrees if isinstance(degrees, tuple) else (-degrees, degrees)
        self.translate = translate
        self.scale = scale
        self.shear = shear if isinstance(shear, tuple) else (-shear, shear)
        self.fill = fill

    def __call__(self, img):
        from PIL import Image
        pil_img = img if isinstance(img, Image.Image) else Image.fromarray(np.array(img).astype(np.uint8))
        w, h = pil_img.size

        angle = np.radians(np.random.uniform(*self.degrees))
        shear_val = np.radians(np.random.uniform(*self.shear))
        s = np.random.uniform(*self.scale) if self.scale else 1.0
        tx = np.random.uniform(-self.translate[0], self.translate[0]) * w if self.translate else 0
        ty = np.random.uniform(-self.translate[1], self.translate[1]) * h if self.translate else 0

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        a = s * (cos_a + np.tan(shear_val) * sin_a)
        b = -s * sin_a
        c = s * sin_a
        d = s * cos_a

        cx, cy = w / 2, h / 2
        matrix = (a, b, tx + cx - a*cx - b*cy,
                  c, d, ty + cy - c*cx - d*cy)
        warped = pil_img.transform((w, h), Image.AFFINE, matrix, resample=Image.BILINEAR, fillcolor=self.fill)
        return np.array(warped)

class ColorJitter:
    """Randomly adjusts brightness/contrast/saturation/hue, in that order
    (matching torchvision's default order), each within +-factor of 1.0.
    Operates on float [0,1] arrays internally regardless of input dtype."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img):
        arr = np.array(img, dtype=np.float32)
        was_uint8_range = arr.max() > 1.5
        if was_uint8_range:
            arr = arr / 255.0

        if self.brightness > 0:
            factor = np.random.uniform(max(0, 1-self.brightness), 1+self.brightness)
            arr = arr * factor

        if self.contrast > 0:
            factor = np.random.uniform(max(0, 1-self.contrast), 1+self.contrast)
            mean = arr.mean(axis=(0, 1), keepdims=True) if arr.ndim == 3 else arr.mean()
            arr = (arr - mean) * factor + mean

        if self.saturation > 0 and arr.ndim == 3 and arr.shape[2] == 3:
            factor = np.random.uniform(max(0, 1-self.saturation), 1+self.saturation)
            gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            gray = gray[..., None]
            arr = gray + (arr - gray) * factor

        if self.hue > 0 and arr.ndim == 3 and arr.shape[2] == 3:
            arr = self._adjust_hue(arr, np.random.uniform(-self.hue, self.hue))

        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8) if was_uint8_range else arr

    @staticmethod
    def _adjust_hue(arr, hue_shift):
        # RGB -> HSV, shift hue, back to RGB. Simple, correct, not the
        # fastest possible implementation — fine for a transform pipeline.
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        delta = maxc - minc
        s = np.where(maxc == 0, 0, delta / np.where(maxc == 0, 1, maxc))

        rc = np.where(delta == 0, 0, (maxc - r) / np.where(delta == 0, 1, delta))
        gc = np.where(delta == 0, 0, (maxc - g) / np.where(delta == 0, 1, delta))
        bc = np.where(delta == 0, 0, (maxc - b) / np.where(delta == 0, 1, delta))

        h = np.zeros_like(maxc)
        h = np.where(maxc == r, bc - gc, h)
        h = np.where(maxc == g, 2.0 + rc - bc, h)
        h = np.where(maxc == b, 4.0 + gc - rc, h)
        h = (h / 6.0) % 1.0
        h = (h + hue_shift) % 1.0

        i = (h * 6.0).astype(int) % 6
        f = (h * 6.0) - (h * 6.0).astype(int)
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        conditions = [i == k for k in range(6)]
        r_out = np.select(conditions, [v, q, p, p, t, v])
        g_out = np.select(conditions, [t, v, v, q, p, p])
        b_out = np.select(conditions, [p, p, t, v, v, q])
        return np.stack([r_out, g_out, b_out], axis=-1)


class RandomErasing:
    """Randomly erases a rectangular region, filling with random noise
    (torchvision default) or a constant value. Operates on the array
    AFTER ToTensor (CHW format) to match torchvision's placement in the
    pipeline (RandomErasing is typically applied post-ToTensor)."""

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value  # 0, a number, or 'random'

    def __call__(self, img_chw):
        if np.random.random() >= self.p:
            return img_chw

        c, h, w = img_chw.shape
        area = h * w

        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            aspect = np.exp(np.random.uniform(np.log(self.ratio[0]), np.log(self.ratio[1])))
            eh = int(round(np.sqrt(target_area * aspect)))
            ew = int(round(np.sqrt(target_area / aspect)))
            if eh < h and ew < w:
                top = np.random.randint(0, h - eh)
                left = np.random.randint(0, w - ew)
                out = img_chw.copy()
                if self.value == 'random':
                    out[:, top:top+eh, left:left+ew] = np.random.rand(c, eh, ew).astype(img_chw.dtype)
                else:
                    out[:, top:top+eh, left:left+ew] = self.value
                return out
        return img_chw


class GaussianBlur:
    """Gaussian blur via a separable convolution (numpy-only, no scipy
    dependency added). Kernel size must be odd."""

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma if isinstance(sigma, tuple) else (sigma, sigma)

    def __call__(self, img):
        arr = np.array(img, dtype=np.float32)
        was_uint8_range = arr.max() > 1.5
        sigma = np.random.uniform(*self.sigma)

        k = self.kernel_size
        ax = np.arange(k) - k // 2
        gauss_1d = np.exp(-(ax**2) / (2 * sigma**2))
        gauss_1d /= gauss_1d.sum()

        pad = k // 2
        if arr.ndim == 3:
            padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
            blurred = np.zeros_like(arr)
            for c in range(arr.shape[2]):
                tmp = np.apply_along_axis(lambda m: np.convolve(m, gauss_1d, mode="valid"), 0, padded[:, :, c])
                blurred[:, :, c] = np.apply_along_axis(lambda m: np.convolve(m, gauss_1d, mode="valid"), 1, tmp)
        else:
            padded = np.pad(arr, pad, mode="reflect")
            tmp = np.apply_along_axis(lambda m: np.convolve(m, gauss_1d, mode="valid"), 0, padded)
            blurred = np.apply_along_axis(lambda m: np.convolve(m, gauss_1d, mode="valid"), 1, tmp)

        blurred = np.clip(blurred, 0, 255 if was_uint8_range else 1.0)
        return blurred.astype(np.uint8) if was_uint8_range else blurred