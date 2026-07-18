"""
aakaar.data — Dataset / DataLoader.

NOT implemented: num_workers > 0 (multiprocessing). Real multiprocess data
loading needs worker-process pickling of tensors/datasets, shared-memory
handoff, and CUDA-context-per-process handling — none of that exists here.
Passing num_workers > 0 raises a clear NotImplementedError rather than
silently ignoring it, since silently running single-threaded when the user
asked for multiple workers could mislead them about actual load performance.
"""

import numpy as np
import aakaar


class Dataset:
    """Base class. Subclasses implement __len__ and __getitem__(should return a single)
    sample (a tuple, e.g. (features, label), or a single array)."""

    def __len__(self):
        raise NotImplementedError("Dataset subclasses must implement __len__")

    def __getitem__(self, idx):
        raise NotImplementedError("Dataset subclasses must implement __getitem__")


class TensorDataset(Dataset):
    """Wraps one or more array-likes (numpy arrays or aakaar Tensors) of
    equal first-dimension length, indexing them together."""

    def __init__(self, *arrays):
        if len(arrays) == 0:
            raise ValueError("TensorDataset requires at least one array")

        self._arrays = []
        for a in arrays:
            if hasattr(a, "to_numpy"):  # an aakaar Tensor
                self._arrays.append(a.to_numpy())
            else:
                self._arrays.append(np.asarray(a))

        lengths = {arr.shape[0] for arr in self._arrays}
        if len(lengths) != 1:
            raise ValueError(f"All arrays must have the same first-dimension length, got {lengths}")
        self._len = lengths.pop()

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        sample = tuple(arr[idx] for arr in self._arrays)
        return sample if len(sample) > 1 else sample[0]


def default_collate(batch):
    """Combines a list of samples into a single batch.a list of tuples (stacks each
    position independently), or a list of plain arrays (stacks directly)."""
    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch")

    first = batch[0]
    if isinstance(first, tuple):
        n_fields = len(first)
        return tuple(np.stack([sample[i] for sample in batch]) for i in range(n_fields))
    else:
        return np.stack(batch)


class DataLoader:
    """Iterating yields batches with each field converted to an aakaar Tensor
    on the requested device/dtype — matching how a training loop would
    consume it directly (no manual aakaar.from_numpy() calls needed).
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False,
                 collate_fn=None, device="cpu", dtype="float32", num_workers=0,
                 seed=None):
        if num_workers != 0:
            raise NotImplementedError(
                "num_workers > 0 (multiprocess data loading) is not implemented. "
                "Use num_workers=0 (the default) for single-process loading."
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate
        self.device = device
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _to_aakaar(self, arr):
        if not isinstance(arr, np.ndarray) or arr.dtype == object:
            return arr

        if np.issubdtype(arr.dtype, np.integer):
            np_dtype = np.int64 if arr.dtype == np.int64 else np.int32
        else:
            np_dtype = np.float64 if self.dtype == "float64" else np.float32

        return aakaar.from_numpy(arr.astype(np_dtype), device=self.device)

    def __iter__(self):
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            self._rng.shuffle(indices)

        for start in range(0, n, self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            batch = [self.dataset[i] for i in batch_indices]
            collated = self.collate_fn(batch)

            if isinstance(collated, tuple):
                yield tuple(self._to_aakaar(c) for c in collated)
            else:
                yield self._to_aakaar(collated)