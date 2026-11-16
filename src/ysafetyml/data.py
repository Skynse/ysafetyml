"""Loaders for the preprocessed SisFall "Three Classes" windows.

The archive ships headerless binaries with no documentation. Layout, inferred
from file sizes and byte patterns (see notebooks/01_explore_sisfall.ipynb):

- ``x_{split}_3``: float32, shape (n, 256, 6). Channels are accel x/y/z then
  gyro x/y/z, in scaled units (not g or deg/s).
- ``y_{split}_3``: uint8 one-hot, shape (n, 3).
- ``weights_3.txt``: max_count / count per class, in *reverse* class order.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SISFALL_3C_DIR = PROJECT_ROOT / "data" / "raw" / "sisfall_three_classes"

WINDOW = 256
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
SPLITS = ("train", "val", "test")
# SisFall was recorded at 200 Hz, so one window is 1.28 s.
SAMPLE_HZ = 200


@dataclass
class Split:
    x: np.ndarray  # (n, WINDOW, 6) float32
    y: np.ndarray  # (n,) int class index

    def __len__(self) -> int:
        return len(self.y)


def load_split(split: str, root: Path = SISFALL_3C_DIR, mmap: bool = True) -> Split:
    """Load one split. With ``mmap`` the windows stay on disk until touched."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    onehot = np.fromfile(root / f"y_{split}_3", dtype=np.uint8).reshape(-1, 3)
    if not (onehot.sum(axis=1) == 1).all():
        raise ValueError(f"y_{split}_3 is not one-hot")
    n = len(onehot)
    x_path = root / f"x_{split}_3"
    expected = n * WINDOW * len(CHANNELS) * 4
    if x_path.stat().st_size != expected:
        raise ValueError(f"{x_path.name}: expected {expected} bytes for {n} windows")
    if mmap:
        x = np.memmap(x_path, dtype=np.float32, mode="r", shape=(n, WINDOW, len(CHANNELS)))
    else:
        x = np.fromfile(x_path, dtype=np.float32).reshape(n, WINDOW, len(CHANNELS))
    return Split(x=x, y=onehot.argmax(axis=1))


def load_all(root: Path = SISFALL_3C_DIR, mmap: bool = True) -> dict[str, Split]:
    return {s: load_split(s, root, mmap) for s in SPLITS}


def load_class_weights(root: Path = SISFALL_3C_DIR) -> np.ndarray:
    """Weights as shipped (file order, which is reversed relative to y)."""
    return np.loadtxt(root / "weights_3.txt")
