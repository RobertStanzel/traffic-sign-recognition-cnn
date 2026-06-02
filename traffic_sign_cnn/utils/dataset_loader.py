"""Dataset and augmentation pipeline for traffic sign classification.

Real-world augmentation strategy:
  Every transform below simulates a condition that occurs in dashcam / road footage
  but is absent from the clean GTSRB crops the model is trained on.

Dataset structure expected on disk:
    data/dataset/
        ClassName1/
            img001.jpg
        ClassName2/
            ...
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
import random

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ---------------------------------------------------------------------------
# Custom transforms
# ---------------------------------------------------------------------------

class MotionBlur:
    """Simulate camera / vehicle motion blur by blurring in a random direction."""

    def __init__(self, max_kernel: int = 7, p: float = 0.35):
        self.max_kernel = max_kernel
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        k = random.choice([3, 5, self.max_kernel])
        # Horizontal or vertical blur
        filt = ImageFilter.BoxBlur(radius=(k // 2, 0) if random.random() < 0.5 else (0, k // 2))
        return img.filter(filt)


class SimulateFog:
    """Blend a white overlay at random opacity to simulate fog/overexposure."""

    def __init__(self, max_alpha: float = 0.25, p: float = 0.15):
        self.max_alpha = max_alpha
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        alpha = random.uniform(0.05, self.max_alpha)
        white = Image.new("RGB", img.size, (255, 255, 255))
        return Image.blend(img, white, alpha)


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_transforms(train: bool = True) -> transforms.Compose:
    """Build the augmentation pipeline for training or validation.

    Training pipeline simulates real-world dashcam conditions:
      - Scale / position variation  → RandomResizedCrop
      - Viewing angle               → RandomPerspective
      - Sign tilt on post           → RandomRotation
      - Lighting / time of day      → ColorJitter (strong)
      - Night / desaturated camera  → RandomGrayscale
      - Rain on lens / distance fog → GaussianBlur + custom SimulateFog
      - Vehicle / object motion     → custom MotionBlur
      - Partial occlusion           → RandomErasing

    No horizontal flip — sign direction matters (left ≠ right turn).
    """
    normalize = transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    sz = config.IMG_SIZE

    if train:
        return transforms.Compose([
            # ── Spatial ───────────────────────────────────────────────────
            # Zoom between 80 % and 100 % of the crop, then resize to sz.
            # Simulates signs at varying distances from the camera.
            transforms.RandomResizedCrop(
                sz,
                scale=(0.75, 1.0),
                ratio=(0.85, 1.15),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            # Tilt up to ±20 ° — signs on bent posts, camera roll
            transforms.RandomRotation(degrees=20, fill=128),
            # Perspective warp — viewing sign at an angle while driving past
            transforms.RandomPerspective(distortion_scale=0.45, p=0.5, fill=128),

            # ── Colour / lighting ─────────────────────────────────────────
            # Strong jitter covers cloudy / sunny / night / tunnel transitions
            transforms.ColorJitter(
                brightness=0.6,
                contrast=0.6,
                saturation=0.5,
                hue=0.08,
            ),
            # Desaturation simulates B&W or night-mode dashcams (5 % of samples)
            transforms.RandomGrayscale(p=0.05),

            # ── Blur / weather ────────────────────────────────────────────
            SimulateFog(max_alpha=0.25, p=0.15),
            # Rain on lens / out-of-focus distant sign
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 2.5))],
                p=0.4,
            ),
            MotionBlur(max_kernel=7, p=0.3),

            # ── Tensor conversion + normalise ─────────────────────────────
            transforms.ToTensor(),
            normalize,

            # ── Occlusion ─────────────────────────────────────────────────
            # Randomly erase a small rectangle — other vehicles, branches, dirt
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ])

    # Validation: deterministic — just resize and normalise
    return transforms.Compose([
        transforms.Resize((sz, sz), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TrafficSignDataset(Dataset):
    """PyTorch Dataset for folder-organised traffic sign images."""

    def __init__(
        self,
        root_dir: str,
        file_list: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
        idx_to_class: Optional[Dict[int, str]] = None,
    ):
        self.root_dir     = root_dir
        self.file_list    = file_list
        self.transform    = transform
        self.idx_to_class: Dict[int, str] = idx_to_class or {}

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    data_dir: str = config.DATA_DIR,
    val_split: float = config.VAL_SPLIT,
) -> Tuple[TrafficSignDataset, TrafficSignDataset, Dict[int, str]]:
    """Scan dataset directory, split into train/val, return Dataset objects.

    The split is stratified so every class is proportionally represented.

    Raises:
        FileNotFoundError: If data_dir doesn't exist or contains no images.
    """
    from sklearn.model_selection import train_test_split

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: '{data_dir}'\n"
            "Place class subfolders inside data/dataset/ before training."
        )

    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class subdirectories found in '{data_dir}'.")

    class_to_idx: Dict[str, int] = {d.name: i for i, d in enumerate(class_dirs)}
    idx_to_class: Dict[int, str] = {i: d.name for i, d in enumerate(class_dirs)}

    supported_ext = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".webp"}
    all_samples: List[Tuple[str, int]] = []

    for class_dir in class_dirs:
        label = class_to_idx[class_dir.name]
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in supported_ext:
                all_samples.append((str(img_file), label))

    if not all_samples:
        raise FileNotFoundError(f"No supported images found in '{data_dir}'.")

    labels = [s[1] for s in all_samples]
    train_samples, val_samples = train_test_split(
        all_samples,
        test_size=val_split,
        stratify=labels,
        random_state=42,
    )

    train_ds = TrafficSignDataset(
        data_dir, train_samples,
        transform=get_transforms(train=True),
        idx_to_class=idx_to_class,
    )
    val_ds = TrafficSignDataset(
        data_dir, val_samples,
        transform=get_transforms(train=False),
        idx_to_class=idx_to_class,
    )

    print(
        f"[INFO] Dataset — Classes: {len(class_dirs)} | "
        f"Train: {len(train_samples)} | Val: {len(val_samples)}"
    )
    return train_ds, val_ds, idx_to_class
