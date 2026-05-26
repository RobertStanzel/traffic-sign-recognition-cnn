"""Custom PyTorch Dataset and data-pipeline utilities.

Dataset structure expected on disk:
    data/dataset/
        ClassName1/
            img001.jpg
            img002.jpg
        ClassName2/
            img001.jpg
        ...

The folder name is used as the class label.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_transforms(train: bool = True) -> transforms.Compose:
    """Build the torchvision transform pipeline for training or validation.

    Training includes random flips, rotation, and colour jitter for augmentation.
    Validation uses only deterministic resize + normalise.

    Args:
        train: If True, return augmented training transforms.

    Returns:
        Composed torchvision transform pipeline.
    """
    normalize = transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)

    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TrafficSignDataset(Dataset):
    """PyTorch Dataset for folder-organised traffic sign images.

    Args:
        root_dir:     Root dataset directory (informational only; paths in
                      file_list are already absolute or relative-to-cwd).
        file_list:    List of (image_path, label_index) tuples.
        transform:    Optional torchvision transform to apply to each image.
        idx_to_class: Mapping {int → class_name_str} exposed for inference.
    """

    def __init__(
        self,
        root_dir: str,
        file_list: List[Tuple[str, int]],
        transform: Optional[Callable] = None,
        idx_to_class: Optional[Dict[int, str]] = None,
    ):
        self.root_dir = root_dir
        self.file_list = file_list
        self.transform = transform
        self.idx_to_class: Dict[int, str] = idx_to_class or {}

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load image at index and return (tensor, label).

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label_index).
        """
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

    The split is stratified so every class has proportional representation in
    both the train and validation sets.

    Args:
        data_dir:  Root directory with one subfolder per class.
        val_split: Fraction of samples reserved for validation (0 < val_split < 1).

    Returns:
        (train_dataset, val_dataset, idx_to_class)

    Raises:
        FileNotFoundError: If data_dir does not exist, contains no class
                           subdirectories, or contains no image files.
    """
    from sklearn.model_selection import train_test_split

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: '{data_dir}'\n"
            "Please place class subfolders inside data/dataset/ before training."
        )

    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(
            f"No class subdirectories found in '{data_dir}'.\n"
            "Expected structure: data/dataset/ClassName/img.jpg"
        )

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
        raise FileNotFoundError(
            f"No supported images found in '{data_dir}'.\n"
            f"Supported extensions: {', '.join(sorted(supported_ext))}"
        )

    labels = [s[1] for s in all_samples]
    train_samples, val_samples = train_test_split(
        all_samples,
        test_size=val_split,
        stratify=labels,
        random_state=42,
    )

    train_ds = TrafficSignDataset(
        data_dir,
        train_samples,
        transform=get_transforms(train=True),
        idx_to_class=idx_to_class,
    )
    val_ds = TrafficSignDataset(
        data_dir,
        val_samples,
        transform=get_transforms(train=False),
        idx_to_class=idx_to_class,
    )

    print(
        f"[INFO] Dataset loaded — Classes: {len(class_dirs)} | "
        f"Train: {len(train_samples)} | Val: {len(val_samples)}"
    )
    return train_ds, val_ds, idx_to_class
