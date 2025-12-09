"""Lightweight helpers extracted from the retired training_helper module.

This config shim keeps notebooks and helper scripts working after
training_helper.py is removed. It includes only the pieces still needed
by the YOLO keypoint pipeline:
- Config dataclasses for detection and pose
- Dataset loaders and split discovery
- Minimal training helpers (device resolution, cache clearing, pose training)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import os
import random

try:
	import torch  # type: ignore
	HAS_TORCH = True
except Exception:  # pragma: no cover - soft dependency
	HAS_TORCH = False

try:
	from ultralytics import YOLO  # type: ignore
	HAS_ULTRALYTICS = True
except Exception:  # pragma: no cover - soft dependency
	HAS_ULTRALYTICS = False


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

# Define project root relative to this config file
ROOT_DIR = Path(__file__).parent.resolve()

@dataclass
class TrainingConfig:
    data_root: Path = field(default_factory=lambda: ROOT_DIR / "Data" / "Sample") # Defaulting to Sample for safety, user can change
    yolo_root: Path = field(default_factory=lambda: ROOT_DIR / "Data" / "yolo_dataset")
    results_dir: Path = field(default_factory=lambda: ROOT_DIR / "Models" / "Training" / "result")

    use_limited_dataset: bool = True
    max_images_limit: int = 10_000

    debug_train_only: bool = False
    debug_val_count: int = 24
    debug_test_count: int = 12

    fast_training: bool = True
    random_seed: int = 42

    img_size: int = 640
    batch_size: int = 16

    def __post_init__(self) -> None:
        if isinstance(self.data_root, str):
            self.data_root = Path(self.data_root)
        if isinstance(self.yolo_root, str):
            self.yolo_root = Path(self.yolo_root)
        if isinstance(self.results_dir, str):
            self.results_dir = Path(self.results_dir)
        random.seed(self.random_seed)

    def get_training_params(self) -> Dict[str, Any]:
        if self.fast_training:
            return {"epochs": 50, "model_size": "n", "batch_size": self.batch_size}
        return {"epochs": 100, "model_size": "m", "batch_size": self.batch_size}


@dataclass
class ObjectDetectionConfig(TrainingConfig):
    class_map: Dict[str, int] = field(default_factory=lambda: {
        "referee": 0,
        "player": 1,
        "goalkeeper": 2,
        "ball": 3,
    })
    class_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "referee": (0, 0, 255),
        "player": (0, 255, 0),
        "goalkeeper": (255, 165, 0),
        "ball": (0, 255, 255),
    })
    project_name: str = "SISIKRIPSI_objects"

    def get_class_names(self) -> List[str]:
        return [k for k, _ in sorted(self.class_map.items(), key=lambda kv: kv[1])]

    def get_yolo_model_name(self) -> str:
        size = self.get_training_params()["model_size"]
        return f"yolo11{size}.pt"


@dataclass
class PoseEstimationConfig(TrainingConfig):
    pitch_keypoints: List[str] = field(default_factory=lambda: [f"kp_{i:02d}" for i in range(1, 33)])
    total_keypoints: int = 32
    canonical_kp_order: List[Tuple[str, int]] = field(default_factory=list)
    project_name: str = "SISIKRIPSI_pitch"
    line_color: Tuple[int, int, int] = (0, 191, 255)

    def get_yolo_model_name(self) -> str:
        size = self.get_training_params()["model_size"]
        return f"yolo11{size}-pose.pt"


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def load_game_annotations(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_game_folders(data_root: Path, split: str = "train") -> List[Path]:
    split_dir = data_root / split
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob("SNGS-*"))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def get_device() -> Union[int, str]:
    if HAS_TORCH and torch.cuda.is_available():
        return 0
    return "cpu"


def clear_cache_files(base_path: Path) -> int:
    cache_files = list(base_path.glob("**/*.cache"))
    removed = 0
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            removed += 1
            print(f"   ✓ Removed: {cache_file}")
        except Exception as exc:  # pragma: no cover - best effort cleanup
            print(f"   ⚠️  Could not remove {cache_file}: {exc}")
    if removed:
        print(f"✅ Cleared {removed} cache file(s)")
    else:
        print("No cache files found")
    return removed


def train_pose_model(
    config: PoseEstimationConfig,
    data_yaml: Path,
    device: Union[int, str] = 0,
    # === GPU Optimization Parameters ===
    patience: int = 15,           # Early stopping patience (epochs without improvement)
    amp: bool = True,             # Mixed precision training (FP16)
    workers: int = 8,             # Data loading workers
    cache: Union[bool, str] = False,  # Dataset caching: False, "disk", or "ram"
    optimizer: str = "AdamW",     # Optimizer: "SGD", "Adam", "AdamW", etc.
    cos_lr: bool = True,          # Cosine learning rate scheduler
    warmup_epochs: int = 3,       # LR warmup epochs
    close_mosaic: int = 10,       # Disable mosaic augmentation in last N epochs
    model_name_override: Optional[str] = None,  # Override model name (e.g., "yolo11l-pose.pt")
) -> Any:
    """
    Train YOLO pose model with GPU optimizations and early stopping.
    
    Key optimizations for rented GPUs (L40S, A100, etc.):
    - Early stopping (patience) prevents wasted epochs when val loss plateaus
    - Mixed precision (amp) gives ~2x speedup with less VRAM usage
    - Higher workers for faster data loading on multi-core CPUs
    - Cosine LR for smoother convergence
    """
    if not HAS_ULTRALYTICS:
        raise ImportError("ultralytics is required. Install with: pip install ultralytics")
    params = config.get_training_params()
    model_name = model_name_override if model_name_override else config.get_yolo_model_name()
    
    print("🚀 Starting YOLO11 Pose training (GPU-Optimized)...")
    print(f"   Model: {model_name}")
    print(f"   Epochs: {params['epochs']} (max, early stopping enabled)")
    print(f"   Early Stopping: patience={patience} epochs")
    print(f"   Mixed Precision (AMP): {amp}")
    print(f"   Data: {data_yaml}")
    print(f"   Keypoints: {config.total_keypoints}")
    print(f"   Batch Size: {params['batch_size']}")
    print(f"   Workers: {workers}")
    print(f"   Cache: {cache}")
    print(f"   Optimizer: {optimizer}")
    print(f"   Cosine LR: {cos_lr}")
    print(f"   Experiment: {config.project_name}\n")

    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=params["epochs"],
        imgsz=config.img_size,
        batch=params["batch_size"],
        project=str(config.results_dir),
        name=config.project_name,
        device=device,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        # === GPU Optimizations ===
        patience=patience,        # Early stopping
        amp=amp,                  # Mixed precision (FP16)
        workers=workers,          # Data loading parallelism
        cache=cache,              # Dataset caching
        optimizer=optimizer,      # Optimizer choice
        cos_lr=cos_lr,            # Cosine LR schedule
        warmup_epochs=warmup_epochs,
        close_mosaic=close_mosaic,
        # === Additional stability settings ===
        val=True,                 # Enable validation
        exist_ok=True,            # Don't fail if project exists
    )
    print(f"\n✅ Training complete! Results saved to: {config.results_dir}/{config.project_name}/")
    return results
