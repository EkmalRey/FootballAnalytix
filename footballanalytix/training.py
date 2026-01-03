import os
from pathlib import Path
from typing import Any, Optional, Union
from .config import PoseEstimationConfig

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
