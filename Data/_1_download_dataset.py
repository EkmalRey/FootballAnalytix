"""Download helper split from the retired training_helper for reuse."""

from pathlib import Path
import sys

# Add project root to sys.path to allow importing config
sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import Any, List, Optional
import glob
import os
import zipfile

try:
    from huggingface_hub import snapshot_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


def download_dataset(config: Any,
                     repo_id: str = "SoccerNet/SN-GSR-2025",
                     patterns: Optional[List[str]] = None,
                     debug_test_only: bool = False) -> Path:
    """Download dataset from Hugging Face Hub and extract splits.

    debug_train_only: only train.zip
    debug_test_only: only test.zip
    """
    if not HAS_HF_HUB:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

    # Normalize config attributes we need
    debug_train_only = bool(getattr(config, "debug_train_only", False))
    data_root = Path(getattr(config, "data_root", Path("./SN-GSR-2025")))
    if hasattr(config, "data_root"):
        config.data_root = data_root

    if patterns is None:
        patterns = ["train.zip", "valid.zip", "test.zip"]
        if debug_train_only and debug_test_only:
            raise ValueError("debug_train_only and debug_test_only cannot both be True")
        if debug_train_only:
            patterns = ["train.zip"]
        if debug_test_only:
            patterns = ["test.zip"]

    print("📥 Downloading SoccerNet GSR dataset...")

    # Check if already downloaded
    if data_root.exists():
        train_dir = data_root / "train"
        valid_dir = data_root / "valid"
        test_dir = data_root / "test"

        if debug_train_only:
            if train_dir.exists() and any(train_dir.glob("SNGS-*")):
                print("   ⏭️  Dataset already downloaded and extracted (train only)")
                return data_root
        elif debug_test_only:
            if test_dir.exists() and any(test_dir.glob("SNGS-*")):
                print("   ⏭️  Dataset already downloaded and extracted (test only)")
                return data_root
        else:
            if all(p.exists() for p in [train_dir, valid_dir, test_dir]) and len(list(data_root.glob("*"))) >= 3:
                print("   ⏭️  Dataset already downloaded and extracted")
                return data_root

    # Download dataset
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(data_root),
        allow_patterns=patterns
    )

    # Extract zip files
    for zip_path in glob.glob(os.path.join(local_dir, "*.zip")):
        name = os.path.splitext(os.path.basename(zip_path))[0]
        target_dir = os.path.join(local_dir, name)

        if not os.path.exists(target_dir):
            print(f"📦 Extracting {os.path.basename(zip_path)}...")
            os.makedirs(target_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(target_dir)

            print(f"   ✅ Extracted to {target_dir}")
            os.remove(zip_path)
        else:
            print(f"   ⏭️  {name}/ already exists, skipping extraction")

    # Verify dataset structure
    train_dir = data_root / "train"
    valid_dir = data_root / "valid"
    test_dir = data_root / "test"

    if train_dir.exists() or test_dir.exists():
        print("\n✅ Dataset ready:")
        if train_dir.exists():
            train_games = list(train_dir.glob("SNGS-*"))
            print(f"   📁 Train games: {len(train_games)}")
        if valid_dir.exists():
            valid_games = list(valid_dir.glob("SNGS-*"))
            print(f"   📁 Valid games: {len(valid_games)}")
        if test_dir.exists():
            test_games = list(test_dir.glob("SNGS-*"))
            print(f"   📁 Test games : {len(test_games)}")
        if debug_train_only:
            print("   🧪 Debug mode: train-only download")
        if debug_test_only:
            print("   🧪 Debug mode: test-only download")
    else:
        print("\n⚠️  Warning: Dataset structure not found. Check download.")

    return data_root