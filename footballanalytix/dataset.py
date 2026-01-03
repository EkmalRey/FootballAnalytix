from pathlib import Path
from typing import Dict, List
import json

def load_game_annotations(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_game_folders(data_root: Path, split: str = "train") -> List[Path]:
    split_dir = data_root / split
    if not split_dir.exists():
        return []
    return sorted(split_dir.glob("SNGS-*"))
