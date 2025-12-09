"""Raw dataset EDA and visualization helper.

Runs lightweight stats and leverages existing helpers to visualize raw
annotations (objects, pitch lines/keypoints) with optional figure saving.
Use as a CLI or import the functions in notebooks.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
import re
from contextlib import contextmanager
from pathlib import Path
import sys

# Add project root to sys.path to allow importing config
sys.path.append(str(Path(__file__).resolve().parents[1]))

from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt

from config import (
	ObjectDetectionConfig,
	PoseEstimationConfig,
	get_game_folders,
	load_game_annotations,
)

# Optional deps for visualization
try:
	import numpy as np  # type: ignore
	import cv2  # type: ignore
except Exception:  # pragma: no cover - soft dependency
	np = None  # type: ignore
	cv2 = None  # type: ignore
try:
	import supervision as sv  # type: ignore
	HAS_SUPERVISION = True
except Exception:  # pragma: no cover - soft dependency
	HAS_SUPERVISION = False


# ---------------------------------------------------------------------------
# Visualization helpers (local to this EDA script)
# ---------------------------------------------------------------------------

def _require_viz_deps() -> None:
	if np is None or cv2 is None:
		raise ImportError("numpy and opencv-python are required for visualization.")


def visualize_raw_objects(
	data: Dict,
	image_dir: Path,
	config: ObjectDetectionConfig,
	num_samples: int = 3,
) -> None:
	if not HAS_SUPERVISION:
		print("⚠️  supervision not installed; skipping object viz")
		return
	_require_viz_deps()

	id_to_role = {v: k for k, v in config.class_map.items()}
	images = {img["image_id"]: img for img in data.get("images", [])}
	by_image = defaultdict(list)
	for ann in data.get("annotations", []):
		if ann.get("supercategory") == "object":
			by_image[ann["image_id"]].append(ann)

	candidates = [i for i in by_image.keys() if i in images]
	sampled = random.sample(candidates, min(num_samples, len(candidates)))

	box_annotator = sv.BoxAnnotator(thickness=2)
	label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5)

	for img_id in sampled:
		info = images[img_id]
		img_path = image_dir / info["file_name"]
		if not img_path.exists():
			continue
		img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
		boxes = []
		class_ids = []
		for ann in by_image[img_id]:
			role = ann.get("attributes", {}).get("role")
			bbox = ann.get("bbox_image")
			if role not in config.class_map or not bbox:
				continue
			x1, y1 = bbox["x"], bbox["y"]
			x2, y2 = x1 + bbox["w"], y1 + bbox["h"]
			boxes.append([x1, y1, x2, y2])
			class_ids.append(config.class_map[role])
		if not boxes:
			continue
		dets = sv.Detections(xyxy=np.array(boxes), class_id=np.array(class_ids))  # type: ignore
		labels = [id_to_role[cid] for cid in dets.class_id]
		annotated = box_annotator.annotate(scene=img.copy(), detections=dets)
		annotated = label_annotator.annotate(scene=annotated, detections=dets, labels=labels)
		plt.figure(figsize=(12, 8))
		plt.imshow(annotated)
		plt.axis("off")
		plt.show()


def visualize_raw_pitch(
	data: Dict,
	image_dir: Path,
	config: PoseEstimationConfig,
	num_samples: int = 3,
) -> None:
	_require_viz_deps()
	images = {img["image_id"]: img for img in data.get("images", [])}
	by_image = defaultdict(list)
	for ann in data.get("annotations", []):
		by_image[ann["image_id"]].append(ann)
	candidates = [i for i in by_image.keys() if i in images]
	sampled = random.sample(candidates, min(num_samples, len(candidates)))

	for img_id in sampled:
		info = images[img_id]
		img_path = image_dir / info["file_name"]
		if not img_path.exists():
			continue
		img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
		w, h = info.get("width", img.shape[1]), info.get("height", img.shape[0])
		for ann in by_image[img_id]:
			if ann.get("supercategory") != "pitch":
				continue
			for line_name, points in (ann.get("lines") or {}).items():
				pts = [(int(p["x"] * w), int(p["y"] * h)) for p in points]
				if len(pts) >= 2:
					for i in range(len(pts) - 1):
						cv2.line(img, pts[i], pts[i + 1], config.line_color, 2)
				for pt in pts:
					cv2.circle(img, pt, 3, config.line_color, -1)
				if pts:
					cv2.putText(img, line_name, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.line_color, 1)
		plt.figure(figsize=(12, 8))
		plt.imshow(img)
		plt.axis("off")
		plt.show()


def visualize_raw_pitch_with_indices(
	data: Dict,
	image_dir: Path,
	num_samples: int = 3,
	canonical_order: Optional[List[Tuple[str, int]]] = None,
) -> None:
	_require_viz_deps()
	images = {img["image_id"]: img for img in data.get("images", [])}
	by_image = defaultdict(list)
	for ann in data.get("annotations", []):
		if ann.get("supercategory") == "pitch":
			by_image[ann["image_id"]].append(ann)
	candidates = [i for i in by_image.keys() if i in images]
	sampled = random.sample(candidates, min(num_samples, len(candidates)))

	for img_id in sampled:
		info = images[img_id]
		img_path = image_dir / info["file_name"]
		if not img_path.exists():
			continue
		img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
		w, h = info.get("width", img.shape[1]), info.get("height", img.shape[0])
		order_map = {(ln, idx): i for i, (ln, idx) in enumerate(canonical_order or [])}
		for ann in by_image[img_id]:
			lines = ann.get("lines") or {}
			for line_name, points in lines.items():
				for local_idx, pt in enumerate(points):
					px, py = int(pt["x"] * w), int(pt["y"] * h)
					label_idx = order_map.get((line_name, local_idx))
					label = f"{label_idx}" if label_idx is not None else f"{local_idx}"
					cv2.circle(img, (px, py), 4, (0, 0, 255), -1)
					cv2.putText(img, label, (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
		plt.figure(figsize=(12, 8))
		plt.imshow(img)
		plt.axis("off")
		plt.show()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_list(items: Sequence[str]) -> List[str]:
	return [str(i) for i in items]


def _default_splits(debug_train_only: bool) -> List[str]:
	return ["train"] if debug_train_only else ["train", "valid", "test"]


def _fmt_stats(number: float) -> float:
	return float(f"{number:.3f}")


def _safe_mean(values: Iterable[float]) -> float:
	vals = list(values)
	return _fmt_stats(mean(vals)) if vals else 0.0


@contextmanager
def capture_show(save: bool, save_dir: Path, prefix: str, display: bool):
	"""Patch matplotlib.show to optionally save figures.

	Any call to plt.show() inside this context will first save the current
	figure (if save=True) and then display/close depending on `display`.
	"""

	save_dir = save_dir.expanduser()
	counter = {"i": 0}
	original_show = plt.show

	def _patched_show(*args, **kwargs):
		counter["i"] += 1
		if save:
			save_dir.mkdir(parents=True, exist_ok=True)
			outfile = save_dir / f"{prefix}_{counter['i']:02d}.png"
			plt.savefig(outfile, bbox_inches="tight")
			print(f"Saved figure -> {outfile}")
		if display:
			original_show(*args, **kwargs)
		else:
			plt.close("all")

	plt.show = _patched_show
	try:
		yield counter
	finally:
		plt.show = original_show


# ---------------------------------------------------------------------------
# Stats collection
# ---------------------------------------------------------------------------

def collect_detection_stats(
	config: ObjectDetectionConfig,
	splits: Sequence[str],
	max_games: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
	stats: Dict[str, Dict[str, object]] = {}
	agg_class_counts: Counter[str] = Counter()
	agg_widths: List[float] = []
	agg_heights: List[float] = []
	agg_aspects: List[float] = []
	agg_images = 0
	agg_images_with = 0
	agg_boxes = 0
	for split in splits:
		class_counts: Counter[str] = Counter()
		widths: List[float] = []
		heights: List[float] = []
		aspects: List[float] = []
		images_total = 0
		images_with_objs = 0
		total_boxes = 0

		for game_idx, game_dir in enumerate(get_game_folders(config.data_root, split)):
			if max_games and game_idx >= max_games:
				break
			json_path = game_dir / "Labels-GameState.json"
			if not json_path.exists():
				continue
			data = load_game_annotations(json_path)
			images = {img["image_id"]: img for img in data.get("images", [])}
			by_image = defaultdict(list)
			for ann in data.get("annotations", []):
				if ann.get("supercategory") == "object":
					by_image[ann["image_id"]].append(ann)

			for img_id, info in images.items():
				images_total += 1
				anns = by_image.get(img_id, [])
				if not anns:
					continue
				images_with_objs += 1
				for ann in anns:
					role = ann.get("attributes", {}).get("role")
					bbox = ann.get("bbox_image")
					if role not in config.class_map or not bbox:
						continue
					class_counts[role] += 1
					total_boxes += 1
					w = float(bbox.get("w", 0))
					h = float(bbox.get("h", 0))
					if w > 0 and h > 0:
						widths.append(w)
						heights.append(h)
						aspects.append(w / h)

		agg_class_counts.update(class_counts)
		agg_widths.extend(widths)
		agg_heights.extend(heights)
		agg_aspects.extend(aspects)
		agg_images += images_total
		agg_images_with += images_with_objs
		agg_boxes += total_boxes

		stats[split] = {
			"images": images_total,
			"images_with_objects": images_with_objs,
			"total_boxes": total_boxes,
			"class_counts": dict(class_counts),
			"bbox_mean_w": _safe_mean(widths),
			"bbox_mean_h": _safe_mean(heights),
			"bbox_mean_aspect": _safe_mean(aspects),
		}

	stats["all"] = {
		"images": agg_images,
		"images_with_objects": agg_images_with,
		"total_boxes": agg_boxes,
		"class_counts": dict(agg_class_counts),
		"bbox_mean_w": _safe_mean(agg_widths),
		"bbox_mean_h": _safe_mean(agg_heights),
		"bbox_mean_aspect": _safe_mean(agg_aspects),
	}

	return stats


def collect_pitch_stats(
	config: PoseEstimationConfig,
	splits: Sequence[str],
	max_games: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
	stats: Dict[str, Dict[str, object]] = {}
	agg_images = 0
	agg_images_with = 0
	agg_line_counts: Counter[str] = Counter()
	agg_kps: List[int] = []
	for split in splits:
		images_total = 0
		images_with_pitch = 0
		line_counts: Counter[str] = Counter()
		keypoints_per_image: List[int] = []

		for game_idx, game_dir in enumerate(get_game_folders(config.data_root, split)):
			if max_games and game_idx >= max_games:
				break
			json_path = game_dir / "Labels-GameState.json"
			if not json_path.exists():
				continue
			data = load_game_annotations(json_path)
			images = {img["image_id"]: img for img in data.get("images", [])}
			by_image = defaultdict(list)
			for ann in data.get("annotations", []):
				if ann.get("supercategory") == "pitch":
					by_image[ann["image_id"]].append(ann)

			for img_id, info in images.items():
				images_total += 1
				anns = by_image.get(img_id, [])
				if not anns:
					continue
				images_with_pitch += 1
				ann = anns[0]
				lines = ann.get("lines", {})
				kp_count = 0
				for line_name, pts in lines.items():
					if not pts:
						continue
					line_counts[line_name] += 1
					kp_count += len(pts)
				keypoints_per_image.append(kp_count)

		agg_images += images_total
		agg_images_with += images_with_pitch
		agg_line_counts.update(line_counts)
		agg_kps.extend(keypoints_per_image)

		stats[split] = {
			"images": images_total,
			"images_with_pitch": images_with_pitch,
			"line_counts": dict(line_counts),
			"mean_keypoints_per_image": _safe_mean(keypoints_per_image),
		}

	stats["all"] = {
		"images": agg_images,
		"images_with_pitch": agg_images_with,
		"line_counts": dict(agg_line_counts),
		"mean_keypoints_per_image": _safe_mean(agg_kps),
	}

	return stats


# ---------------------------------------------------------------------------
# Tracking/Clustering ground truth export
# ---------------------------------------------------------------------------

TeamValue = Union[str, int, None]


def _normalize_team(team: TeamValue) -> Optional[Union[int, str]]:
	"""Map common team strings to ints while keeping unknowns usable."""
	if team is None:
		return None
	if isinstance(team, int):
		return team
	val = str(team).lower().strip()
	mapping = {
		"left": 0,
		"home": 0,
		"team_a": 0,
		"team0": 0,
		"0": 0,
		"right": 1,
		"away": 1,
		"team_b": 1,
		"team1": 1,
		"1": 1,
	}
	return mapping.get(val, team)


def _extract_track_id(ann: Dict) -> Optional[Union[int, str]]:
	attrs = ann.get("attributes", {}) or {}
	for key in ("track_id", "player_id", "player", "id", "global_track_id", "jersey_number"):
		val = attrs.get(key)
		if val is None:
			val = ann.get(key)
		if val is not None:
			try:
				return int(val)
			except Exception:
				return val
	return None


def _extract_frame_idx(info: Dict) -> int:
	file_name = info.get("file_name", "")
	stem = Path(file_name).stem
	match = re.findall(r"\d+", stem)
	if match:
		return int(match[-1])
	return int(info.get("frame_id", info.get("image_id", 0)))


def export_tracking_groundtruth(
	config: ObjectDetectionConfig,
	splits: Sequence[str],
	output_dir: Path,
	max_games: Optional[int] = None,
) -> Dict[str, object]:
	"""Export per-frame GT with ids/teams for tracking + clustering eval.

	Each game -> JSON with frames:{frame_idx:[{track_id, team, role, class_id, bbox}]}
	BBox is [x1, y1, x2, y2] in image coords.
	"""
	output_dir = output_dir.expanduser()
	output_dir.mkdir(parents=True, exist_ok=True)

	summary: Dict[str, object] = {}
	for split in splits:
		split_games = get_game_folders(config.data_root, split)
		if max_games:
			split_games = split_games[:max_games]
		if not split_games:
			print(f"No games found for split '{split}' - skipping GT export")
			summary[split] = {"games": 0, "frames": 0, "objects": 0}
			continue

		split_dir = output_dir / split
		split_dir.mkdir(parents=True, exist_ok=True)
		total_frames = 0
		total_objects = 0
		for game_dir in split_games:
			json_path = game_dir / "Labels-GameState.json"
			if not json_path.exists():
				print(f"Missing {json_path}, skipping game {game_dir.name}")
				continue
			data = load_game_annotations(json_path)
			images = {img["image_id"]: img for img in data.get("images", [])}
			frames: Dict[int, List[Dict[str, object]]] = defaultdict(list)

			for ann in data.get("annotations", []):
				if ann.get("supercategory") != "object":
					continue
				attrs = ann.get("attributes", {}) or {}
				role = attrs.get("role") or ann.get("category") or ann.get("role")
				if role is None:
					continue
				if role not in config.class_map:
					continue
				bbox = ann.get("bbox_image") or ann.get("bbox")
				if not bbox:
					continue
				x1, y1 = float(bbox.get("x", 0)), float(bbox.get("y", 0))
				w, h = float(bbox.get("w", 0)), float(bbox.get("h", 0))
				x2, y2 = x1 + w, y1 + h
				img_id = ann.get("image_id")
				img_info = images.get(img_id, {})
				frame_idx = _extract_frame_idx(img_info)
				track_id = _extract_track_id(ann)
				team = _normalize_team(attrs.get("team") or ann.get("team"))
				frames[frame_idx].append(
					{
						"track_id": track_id,
						"team": team,
						"role": role,
						"class_id": config.class_map.get(role),
						"bbox": [x1, y1, x2, y2],
					}
				)

			if not frames:
				print(f"No object annotations found in {game_dir.name}, skipping export")
				continue

			out_path = split_dir / f"{game_dir.name}_tracking_gt.json"
			payload = {
				"game": game_dir.name,
				"split": split,
				"class_map": config.class_map,
				"frames": {int(k): v for k, v in frames.items()},
			}
			with out_path.open("w", encoding="utf-8") as f:
				json.dump(payload, f, indent=2)
			print(f"✓ Saved tracking GT -> {out_path}")
			total_frames += len(frames)
			total_objects += sum(len(v) for v in frames.values())

		summary[split] = {
			"games": len(split_games),
			"frames": total_frames,
			"objects": total_objects,
		}

	return summary


def print_block(title: str, payload: Dict[str, object]) -> None:
	print("\n" + "=" * 64)
	print(title)
	print("=" * 64)
	for key, val in payload.items():
		print(f"{key}: {val}")


# ---------------------------------------------------------------------------
# Visualization runners
# ---------------------------------------------------------------------------

def visualize_detection_samples(
	config: ObjectDetectionConfig,
	split: str,
	num_samples: int,
	save_figs: bool,
	display: bool,
	out_dir: Path,
) -> None:
	game_dirs = get_game_folders(config.data_root, split)
	if not game_dirs:
		print(f"No games found for split '{split}'. Skipping detection viz.")
		return
	json_path = game_dirs[0] / "Labels-GameState.json"
	img_dir = game_dirs[0] / "img1"
	if not json_path.exists() or not img_dir.exists():
		print(f"Missing data for split '{split}'. Skipping detection viz.")
		return
	data = load_game_annotations(json_path)
	with capture_show(save_figs, out_dir, f"det_{split}", display):
		visualize_raw_objects(data, img_dir, config, num_samples=num_samples)


def visualize_pitch_samples(
	config: PoseEstimationConfig,
	split: str,
	num_samples: int,
	save_figs: bool,
	display: bool,
	out_dir: Path,
	with_indices: bool,
) -> None:
	game_dirs = get_game_folders(config.data_root, split)
	if not game_dirs:
		print(f"No games found for split '{split}'. Skipping pitch viz.")
		return
	json_path = game_dirs[0] / "Labels-GameState.json"
	img_dir = game_dirs[0] / "img1"
	if not json_path.exists() or not img_dir.exists():
		print(f"Missing data for split '{split}'. Skipping pitch viz.")
		return
	data = load_game_annotations(json_path)
	prefix = f"pitch_{split}" if not with_indices else f"pitch_idx_{split}"
	with capture_show(save_figs, out_dir, prefix, display):
		if with_indices:
			visualize_raw_pitch_with_indices(
				data,
				img_dir,
				num_samples=num_samples,
				canonical_order=config.canonical_kp_order,
			)
		else:
			visualize_raw_pitch(data, img_dir, config, num_samples=num_samples)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="EDA and visualization for raw SoccerNet GSR data")
	parser.add_argument("--data-root", type=Path, default=Path("./SN-GSR-2025"), help="Dataset root")
	parser.add_argument(
		"--mode",
		choices=["detection", "pitch", "both"],
		default="both",
		help="Which annotation types to process",
	)
	parser.add_argument(
		"--splits",
		nargs="+",
		help="Splits to scan (default: train/valid/test or train only when debug_train_only)",
	)
	parser.add_argument("--max-games", type=int, default=None, help="Limit number of games per split for stats")
	parser.add_argument("--num-samples-detection", type=int, default=3, help="Samples to visualize for objects")
	parser.add_argument("--num-samples-pitch", type=int, default=3, help="Samples to visualize for pitch")
	parser.add_argument("--save-figs", action="store_true", help="Save figures instead of display-only")
	parser.add_argument("--no-display", action="store_true", help="Do not display figures (saves only)")
	parser.add_argument("--with-indices", action="store_true", help="Use indexed pitch visualization")
	parser.add_argument(
		"--export-tracking-gt",
		action="store_true",
		help="Export per-frame tracking GT with ids/teams for evaluation",
	)
	parser.add_argument(
		"--tracking-out",
		type=Path,
		default=Path("./eda_outputs/tracking_gt"),
		help="Directory to dump tracking/clustering ground truth JSON",
	)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=Path("./eda_outputs"),
		help="Directory to save figures and summaries",
	)
	parser.add_argument("--export-summary", type=Path, default=None, help="Optional path to dump stats JSON")
	return parser.parse_args()


def main():
	args = parse_args()

	det_config = ObjectDetectionConfig(data_root=args.data_root)
	pose_config = PoseEstimationConfig(data_root=args.data_root)

	splits = _ensure_list(args.splits) if args.splits else _default_splits(det_config.debug_train_only)
	show_figs = not args.no_display
	save_figs = bool(args.save_figs)

	summary: Dict[str, Dict[str, object]] = {}

	if args.mode in {"detection", "both"}:
		det_stats = collect_detection_stats(det_config, splits, max_games=args.max_games)
		summary["detection"] = det_stats
		print_block("[Detection] Stats (per split)", {k: v for k, v in det_stats.items() if k != "all"})
		print_block("[Detection] Stats (all)", det_stats.get("all", {}))
		if args.num_samples_detection > 0:
			visualize_detection_samples(
				det_config,
				splits[0],
				num_samples=args.num_samples_detection,
				save_figs=save_figs,
				display=show_figs,
				out_dir=args.out_dir,
			)

	if args.mode in {"pitch", "both"}:
		pitch_stats = collect_pitch_stats(pose_config, splits, max_games=args.max_games)
		summary["pitch"] = pitch_stats
		print_block("[Pitch] Stats (per split)", {k: v for k, v in pitch_stats.items() if k != "all"})
		print_block("[Pitch] Stats (all)", pitch_stats.get("all", {}))
		if args.num_samples_pitch > 0:
			visualize_pitch_samples(
				pose_config,
				splits[0],
				num_samples=args.num_samples_pitch,
				save_figs=save_figs,
				display=show_figs,
				out_dir=args.out_dir,
				with_indices=args.with_indices,
			)

	if args.export_tracking_gt:
		tracking_summary = export_tracking_groundtruth(det_config, splits, args.tracking_out, max_games=args.max_games)
		summary["tracking_gt"] = tracking_summary
		print_block("[Tracking GT] Export", tracking_summary)

	if args.export_summary:
		args.out_dir.mkdir(parents=True, exist_ok=True)
		export_path = args.export_summary
		with export_path.open("w", encoding="utf-8") as f:
			json.dump(summary, f, indent=2)
		print(f"Summary saved to {export_path}")


if __name__ == "__main__":
	main()
