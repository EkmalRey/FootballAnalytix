"""Convert raw SoccerNet-style pitch annotations to YOLO (SoccerNet keypoint variant).

Credits: derived from the Soccer analysis utilities in
https://github.com/Adit-jain/Soccer_Analysis

Runs after `_2_process_raw_dataset.py`. Leverages the SoccerNet keypoint
pipeline (line intersections + pitch detection) to produce:

- YOLO pose labels with 29 field keypoints (SoccerNet variant order)
- Copied images with unique stems per game
- Optional unified JSON artifacts for debugging
- A dataset.yaml targeting the generated layout

Relies on `config.py` for config defaults and dataset discovery.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path to allow importing config
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import math
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tqdm

# Project-relative imports
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent  # .../YOLO
EXTERNAL_HELP_DIR = PROJECT_ROOT / "External Help"
if str(EXTERNAL_HELP_DIR) not in sys.path:
	sys.path.append(str(EXTERNAL_HELP_DIR))

from footballanalytix.config import PoseEstimationConfig
from footballanalytix.dataset import (
	get_game_folders,
	load_game_annotations,
)


class LineIntersectionCalculator:
	"""Calculate field keypoints from SoccerNet line endpoints by computing line intersections."""

	def __init__(self) -> None:
		self.field_keypoints: Dict[str, Tuple[float, float]] = {}
		self.lines: Dict = {}

	def load_soccernet_data(self, json_path: str) -> Dict:
		with open(json_path, "r") as f:
			data = json.load(f)
		self.lines = data
		return data

	def normalize_coordinates(self, point: Dict[str, float], image_shape: Tuple[int, int]) -> Tuple[int, int]:
		height, width = image_shape[:2]
		x = int(point["x"] * width)
		y = int(point["y"] * height)
		return x, y

	def line_intersection(self, line1: List[Dict], line2: List[Dict]) -> Optional[Tuple[float, float]]:
		if len(line1) < 2 or len(line2) < 2:
			return None

		x1, y1 = line1[0]["x"], line1[0]["y"]
		x2, y2 = line1[1]["x"], line1[1]["y"]
		x3, y3 = line2[0]["x"], line2[0]["y"]
		x4, y4 = line2[1]["x"], line2[1]["y"]

		denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
		if abs(denom) < 1e-10:
			return None

		t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
		u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

		x_intersect = x1 + t * (x2 - x1)
		y_intersect = y1 + t * (y2 - y1)

		if x_intersect < 0.0 or x_intersect > 1.0 or y_intersect < 0.0 or y_intersect > 1.0:
			return None

		return x_intersect, y_intersect

	def point_to_line_distance(self, point: Dict[str, float], line: List[Dict]) -> float:
		if len(line) < 2:
			return float("inf")

		x0, y0 = point["x"], point["y"]
		x1, y1 = line[0]["x"], line[0]["y"]
		x2, y2 = line[1]["x"], line[1]["y"]

		a = y2 - y1
		b = x1 - x2
		c = (x2 - x1) * y1 - (y2 - y1) * x1

		if a == 0 and b == 0:
			return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

		return abs(a * x0 + b * y0 + c) / math.sqrt(a * a + b * b)

	def circle_line_intersection(self, circle_points: List[Dict], line: List[Dict]) -> List[Tuple[float, float]]:
		if len(line) < 2 or len(circle_points) < 3:
			return []

		xs = [p["x"] for p in circle_points]
		ys = [p["y"] for p in circle_points]
		center_x = sum(xs) / len(xs)
		center_y = sum(ys) / len(ys)
		distances = [math.sqrt((p["x"] - center_x) ** 2 + (p["y"] - center_y) ** 2) for p in circle_points]
		radius = sum(distances) / len(distances)

		x1, y1 = line[0]["x"], line[0]["y"]
		x2, y2 = line[1]["x"], line[1]["y"]

		if abs(x2 - x1) < 1e-10:
			a = 1
			b = 0
			c = -x1
		else:
			slope = (y2 - y1) / (x2 - x1)
			a = slope
			b = -1
			c = y1 - slope * x1

		intersections: List[Tuple[float, float]] = []

		if abs(b) > 1e-10:
			A = 1 + (a / b) ** 2
			B = 2 * ((a * c) / (b ** 2) + (a * center_y) / b - center_x)
			C = (c / b + center_y) ** 2 + center_x ** 2 - radius ** 2
			discriminant = B ** 2 - 4 * A * C
			if discriminant >= 0:
				sqrt_d = math.sqrt(discriminant)
				x_int1 = (-B + sqrt_d) / (2 * A)
				x_int2 = (-B - sqrt_d) / (2 * A)
				y_int1 = (-a * x_int1 - c) / b
				y_int2 = (-a * x_int2 - c) / b
				intersections.append((x_int1, y_int1))
				if discriminant > 0:
					intersections.append((x_int2, y_int2))
		else:
			y_const = -c / a
			dx_squared = radius ** 2 - (y_const - center_y) ** 2
			if dx_squared >= 0:
				dx = math.sqrt(dx_squared)
				intersections.append((center_x + dx, y_const))
				if dx > 0:
					intersections.append((center_x - dx, y_const))

		return intersections

	def extend_line(self, line: List[Dict], extension_factor: float = 2.0) -> List[Dict]:
		if len(line) < 2:
			return line

		x1, y1 = line[0]["x"], line[0]["y"]
		x2, y2 = line[1]["x"], line[1]["y"]
		dx = x2 - x1
		dy = y2 - y1

		new_x1 = x1 - dx * (extension_factor - 1) / 2
		new_y1 = y1 - dy * (extension_factor - 1) / 2
		new_x2 = x2 + dx * (extension_factor - 1) / 2
		new_y2 = y2 + dy * (extension_factor - 1) / 2

		return [{"x": new_x1, "y": new_y1}, {"x": new_x2, "y": new_y2}]

	def calculate_field_keypoints(self) -> Dict[str, Tuple[float, float]]:
		keypoints: Dict[str, Tuple[float, float]] = {}

		def get_line(key: str) -> List[Dict]:
			return self.lines.get(key, [])

		side_line_top = get_line("Side line top")
		side_line_bottom = get_line("Side line bottom")
		side_line_left = get_line("Side line left")
		side_line_right = get_line("Side line right")
		big_rect_left_top = get_line("Big rect. left top")
		big_rect_left_main = get_line("Big rect. left main")
		big_rect_left_bottom = get_line("Big rect. left bottom")
		big_rect_right_top = get_line("Big rect. right top")
		big_rect_right_main = get_line("Big rect. right main")
		big_rect_right_bottom = get_line("Big rect. right bottom")
		small_rect_left_top = get_line("Small rect. left top")
		small_rect_left_main = get_line("Small rect. left main")
		small_rect_left_bottom = get_line("Small rect. left bottom")
		small_rect_right_top = get_line("Small rect. right top")
		small_rect_right_main = get_line("Small rect. right main")
		small_rect_right_bottom = get_line("Small rect. right bottom")
		middle_line = get_line("Middle line")
		circle_central = get_line("Circle central")
		circle_left = get_line("Circle left")
		circle_right = get_line("Circle right")

		if side_line_top and side_line_left:
			pt = self.line_intersection(side_line_top, side_line_left)
			if pt:
				keypoints["0_sideline_top_left"] = pt

		if side_line_left and big_rect_left_top:
			pt = self.line_intersection(side_line_left, big_rect_left_top)
			if pt:
				keypoints["1_big_rect_left_top_pt1"] = pt

		if big_rect_left_top and big_rect_left_main:
			pt = self.line_intersection(big_rect_left_top, big_rect_left_main)
			if pt:
				keypoints["2_big_rect_left_top_pt2"] = pt

		if side_line_left and big_rect_left_bottom:
			pt = self.line_intersection(side_line_left, big_rect_left_bottom)
			if pt:
				keypoints["3_big_rect_left_bottom_pt1"] = pt

		if big_rect_left_bottom and big_rect_left_main:
			pt = self.line_intersection(big_rect_left_bottom, big_rect_left_main)
			if pt:
				keypoints["4_big_rect_left_bottom_pt2"] = pt

		if side_line_left and small_rect_left_top:
			pt = self.line_intersection(side_line_left, small_rect_left_top)
			if pt:
				keypoints["5_small_rect_left_top_pt1"] = pt

		if small_rect_left_top and small_rect_left_main:
			pt = self.line_intersection(small_rect_left_top, small_rect_left_main)
			if pt:
				keypoints["6_small_rect_left_top_pt2"] = pt

		if side_line_left and small_rect_left_bottom:
			pt = self.line_intersection(side_line_left, small_rect_left_bottom)
			if pt:
				keypoints["7_small_rect_left_bottom_pt1"] = pt

		if small_rect_left_bottom and small_rect_left_main:
			pt = self.line_intersection(small_rect_left_bottom, small_rect_left_main)
			if pt:
				keypoints["8_small_rect_left_bottom_pt2"] = pt

		if side_line_bottom and side_line_left:
			pt = self.line_intersection(side_line_bottom, side_line_left)
			if pt:
				keypoints["9_sideline_bottom_left"] = pt

		if circle_left and big_rect_left_main:
			far = max(circle_left, key=lambda p: self.point_to_line_distance(p, big_rect_left_main))
			if 0.0 <= far["x"] <= 1.0 and 0.0 <= far["y"] <= 1.0:
				keypoints["10_left_semicircle_right"] = (far["x"], far["y"])

		if middle_line and side_line_top:
			pt = self.line_intersection(middle_line, side_line_top)
			if pt:
				keypoints["11_center_line_top"] = pt

		if middle_line and side_line_bottom:
			pt = self.line_intersection(middle_line, side_line_bottom)
			if pt:
				keypoints["12_center_line_bottom"] = pt

		if circle_central and middle_line:
			ys = [p["y"] for p in circle_central]
			median_y = sorted(ys)[len(ys) // 2]
			upper = [p for p in circle_central if p["y"] <= median_y]
			if len(upper) >= 2:
				closest = sorted(upper, key=lambda p: self.point_to_line_distance(p, middle_line))[:2]
				top_line = [closest[0], closest[1]]
				pt = self.line_intersection(top_line, middle_line)
				if pt:
					keypoints["13_center_circle_top"] = pt

		if circle_central and middle_line:
			ys = [p["y"] for p in circle_central]
			median_y = sorted(ys)[len(ys) // 2]
			lower = [p for p in circle_central if p["y"] > median_y]
			if len(lower) >= 2:
				closest = sorted(lower, key=lambda p: self.point_to_line_distance(p, middle_line))[:2]
				bottom_line = [closest[0], closest[1]]
				pt = self.line_intersection(bottom_line, middle_line)
				if pt:
					keypoints["14_center_circle_bottom"] = pt

		if "13_center_circle_top" in keypoints and "14_center_circle_bottom" in keypoints:
			tx, ty = keypoints["13_center_circle_top"]
			bx, by = keypoints["14_center_circle_bottom"]
			keypoints["15_field_center"] = ((tx + bx) / 2, (ty + by) / 2)

		if side_line_top and side_line_right:
			pt = self.line_intersection(side_line_top, side_line_right)
			if pt:
				keypoints["16_sideline_top_right"] = pt

		if side_line_right and big_rect_right_top:
			pt = self.line_intersection(side_line_right, big_rect_right_top)
			if pt:
				keypoints["17_big_rect_right_top_pt1"] = pt

		if big_rect_right_top and big_rect_right_main:
			pt = self.line_intersection(big_rect_right_top, big_rect_right_main)
			if pt:
				keypoints["18_big_rect_right_top_pt2"] = pt

		if side_line_right and big_rect_right_bottom:
			pt = self.line_intersection(side_line_right, big_rect_right_bottom)
			if pt:
				keypoints["19_big_rect_right_bottom_pt1"] = pt

		if big_rect_right_bottom and big_rect_right_main:
			pt = self.line_intersection(big_rect_right_bottom, big_rect_right_main)
			if pt:
				keypoints["20_big_rect_right_bottom_pt2"] = pt

		if side_line_right and small_rect_right_top:
			pt = self.line_intersection(side_line_right, small_rect_right_top)
			if pt:
				keypoints["21_small_rect_right_top_pt1"] = pt

		if small_rect_right_top and small_rect_right_main:
			pt = self.line_intersection(small_rect_right_top, small_rect_right_main)
			if pt:
				keypoints["22_small_rect_right_top_pt2"] = pt

		if side_line_right and small_rect_right_bottom:
			pt = self.line_intersection(side_line_right, small_rect_right_bottom)
			if pt:
				keypoints["23_small_rect_right_bottom_pt1"] = pt

		if small_rect_right_bottom and small_rect_right_main:
			pt = self.line_intersection(small_rect_right_bottom, small_rect_right_main)
			if pt:
				keypoints["24_small_rect_right_bottom_pt2"] = pt

		if side_line_bottom and side_line_right:
			pt = self.line_intersection(side_line_bottom, side_line_right)
			if pt:
				keypoints["25_sideline_bottom_right"] = pt

		if circle_right and big_rect_right_main:
			far = max(circle_right, key=lambda p: self.point_to_line_distance(p, big_rect_right_main))
			if 0.0 <= far["x"] <= 1.0 and 0.0 <= far["y"] <= 1.0:
				keypoints["26_right_semicircle_left"] = (far["x"], far["y"])

		if circle_central and middle_line:
			xs = [p["x"] for p in circle_central]
			median_x = sorted(xs)[len(xs) // 2]
			left_points = [p for p in circle_central if p["x"] <= median_x]
			if left_points:
				far_left = max(left_points, key=lambda p: self.point_to_line_distance(p, middle_line))
				if 0.0 <= far_left["x"] <= 1.0 and 0.0 <= far_left["y"] <= 1.0:
					keypoints["27_center_circle_left"] = (far_left["x"], far_left["y"])

		if circle_central and middle_line:
			xs = [p["x"] for p in circle_central]
			median_x = sorted(xs)[len(xs) // 2]
			right_points = [p for p in circle_central if p["x"] > median_x]
			if right_points:
				far_right = max(right_points, key=lambda p: self.point_to_line_distance(p, middle_line))
				if 0.0 <= far_right["x"] <= 1.0 and 0.0 <= far_right["y"] <= 1.0:
					keypoints["28_center_circle_right"] = (far_right["x"], far_right["y"])

		self.field_keypoints = keypoints
		return self.field_keypoints, self.lines

	def visualize_keypoints(self, image_path: str, keypoints: Dict = None, lines: Dict = None, output_path: str = None) -> None:
		image = cv2.imread(image_path)
		if image is None:
			return

		height, width = image.shape[:2]

		if lines is not None:
			for line_name, line_points in lines.items():
				if len(line_points) >= 2 and line_name not in ["Circle left", "Circle right"]:
					pt1 = self.normalize_coordinates(line_points[0], image.shape)
					pt2 = self.normalize_coordinates(line_points[1], image.shape)
					cv2.line(image, pt1, pt2, (0, 255, 0), 2)
					cv2.putText(image, line_name[:10], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

			if "Circle left" in lines:
				for point in lines["Circle left"]:
					pt = self.normalize_coordinates(point, image.shape)
					cv2.circle(image, pt, 3, (0, 255, 0), -1)

			if "Circle right" in lines:
				for point in lines["Circle right"]:
					pt = self.normalize_coordinates(point, image.shape)
					cv2.circle(image, pt, 3, (0, 255, 0), -1)

		if keypoints is not None:
			for keypoint_name, (x, y) in keypoints.items():
				pt = (int(x * width), int(y * height))
				cv2.circle(image, pt, 8, (0, 0, 255), -1)
				cv2.putText(image, keypoint_name, (pt[0] + 10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

		if output_path:
			cv2.imwrite(output_path, image)
		else:
			cv2.imshow("Field Keypoints", image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()


class PitchDetector:
	"""Detect the pitch object (green area) in soccer field images."""

	def __init__(self) -> None:
		self.lower_green = np.array([35, 40, 40])
		self.upper_green = np.array([85, 255, 255])

	def detect_green_area(self, image: np.ndarray) -> np.ndarray:
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
		kernel = np.ones((5, 5), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		return mask

	def find_largest_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if not contours:
			return None
		return max(contours, key=cv2.contourArea)

	def get_pitch_bounding_box(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> Optional[Dict[str, float]]:
		if contour is None:
			return None

		height, width = image_shape[:2]
		x, y, w, h = cv2.boundingRect(contour)
		x_min_norm = x / width
		y_min_norm = y / height
		x_max_norm = (x + w) / width
		y_max_norm = (y + h) / height
		center_x = (x_min_norm + x_max_norm) / 2
		center_y = (y_min_norm + y_max_norm) / 2
		bbox_width = x_max_norm - x_min_norm
		bbox_height = y_max_norm - y_min_norm

		return {
			"class_id": 0,
			"class_name": "pitch",
			"center_x": center_x,
			"center_y": center_y,
			"width": bbox_width,
			"height": bbox_height,
			"x_min": x_min_norm,
			"y_min": y_min_norm,
			"x_max": x_max_norm,
			"y_max": y_max_norm,
			"area": bbox_width * bbox_height,
			"contour_area": cv2.contourArea(contour) / (width * height),
		}

	def detect_pitch_from_image(self, image_path: str) -> Optional[Dict]:
		image = cv2.imread(image_path)
		if image is None:
			return None

		green_mask = self.detect_green_area(image)
		largest_contour = self.find_largest_contour(green_mask)
		if largest_contour is None:
			return None

		pitch_bbox = self.get_pitch_bounding_box(largest_contour, image.shape)
		if pitch_bbox is None:
			return None

		return {
			"image_path": image_path,
			"image_shape": {"height": image.shape[0], "width": image.shape[1]},
			"pitch_detection": pitch_bbox,
		}

	def visualize_detection(self, image_path: str, detection_result: Dict, output_path: Optional[str] = None) -> None:
		image = cv2.imread(image_path)
		if image is None:
			return

		height, width = image.shape[:2]
		pitch_data = detection_result["pitch_detection"]

		x_min = int(pitch_data["x_min"] * width)
		y_min = int(pitch_data["y_min"] * height)
		x_max = int(pitch_data["x_max"] * width)
		y_max = int(pitch_data["y_max"] * height)

		cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
		text = f"Pitch (Area: {pitch_data['area']:.3f})"
		cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

		green_mask = self.detect_green_area(image)
		green_overlay = cv2.applyColorMap(green_mask, cv2.COLORMAP_JET)
		overlay = cv2.addWeighted(image, 0.7, green_overlay, 0.3, 0)

		if output_path:
			cv2.imwrite(output_path, overlay)
		else:
			cv2.imshow("Pitch Detection", overlay)
			cv2.waitKey(0)
			cv2.destroyAllWindows()


# Canonical SoccerNet variant keypoint order (29 pts)
KEYPOINT_ORDER: List[str] = [
	"0_sideline_top_left",
	"1_big_rect_left_top_pt1",
	"2_big_rect_left_top_pt2",
	"3_big_rect_left_bottom_pt1",
	"4_big_rect_left_bottom_pt2",
	"5_small_rect_left_top_pt1",
	"6_small_rect_left_top_pt2",
	"7_small_rect_left_bottom_pt1",
	"8_small_rect_left_bottom_pt2",
	"9_sideline_bottom_left",
	"10_left_semicircle_right",
	"11_center_line_top",
	"12_center_line_bottom",
	"13_center_circle_top",
	"14_center_circle_bottom",
	"15_field_center",
	"16_sideline_top_right",
	"17_big_rect_right_top_pt1",
	"18_big_rect_right_top_pt2",
	"19_big_rect_right_bottom_pt1",
	"20_big_rect_right_bottom_pt2",
	"21_small_rect_right_top_pt1",
	"22_small_rect_right_top_pt2",
	"23_small_rect_right_bottom_pt1",
	"24_small_rect_right_bottom_pt2",
	"25_sideline_bottom_right",
	"26_right_semicircle_left",
	"27_center_circle_left",
	"28_center_circle_right",
]


@dataclass
class ConvertConfig:
	data_root: Path
	yolo_root: Path
	splits: List[str]
	max_games: Optional[int]
	max_images: Optional[int]
	overwrite: bool
	save_unified_json: bool
	convert_workers: Optional[int] = None


def _at_image_limit(counter: Optional[Any], lock: Optional[Any], max_images: Optional[int]) -> bool:
	if counter is None or lock is None or max_images is None:
		return False
	with lock:
		return counter.value >= max_images


def _reserve_slot(counter: Optional[Any], lock: Optional[Any], max_images: Optional[int]) -> bool:
	if counter is None or lock is None or max_images is None:
		return True
	with lock:
		if counter.value >= max_images:
			return False
		counter.value += 1
		return True


def split_train_to_subsets(
	base_out: Path,
	val_ratio: float = 0.15,
	test_ratio: float = 0.10,
	seed: int = 42,
) -> Dict[str, int]:
	"""Split train images/labels into train/val/test folders.
	
	Moves a portion of the converted train data into val/ and test/ directories
	for use when DEBUG_TRAIN_ONLY mode is enabled (no separate val/test downloads).
	
	Args:
		base_out: Base YOLO dataset directory (contains images/ and labels/).
		val_ratio: Fraction of train data to move to val split.
		test_ratio: Fraction of train data to move to test split.
		seed: Random seed for reproducible shuffling.
	
	Returns:
		Dict with final counts: {"train": N, "val": M, "test": K}
	"""
	import random
	random.seed(seed)
	
	train_imgs = base_out / "images" / "train"
	train_lbls = base_out / "labels" / "train"
	
	all_files = sorted(train_imgs.glob("*.jpg"))
	if not all_files:
		print("No train images found to split.")
		return {"train": 0, "val": 0, "test": 0}
	
	random.shuffle(all_files)
	
	n = len(all_files)
	n_test = int(n * test_ratio)
	n_val = int(n * val_ratio)
	
	test_files = all_files[:n_test]
	val_files = all_files[n_test : n_test + n_val]
	# Remaining files stay in train
	
	for split, files in [("val", val_files), ("test", test_files)]:
		img_dst = base_out / "images" / split
		lbl_dst = base_out / "labels" / split
		img_dst.mkdir(parents=True, exist_ok=True)
		lbl_dst.mkdir(parents=True, exist_ok=True)
		
		for src_img in files:
			src_lbl = train_lbls / f"{src_img.stem}.txt"
			dst_img_path = img_dst / src_img.name
			dst_lbl_path = lbl_dst / src_lbl.name
			
			# Move image
			shutil.move(str(src_img), str(dst_img_path))
			# Move label if exists
			if src_lbl.exists():
				shutil.move(str(src_lbl), str(dst_lbl_path))
	
	final_counts = {
		"train": n - n_val - n_test,
		"val": n_val,
		"test": n_test,
	}
	print(f"Split train into subsets: {final_counts}")
	return final_counts


def _create_dirs(base: Path, split: str) -> Dict[str, Path]:
	paths = {
		"images": base / "images" / split,
		"labels": base / "labels" / split,
		"json": base / "annotations_json" / split,
	}
	for p in paths.values():
		p.mkdir(parents=True, exist_ok=True)
	return paths


def _calc_keypoints(lines: Dict) -> Optional[Dict[str, Tuple[float, float]]]:
	calc = LineIntersectionCalculator()
	calc.lines = lines or {}
	# calculate_field_keypoints historically returned only the keypoints dict;
	# guard against older tuple-return signatures to keep notebook stable.
	result = calc.calculate_field_keypoints()
	keypoints = result[0] if isinstance(result, (list, tuple)) and len(result) > 0 else result
	return keypoints if keypoints else None


def _format_yolo_annotation(pitch_data: Dict, keypoints: Dict[str, Tuple[float, float]]) -> str:
	parts: List[str] = [
		"0",
		f"{pitch_data['center_x']:.6f}",
		f"{pitch_data['center_y']:.6f}",
		f"{pitch_data['width']:.6f}",
		f"{pitch_data['height']:.6f}",
	]

	for kp_name in KEYPOINT_ORDER:
		if kp_name in keypoints:
			x, y = keypoints[kp_name]
			parts.extend([f"{x:.6f}", f"{y:.6f}", "2"])
		else:
			parts.extend(["0.0", "0.0", "0"])

	return " ".join(parts)


def visualize_yolo_sample(image_path: Path, label_path: Path, ax=None):
	"""Visualize a YOLO image with its label annotations on-demand.
	
	Reads raw image and YOLO label file, draws bounding box and keypoints.
	Useful for QA without saving pre-rendered images during conversion.
	
	Args:
		image_path: Path to the image file.
		label_path: Path to the corresponding YOLO label file.
		ax: Optional matplotlib axis. If None, creates a new figure.
	
	Returns:
		The annotated image as a numpy array (RGB).
	"""
	import cv2
	import numpy as np
	
	img = cv2.imread(str(image_path))
	if img is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")
	
	h, w = img.shape[:2]
	
	# Parse label file
	if not label_path.exists():
		raise FileNotFoundError(f"Label file not found: {label_path}")
	
	lines = label_path.read_text().strip().splitlines()
	if not lines:
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	# Parse first annotation line
	parts = lines[0].split()
	if len(parts) < 5:
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	# Extract bounding box (YOLO format: center_x, center_y, width, height)
	center_x, center_y, box_w, box_h = map(float, parts[1:5])
	x_min = int((center_x - box_w / 2) * w)
	x_max = int((center_x + box_w / 2) * w)
	y_min = int((center_y - box_h / 2) * h)
	y_max = int((center_y + box_h / 2) * h)
	
	# Draw bounding box
	cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
	cv2.putText(img, "pitch", (x_min, max(20, y_min - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
	
	# Extract and draw keypoints (29 keypoints, each with x, y, visibility)
	K = len(KEYPOINT_ORDER)
	kp_start = 5
	kp_end = kp_start + 3 * K
	
	if len(parts) >= kp_end:
		kp_fields = parts[kp_start:kp_end]
		for idx in range(K):
			try:
				kp_x = float(kp_fields[idx * 3])
				kp_y = float(kp_fields[idx * 3 + 1])
				visibility = int(float(kp_fields[idx * 3 + 2]))
			except (ValueError, IndexError):
				continue
			
			if visibility > 0:
				px, py = int(kp_x * w), int(kp_y * h)
				cv2.circle(img, (px, py), 4, (0, 0, 255), -1)
				cv2.putText(img, str(idx), (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
	
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	if ax is not None:
		ax.imshow(img_rgb)
		ax.set_title(image_path.name, fontsize=10)
		ax.axis("off")
	
	return img_rgb


def _create_dataset_yaml(base: Path) -> Path:
	yaml_path = base / "dataset.yaml"
	yaml_content = f"""# SoccerNet Keypoints Dataset (29-point variant)
path: {base.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: pitch

kpt_shape: [29, 3]
# Keypoint order reference
keypoint_names:
"""
	for idx, name in enumerate(KEYPOINT_ORDER):
		yaml_content += f"  {idx}: {name}\n"

	yaml_path.write_text(yaml_content)
	return yaml_path


def process_game(
	game_dir: Path,
	paths: Dict[str, Path],
	output_split: str,
	cfg: ConvertConfig,
	max_images: Optional[int],
	global_counter: Optional[Any],
	counter_lock: Optional[Any],
) -> Dict[str, int]:
	detector = PitchDetector()
	processed_images = 0
	written_labels = 0
	skipped = 0

	json_path = game_dir / "Labels-GameState.json"
	img_dir = game_dir / "img1"
	if not json_path.exists() or not img_dir.exists():
		return {"images": 0, "labels": 0, "skipped": 0}

	data = load_game_annotations(json_path)
	images = {img["image_id"]: img for img in data.get("images", [])}
	anns_by_image: Dict = {}
	for ann in data.get("annotations", []):
		anns_by_image.setdefault(ann.get("image_id"), []).append(ann)

	for img_id, info in images.items():
		if _at_image_limit(global_counter, counter_lock, max_images):
			break

		img_name = info["file_name"]
		src_img = img_dir / img_name
		if not src_img.exists():
			skipped += 1
			continue

		pitch_anns = [a for a in anns_by_image.get(img_id, []) if a.get("supercategory") == "pitch"]
		if not pitch_anns:
			skipped += 1
			continue

		lines = pitch_anns[0].get("lines", {})
		keypoints = _calc_keypoints(lines)
		if not keypoints:
			skipped += 1
			continue

		pitch_result = detector.detect_pitch_from_image(str(src_img))
		if not pitch_result:
			skipped += 1
			continue

		if not _reserve_slot(global_counter, counter_lock, max_images):
			break

		pitch_data = pitch_result["pitch_detection"]
		yolo_line = _format_yolo_annotation(pitch_data, keypoints)

		unique_stem = f"{game_dir.name}_{Path(img_name).stem}"
		dst_img = paths["images"] / f"{unique_stem}.jpg"
		dst_lbl = paths["labels"] / f"{unique_stem}.txt"

		if cfg.overwrite or not dst_img.exists():
			shutil.copy2(src_img, dst_img)

		if cfg.overwrite or not dst_lbl.exists():
			dst_lbl.write_text(yolo_line + "\n")
			written_labels += 1

		if cfg.save_unified_json:
			unified = {
				"image_info": {
					"file_name": dst_img.name,
					"width": info.get("width"),
					"height": info.get("height"),
				},
				"pitch_object": pitch_data,
				"keypoints": keypoints,
				"original_lines": lines,
				"dataset_split": output_split,
			}
			json_path_out = paths["json"] / f"{unique_stem}.json"
			json_path_out.write_text(json.dumps(unified, indent=2))

		processed_images += 1

	return {"images": processed_images, "labels": written_labels, "skipped": skipped}


def process_split(cfg: ConvertConfig, split: str, base_out: Path) -> Dict[str, int]:
	# Source folder uses "valid", YOLO output uses "val"
	source_split = "valid" if split == "val" else split  # For raw dataset lookup
	output_split = "val" if split == "valid" else split   # For YOLO output folders
	paths = _create_dirs(base_out, output_split)

	# If not overwriting, optionally early exit when labels already exist
	if not cfg.overwrite and any(paths["labels"].glob("*.txt")):
		return {"images": 0, "labels": 0, "skipped": 0}

	game_dirs = get_game_folders(cfg.data_root, source_split)
	if cfg.max_games is not None:
		game_dirs = game_dirs[: cfg.max_games]

	if not game_dirs:
		return {"images": 0, "labels": 0, "skipped": 0}

	auto_workers = max(1, multiprocessing.cpu_count() // 2)
	worker_count = cfg.convert_workers if cfg.convert_workers is not None else min(len(game_dirs), auto_workers)
	use_parallel = worker_count > 1

	manager = multiprocessing.Manager() if cfg.max_images is not None else None
	global_counter = manager.Value("i", 0) if manager else None
	counter_lock = manager.Lock() if manager else None

	processed_images = 0
	written_labels = 0
	skipped = 0

	def _accumulate(res: Dict[str, int]) -> None:
		nonlocal processed_images, written_labels, skipped
		processed_images += res.get("images", 0)
		written_labels += res.get("labels", 0)
		skipped += res.get("skipped", 0)

	if use_parallel:
		print(f"Using {worker_count} conversion workers for split '{output_split}' (games-level parallelism).")
		try:
			with ProcessPoolExecutor(max_workers=worker_count) as executor:
				futures = {
					executor.submit(
						process_game,
						game_dir,
						paths,
						output_split,
						cfg,
						cfg.max_images,
						global_counter,
						counter_lock,
					): game_dir
					for game_dir in game_dirs
				}
				with tqdm.tqdm(total=len(futures), desc=f"{output_split}:games") as pbar:
					for future in as_completed(futures):
						res = future.result()
						_accumulate(res)
						pbar.update(1)
		except Exception as exc:
			print(f"Parallel conversion failed ({exc}); falling back to sequential.")
			use_parallel = False

	if not use_parallel:
		for game_dir in tqdm.tqdm(game_dirs, desc=f"{output_split}:games"):
			res = process_game(
				game_dir,
				paths,
				output_split,
				cfg,
				cfg.max_images,
				global_counter,
				counter_lock,
			)
			_accumulate(res)

	return {"images": processed_images, "labels": written_labels, "skipped": skipped}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Convert raw pitch annotations to YOLO (SoccerNet 29-kp variant)")
	parser.add_argument("--data-root", type=Path, default=Path("./SN-GSR-2025"), help="Raw dataset root")
	parser.add_argument("--yolo-root", type=Path, default=Path("./yolo_soccernet_pitch"), help="Output base for YOLO data")
	parser.add_argument("--splits", nargs="+", default=None, help="Splits to process (default train/valid/test)")
	parser.add_argument("--max-games", type=int, default=None, help="Optional limit on number of games per split")
	parser.add_argument("--max-images", type=int, default=None, help="Optional global image limit per split")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labels/images")
	parser.add_argument("--save-unified-json", action="store_true", help="Write combined JSON artifacts")
	parser.add_argument("--convert-workers", type=int, default=None, help="Conversion workers (process per game). Default: cpu_count//2")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	# Use PoseEstimationConfig for shared defaults
	base_cfg = PoseEstimationConfig(data_root=args.data_root, yolo_root=args.yolo_root)

	splits = args.splits or (["train"] if base_cfg.debug_train_only else ["train", "valid", "test"])

	convert_cfg = ConvertConfig(
		data_root=base_cfg.data_root,
		yolo_root=base_cfg.yolo_root,
		splits=splits,
		max_games=args.max_games,
		max_images=args.max_images,
		overwrite=args.overwrite,
		save_unified_json=args.save_unified_json,
		convert_workers=args.convert_workers,
	)

	base_output = convert_cfg.yolo_root / "soccernet_pitch"
	base_output.mkdir(parents=True, exist_ok=True)

	totals = {}
	for split in splits:
		stats = process_split(convert_cfg, split, base_output)
		totals[split] = stats

	yaml_path = _create_dataset_yaml(base_output)

	print("\nSummary:")
	for split, stats in totals.items():
		print(f"  {split}: images={stats['images']}, labels={stats['labels']}, skipped={stats['skipped']}")
	print(f"  dataset.yaml: {yaml_path}")


if __name__ == "__main__":
	main()
