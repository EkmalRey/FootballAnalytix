"""End-to-end evaluation helpers for SoccerNet GSR.

Includes dataset prep (download + GT export), detection metrics,
tracking metrics (MOTA/IDF1/MT/ML/IDSW), clustering purity,
homography reprojection error, and throughput utilities.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
import sys

# Add project root to sys.path to allow importing config
sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Dict, List, Tuple, Optional, Sequence, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover - optional
    KMeans = None  # type: ignore

try:
    import cv2
except Exception:  # pragma: no cover - optional
    cv2 = None  # type: ignore

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional
    YOLO = None  # type: ignore

# Local helpers
from config import ObjectDetectionConfig, load_game_annotations
from _1_download_dataset import download_dataset
from _2_process_raw_dataset import export_tracking_groundtruth

TrackingPipeline = None  # legacy placeholder; we now use the built-in simple tracker below

# ---------------------------------------------------------------------------
# Dataset prep
# ---------------------------------------------------------------------------

def prepare_dataset_and_gt(
    data_root: Path,
    debug_train_only: bool = False,
    debug_test_only: bool = False,
    tracking_out: Path = Path("./eda_outputs/tracking_gt"),
    max_games: Optional[int] = None,
) -> Dict[str, object]:
    """Download SoccerNet GSR (if needed) and export tracking GT JSONs."""
    cfg = ObjectDetectionConfig(data_root=data_root, debug_train_only=debug_train_only)
    download_dataset(cfg, debug_test_only=debug_test_only)
    if debug_train_only:
        splits = ["train"]
    elif debug_test_only:
        splits = ["test"]
    else:
        splits = ["train", "valid", "test"]
    tracking_summary = export_tracking_groundtruth(cfg, splits, tracking_out, max_games=max_games)
    return tracking_summary


# ---------------------------------------------------------------------------
# GT helpers
# ---------------------------------------------------------------------------

TeamValue = Union[str, int, None]

def load_tracking_gt(gt_path: Path) -> Dict[int, List[Dict]]:
    with gt_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    frames = payload.get("frames", {})
    return {int(k): v for k, v in frames.items()}


def summarize_gt(frames: Dict[int, List[Dict]]) -> Dict[str, int]:
    total = sum(len(v) for v in frames.values())
    unique_tracks = set()
    for objs in frames.values():
        for obj in objs:
            tid = obj.get("track_id")
            if tid is not None:
                unique_tracks.add(tid)
    return {"frames": len(frames), "objects": total, "tracks": len(unique_tracks)}


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return np.where(union > 0, inter / union, 0)


def average_precision(recalls: List[float], precisions: List[float]) -> float:
    if not recalls:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def compute_detection_metrics(samples: List[Dict], iou_thresh: float = 0.5) -> Dict[str, object]:
    """Compute per-class AP/precision/recall and IoU summaries for matched detections.

    samples: list of dicts with keys pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels.
    Returns mAP and per-class stats including mean IoU over true positives.
    """
    class_ids = set()
    for s in samples:
        class_ids.update(map(int, s.get("gt_labels", [])))
        class_ids.update(map(int, s.get("pred_labels", [])))

    results = {}
    aps: List[float] = []
    tp_ious_all: List[float] = []

    for cid in sorted(class_ids):
        detections: List[Tuple[float, bool]] = []
        tp_ious: List[float] = []
        total_gts = 0
        for s in samples:
            gt_mask = [i for i, g in enumerate(s.get("gt_labels", [])) if g == cid]
            pred_mask = [i for i, p in enumerate(s.get("pred_labels", [])) if p == cid]
            gt_boxes = np.array(s.get("gt_boxes", []), dtype=float)[gt_mask]
            pred_boxes = np.array(s.get("pred_boxes", []), dtype=float)[pred_mask]
            scores = np.array(s.get("pred_scores", []), dtype=float)[pred_mask]
            total_gts += len(gt_boxes)
            order = np.argsort(-scores)
            pred_boxes = pred_boxes[order]
            scores = scores[order]
            ious = box_iou_matrix(pred_boxes, gt_boxes)
            matched_gt = set()
            for pb, sc, row in zip(pred_boxes, scores, ious):
                best_idx = int(row.argmax()) if row.size else -1
                best_iou = float(row.max()) if row.size else 0.0
                is_tp = best_iou >= iou_thresh and best_idx not in matched_gt
                if is_tp:
                    matched_gt.add(best_idx)
                    tp_ious.append(best_iou)
                    tp_ious_all.append(best_iou)
                detections.append((float(sc), is_tp))
        if total_gts == 0:
            results[cid] = {"precision": 0.0, "recall": 0.0, "ap": 0.0, "mean_iou": 0.0, "tp": 0, "fp": 0, "gt": 0}
            continue
        detections.sort(key=lambda x: -x[0])
        tps = np.cumsum([int(d[1]) for d in detections])
        fps = np.cumsum([int(not d[1]) for d in detections])
        precisions = (tps / np.maximum(tps + fps, 1e-9)).tolist()
        recalls = (tps / total_gts).tolist()
        ap = average_precision(recalls, precisions)
        aps.append(ap)
        precision = precisions[-1] if precisions else 0.0
        recall = recalls[-1] if recalls else 0.0
        tp_count = int(tps[-1]) if len(tps) else 0
        fp_count = len(detections) - tp_count
        mean_iou = float(np.mean(tp_ious)) if tp_ious else 0.0
        results[cid] = {
            "precision": precision,
            "recall": recall,
            "ap": ap,
            "mean_iou": mean_iou,
            "tp": tp_count,
            "fp": fp_count,
            "gt": int(total_gts),
        }
    map_val = float(np.mean(aps)) if aps else 0.0
    mean_iou_all = float(np.mean(tp_ious_all)) if tp_ious_all else 0.0
    return {"per_class": results, "mAP": map_val, "mean_iou": mean_iou_all}


# ---------------------------------------------------------------------------
# Tracking metrics
# ---------------------------------------------------------------------------

def greedy_match(gt_boxes: np.ndarray, pred_boxes: np.ndarray, iou_thresh: float) -> List[Tuple[int, int, float]]:
    """Greedy matching with sorted IOUs to avoid O(n^3) argmax loops."""
    matches: List[Tuple[int, int, float]] = []
    if gt_boxes.size == 0 or pred_boxes.size == 0:
        return matches

    ious = box_iou_matrix(pred_boxes, gt_boxes)
    flat = ious.ravel()
    order = np.argsort(-flat)  # descending

    used_gt = set()
    used_pred = set()
    for idx in order:
        best = float(flat[idx])
        if best < iou_thresh:
            break
        p_idx = int(idx // ious.shape[1])
        g_idx = int(idx % ious.shape[1])
        if p_idx in used_pred or g_idx in used_gt:
            continue
        matches.append((g_idx, p_idx, best))
        used_pred.add(p_idx)
        used_gt.add(g_idx)
    return matches


def compute_tracking_metrics(
    gt_frames: Dict[int, List[Dict]],
    pred_frames: Dict[int, List[Dict]],
    class_filter: Optional[Sequence[int]] = None,
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    """Compute lightweight MOT metrics (IDF1/MOTA/MT/ML/IDSW)."""
    last_matches: Dict[Union[int, str], Union[int, str]] = {}
    id_switches = 0
    idtp = idfp = idfn = 0

    gt_track_frames: Dict[Union[int, str], int] = defaultdict(int)
    gt_track_hits: Dict[Union[int, str], int] = defaultdict(int)

    frame_ids = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))
    for frame_id in frame_ids:
        gt_objs = gt_frames.get(frame_id, [])
        pred_objs = pred_frames.get(frame_id, [])
        if class_filter is not None:
            gt_objs = [o for o in gt_objs if o.get("class_id") in class_filter]
            pred_objs = [o for o in pred_objs if o.get("class_id") in class_filter]

        gt_boxes = np.array([o["bbox"] for o in gt_objs], dtype=float)
        pred_boxes = np.array([o["bbox"] for o in pred_objs], dtype=float)
        matches = greedy_match(gt_boxes, pred_boxes, iou_thresh)

        matched_gt = set()
        matched_pred = set()
        for g_idx, p_idx, _ in matches:
            gt_obj = gt_objs[g_idx]
            pred_obj = pred_objs[p_idx]
            gt_id = gt_obj.get("track_id")
            pred_id = pred_obj.get("track_id")
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            gt_track_frames[gt_id] += 1
            gt_track_hits[gt_id] += 1
            if gt_id in last_matches and last_matches[gt_id] != pred_id:
                id_switches += 1
            last_matches[gt_id] = pred_id
            idtp += 1

        for g_idx, gt_obj in enumerate(gt_objs):
            gt_id = gt_obj.get("track_id")
            gt_track_frames[gt_id] += 1
            if g_idx not in matched_gt:
                idfn += 1

        for p_idx in range(len(pred_objs)):
            if p_idx not in matched_pred:
                idfp += 1

    mota = 1.0 - float(idfn + idfp + id_switches) / float(max(idtp + idfn, 1))
    idf1 = 0.0 if (2 * idtp + idfp + idfn) == 0 else (2 * idtp) / float(2 * idtp + idfp + idfn)

    mt = ml = 0
    for track_id, total_frames in gt_track_frames.items():
        hits = gt_track_hits.get(track_id, 0)
        if total_frames == 0:
            continue
        ratio = hits / total_frames
        if ratio >= 0.8:
            mt += 1
        if ratio <= 0.2:
            ml += 1

    return {
        "IDF1": float(idf1),
        "MOTA": float(mota),
        "IDTP": float(idtp),
        "IDFP": float(idfp),
        "IDFN": float(idfn),
        "IDSW": float(id_switches),
        "MT": int(mt),
        "ML": int(ml),
    }


# ---------------------------------------------------------------------------
# Clustering & homography metrics
# ---------------------------------------------------------------------------

def clustering_purity(gt_labels: List, pred_labels: List) -> float:
    if not gt_labels or not pred_labels or len(gt_labels) != len(pred_labels):
        return 0.0
    df = pd.DataFrame({"gt": gt_labels, "pred": pred_labels})
    correct = 0
    for _, group in df.groupby("pred"):
        majority = group["gt"].value_counts().max()
        correct += majority
    return float(correct / len(df))


def reprojection_error(h_mat: np.ndarray, src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, float]:
    if cv2 is None:
        raise ImportError("cv2 is required for homography evaluation")
    if src.size == 0 or dst.size == 0:
        return np.array([]), 0.0
    src = src.astype(np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, h_mat).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)
    return errors, float(errors.mean()) if errors.size else 0.0


# ---------------------------------------------------------------------------
# Throughput helpers
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def time_block(name: str = "block"):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name} took {elapsed:.3f}s")


def measure_fps(durations: List[float]) -> float:
    if not durations:
        return 0.0
    mean_time = float(np.mean(durations))
    return float(1.0 / mean_time) if mean_time > 0 else 0.0


def _iterate_frames(
    frames_path: Path,
    max_frames: Optional[int],
    show_progress: bool,
    desc: str,
):
    """Yield (frame_idx, frame) from either a directory of images or a video file."""
    if cv2 is None:
        raise ImportError("cv2 is required for frame iteration")

    if frames_path.is_dir():
        frame_files = sorted(frames_path.glob("*.jpg"))
        if not frame_files:
            raise FileNotFoundError(f"No frames found in {frames_path}")
        total = len(frame_files) if max_frames is None else min(len(frame_files), int(max_frames))
        pbar = tqdm(total=total, desc=desc, unit="f", leave=True, dynamic_ncols=True) if show_progress else None
        try:
            for frame_idx, fp in enumerate(frame_files):
                if frame_idx >= total:
                    break
                frame = cv2.imread(str(fp))
                if frame is None:
                    continue
                if pbar:
                    pbar.update(1)
                yield frame_idx, frame
        finally:
            if pbar:
                pbar.close()
    else:
        cap = cv2.VideoCapture(str(frames_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {frames_path}")
        pbar = tqdm(total=max_frames, desc=desc, unit="f", leave=True, dynamic_ncols=True) if show_progress else None
        try:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok or (max_frames is not None and frame_idx >= max_frames):
                    break
                if pbar:
                    pbar.update(1)
                yield frame_idx, frame
                frame_idx += 1
        finally:
            if pbar:
                pbar.close()
            cap.release()


def _clamp_bbox(bbox: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(x1), float(width - 1)))
    y1 = max(0.0, min(float(y1), float(height - 1)))
    x2 = max(0.0, min(float(x2), float(width - 1)))
    y2 = max(0.0, min(float(y2), float(height - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _auto_device(preferred: Optional[str] = None) -> str:
    """Pick a device string; default to CUDA if available."""
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def stitch_frames_to_video(
    frames_dir: Path,
    output_path: Path,
    max_frames: int = 750,
    fps: int = 25,
) -> Path:
    """Combine ordered frames (img1 folder) into a video file.

    Assumes frame files are named in ascending numeric order (e.g., 1.jpg or 000001.jpg).
    """
    if cv2 is None:
        raise ImportError("cv2 is required to stitch frames into video")
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    frames = frames[:max_frames]
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {frames[0]}")
    h, w = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        writer.write(img)
    writer.release()
    return output_path


# ---------------------------------------------------------------------------
# Pipeline execution (tracking/detection)
# ---------------------------------------------------------------------------


class SimpleIOUTracker:
    """Minimal IOU-based tracker to replace external TrackingPipeline.

    Keeps last seen boxes and assigns a persistent id based on IOU matching.
    This is intentionally lightweight for evaluation and avoids external deps.
    """

    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, object]] = {}

    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        frame_idx: int,
    ) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        boxes = boxes.astype(float)
        scores = scores.astype(float)
        labels = labels.astype(int)
        active_ids = set()

        if boxes.size > 0:
            track_ids = list(self.tracks.keys())
            track_boxes = np.array([self.tracks[tid]["bbox"] for tid in track_ids], dtype=float) if track_ids else np.empty((0, 4))
            ious = box_iou_matrix(boxes, track_boxes) if track_boxes.size else np.zeros((len(boxes), 0))

            for det_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                best_id = None
                best_iou = 0.0
                for tcol, track_id in enumerate(track_ids):
                    if ious.shape[1] == 0:
                        break
                    iou_val = float(ious[det_idx, tcol]) if ious.size else 0.0
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_id = track_id
                if best_id is not None and best_iou >= self.iou_thresh:
                    tid = best_id
                else:
                    tid = self.next_id
                    self.next_id += 1
                self.tracks[tid] = {"bbox": box.tolist(), "label": int(label), "last_frame": frame_idx}
                active_ids.add(tid)
                results.append({"track_id": tid, "bbox": box.tolist(), "class_id": int(label), "score": float(score)})

        stale = [tid for tid, t in self.tracks.items() if frame_idx - int(t.get("last_frame", 0)) > self.max_age]
        for tid in stale:
            self.tracks.pop(tid, None)
        return results


def load_pitch_keypoints(json_path: Path) -> Dict[int, np.ndarray]:
    """Load pitch keypoints per frame from SoccerNet Labels-GameState.json."""
    try:
        data = load_game_annotations(json_path)
    except Exception:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    images = {img.get("image_id"): img for img in data.get("images", [])}
    frame_kps: Dict[int, np.ndarray] = {}
    for ann in data.get("annotations", []):
        if ann.get("supercategory") != "pitch":
            continue
        img_id = ann.get("image_id")
        info = images.get(img_id, {})
        file_name = info.get("file_name", "")
        stem = Path(file_name).stem
        idx_digits = [int(x) for x in stem.split("_") if x.isdigit()]
        frame_idx = idx_digits[-1] if idx_digits else int(info.get("frame_id", info.get("image_id", 0)))
        w = info.get("width") or 0
        h = info.get("height") or 0
        pts: List[Tuple[float, float]] = []
        for _, kp_list in (ann.get("lines") or {}).items():
            for pt in kp_list:
                x = float(pt.get("x", 0.0)) * float(w)
                y = float(pt.get("y", 0.0)) * float(h)
                pts.append((x, y))
        if pts:
            frame_kps[int(frame_idx)] = np.array(pts, dtype=np.float32)
    return frame_kps

def track_players_with_yolo(
    video_path: Path,
    model_path: Path,
    max_frames: int = 200,
    conf: float = 0.25,
    iou: float = 0.5,
    show_progress: bool = False,
    device: Optional[str] = None,
) -> Dict[int, List[Dict[str, object]]]:
    """Run YOLO detection with a simple IOU tracker to produce per-frame tracks.

    `video_path` can be a video file or a directory of frames.
    """
    if cv2 is None:
        raise ImportError("cv2 is required for tracking evaluation")
    if YOLO is None:
        raise ImportError("ultralytics is required for tracking evaluation")

    device = _auto_device(device)
    model = YOLO(str(model_path)).to(device)
    tracker = SimpleIOUTracker(iou_thresh=0.3)
    pred_frames: Dict[int, List[Dict[str, object]]] = defaultdict(list)

    for frame_idx, frame in _iterate_frames(video_path, max_frames, show_progress, desc="tracking frames"):
        result = model(frame, conf=conf, iou=iou, verbose=False, device=device)[0]
        boxes = result.boxes
        pred_boxes = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.empty((0, 4))
        pred_scores = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.empty((0,))
        pred_labels = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else np.empty((0,), dtype=int)
        mask = pred_labels == 1  # players only for tracking
        tracked = tracker.update(pred_boxes[mask], pred_scores[mask], pred_labels[mask], frame_idx)
        pred_frames[frame_idx].extend(tracked)
    return pred_frames


def evaluate_tracking_on_video(
    video_path: Path,
    gt_path: Path,
    model_path: Path,
    max_frames: int = 200,
    show_progress: bool = False,
    device: Optional[str] = None,
) -> Dict[str, float]:
    gt_frames = load_tracking_gt(gt_path)
    pred_frames = track_players_with_yolo(
        video_path, model_path, max_frames=max_frames, show_progress=show_progress, device=device
    )
    metrics = compute_tracking_metrics(gt_frames, pred_frames, class_filter=[1], iou_thresh=0.5)
    return metrics


def collect_detection_samples_from_video(
    video_path: Path,
    gt_frames: Dict[int, List[Dict]],
    model_path: Path,
    max_frames: int = 200,
    show_progress: bool = False,
    device: Optional[str] = None,
) -> Tuple[List[Dict], List[float]]:
    if cv2 is None:
        raise ImportError("cv2 is required for detection evaluation")
    if YOLO is None:
        raise ImportError("ultralytics is required for detection evaluation")
    device = _auto_device(device)
    model = YOLO(str(model_path)).to(device)
    samples: List[Dict] = []
    durations: List[float] = []

    for frame_idx, frame in _iterate_frames(video_path, max_frames, show_progress, desc="detection frames"):
        gt_objs = gt_frames.get(frame_idx, [])
        gt_boxes = [o["bbox"] for o in gt_objs]
        gt_labels = [o.get("class_id", 1) for o in gt_objs]
        t0 = time.time()
        result = model(frame, verbose=False, device=device)[0]
        durations.append(time.time() - t0)
        boxes = result.boxes
        pred_boxes = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.empty((0, 4))
        pred_scores = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.empty((0,))
        pred_labels = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else np.empty((0,), dtype=int)
        samples.append(
            {
                "gt_boxes": gt_boxes,
                "gt_labels": gt_labels,
                "pred_boxes": pred_boxes,
                "pred_scores": pred_scores,
                "pred_labels": pred_labels,
            }
        )
    return samples, durations


def _crop_center_color(frame: np.ndarray, bbox: Sequence[float]) -> List[float]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(bbox, w, h)
    cx1 = int(x1 + 0.25 * (x2 - x1))
    cy1 = int(y1 + 0.25 * (y2 - y1))
    cx2 = int(x1 + 0.75 * (x2 - x1))
    cy2 = int(y1 + 0.75 * (y2 - y1))
    cx1, cy1 = max(0, cx1), max(0, cy1)
    cx2, cy2 = min(w - 1, cx2), min(h - 1, cy2)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return [0.0, 0.0, 0.0]
    mean_bgr = crop.reshape(-1, 3).mean(axis=0)
    return [float(v) / 255.0 for v in mean_bgr]


def collect_clustering_samples_from_video(
    video_path: Path,
    gt_frames: Dict[int, List[Dict]],
    model_path: Path,
    max_frames: int = 200,
    iou_match_thresh: float = 0.3,
    show_progress: bool = False,
    device: Optional[str] = None,
) -> Tuple[List[List[float]], List[int]]:
    if cv2 is None:
        raise ImportError("cv2 is required for clustering evaluation")
    if YOLO is None:
        raise ImportError("ultralytics is required for clustering evaluation")
    device = _auto_device(device)
    model = YOLO(str(model_path)).to(device)
    features: List[List[float]] = []
    gt_labels: List[int] = []

    for frame_idx, frame in _iterate_frames(video_path, max_frames, show_progress, desc="clustering frames"):
        gt_objs = [o for o in gt_frames.get(frame_idx, []) if o.get("class_id") == 1]
        gt_boxes = np.array([o["bbox"] for o in gt_objs], dtype=float)
        gt_teams = [o.get("team") for o in gt_objs]

        result = model(frame, verbose=False, device=device)[0]
        boxes = result.boxes
        pred_boxes = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.empty((0, 4))
        pred_labels = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else np.empty((0,), dtype=int)
        player_mask = pred_labels == 1
        pred_boxes = pred_boxes[player_mask]
        if pred_boxes.size == 0 or gt_boxes.size == 0:
            continue

        ious = box_iou_matrix(pred_boxes, gt_boxes)
        for p_idx, row in enumerate(ious):
            g_idx = int(row.argmax())
            if row[g_idx] < iou_match_thresh:
                continue
            gt_team = gt_teams[g_idx]
            if gt_team is None:
                continue
            feat = _crop_center_color(frame, pred_boxes[p_idx])
            features.append(feat)
            gt_labels.append(int(gt_team))
    return features, gt_labels


def evaluate_clustering_on_video(
    video_path: Path,
    gt_frames: Dict[int, List[Dict]],
    model_path: Path,
    max_frames: int = 200,
    show_progress: bool = False,
    device: Optional[str] = None,
) -> Dict[str, float]:
    if KMeans is None:
        raise ImportError("scikit-learn is required for clustering evaluation")
    feats, gt_labels = collect_clustering_samples_from_video(
        video_path, gt_frames, model_path, max_frames=max_frames, show_progress=show_progress, device=device
    )
    if not feats or not gt_labels:
        return {"purity": 0.0, "samples": 0}
    kmeans = KMeans(n_clusters=len(set(gt_labels)), n_init=10, random_state=42)
    pred_labels = kmeans.fit_predict(np.array(feats))
    purity = clustering_purity(gt_labels, pred_labels.tolist())
    return {"purity": float(purity), "samples": len(gt_labels)}


def evaluate_homography_on_video(
    video_path: Path,
    pitch_json: Optional[Path],
    field_model_path: Optional[Path],
    max_frames: int = 50,
    show_progress: bool = False,
    device: Optional[str] = None,
) -> Dict[str, float]:
    if pitch_json is None or field_model_path is None:
        return {"mean_reprojection_error": 0.0, "frames_used": 0}
    if cv2 is None:
        raise ImportError("cv2 is required for homography evaluation")
    if YOLO is None:
        raise ImportError("ultralytics is required for homography evaluation")

    gt_kps = load_pitch_keypoints(pitch_json)
    if not gt_kps:
        return {"mean_reprojection_error": 0.0, "frames_used": 0}
    device = _auto_device(device)
    model = YOLO(str(field_model_path)).to(device)
    errors: List[float] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    pbar = tqdm(total=max_frames, desc="homography frames", unit="f", leave=True, dynamic_ncols=True) if show_progress else None
    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame_idx >= max_frames:
                break
            if frame_idx not in gt_kps:
                frame_idx += 1
                continue
            result = model(frame, verbose=False, device=device)[0]
            kp_obj = getattr(result, "keypoints", None)
            if kp_obj is None or not hasattr(kp_obj, "xy"):
                frame_idx += 1
                continue
            pred_kps = kp_obj.xy.cpu().numpy()
            if pred_kps.size == 0:
                frame_idx += 1
                continue
            pred_pts = pred_kps[0]
            gt_pts = gt_kps.get(frame_idx)
            if gt_pts is None:
                frame_idx += 1
                continue
            limit = min(len(pred_pts), len(gt_pts))
            if limit < 4:
                frame_idx += 1
                continue
            src = pred_pts[:limit].astype(np.float32)
            dst = gt_pts[:limit].astype(np.float32)
            h_mat, _ = cv2.findHomography(src, dst, method=cv2.RANSAC)
            if h_mat is None:
                frame_idx += 1
                continue
            _, mean_err = reprojection_error(h_mat, src, dst)
            errors.append(mean_err)
            frame_idx += 1
            if pbar:
                pbar.update(1)
    finally:
        if pbar:
            pbar.close()
        cap.release()

    mean_error = float(np.mean(errors)) if errors else 0.0
    return {"mean_reprojection_error": mean_error, "frames_used": len(errors)}


def run_full_evaluation(
    video_path: Path,
    gt_path: Path,
    model_path: Path,
    max_frames: int = 200,
    field_model_path: Optional[Path] = None,
    pitch_json: Optional[Path] = None,
    verbose: bool = True,
    device: Optional[str] = None,
) -> Dict[str, object]:
    gt_frames = load_tracking_gt(gt_path)
    device = _auto_device(device)

    if verbose:
        print("[1/5] Tracking players with YOLO + IOU tracker...")
    with time_block("tracking"):
        tracking = evaluate_tracking_on_video(
            video_path, gt_path, model_path, max_frames=max_frames, show_progress=verbose, device=device
        )

    if verbose:
        print("[2/5] Collecting detection samples...")
    with time_block("detection_sampling"):
        samples, det_times = collect_detection_samples_from_video(
            video_path, gt_frames, model_path, max_frames=max_frames, show_progress=verbose, device=device
        )
    if verbose:
        print(f"[2/5] detection sampling done: {len(samples)} frames, {len(det_times)} timings")

    if verbose:
        print("[3/5] Computing detection metrics and throughput...")
    detection = compute_detection_metrics(samples, iou_thresh=0.5)
    throughput = {"mean_inference_time_s": float(np.mean(det_times)) if det_times else 0.0, "fps": measure_fps(det_times)}
    if verbose:
        print(
            f"[3/5] detection metrics ready: mAP={detection.get('mAP', 0):.4f}, "
            f"mean_iou={detection.get('mean_iou', 0):.4f}, fps={throughput.get('fps', 0):.2f}"
        )

    if verbose:
        print("[4/5] Evaluating clustering (team colors)...")
    with time_block("clustering"):
        clustering = evaluate_clustering_on_video(
            video_path, gt_frames, model_path, max_frames=max_frames, show_progress=verbose, device=device
        )
    if verbose:
        print(f"[4/5] clustering done: samples={clustering.get('samples', 0)}, purity={clustering.get('purity', 0):.4f}")

    if verbose:
        print("[5/5] Evaluating homography (field keypoints)...")
    with time_block("homography"):
        homography = evaluate_homography_on_video(
            video_path, pitch_json, field_model_path, max_frames=min(max_frames, 50), show_progress=verbose, device=device
        )
    if verbose:
        print(f"[5/5] homography done: frames_used={homography.get('frames_used', 0)}, mean_err={homography.get('mean_reprojection_error', 0):.4f}")

    if verbose:
        print("[done] Aggregating results.")
    return {
        "tracking": tracking,
        "detection": detection,
        "clustering": clustering,
        "homography": homography,
        "throughput": throughput,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SoccerNet evaluation helper")
    parser.add_argument("--data-root", type=Path, default=Path("./SN-GSR-2025"))
    parser.add_argument("--video", type=Path, help="Path to evaluation video")
    parser.add_argument("--gt", type=Path, help="Path to exported tracking GT JSON")
    parser.add_argument("--model", type=Path, default=Path("./Resources/Models/players.pt"))
    parser.add_argument("--field-model", type=Path, default=None, help="Path to field keypoint model for homography eval")
    parser.add_argument("--pitch-json", type=Path, default=None, help="Path to Labels-GameState.json for pitch GT")
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--export-gt", action="store_true", help="Export tracking GT before eval")
    parser.add_argument("--tracking-out", type=Path, default=Path("./eda_outputs/tracking_gt"))
    parser.add_argument("--debug-train-only", action="store_true")
    parser.add_argument("--max-games", type=int, default=None)
    args = parser.parse_args()

    if args.export_gt:
        print("Preparing dataset and exporting tracking GT...")
        summary = prepare_dataset_and_gt(
            data_root=args.data_root,
            debug_train_only=args.debug_train_only,
            tracking_out=args.tracking_out,
            max_games=args.max_games,
        )
        print(summary)

    if args.video and args.gt:
        print("Running full evaluation...")
        res = run_full_evaluation(
            args.video,
            args.gt,
            args.model,
            max_frames=args.max_frames,
            field_model_path=args.field_model,
            pitch_json=args.pitch_json,
        )
        print(json.dumps(res, indent=2))
    else:
        print("No video/gt provided; only dataset/GT prep done.")


if __name__ == "__main__":
    main()
