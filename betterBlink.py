#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import csv
import math
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold("error")

import cv2
import mediapipe as mp
import numpy as np


VIDEO_ROOT = Path("video")
DRY_RUN_DIR = VIDEO_ROOT / "dry"
DETAILS_DIR = Path("details")
SEUILS_CSV = DETAILS_DIR / "seuil.csv"

REFERENCE_VIDEO_NAME = "etalon.mp4"
EXPECTED_FILE_NAME = "attendu.txt"
ERROR_REPORT_FILE = Path("Erreur relative.txt")

ESSENTIAL_OUTPUT_CSV = Path("Essential.csv")
MEAN_BLINK_TIME_CSV = Path("mean_blink_time.csv")
MIN_MAX_BLINK_CSV = Path("min_max_blink.csv")

CATEGORY_FILES = {
    "Coloriage": "Coloriage.mp4",
    "Jeu SANS chrono": "Jeu SANS chrono.mp4",
    "Jeu AVEC chrono": "Jeu AVEC chrono.mp4",
    "Parinaud 5 ordi": "Parinaud 5 ordi.mp4",
    "Parinaud 5 papier": "Parinaud 5 papier.mp4",
}

DEFAULT_OUTPUT_CSV = "results.csv"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

DEFAULT_MIN_CLOSED_FRAMES = 1
DEFAULT_MAX_CLOSED_FRAMES = 16

OPEN_PERCENTILE = 95
CLOSED_PERCENTILE = 5

DEFAULT_CLOSE_THRESHOLD = 0.25
DEFAULT_OPEN_THRESHOLD = 0.45

DRY_SHOW_WARMUP_VALID_FRAMES = 30
SMOOTH_KERNEL = 3
DEEPDATA_STEP_FRAMES = 5


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Valeur booléenne attendue: true/false")


def canonicalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value)
    without_accents = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_accents.casefold()


def resolve_child_case_insensitive(parent: Path, target_name: str) -> Optional[Path]:
    if not parent.exists() or not parent.is_dir():
        return None
    target_key = canonicalize_name(target_name)
    for child in parent.iterdir():
        if canonicalize_name(child.name) == target_key:
            return child
    return None


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def landmark_to_pixel(landmark, width, height):
    return landmark.x * width, landmark.y * height


def eye_aspect_ratio(landmarks, eye_indices, width, height):
    pts = [landmark_to_pixel(landmarks[i], width, height) for i in eye_indices]
    p1, p2, p3, p4, p5, p6 = pts

    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_ear(face_landmarks, width, height):
    left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE, width, height)
    right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE, width, height)
    return (left_ear + right_ear) / 2.0


def median_filter_nan(values: np.ndarray, k: int = 3) -> np.ndarray:
    if len(values) == 0 or k <= 1:
        return values.copy()

    k = int(k)
    if k % 2 == 0:
        k += 1

    out = values.copy()
    radius = k // 2

    for i in range(len(values)):
        start = max(0, i - radius)
        end = min(len(values), i + radius + 1)
        chunk = values[start:end]
        finite = chunk[np.isfinite(chunk)]
        out[i] = np.median(finite) if finite.size > 0 else np.nan

    return out


def read_expected_blinks(expected_file: Path) -> int:
    content = expected_file.read_text(encoding="utf-8").strip()
    return int(content)


def format_optional_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def compute_blinks_per_minute(blink_count: int, duration_seconds: Optional[float]) -> Optional[float]:
    if duration_seconds is None or duration_seconds <= 0:
        return None
    return float((blink_count * 60.0) / duration_seconds)


def compute_blink_interval_stats(
    blink_timestamps: List[float],
    duration_seconds: Optional[float],
) -> Dict[str, Optional[float]]:
    """
    Règles :
    - 0 clignement :
        intervalle = durée totale de la vidéo

    - 1 clignement :
        intervalle = max(
            temps entre début vidéo et clignement,
            temps entre clignement et fin vidéo
        )

    - 2 clignements ou plus :
        intervalle = écarts entre clignements successifs
    """
    if duration_seconds is None or duration_seconds < 0:
        duration_seconds = 0.0

    if blink_timestamps is None or len(blink_timestamps) == 0:
        fallback = float(duration_seconds)
        return {
            "mean_interval": fallback,
            "min_interval": fallback,
            "max_interval": fallback,
        }

    if len(blink_timestamps) == 1:
        ts = float(blink_timestamps[0])
        before = max(0.0, ts)
        after = max(0.0, float(duration_seconds) - ts)
        fallback = max(before, after)

        return {
            "mean_interval": fallback,
            "min_interval": fallback,
            "max_interval": fallback,
        }

    ts = np.array(blink_timestamps, dtype=np.float32)
    intervals = np.diff(ts)

    return {
        "mean_interval": float(np.mean(intervals)),
        "min_interval": float(np.min(intervals)),
        "max_interval": float(np.max(intervals)),
    }

def compute_reference_error_metrics(reference_errors: List[Dict]) -> Dict[str, Optional[float]]:
    if not reference_errors:
        return {
            "n_subjects": 0,
            "relative_mean_error": None,
            "mean_bias": None,
            "error_std": None,
        }

    signed_errors = np.array([item["signed_error"] for item in reference_errors], dtype=np.float32)
    relative_errors = [
        item["relative_error"]
        for item in reference_errors
        if item["relative_error"] is not None
    ]
    relative_errors = np.array(relative_errors, dtype=np.float32) if relative_errors else np.array([], dtype=np.float32)

    return {
        "n_subjects": int(len(reference_errors)),
        "relative_mean_error": float(np.mean(relative_errors)) if relative_errors.size > 0 else None,
        "mean_bias": float(np.mean(signed_errors)),
        "error_std": float(np.std(signed_errors, ddof=0)),
    }


def write_reference_error_report(report_path: Path, reference_errors: List[Dict], metrics: Dict[str, Optional[float]]):
    lines = []
    lines.append("Rapport global d'erreur sur les vidéos étalon")
    lines.append("")

    rme = metrics.get("relative_mean_error")
    mean_bias = metrics.get("mean_bias")
    error_std = metrics.get("error_std")

    lines.append(f"Nombre de sujets analysés : {metrics.get('n_subjects', 0)}")
    lines.append("Erreur moyenne relative : " + (f"{rme:.6f} ({rme * 100:.2f}%)" if rme is not None else "NA"))
    lines.append("Biais moyen : " + (f"{mean_bias:.6f}" if mean_bias is not None else "NA"))
    lines.append("Ecart type de l'erreur : " + (f"{error_std:.6f}" if error_std is not None else "NA"))
    lines.append("")
    lines.append("Détail par sujet :")
    lines.append("subject\tattendu\tpredit\terreur_signee\terreur_absolue\terreur_relative")

    for item in reference_errors:
        rel = item["relative_error"]
        rel_str = f"{rel:.6f}" if rel is not None else "NA"
        lines.append(
            f"{item['subject']}\t"
            f"{item['expected']}\t"
            f"{item['predicted']}\t"
            f"{item['signed_error']}\t"
            f"{item['absolute_error']}\t"
            f"{rel_str}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def full_metric_fields(base_name: str) -> List[str]:
    return [
        base_name,
        f"{base_name} mean",
        f"{base_name} low",
        f"{base_name} high",
        f"{base_name} per minute",
        f"{base_name} Face Detect Rate",
    ]


def build_subject_dirs() -> List[Path]:
    return [
        p for p in sorted(VIDEO_ROOT.iterdir())
        if p.is_dir() and p.name != "dry"
    ]


def get_all_category_files() -> Dict[str, str]:
    return {"étalon": REFERENCE_VIDEO_NAME, **CATEGORY_FILES}


def denormalize_threshold(norm_threshold: float, profile: Optional[Dict]) -> Optional[float]:
    if profile is None:
        return None
    return profile["ear_closed_ref"] + norm_threshold * (
        profile["ear_open_ref"] - profile["ear_closed_ref"]
    )


def sample_series(values: np.ndarray, step: int) -> List[str]:
    out = []
    for i in range(0, len(values), step):
        v = values[i]
        out.append("" if not np.isfinite(v) else f"{v:.3f}")
    return out


def sample_states(states: List[str], step: int) -> List[str]:
    out = []
    for i in range(0, len(states), step):
        out.append(states[i] if i < len(states) else "")
    return out


def append_deepdata_rows(
    detail_rows: List[List[str]],
    detail_ear_rows: List[List[str]],
    detail_ear_norm_rows: List[List[str]],
    video_label: str,
    profile: Optional[Dict],
    thresholds: Dict,
    ears: np.ndarray,
    norm_ears: np.ndarray,
    states: List[str],
):
    low_ear = denormalize_threshold(thresholds["close_threshold"], profile)
    high_ear = denormalize_threshold(thresholds["open_threshold"], profile)

    detail_rows.append([
        video_label,
        "" if low_ear is None else f"{low_ear:.3f}",
        "" if high_ear is None else f"{high_ear:.3f}",
        *sample_series(ears, DEEPDATA_STEP_FRAMES),
    ])
    detail_rows.append([
        "",
        f"{thresholds['close_threshold']:.3f}",
        f"{thresholds['open_threshold']:.3f}",
        *sample_series(norm_ears, DEEPDATA_STEP_FRAMES),
    ])
    detail_rows.append([
        "", "", "",
        *sample_states(states, DEEPDATA_STEP_FRAMES),
    ])

    detail_ear_rows.append([video_label, *sample_series(ears, DEEPDATA_STEP_FRAMES)])
    detail_ear_norm_rows.append([video_label, *sample_series(norm_ears, DEEPDATA_STEP_FRAMES)])


def write_detail_csv(detail_rows: List[List[str]], output_path: Path):
    if not detail_rows:
        return

    max_len = max(len(r) for r in detail_rows)
    headers = ["Nomdedossier/nom_video", "Seuil bas EAR", "Seuil haut EAR"]
    headers.extend([f"frame_{i * DEEPDATA_STEP_FRAMES}" for i in range(max_len - 3)])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in detail_rows:
            writer.writerow(row + [""] * (max_len - len(row)))


def write_detail_series_csv(series_rows: List[List[str]], output_path: Path):
    if not series_rows:
        return

    max_len = max(len(r) for r in series_rows)
    headers = ["Nomdedossier/nom_video"]
    headers.extend([f"frame_{i * DEEPDATA_STEP_FRAMES}" for i in range(max_len - 1)])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in series_rows:
            writer.writerow(row + [""] * (max_len - len(row)))


def write_subject_deepdata_files(
    subject_name: str,
    detail_rows: List[List[str]],
    detail_ear_rows: List[List[str]],
    detail_ear_norm_rows: List[List[str]],
):
    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    write_detail_csv(detail_rows, DETAILS_DIR / f"{subject_name}_detail.csv")
    write_detail_series_csv(detail_ear_rows, DETAILS_DIR / f"{subject_name}_detailEAR.csv")
    write_detail_series_csv(detail_ear_norm_rows, DETAILS_DIR / f"{subject_name}_detailEARNorm.csv")


def compute_reference_peak_stats(
    ears: np.ndarray,
    norm_ears: np.ndarray,
    close_threshold: float,
    open_threshold: float,
    min_closed_frames: int,
    max_closed_frames: int,
) -> Dict[str, Optional[float]]:
    finite_ears = ears[np.isfinite(ears)]
    abs_ear_min = float(np.min(finite_ears)) if finite_ears.size > 0 else None
    abs_ear_max = float(np.max(finite_ears)) if finite_ears.size > 0 else None

    low_peaks_ear = []
    high_peaks_ear = []
    low_peaks_norm = []
    high_peaks_norm = []

    in_closed = False
    closed_len = 0
    current_min_norm = None
    current_min_ear = None
    current_high_norm = None
    current_high_ear = None

    for i in range(len(norm_ears)):
        val_norm = norm_ears[i]
        val_ear = ears[i]

        if not np.isfinite(val_norm):
            in_closed = False
            closed_len = 0
            current_min_norm = None
            current_min_ear = None
            current_high_norm = None
            current_high_ear = None
            continue

        if not in_closed:
            if val_norm < close_threshold:
                in_closed = True
                closed_len = 1
                current_min_norm = float(val_norm)
                current_min_ear = float(val_ear) if np.isfinite(val_ear) else None
                current_high_norm = float(val_norm)
                current_high_ear = float(val_ear) if np.isfinite(val_ear) else None
        else:
            closed_len += 1

            if current_min_norm is None or val_norm < current_min_norm:
                current_min_norm = float(val_norm)
                current_min_ear = float(val_ear) if np.isfinite(val_ear) else current_min_ear

            if current_high_norm is None or val_norm > current_high_norm:
                current_high_norm = float(val_norm)
                current_high_ear = float(val_ear) if np.isfinite(val_ear) else current_high_ear

            if val_norm > open_threshold:
                if min_closed_frames <= closed_len <= max_closed_frames:
                    if current_min_ear is not None:
                        low_peaks_ear.append(current_min_ear)
                    if current_high_ear is not None:
                        high_peaks_ear.append(current_high_ear)
                    if current_min_norm is not None:
                        low_peaks_norm.append(current_min_norm)
                    if current_high_norm is not None:
                        high_peaks_norm.append(current_high_norm)

                in_closed = False
                closed_len = 0
                current_min_norm = None
                current_min_ear = None
                current_high_norm = None
                current_high_ear = None

    return {
        "abs_ear_min": abs_ear_min,
        "abs_ear_max": abs_ear_max,
        "mean_low_peak_ear": float(np.mean(low_peaks_ear)) if low_peaks_ear else None,
        "mean_high_peak_ear": float(np.mean(high_peaks_ear)) if high_peaks_ear else None,
        "mean_low_peak_norm": float(np.mean(low_peaks_norm)) if low_peaks_norm else None,
        "mean_high_peak_norm": float(np.mean(high_peaks_norm)) if high_peaks_norm else None,
    }


def write_seuils_csv(rows: List[Dict]):
    if not rows:
        return

    DETAILS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "min seuil EAR",
        "max seuil EAR",
        "min seuil EAR norm",
        "max seuil EAR norm",
        "min absolu EAR etalon",
        "max absolu EAR etalon",
        "moyenne pics bas EAR",
        "moyenne pics haut EAR",
        "moyenne pics bas EAR norm",
        "moyenne pics haut EAR norm",
    ]

    with open(SEUILS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_landmarker(model_path: Path):
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def draw_face_landmarks(frame, face_landmarks):
    h, w = frame.shape[:2]

    for lm in face_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    for idx in LEFT_EYE:
        lm = face_landmarks[idx]
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 0, 0), -1)

    for idx in RIGHT_EYE:
        lm = face_landmarks[idx]
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 0, 255), -1)

    for eye, color in [(LEFT_EYE, (255, 0, 0)), (RIGHT_EYE, (255, 0, 255))]:
        pts = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in eye]
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]:
            cv2.line(frame, pts[a], pts[b], color, 1)

    return frame


class OnlineEarNormalizer:
    def __init__(self, warmup_valid_frames: int = 30):
        self.history: List[float] = []
        self.warmup_valid_frames = warmup_valid_frames

    def update(self, ear: Optional[float]) -> Optional[float]:
        if ear is None or not np.isfinite(ear):
            return None

        self.history.append(float(ear))

        if len(self.history) < self.warmup_valid_frames:
            return None

        arr = np.array(self.history, dtype=np.float32)
        closed_ref = float(np.percentile(arr, CLOSED_PERCENTILE))
        open_ref = float(np.percentile(arr, OPEN_PERCENTILE))
        denom = open_ref - closed_ref

        if denom <= 1e-8:
            return None

        return float((ear - closed_ref) / denom)

    def get_refs(self):
        if len(self.history) < self.warmup_valid_frames:
            return None, None
        arr = np.array(self.history, dtype=np.float32)
        closed_ref = float(np.percentile(arr, CLOSED_PERCENTILE))
        open_ref = float(np.percentile(arr, OPEN_PERCENTILE))
        return open_ref, closed_ref


class OnlineBlinkDetector:
    def __init__(
        self,
        fps: float,
        close_threshold: float,
        open_threshold: float,
        min_closed_frames: int,
        max_closed_frames: int,
    ):
        self.fps = fps
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.min_closed_frames = min_closed_frames
        self.max_closed_frames = max_closed_frames

        self.in_closed = False
        self.closed_start_frame = None
        self.closed_len = 0

        self.blink_count = 0
        self.blink_timestamps: List[float] = []
        self.blink_states: List[str] = []

    def update(self, value: Optional[float], frame_idx: int):
        state = "o"

        if value is None or not np.isfinite(value):
            self.in_closed = False
            self.closed_start_frame = None
            self.closed_len = 0
            self.blink_states.append(state)
            return

        if not self.in_closed:
            if value < self.close_threshold:
                self.in_closed = True
                self.closed_start_frame = frame_idx
                self.closed_len = 1
                state = "-"
        else:
            self.closed_len += 1
            state = "+"

            if value > self.open_threshold:
                if self.min_closed_frames <= self.closed_len <= self.max_closed_frames:
                    self.blink_count += 1
                    ts = (self.closed_start_frame + self.closed_len / 2.0) / self.fps
                    self.blink_timestamps.append(ts)

                self.in_closed = False
                self.closed_start_frame = None
                self.closed_len = 0
                state = "o"

        self.blink_states.append(state)


def build_profile_from_reference_ears(ears: np.ndarray) -> Dict:
    valid = np.isfinite(ears)
    if valid.sum() < 10:
        raise RuntimeError("Pas assez de frames valides pour construire le profil.")

    valid_ears = ears[valid].astype(np.float32)
    valid_ears = median_filter_nan(valid_ears, k=SMOOTH_KERNEL)

    ear_open_ref = float(np.percentile(valid_ears, OPEN_PERCENTILE))
    ear_closed_ref = float(np.percentile(valid_ears, CLOSED_PERCENTILE))

    if ear_open_ref <= ear_closed_ref:
        raise RuntimeError("Profil invalide: ear_open_ref <= ear_closed_ref")

    return {
        "ear_open_ref": ear_open_ref,
        "ear_closed_ref": ear_closed_ref,
        "dynamic_range": ear_open_ref - ear_closed_ref,
    }


def normalize_ear_static(ear: Optional[float], profile: Optional[Dict]) -> Optional[float]:
    if ear is None or not np.isfinite(ear) or profile is None:
        return None

    denom = profile["ear_open_ref"] - profile["ear_closed_ref"]
    if denom <= 1e-8:
        return None

    return float((ear - profile["ear_closed_ref"]) / denom)


def normalize_series_static(ears: np.ndarray, profile: Dict) -> np.ndarray:
    denom = profile["ear_open_ref"] - profile["ear_closed_ref"]
    out = np.full_like(ears, np.nan, dtype=np.float32)
    if denom <= 1e-8:
        return out

    valid = np.isfinite(ears)
    out[valid] = (ears[valid] - profile["ear_closed_ref"]) / denom
    return out.astype(np.float32)


def count_blinks_from_normalized_series(
    norm_ears: np.ndarray,
    fps: float,
    close_threshold: float,
    open_threshold: float,
    min_closed_frames: int,
    max_closed_frames: int,
) -> Dict:
    detector = OnlineBlinkDetector(
        fps=fps,
        close_threshold=close_threshold,
        open_threshold=open_threshold,
        min_closed_frames=min_closed_frames,
        max_closed_frames=max_closed_frames,
    )

    for i, value in enumerate(norm_ears, start=1):
        detector.update(value if np.isfinite(value) else None, i)

    return {
        "blink_count": detector.blink_count,
        "blink_timestamps": detector.blink_timestamps,
    }


def calibrate_thresholds_on_reference(
    norm_ears: np.ndarray,
    fps: float,
    expected_blinks: int,
    min_closed_frames: int,
    max_closed_frames: int,
) -> Dict:
    best = None

    close_candidates = np.arange(0.10, 0.46, 0.02)
    open_candidates = np.arange(0.30, 0.86, 0.02)

    for close_t in close_candidates:
        for open_t in open_candidates:
            if open_t <= close_t + 0.08:
                continue

            result = count_blinks_from_normalized_series(
                norm_ears=norm_ears,
                fps=fps,
                close_threshold=float(close_t),
                open_threshold=float(open_t),
                min_closed_frames=min_closed_frames,
                max_closed_frames=max_closed_frames,
            )

            predicted = result["blink_count"]
            diff = abs(predicted - expected_blinks)

            regularization = abs(close_t - DEFAULT_CLOSE_THRESHOLD) + abs(open_t - DEFAULT_OPEN_THRESHOLD)
            score = (diff, regularization)

            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "close_threshold": float(close_t),
                    "open_threshold": float(open_t),
                    "predicted_reference_blinks": int(predicted),
                    "expected_reference_blinks": int(expected_blinks),
                }

    if best is None:
        raise RuntimeError("Calibration impossible.")

    return best


def analyze_video_one_pass(
    model_path: Path,
    video_path: Path,
    profile: Optional[Dict],
    thresholds: Dict,
    min_closed_frames: int,
    max_closed_frames: int,
    show: bool = False,
    dynamic_normalization: bool = False,
    show_landmarks: bool = False,
) -> Dict:
    start_time = time.perf_counter()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    detector = OnlineBlinkDetector(
        fps=fps,
        close_threshold=thresholds["close_threshold"],
        open_threshold=thresholds["open_threshold"],
        min_closed_frames=min_closed_frames,
        max_closed_frames=max_closed_frames,
    )

    normalizer = OnlineEarNormalizer(DRY_SHOW_WARMUP_VALID_FRAMES) if dynamic_normalization else None

    if show:
        cv2.namedWindow("Blink Counter", cv2.WINDOW_NORMAL)

    frame_idx = 0
    valid_frames = 0
    ears: List[float] = []
    norm_ears: List[float] = []

    with create_landmarker(model_path) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            height, width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            current_ear = None
            current_ear_norm = None
            face_landmarks = None

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                current_ear = float(get_ear(face_landmarks, width, height))
                valid_frames += 1

                if dynamic_normalization and normalizer is not None:
                    current_ear_norm = normalizer.update(current_ear)
                elif profile is not None:
                    current_ear_norm = normalize_ear_static(current_ear, profile)

            ears.append(np.nan if current_ear is None else current_ear)
            norm_ears.append(np.nan if current_ear_norm is None else current_ear_norm)

            detector.update(current_ear_norm, frame_idx)

            if show:
                overlay = frame.copy()

                if show_landmarks and face_landmarks is not None:
                    overlay = draw_face_landmarks(overlay, face_landmarks)

                ear_text = f"EAR={current_ear:.3f}" if current_ear is not None else "EAR=NA"
                norm_text = f"EAR_norm={current_ear_norm:.3f}" if current_ear_norm is not None else "EAR_norm=NA"

                if dynamic_normalization and normalizer is not None:
                    open_ref, closed_ref = normalizer.get_refs()
                    ref_text = (
                        f"open={open_ref:.3f} closed={closed_ref:.3f}"
                        if open_ref is not None and closed_ref is not None
                        else f"warmup<{DRY_SHOW_WARMUP_VALID_FRAMES} valid"
                    )
                else:
                    ref_text = (
                        f"open={profile['ear_open_ref']:.3f} closed={profile['ear_closed_ref']:.3f}"
                        if profile is not None
                        else "open=NA closed=NA"
                    )

                thr_text = f"close<{thresholds['close_threshold']:.2f} open>{thresholds['open_threshold']:.2f}"

                cv2.putText(overlay, f"Blinks: {detector.blink_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(overlay, ear_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(overlay, norm_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(overlay, ref_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(overlay, thr_text, (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(overlay, f"Frame: {frame_idx}/{total_frames if total_frames > 0 else '?'}", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("Blink Counter", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    duration_seconds = (frame_idx / fps) if (fps > 0 and frame_idx > 0) else None
    interval_stats = compute_blink_interval_stats(detector.blink_timestamps, duration_seconds)
    blinks_per_minute = compute_blinks_per_minute(detector.blink_count, duration_seconds)

    return {
        "video_path": str(video_path),
        "fps": float(fps),
        "total_frames": int(total_frames) if total_frames > 0 else frame_idx,
        "processed_frames": int(frame_idx),
        "valid_frames": int(valid_frames),
        "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
        "face_detect_rate": float(valid_frames / frame_idx) if frame_idx > 0 else 0.0,
        "blink_count": int(detector.blink_count),
        "blink_timestamps": detector.blink_timestamps,
        "blink_states": detector.blink_states,
        "blink_interval_mean": interval_stats["mean_interval"],
        "blink_interval_min": interval_stats["min_interval"],
        "blink_interval_max": interval_stats["max_interval"],
        "blinks_per_minute": blinks_per_minute,
        "ears": np.array(ears, dtype=np.float32),
        "norm_ears": np.array(norm_ears, dtype=np.float32),
        "elapsed_seconds": float(elapsed),
    }


def find_first_valid_subject() -> Optional[Path]:
    for subject_dir in build_subject_dirs():
        ref_video = resolve_child_case_insensitive(subject_dir, REFERENCE_VIDEO_NAME)
        expected_file = resolve_child_case_insensitive(subject_dir, EXPECTED_FILE_NAME)
        if ref_video is not None and expected_file is not None:
            return subject_dir
    return None


def find_first_available_category_video(subject_dir: Path) -> Tuple[Optional[str], Optional[Path]]:
    for category, filename in CATEGORY_FILES.items():
        video_path = resolve_child_case_insensitive(subject_dir, filename)
        if video_path is not None and video_path.is_file():
            return category, video_path
    return None, None


def find_single_dry_video() -> Path:
    if not DRY_RUN_DIR.exists() or not DRY_RUN_DIR.is_dir():
        raise FileNotFoundError(f"Dossier dry introuvable: {DRY_RUN_DIR}")

    videos = sorted([
        p for p in DRY_RUN_DIR.iterdir()
        if p.is_file() and p.suffix.casefold() == ".mp4"
    ])

    if len(videos) == 0:
        raise FileNotFoundError(f"Aucun fichier .mp4 trouvé dans {DRY_RUN_DIR}")

    if len(videos) > 1:
        print(f"[WARN] Plusieurs MP4 trouvés dans {DRY_RUN_DIR}. Le premier sera utilisé: {videos[0]}")

    return videos[0]


def run_dry_run_faithful(
    model_path: Path,
    show: bool,
    min_closed_frames: int,
    max_closed_frames: int,
    deepData: bool,
    show_landmarks: bool,
):
    subject_dir = find_first_valid_subject()
    if subject_dir is None:
        raise FileNotFoundError(
            f"Aucun dossier sujet valide trouvé dans {VIDEO_ROOT} "
            f"(il faut {REFERENCE_VIDEO_NAME} et {EXPECTED_FILE_NAME})."
        )

    subject_id = subject_dir.name
    ref_video = resolve_child_case_insensitive(subject_dir, REFERENCE_VIDEO_NAME)
    expected_file = resolve_child_case_insensitive(subject_dir, EXPECTED_FILE_NAME)
    if ref_video is None or expected_file is None:
        raise FileNotFoundError("Impossible de résoudre l'étalon ou attendu.txt.")

    expected_blinks = read_expected_blinks(expected_file)
    category_name, category_video = find_first_available_category_video(subject_dir)

    print(f"[DRY RUN faithful] Sujet test: {subject_id}")
    print(f"[DRY RUN faithful] Etalon: {ref_video}")

    ref_result = analyze_video_one_pass(
        model_path=model_path,
        video_path=ref_video,
        profile=None,
        thresholds={"close_threshold": DEFAULT_CLOSE_THRESHOLD, "open_threshold": DEFAULT_OPEN_THRESHOLD},
        min_closed_frames=min_closed_frames,
        max_closed_frames=max_closed_frames,
        show=False,
        dynamic_normalization=False,
    )

    ears = ref_result.get("ears")
    if ears is None or np.isfinite(ears).sum() < 10:
        raise RuntimeError("Vidéo étalon invalide.")

    profile = build_profile_from_reference_ears(ears)
    ref_norm = median_filter_nan(normalize_series_static(ref_result["ears"], profile), k=SMOOTH_KERNEL)

    thresholds = calibrate_thresholds_on_reference(
        norm_ears=ref_norm,
        fps=ref_result["fps"],
        expected_blinks=expected_blinks,
        min_closed_frames=min_closed_frames,
        max_closed_frames=max_closed_frames,
    )

    target_video = category_video if category_video is not None else ref_video
    target_name = category_name if category_name is not None else "étalon"

    detail_rows, detail_ear_rows, detail_ear_norm_rows = [], [], []

    if deepData:
        append_deepdata_rows(
            detail_rows,
            detail_ear_rows,
            detail_ear_norm_rows,
            f"{subject_id}/{ref_video.name}",
            profile,
            thresholds,
            ref_result["ears"],
            ref_norm,
            ["o"] * len(ref_result["ears"]),
        )

    shown_result = analyze_video_one_pass(
        model_path=model_path,
        video_path=target_video,
        profile=profile,
        thresholds=thresholds,
        min_closed_frames=min_closed_frames,
        max_closed_frames=max_closed_frames,
        show=show,
        dynamic_normalization=False,
        show_landmarks=show_landmarks,
    )

    if deepData:
        append_deepdata_rows(
            detail_rows,
            detail_ear_rows,
            detail_ear_norm_rows,
            f"{subject_id}/{target_video.name}",
            profile,
            thresholds,
            shown_result["ears"],
            shown_result["norm_ears"],
            shown_result["blink_states"],
        )
        write_subject_deepdata_files(subject_id, detail_rows, detail_ear_rows, detail_ear_norm_rows)

    print("\n--- DRY RUN faithful RESULT ---")
    print(f"Sujet               : {subject_id}")
    print(f"Vidéo affichée      : {target_name}")
    print(f"EAR open ref        : {profile['ear_open_ref']:.4f}")
    print(f"EAR closed ref      : {profile['ear_closed_ref']:.4f}")
    print(f"Close threshold     : {thresholds['close_threshold']:.2f}")
    print(f"Open threshold      : {thresholds['open_threshold']:.2f}")
    print(f"Etalon attendu      : {expected_blinks}")
    print(f"Etalon prédit       : {thresholds['predicted_reference_blinks']}")
    print(f"Clignements détectés: {shown_result['blink_count']}")
    print(f"Clign./minute       : {format_optional_float(shown_result['blinks_per_minute'], 1)}")
    print(f"Face detect rate    : {shown_result['face_detect_rate']:.4f}")
    print(f"Temps traitement    : {shown_result['elapsed_seconds']:.2f} s")


def run_dry_run_dynamic(
    model_path: Path,
    show: bool,
    min_closed_frames: int,
    max_closed_frames: int,
    deepData: bool,
    show_landmarks: bool,
):
    video_path = find_single_dry_video()

    print(f"[DRY RUN dynamic] Vidéo test: {video_path}")
    print("[DRY RUN dynamic] Normalisation dynamique.")

    result = analyze_video_one_pass(
        model_path=model_path,
        video_path=video_path,
        profile=None,
        thresholds={"close_threshold": DEFAULT_CLOSE_THRESHOLD, "open_threshold": DEFAULT_OPEN_THRESHOLD},
        min_closed_frames=min_closed_frames,
        max_closed_frames=max_closed_frames,
        show=show,
        dynamic_normalization=True,
        show_landmarks=show_landmarks,
    )

    valid_norm = result["norm_ears"][np.isfinite(result["norm_ears"])]
    norm_p05 = float(np.percentile(valid_norm, 5)) if valid_norm.size > 0 else float("nan")
    norm_p95 = float(np.percentile(valid_norm, 95)) if valid_norm.size > 0 else float("nan")

    if deepData:
        detail_rows, detail_ear_rows, detail_ear_norm_rows = [], [], []
        append_deepdata_rows(
            detail_rows,
            detail_ear_rows,
            detail_ear_norm_rows,
            f"dry/{video_path.name}",
            None,
            {"close_threshold": DEFAULT_CLOSE_THRESHOLD, "open_threshold": DEFAULT_OPEN_THRESHOLD},
            result["ears"],
            result["norm_ears"],
            result["blink_states"],
        )
        write_subject_deepdata_files("dry", detail_rows, detail_ear_rows, detail_ear_norm_rows)

    print("\n--- DRY RUN dynamic RESULT ---")
    print(f"Vidéo               : {video_path}")
    print(f"Frames traitées     : {result['processed_frames']}")
    print(f"Face detect rate    : {result['face_detect_rate']:.3f}")
    print(f"Close threshold     : {DEFAULT_CLOSE_THRESHOLD:.2f}")
    print(f"Open threshold      : {DEFAULT_OPEN_THRESHOLD:.2f}")
    print(f"EAR_norm p05/p95    : {norm_p05:.3f} / {norm_p95:.3f}")
    print(f"Clignements détectés: {result['blink_count']}")
    print(f"Clign./minute       : {format_optional_float(result['blinks_per_minute'], 1)}")
    print(f"Temps traitement    : {result['elapsed_seconds']:.2f} s")


def run_dry_run(
    model_path: Path,
    show: bool,
    min_closed_frames: int,
    max_closed_frames: int,
    mode: str,
    deepData: bool,
    show_landmarks: bool,
):
    if mode == "faithful":
        run_dry_run_faithful(model_path, show, min_closed_frames, max_closed_frames, deepData, show_landmarks)
    elif mode == "dynamic":
        run_dry_run_dynamic(model_path, show, min_closed_frames, max_closed_frames, deepData, show_landmarks)
    else:
        raise ValueError(f"Mode dry-run inconnu: {mode}")


def run_real(
    model_path: Path,
    output_csv: Path,
    min_closed_frames: int,
    max_closed_frames: int,
    deepData: bool,
):
    subject_dirs = build_subject_dirs()
    if not subject_dirs:
        raise FileNotFoundError(f"Aucun dossier sujet trouvé dans {VIDEO_ROOT}")

    rows = []
    essential_rows = []
    mean_blink_rows = []
    min_max_blink_rows = []
    reference_errors = []
    seuils_rows = []
    all_categories = get_all_category_files()

    if deepData:
        DETAILS_DIR.mkdir(parents=True, exist_ok=True)

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        ref_video = resolve_child_case_insensitive(subject_dir, REFERENCE_VIDEO_NAME)
        expected_file = resolve_child_case_insensitive(subject_dir, EXPECTED_FILE_NAME)

        print(f"\n[SUBJECT] {subject_id}")

        if ref_video is None:
            print(f"  - ignoré: référence absente ({REFERENCE_VIDEO_NAME})")
            continue

        if expected_file is None:
            print(f"  - ignoré: attendu absent ({EXPECTED_FILE_NAME})")
            continue

        try:
            expected_blinks = read_expected_blinks(expected_file)
        except Exception as exc:
            print(f"  - ignoré: lecture attendu.txt impossible ({exc})")
            continue

        try:
            ref_result = analyze_video_one_pass(
                model_path=model_path,
                video_path=ref_video,
                profile=None,
                thresholds={"close_threshold": DEFAULT_CLOSE_THRESHOLD, "open_threshold": DEFAULT_OPEN_THRESHOLD},
                min_closed_frames=min_closed_frames,
                max_closed_frames=max_closed_frames,
                show=False,
                dynamic_normalization=False,
            )
            print(
                f"  - étalon analysé en {ref_result['elapsed_seconds']:.2f} s "
                f"| face_detect_rate={ref_result['face_detect_rate']:.4f}"
            )
        except Exception as exc:
            print(f"  - erreur analyse étalon: {exc}")
            continue

        try:
            ears = ref_result.get("ears")
            if ears is None or np.isfinite(ears).sum() < 10:
                print("  - vidéo étalon invalide")
                continue

            profile = build_profile_from_reference_ears(ears)
            ref_norm = median_filter_nan(normalize_series_static(ref_result["ears"], profile), k=SMOOTH_KERNEL)

            thresholds = calibrate_thresholds_on_reference(
                norm_ears=ref_norm,
                fps=ref_result["fps"],
                expected_blinks=expected_blinks,
                min_closed_frames=min_closed_frames,
                max_closed_frames=max_closed_frames,
            )

            peak_stats = compute_reference_peak_stats(
                ears=ref_result["ears"],
                norm_ears=ref_norm,
                close_threshold=thresholds["close_threshold"],
                open_threshold=thresholds["open_threshold"],
                min_closed_frames=min_closed_frames,
                max_closed_frames=max_closed_frames,
            )

            seuils_rows.append({
                "subject": subject_id,
                "min seuil EAR": format_optional_float(denormalize_threshold(thresholds["close_threshold"], profile), 6),
                "max seuil EAR": format_optional_float(denormalize_threshold(thresholds["open_threshold"], profile), 6),
                "min seuil EAR norm": format_optional_float(thresholds["close_threshold"], 6),
                "max seuil EAR norm": format_optional_float(thresholds["open_threshold"], 6),
                "min absolu EAR etalon": format_optional_float(peak_stats["abs_ear_min"], 6),
                "max absolu EAR etalon": format_optional_float(peak_stats["abs_ear_max"], 6),
                "moyenne pics bas EAR": format_optional_float(peak_stats["mean_low_peak_ear"], 6),
                "moyenne pics haut EAR": format_optional_float(peak_stats["mean_high_peak_ear"], 6),
                "moyenne pics bas EAR norm": format_optional_float(peak_stats["mean_low_peak_norm"], 6),
                "moyenne pics haut EAR norm": format_optional_float(peak_stats["mean_high_peak_norm"], 6),
            })

            predicted_ref = thresholds["predicted_reference_blinks"]
            signed_error = predicted_ref - expected_blinks
            absolute_error = abs(signed_error)
            relative_error = absolute_error / expected_blinks if expected_blinks > 0 else None

            reference_errors.append({
                "subject": subject_id,
                "expected": int(expected_blinks),
                "predicted": int(predicted_ref),
                "signed_error": int(signed_error),
                "absolute_error": int(absolute_error),
                "relative_error": float(relative_error) if relative_error is not None else None,
            })

            calib_msg = (
                f"  - calibration: attendu={expected_blinks}, "
                f"prédit_ref={predicted_ref}, "
                f"close={thresholds['close_threshold']:.2f}, "
                f"open={thresholds['open_threshold']:.2f}, "
                f"open_ref={profile['ear_open_ref']:.3f}, "
                f"closed_ref={profile['ear_closed_ref']:.3f}, "
            )
            calib_msg += f"err_rel={relative_error:.4f}" if relative_error is not None else "err_rel=NA"
            print(calib_msg)

        except Exception as exc:
            print(f"  - erreur calibration: {exc}")
            continue

        detail_rows, detail_ear_rows, detail_ear_norm_rows = [], [], []
        row = {"subject": subject_id}
        essential_row = {"subject": subject_id}
        mean_blink_row = {"subject": subject_id}
        min_max_blink_row = {"subject": subject_id}

        for category, filename in all_categories.items():
            video_path = resolve_child_case_insensitive(subject_dir, filename)

            if video_path is None or not video_path.is_file():
                row[category] = ""
                row[f"{category} mean"] = ""
                row[f"{category} low"] = ""
                row[f"{category} high"] = ""
                row[f"{category} per minute"] = ""
                row[f"{category} Face Detect Rate"] = ""

                if category != "étalon":
                    essential_row[category] = ""
                    mean_blink_row[category] = ""
                    min_max_blink_row[f"{category} min"] = ""
                    min_max_blink_row[f"{category} max"] = ""

                print(f"  - {category}: absent")
                continue

            try:
                result = analyze_video_one_pass(
                    model_path=model_path,
                    video_path=video_path,
                    profile=profile,
                    thresholds=thresholds,
                    min_closed_frames=min_closed_frames,
                    max_closed_frames=max_closed_frames,
                    show=False,
                    dynamic_normalization=False,
                )

                if deepData:
                    append_deepdata_rows(
                        detail_rows,
                        detail_ear_rows,
                        detail_ear_norm_rows,
                        f"{subject_id}/{video_path.name}",
                        profile,
                        thresholds,
                        result["ears"],
                        result["norm_ears"],
                        result["blink_states"],
                    )

                row[category] = result["blink_count"]
                row[f"{category} mean"] = format_optional_float(result["blink_interval_mean"], 3)
                row[f"{category} low"] = format_optional_float(result["blink_interval_min"], 3)
                row[f"{category} high"] = format_optional_float(result["blink_interval_max"], 3)
                row[f"{category} per minute"] = format_optional_float(result["blinks_per_minute"], 1)
                row[f"{category} Face Detect Rate"] = f"{result['face_detect_rate']:.4f}"

                if category != "étalon":
                    essential_row[category] = format_optional_float(result["blinks_per_minute"], 1)
                    mean_blink_row[category] = format_optional_float(result["blink_interval_mean"], 3)
                    min_max_blink_row[f"{category} min"] = format_optional_float(result["blink_interval_min"], 3)
                    min_max_blink_row[f"{category} max"] = format_optional_float(result["blink_interval_max"], 3)

                print(
                    f"  - {category}: {result['blink_count']} | "
                    f"mean={result['blink_interval_mean'] if result['blink_interval_mean'] is not None else 'NA'} | "
                    f"low={result['blink_interval_min'] if result['blink_interval_min'] is not None else 'NA'} | "
                    f"high={result['blink_interval_max'] if result['blink_interval_max'] is not None else 'NA'} | "
                    f"per_min={format_optional_float(result['blinks_per_minute'], 1) if result['blinks_per_minute'] is not None else 'NA'} | "
                    f"Face Detect Rate={result['face_detect_rate']:.4f} | "
                    f"time={result['elapsed_seconds']:.2f}s"
                )

            except Exception as exc:
                row[category] = ""
                row[f"{category} mean"] = ""
                row[f"{category} low"] = ""
                row[f"{category} high"] = ""
                row[f"{category} per minute"] = ""
                row[f"{category} Face Detect Rate"] = ""

                if category != "étalon":
                    essential_row[category] = ""
                    mean_blink_row[category] = ""
                    min_max_blink_row[f"{category} min"] = ""
                    min_max_blink_row[f"{category} max"] = ""

                print(f"  - {category}: erreur ({exc})")

        if deepData:
            write_subject_deepdata_files(subject_id, detail_rows, detail_ear_rows, detail_ear_norm_rows)

        rows.append(row)
        essential_rows.append(essential_row)
        mean_blink_rows.append(mean_blink_row)
        min_max_blink_rows.append(min_max_blink_row)

    fieldnames = ["subject"]
    for category in all_categories.keys():
        fieldnames.extend(full_metric_fields(category))

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    essential_fieldnames = ["subject"] + list(CATEGORY_FILES.keys())
    with open(ESSENTIAL_OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=essential_fieldnames)
        writer.writeheader()
        writer.writerows(essential_rows)

    mean_blink_fieldnames = ["subject"] + list(CATEGORY_FILES.keys())
    with open(MEAN_BLINK_TIME_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mean_blink_fieldnames)
        writer.writeheader()
        writer.writerows(mean_blink_rows)

    min_max_fieldnames = ["subject"]
    for category in CATEGORY_FILES.keys():
        min_max_fieldnames.extend([f"{category} min", f"{category} max"])

    with open(MIN_MAX_BLINK_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=min_max_fieldnames)
        writer.writeheader()
        writer.writerows(min_max_blink_rows)

    metrics = compute_reference_error_metrics(reference_errors)
    write_reference_error_report(ERROR_REPORT_FILE, reference_errors, metrics)
    write_seuils_csv(seuils_rows)

    print(f"\nCSV écrit: {output_csv}")
    print(f"CSV essentiel écrit: {ESSENTIAL_OUTPUT_CSV}")
    print(f"CSV mean blink time écrit: {MEAN_BLINK_TIME_CSV}")
    print(f"CSV min/max blink time écrit: {MIN_MAX_BLINK_CSV}")
    print(f"Rapport erreur écrit: {ERROR_REPORT_FILE}")
    print(f"Seuils écrits: {SEUILS_CSV}")
    if deepData:
        print(f"Détails écrits dans: {DETAILS_DIR}")

    if metrics["n_subjects"] > 0:
        rme = metrics["relative_mean_error"]
        mean_bias = metrics["mean_bias"]
        error_std = metrics["error_std"]

        summary = f"Résumé erreur étalons: n={metrics['n_subjects']} | "
        summary += f"erreur moyenne relative={(rme * 100):.2f}% | " if rme is not None else "erreur moyenne relative=NA | "
        summary += (
            f"biais moyen={mean_bias:.4f} | écart type erreur={error_std:.4f}"
            if mean_bias is not None and error_std is not None
            else "biais moyen=NA | écart type erreur=NA"
        )
        print(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Compteur de clignements avec EAR normalisé, dry-run et batch par sujet."
    )

    parser.add_argument("--model", required=True, help="Chemin vers face_landmarker.task")
    parser.add_argument(
        "--dry-run",
        nargs="?",
        const=True,
        default=True,
        type=str2bool,
        help="true par défaut. Passer false pour lancer le real run.",
    )
    parser.add_argument(
        "--dry-run-mode",
        choices=["dynamic", "faithful"],
        default="dynamic",
        help="Mode dry run: dynamic utilise le seul MP4 dans video/dry. faithful utilise un vrai dossier sujet.",
    )
    parser.add_argument("--show", action="store_true", help="Affiche la vidéo annotée. Dry run uniquement.")
    parser.add_argument("--show-landmarks", action="store_true", help="Affiche les points MediaPipe sur la vidéo. Dry run uniquement.")
    parser.add_argument("--deepData", type=str2bool, default=False, help="Génère les CSV de détail par sujet dans /details.")
    parser.add_argument("--csv", default=DEFAULT_OUTPUT_CSV, help=f"CSV de sortie principal. Défaut: {DEFAULT_OUTPUT_CSV}")
    parser.add_argument("--min-closed-frames", type=int, default=DEFAULT_MIN_CLOSED_FRAMES)
    parser.add_argument("--max-closed-frames", type=int, default=DEFAULT_MAX_CLOSED_FRAMES)

    args = parser.parse_args()

    model_path = Path(args.model)
    output_csv = Path(args.csv)

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")

    if args.dry_run is False and args.show:
        raise ValueError("--show est interdit en real run.")

    if args.dry_run is False and args.show_landmarks:
        raise ValueError("--show-landmarks est valide uniquement en dry run.")

    if args.show_landmarks and not args.show:
        raise ValueError("--show-landmarks nécessite --show.")

    t0 = time.perf_counter()

    if args.dry_run:
        run_dry_run(
            model_path=model_path,
            show=args.show,
            min_closed_frames=args.min_closed_frames,
            max_closed_frames=args.max_closed_frames,
            mode=args.dry_run_mode,
            deepData=args.deepData,
            show_landmarks=args.show_landmarks,
        )
    else:
        run_real(
            model_path=model_path,
            output_csv=output_csv,
            min_closed_frames=args.min_closed_frames,
            max_closed_frames=args.max_closed_frames,
            deepData=args.deepData,
        )

    elapsed = time.perf_counter() - t0
    print(f"\nTemps total d'exécution : {elapsed:.2f} s")


if __name__ == "__main__":
    main()