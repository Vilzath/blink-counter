#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# Indices classiques des landmarks des yeux sur le mesh visage MediaPipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def landmark_to_pixel(landmark, width, height):
    return (landmark.x * width, landmark.y * height)


def eye_aspect_ratio(landmarks, eye_indices, width, height):
    """
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Avec les 6 points de l'oeil.
    """
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


def main():
    parser = argparse.ArgumentParser(description="Compteur de clignements sur un fichier MP4")
    parser.add_argument("--video", required=True, help="Chemin vers le fichier vidéo MP4")
    parser.add_argument("--model", required=True, help="Chemin vers face_landmarker.task")
    parser.add_argument("--threshold", type=float, default=0.21,
                        help="Seuil EAR sous lequel l'oeil est considéré fermé (défaut: 0.21)")
    parser.add_argument("--min-closed-frames", type=int, default=2,
                        help="Nombre minimal de frames fermées pour valider un clignement (défaut: 2)")
    parser.add_argument("--max-closed-frames", type=int, default=10,
                        help="Nombre maximal de frames fermées pour éviter de compter un long fermé comme un blink (défaut: 10)")
    parser.add_argument("--csv", default="blinks.csv",
                        help="Fichier CSV de sortie pour les timestamps des clignements")
    parser.add_argument("--show", action="store_true",
                        help="Affiche la vidéo annotée pendant l'analyse")
    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    csv_path = Path(args.csv)

    if not video_path.exists():
        raise FileNotFoundError(f"Vidéo introuvable: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

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

    blink_count = 0
    blink_timestamps = []
    closed_frames = 0
    frame_idx = 0

    with FaceLandmarker.create_from_options(options) as landmarker:
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

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]
                current_ear = get_ear(face_landmarks, width, height)

                if current_ear < args.threshold:
                    closed_frames += 1
                else:
                    if args.min_closed_frames <= closed_frames <= args.max_closed_frames:
                        blink_count += 1
                        blink_time_sec = frame_idx / fps
                        blink_timestamps.append(blink_time_sec)
                    closed_frames = 0
            else:
                # Si le visage disparaît, on réinitialise pour éviter un faux blink
                closed_frames = 0

            if args.show:
                overlay = frame.copy()
                status = "NO FACE"
                if current_ear is not None:
                    status = f"EAR={current_ear:.3f}"

                cv2.putText(
                    overlay,
                    f"Blinks: {blink_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    status,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    overlay,
                    f"Frame: {frame_idx}/{total_frames if total_frames > 0 else '?'}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Blink Counter", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if frame_idx % 100 == 0:
                print(f"Progression: {frame_idx}/{total_frames if total_frames > 0 else '?'} frames | blinks={blink_count}")

    cap.release()
    cv2.destroyAllWindows()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["blink_index", "timestamp_seconds"])
        for i, ts in enumerate(blink_timestamps, start=1):
            writer.writerow([i, f"{ts:.3f}"])

    print("\n--- Résultat ---")
    print(f"Vidéo              : {video_path}")
    print(f"Modèle             : {model_path}")
    print(f"Seuil EAR          : {args.threshold}")
    print(f"Clignements        : {blink_count}")
    print(f"CSV                : {csv_path}")

    if blink_timestamps:
        print("Timestamps (s)     :")
        for i, ts in enumerate(blink_timestamps, start=1):
            print(f"  {i:03d} -> {ts:.3f}s")


if __name__ == "__main__":
    main()
