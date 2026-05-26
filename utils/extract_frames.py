"""Extract frames from video files at a configurable time interval.

Usage:
    python utils/extract_frames.py
    (runs extract_all_videos() using paths from config.py)
"""

import os
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    interval_sec: float = config.FRAME_EXTRACT_INTERVAL_SEC,
) -> int:
    """Extract frames from a single video file at fixed time intervals.

    Args:
        video_path:   Absolute or relative path to the video file.
        output_dir:   Directory where extracted frames will be saved as JPEGs.
        interval_sec: Seconds between successive extracted frames.

    Returns:
        Number of frames successfully extracted. Returns 0 if the video
        could not be opened or has invalid FPS metadata.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARNING] Cannot open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print(f"[WARNING] Could not read FPS for {video_path} — skipping.")
        cap.release()
        return 0

    frame_interval = max(1, int(fps * interval_sec))
    os.makedirs(output_dir, exist_ok=True)

    extracted = 0
    frame_idx = 0

    with tqdm(total=total_frames, desc=f"  {Path(video_path).name}", unit="frame", leave=True) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                out_path = os.path.join(output_dir, f"frame_{extracted:06d}.jpg")
                cv2.imwrite(out_path, frame)
                extracted += 1

            frame_idx += 1
            pbar.update(1)

    cap.release()
    return extracted


def extract_all_videos(
    raw_dir: str = config.RAW_DIR,
    frames_dir: str = config.FRAMES_DIR,
    interval_sec: float = config.FRAME_EXTRACT_INTERVAL_SEC,
) -> None:
    """Scan a folder for supported videos and extract frames from each.

    Each video gets its own subfolder under frames_dir named after the video
    stem (e.g. data/frames/my_video/frame_000000.jpg).

    Args:
        raw_dir:      Directory containing raw video files.
        frames_dir:   Root directory where per-video frame subfolders are saved.
        interval_sec: Seconds between successive extracted frames.
    """
    supported_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        print(f"[ERROR] Raw directory not found: {raw_dir}")
        return

    videos = [f for f in raw_path.iterdir() if f.suffix.lower() in supported_exts]

    if not videos:
        print(f"[WARNING] No supported video files found in '{raw_dir}'.")
        print(f"          Supported formats: {', '.join(sorted(supported_exts))}")
        return

    print(f"[INFO] Found {len(videos)} video(s) in '{raw_dir}'")
    print(f"[INFO] Extracting 1 frame every {interval_sec}s\n")

    total_extracted = 0
    for video in sorted(videos):
        video_out_dir = os.path.join(frames_dir, video.stem)
        print(f"[INFO] Processing: {video.name}")
        count = extract_frames_from_video(str(video), video_out_dir, interval_sec)
        print(f"       → {count} frames saved to '{video_out_dir}'\n")
        total_extracted += count

    print(f"[DONE] Total frames extracted: {total_extracted}")


if __name__ == "__main__":
    extract_all_videos()
