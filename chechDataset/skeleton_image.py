import os
import sys
import yaml
from pathlib import Path
import re
import numpy as np
from PIL import Image, ImageOps
import cv2
from collections import defaultdict

from skeletonModule import SkeletonDetector

_CAMERA_INDEX_RE = re.compile(r"camera_(\d+)_image_raw")
_IMAGE_RE = re.compile(r"(racecar_camera_camera_\d+_image_raw)_(\d{5})\.jpg")
_MASK_RE = re.compile(r"(racecar_camera_camera_\d+_image_raw)_(\d{5})_[0-9a-f]{8}_mask\.(?:png|jpg)", re.IGNORECASE)

def _camera_index(cam_name: str) -> int:
    """Extract camera number from cam_name for consistent ordering (e.g. camera_0 -> 0)."""
    m = _CAMERA_INDEX_RE.search(cam_name)
    return int(m.group(1)) if m else 0

class SkeletonImage:
    def __init__(self, root: Path):
        self.image_dir = root / "individual"
        self.samples = self._index_samples()
        self.input_resolution = None
        self.set_input_resolution()

        print(f"Input resolution: {self.input_resolution}")

    def set_input_resolution(self):
        # get the first image from the dataset (samples are keyed by scene id, not 0-based index)
        indices = self.samples.keys()
        if not indices:
            raise ValueError("Dataset has no samples")
        images = self.load_images(next(iter(indices)))
        # load_images returns dict of cam_name -> array; use first image for resolution
        first_img = next(iter(images.values()))
        self.input_resolution = first_img.shape[:2]

    def _index_samples(self):
        samples = defaultdict(lambda: {"images": {}})

        for img_path in self.image_dir.glob("*.jpg"):
            m = _IMAGE_RE.match(img_path.name)
            if not m:
                continue
            cam_name = m.group(1)
            idx = int(m.group(2))
            samples[idx]["images"][cam_name] = img_path

        synced = {
            idx: s
            for idx, s in samples.items()
            if len(s["images"]) > 0
        }
        return dict(sorted(synced.items()))

    def load_images(self, idx: int):
        items = sorted(
            self.samples[idx]["images"].items(),
            key=lambda x: _camera_index(x[0]),
        )
        imgs = {}
        for cam_name, path in items:
            with Image.open(path) as pil_img:
                pil_img = pil_img.convert("RGB")
                img = np.array(pil_img)
            imgs[cam_name] = img
        return imgs

    def _create_waiting_placeholder(self, width=640, height=480):
        """Create a placeholder image when waiting for camera frames"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        text = 'No Signal'
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 1.0, 2)
        x = (width - tw) // 2
        y = (height + th) // 2
        cv2.putText(img, text, (x, y), font, 1.0, (128, 128, 128), 2, cv2.LINE_AA)
        return img

    def create_combined_view(self, frames: list, max_height: int = 1080) -> np.ndarray | None:
        """Arrange frames into a grid. Frames with no mask show the raw image."""
        if not frames:
            return None

        height, width = frames[0].shape[:2]
        placeholder = self._create_waiting_placeholder(width, height)
        num = len(frames)

        if num <= 2:
            combined = np.hstack(frames)
        elif num == 3:
            top_row = np.hstack(frames[:2])
            bottom_frame = frames[2]
            pad_total = top_row.shape[1] - bottom_frame.shape[1]
            lp = pad_total // 2
            rp = pad_total - lp
            bottom_row = np.hstack([
                np.zeros((height, lp, 3), dtype=np.uint8) if lp > 0 else np.empty((height, 0, 3), dtype=np.uint8),
                bottom_frame,
                np.zeros((height, rp, 3), dtype=np.uint8) if rp > 0 else np.empty((height, 0, 3), dtype=np.uint8),
            ])
            combined = np.vstack([top_row, bottom_row])
        elif num == 4:
            combined = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:4])])
        elif num == 5:
            top_row = np.hstack(frames[:2])
            mid_row = np.hstack(frames[2:4])
            bottom_frame = frames[4]
            pad_total = top_row.shape[1] - bottom_frame.shape[1]
            lp = pad_total // 2
            rp = pad_total - lp
            bottom_row = np.hstack([
                np.zeros((height, lp, 3), dtype=np.uint8) if lp > 0 else np.empty((height, 0, 3), dtype=np.uint8),
                bottom_frame,
                np.zeros((height, rp, 3), dtype=np.uint8) if rp > 0 else np.empty((height, 0, 3), dtype=np.uint8),
            ])
            combined = np.vstack([top_row, mid_row, bottom_row])
        else:
            cols = 2
            grid_rows = []
            for i in range(0, num, cols):
                row_frames = list(frames[i:i + cols])
                while len(row_frames) < cols:
                    row_frames.append(placeholder.copy())
                grid_rows.append(np.hstack(row_frames))
            combined = np.vstack(grid_rows)

        if max_height > 0 and combined.shape[0] > max_height:
            scale = max_height / combined.shape[0]
            combined = cv2.resize(
                combined,
                (int(combined.shape[1] * scale), int(combined.shape[0] * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        return combined


def process_scene(
    skeleton_image: SkeletonImage,
    idx: int,
    skeleton_detector: SkeletonDetector | None = None,
) -> tuple[np.ndarray | None, list[dict], list[tuple[int, np.ndarray]]]:
    """
    Load images and optionally run pose detection per camera for a single scene.
    No mask overlay; only skeleton detection and multicamera combined view.

    Returns:
        combined        – BGR grid image (or None if no frames).
        cam_detections  – list of {cam, persons} dicts with raw detection data.
        individual_frames – list of (cam_idx, frame) for each camera.
    """
    images = skeleton_image.load_images(idx)
    cam_names = list(images.keys())
    img_list = [cv2.cvtColor(images[c], cv2.COLOR_RGB2BGR) for c in cam_names]

    # Pose detection on raw frames; store detections and optionally draw
    cam_detections: list[dict] = []
    if skeleton_detector is not None:
        annotated = []
        for cam_name, raw_frame in zip(cam_names, img_list):
            persons = skeleton_detector.detect(raw_frame)
            cam_detections.append({
                "cam": _camera_index(cam_name),
                "persons": persons,
            })
            annotated.append(skeleton_detector.draw(raw_frame, persons))
        img_list = annotated

    frames: list[np.ndarray] = []
    individual_frames: list[tuple[int, np.ndarray]] = []
    for cam_name, frame in zip(cam_names, img_list):
        if frame is None:
            continue
        cam_idx = _camera_index(cam_name)
        tags = f"cam_{cam_idx}"
        if skeleton_detector is not None:
            tags += " + pose"
        cv2.putText(frame, tags, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(frame)
        individual_frames.append((cam_idx, frame))

    return skeleton_image.create_combined_view(frames), cam_detections, individual_frames


if __name__ == "__main__":
    root = Path("../dataset-sdv-mar6-2")
    skeleton_image = SkeletonImage(root)

    detector = SkeletonDetector(
        os.path.join(os.path.dirname(__file__), "yolo26l-pose.pt")
    )

    out_dir = Path(__file__).parent / "skeleton_detections"
    out_dir.mkdir(exist_ok=True)
    individuals_dir = out_dir / "individuals"
    individuals_dir.mkdir(exist_ok=True)

    indices = list(skeleton_image.samples.keys())
    print(f"Processing {len(indices)} scenes → {out_dir}")

    all_scenes: list[dict] = []

    for i, idx in enumerate(indices):
        combined, cam_detections, individual_frames = process_scene(skeleton_image, idx, skeleton_detector=detector)
        if combined is None:
            print(f"  [{i+1}/{len(indices)}] idx={idx:05d}  skipped (no frames)")
            continue

        out_path = out_dir / f"scene_{idx:05d}.jpg"
        cv2.imwrite(str(out_path), combined)

        for cam_idx, frame in individual_frames:
            ind_path = individuals_dir / f"scene_{idx:05d}_cam{cam_idx}.jpg"
            cv2.imwrite(str(ind_path), frame)

        print(f"  [{i+1}/{len(indices)}] idx={idx:05d}  saved → {out_path.name}  ({len(individual_frames)} individual frames)")

        all_scenes.append({
            "scene": f"{idx:05d}",
            "cameras": cam_detections,
        })

    # Save detection YAML
    yaml_path = out_dir / "detections.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({"scenes": all_scenes}, f,
                  default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"Detections saved → {yaml_path}")

    print("Done.")

