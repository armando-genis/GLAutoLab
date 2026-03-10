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

class MaskImage:
    def __init__(self, root: Path):
        self.image_dir = root / "individual"
        self.images_dir_mask = root / "individual_mask"
        self.mask_resolution = None  # (width, height) when mask images exist
        self.samples = self._index_samples()
        self._index_masks()
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

    def _index_masks(self):
        """
        Index mask images from images_dir_mask if the folder exists and contains images.
        Only sets mask paths on samples when the folder exists and has mask images.
        """
        if not self.images_dir_mask.exists() or not self.images_dir_mask.is_dir():
            return
        mask_paths = list(self.images_dir_mask.glob("*.jpg")) + list(self.images_dir_mask.glob("*.png"))
        if not mask_paths:
            return
        for path in mask_paths:
            m = _MASK_RE.match(path.name)
            if not m:
                continue
            cam_name = m.group(1)
            idx = int(m.group(2))
            if idx in self.samples:
                self.samples[idx].setdefault("masks", {})[cam_name] = path

        # Store mask resolution from first available mask image
        if self.has_masks():
            first_path = None
            for s in self.samples.values():
                if "masks" in s and s["masks"]:
                    first_path = next(iter(s["masks"].values()))
                    break
            if first_path is not None:
                with Image.open(first_path) as pil_img:
                    self.mask_resolution = pil_img.size  # (width, height)

    def has_masks(self) -> bool:
        """True if any sample has mask images (mask folder existed and had images)."""
        return any("masks" in s and s["masks"] for s in self.samples.values())

    def load_masks(self, idx: int):
        """
        Load mask images for scene idx. Returns None if no masks are available
        for this scene; otherwise returns a dict cam_name -> np.ndarray (grayscale mask).
        """
        masks_entry = self.samples[idx].get("masks")
        if not masks_entry:
            return None
        items = sorted(
            masks_entry.items(),
            key=lambda x: _camera_index(x[0]),
        )
        masks = {}
        for cam_name, path in items:
            with Image.open(path) as pil_img:
                pil_img = pil_img.convert("L")
                mask = np.array(pil_img)
            masks[cam_name] = mask
        return masks if masks else None

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


    def procces_mask_with_images(self, images: list[np.ndarray | None],
                    images_mask: list[np.ndarray | None] | None = None) -> list[np.ndarray | None]:
        """
        For each image, overlay the corresponding mask when available.
        Images with no matching mask are returned unchanged.
        Returns a new list of the same length as `images`.
        """
        result = []
        for i, img in enumerate(images):
            if img is None:
                result.append(None)
                continue

            if images_mask is not None and i < len(images_mask) and images_mask[i] is not None:
                color_mask, _ = self.colorize_mask(images_mask[i])
                img = self.overlay_mask(img, color_mask, alpha=0.4)

            result.append(img)
        return result

                
    def overlay_mask(self, image: np.ndarray, color_mask: np.ndarray, alpha=0.4):
        """
        Blend color mask over image.
        """
        return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)

    def colorize_mask(self, mask: np.ndarray):
        """
        Convert class-id mask into BGR color mask.
        """
        # Resize mask to input resolution
        mask_resized = cv2.resize(
            mask,
            (self.input_resolution[1], self.input_resolution[0]),
            interpolation=cv2.INTER_NEAREST
        )

        # Create empty color image
        color_mask = np.zeros((mask_resized.shape[0], mask_resized.shape[1], 3), dtype=np.uint8)

        # Class 1: sidewalk (BGR: [255,0,0])
        color_mask[mask_resized == 1] = (255, 0, 0)

        return color_mask, mask_resized


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
    mask_image: MaskImage,
    idx: int,
    skeleton_detector: SkeletonDetector | None = None,
) -> tuple[np.ndarray | None, list[dict], list[tuple[int, np.ndarray]]]:
    """
    Load, overlay mask and optionally draw pose skeletons for a single scene.

    Returns:
        combined        – BGR grid image (or None if no frames).
        cam_detections  – list of {cam, persons} dicts with raw detection data.
        individual_frames – list of (cam_idx, frame) for each camera that
                           produced a valid frame (mask + annotations applied).
    """
    images = mask_image.load_images(idx)
    masks  = mask_image.load_masks(idx)

    cam_names = list(images.keys())
    img_list  = [cv2.cvtColor(images[c], cv2.COLOR_RGB2BGR) for c in cam_names]
    mask_list = [masks.get(c) if masks else None for c in cam_names]

    # 1. Pose detection on raw frames (clean pixels → better accuracy)
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

    # 2. Mask overlay on top of annotated frames
    processed = mask_image.procces_mask_with_images(img_list, mask_list)

    frames: list[np.ndarray] = []
    individual_frames: list[tuple[int, np.ndarray]] = []
    for cam_name, frame, mask in zip(cam_names, processed, mask_list):
        if frame is None:
            continue

        cam_idx = _camera_index(cam_name)
        tags = f"cam_{cam_idx}"
        if mask is not None:
            tags += " + mask"
        if skeleton_detector is not None:
            tags += " + pose"
        cv2.putText(frame, tags, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(frame)
        individual_frames.append((cam_idx, frame))

    return mask_image.create_combined_view(frames), cam_detections, individual_frames


if __name__ == "__main__":
    root = Path("../dataset-sdv-feb28")
    mask_image = MaskImage(root)

    detector = SkeletonDetector(
        os.path.join(os.path.dirname(__file__), "yolo26l-pose.pt")
    )

    out_dir = Path(__file__).parent / "masked_applied"
    out_dir.mkdir(exist_ok=True)
    individuals_dir = out_dir / "individuals"
    individuals_dir.mkdir(exist_ok=True)

    indices = list(mask_image.samples.keys())
    print(f"Processing {len(indices)} scenes → {out_dir}")

    all_scenes: list[dict] = []

    for i, idx in enumerate(indices):
        combined, cam_detections, individual_frames = process_scene(mask_image, idx, skeleton_detector=detector)
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

