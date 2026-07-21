"""
background_randomization.py
Train-only leaf segmentation + background compositing for tomato leaf disease
domain-generalization experiments (Plan 1).

Pipeline per image (applied with probability `prob` at TRAIN time only):
    1. Segment the leaf from PlantVillage's near-uniform background (HSV threshold).
    2. Composite the segmented leaf onto a random natural background.
    3. Hand the RGB composite back to the rest of the train transform chain.

HYGIENE (enforced by the caller, see notes at bottom):
    - TRAIN ONLY. Never wrap val/test datasets with this.
    - `background_dir` must be domain-neutral textures, NOT the real-world test source.
    - Keep this BEFORE resize-to-224 + ImageNet normalization in the transform chain.

Dependencies: numpy, opencv (cv2), Pillow. (No torch dependency here so it's unit-testable
in isolation; the torchvision wrapper at the bottom is optional.)
"""

from __future__ import annotations
import os
import glob
import random
from typing import List, Optional

import numpy as np
import cv2
from PIL import Image


# --------------------------------------------------------------------------- #
# 1. Leaf segmentation (classical, no neural model needed for PlantVillage)    #
# --------------------------------------------------------------------------- #
def segment_leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Return a binary uint8 mask (H, W) where 255 = leaf, 0 = background.

    Strategy: PlantVillage leaves are green/brown foliage on a near-uniform
    (usually grey/black) background. We threshold in HSV for vegetation-like
    hue+saturation, then clean up and keep the largest connected component.

    This is intentionally conservative. If your PlantVillage variant has a
    different background colour, adjust the HSV bounds or fall back to the
    pretrained-segmentation route (see Plan 1 §3.3 Route B).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Vegetation hue band is broad because diseased leaves go yellow/brown.
    # Hue in OpenCV is 0-179. We accept green->yellow->brown by using a
    # moderately wide band plus a saturation floor to reject grey backgrounds.
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Leaf-ish: some saturation (not grey), not pure-black, hue in green/yellow/brown.
    sat_ok = s > 30
    val_ok = v > 30
    hue_ok = (h < 45) | (h > 20)  # effectively broad; refine per-dataset if needed
    mask = (sat_ok & val_ok & hue_ok).astype(np.uint8) * 255

    # Morphological cleanup: close holes inside the leaf, open to drop speckle.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep the largest connected component (the leaf), drop stray blobs.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        # Nothing found -> return an all-leaf mask so we fall back to original image.
        return np.full(rgb.shape[:2], 255, np.uint8)
    # stats[0] is background; pick largest of the rest by area.
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    return mask


def _mask_quality_ok(mask: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> bool:
    """Reject masks that are almost-empty or almost-full (segmentation failed)."""
    frac = (mask > 0).mean()
    return lo <= frac <= hi


# --------------------------------------------------------------------------- #
# 2. Compositing                                                              #
# --------------------------------------------------------------------------- #
def composite_leaf_on_background(
    rgb: np.ndarray,
    mask: np.ndarray,
    background: np.ndarray,
    scale_range: tuple = (0.6, 1.0),
    boundary_blur: bool = True,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """
    Paste the masked leaf onto `background` at a random scale/position.
    Returns an RGB uint8 image the same size as the ORIGINAL image.

    Lesion-preservation: scale_range floor is kept high enough that the diseased
    region is not shrunk into oblivion. Do not lower below ~0.5.
    """
    rng = rng or random
    H, W = rgb.shape[:2]

    # Resize background to the working canvas size.
    bg = cv2.resize(background, (W, H), interpolation=cv2.INTER_LINEAR)

    # Random scale of the leaf.
    scale = rng.uniform(*scale_range)
    lh, lw = max(1, int(H * scale)), max(1, int(W * scale))
    leaf_small = cv2.resize(rgb, (lw, lh), interpolation=cv2.INTER_LINEAR)
    mask_small = cv2.resize(mask, (lw, lh), interpolation=cv2.INTER_NEAREST)

    # Soften the cut-out edge so it doesn't look pasted (helps realism, mildly).
    if boundary_blur:
        mask_blur = cv2.GaussianBlur(mask_small, (7, 7), 0).astype(np.float32) / 255.0
    else:
        mask_blur = (mask_small > 0).astype(np.float32)
    mask_blur = mask_blur[..., None]  # (lh, lw, 1)

    # Random top-left placement so the leaf isn't always centred.
    top = rng.randint(0, H - lh)
    left = rng.randint(0, W - lw)

    canvas = bg.astype(np.float32)
    roi = canvas[top:top + lh, left:left + lw, :]
    blended = leaf_small.astype(np.float32) * mask_blur + roi * (1.0 - mask_blur)
    canvas[top:top + lh, left:left + lw, :] = blended
    return np.clip(canvas, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 3. The transform object                                                     #
# --------------------------------------------------------------------------- #
class BackgroundRandomize:
    """
    Callable transform. Input/Output: PIL.Image (RGB), so it slots into a
    torchvision Compose BEFORE Resize(224)+ToTensor()+Normalize().

    Args:
        background_dir: folder of domain-neutral background images
                        (soil/foliage/wood/sky/etc). MUST NOT be the real-world
                        test source.
        prob:           fraction of images to composite (tune on VAL only).
        scale_range:    leaf scale relative to canvas; floor keeps lesions visible.
        boundary_blur:  soften the leaf edge.
        seed:           optional per-worker seed for reproducibility.
    """

    def __init__(
        self,
        background_dir: str,
        prob: float = 0.5,
        scale_range: tuple = (0.6, 1.0),
        boundary_blur: bool = True,
        seed: Optional[int] = None,
    ):
        self.background_dir = background_dir
        self.prob = float(prob)
        self.scale_range = scale_range
        self.boundary_blur = boundary_blur
        self.rng = random.Random(seed)

        self.bg_paths: List[str] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            self.bg_paths += glob.glob(os.path.join(background_dir, "**", ext),
                                       recursive=True)
        if not self.bg_paths:
            raise FileNotFoundError(
                f"No background images found under {background_dir!r}. "
                "Populate it with domain-neutral textures (NOT real-world test images)."
            )

    def _load_background(self) -> np.ndarray:
        path = self.rng.choice(self.bg_paths)
        bg = cv2.imread(path, cv2.IMREAD_COLOR)          # BGR
        if bg is None:
            # Corrupt file -> neutral grey fallback so training never crashes.
            return np.full((224, 224, 3), 127, np.uint8)
        return cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    def __call__(self, img: Image.Image) -> Image.Image:
        # Roll the dice: leave (1 - prob) of images as clean originals.
        if self.rng.random() > self.prob:
            return img

        rgb = np.asarray(img.convert("RGB"))
        mask = segment_leaf_mask(rgb)

        # If segmentation clearly failed, skip compositing (return original).
        if not _mask_quality_ok(mask):
            return img

        bg = self._load_background()
        out = composite_leaf_on_background(
            rgb, mask, bg,
            scale_range=self.scale_range,
            boundary_blur=self.boundary_blur,
            rng=self.rng,
        )
        return Image.fromarray(out)


# --------------------------------------------------------------------------- #
# 4. Optional: how it slots into the TRAIN transform chain                     #
# --------------------------------------------------------------------------- #
def build_train_transform(background_dir: str, prob: float, seed: int = 42):
    """
    Reference wiring. Requires torchvision at call time (imported lazily so this
    file stays unit-testable without torch).

    ORDER MATTERS:
      BackgroundRandomize  -> geometric/photometric aug -> Resize(224) -> Normalize
    Advanced albumentations (Stack ON) also go in this train-only chain, on top.
    NEVER attach BackgroundRandomize to the val/test transform.
    """
    from torchvision import transforms

    return transforms.Compose([
        BackgroundRandomize(background_dir=background_dir, prob=prob, seed=seed),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform():
    """Val/test: resize + normalize ONLY. No augmentation, no background swap."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
