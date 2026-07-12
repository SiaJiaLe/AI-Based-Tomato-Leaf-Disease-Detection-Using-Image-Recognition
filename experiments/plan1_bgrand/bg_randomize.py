"""Background-randomization transform (train-only).

The one new variable Plan 1 adds on top of the EfficientNetB0 Stack-ON recipe.
For a fraction `p` of training images it (a) segments the leaf from
PlantVillage's near-uniform backdrop, (b) composites it onto a random generic
background, so the network can no longer lean on "clean uniform background" as
a shortcut. The remaining `1 - p` of images pass through untouched, so the
model still sees the originals (this is why no image copying is needed).

Segmentation deliberately keys on *foreground-vs-uniform-background* (distance
from the corner/background colour) rather than a green-hue threshold, so that
brown/yellow lesions — the diseased pixels the label depends on — are kept
inside the mask instead of being cut away.

Only imports stdlib + cv2 + albumentations (both already dependencies). Nothing
here touches experiments.common.
"""
import os

import albumentations as A
import cv2
import numpy as np

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _load_backgrounds(background_dir: str, cache_size: int = 384) -> list:
    """Read every image in background_dir once, as RGB uint8 arrays."""
    if not os.path.isdir(background_dir):
        raise FileNotFoundError(f"background_dir does not exist: {background_dir}")
    paths = [os.path.join(background_dir, f) for f in sorted(os.listdir(background_dir))
             if f.lower().endswith(_IMAGE_EXTS)]
    backgrounds = []
    for p in paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Downsize to keep the in-memory/pickled-to-workers footprint small.
        h, w = rgb.shape[:2]
        if max(h, w) > cache_size:
            scale = cache_size / max(h, w)
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        backgrounds.append(rgb)
    if not backgrounds:
        raise FileNotFoundError(
            f"No readable images found in background_dir: {background_dir}. "
            "Generate some first (make_synthetic_backgrounds.py) or point at a "
            "folder of generic CC0 textures.")
    return backgrounds


def _background_color(img: np.ndarray) -> np.ndarray:
    """Median colour of the four image corners — the uniform PlantVillage backdrop."""
    h, w = img.shape[:2]
    c = max(2, min(h, w) // 20)
    corners = np.concatenate([
        img[:c, :c].reshape(-1, 3),
        img[:c, -c:].reshape(-1, 3),
        img[-c:, :c].reshape(-1, 3),
        img[-c:, -c:].reshape(-1, 3),
    ], axis=0)
    return np.median(corners, axis=0)


def segment_leaf(img: np.ndarray) -> np.ndarray:
    """Return a binary (0/255) leaf mask via background-colour distance + Otsu.

    Robust to lesions because it segments 'not the backdrop', not 'green'.
    """
    bg = _background_color(img).astype(np.float32)
    dist = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    dist = (dist / (dist.max() + 1e-6) * 255.0).astype(np.uint8)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def composite(img: np.ndarray, mask: np.ndarray, bg_img: np.ndarray,
              boundary_blur: bool = True) -> np.ndarray:
    """Alpha-blend the masked leaf onto a resized background."""
    h, w = img.shape[:2]
    bg = cv2.resize(bg_img, (w, h), interpolation=cv2.INTER_LINEAR)
    # Mild random brightness so lighting varies (kept gentle to preserve realism).
    bright = np.random.uniform(0.75, 1.25)
    bg = np.clip(bg.astype(np.float32) * bright, 0, 255)

    m = mask.astype(np.float32) / 255.0
    if boundary_blur:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=2.0)  # soften the cut-out edge
    m = m[..., None]
    out = img.astype(np.float32) * m + bg * (1.0 - m)
    return np.clip(out, 0, 255).astype(np.uint8)


class BackgroundRandomize(A.ImageOnlyTransform):
    """Albumentations transform: with probability `prob`, replace the leaf's
    uniform backdrop with a random generic background.

    Placed at the FRONT of the train pipeline, so the existing geometric and
    photometric augmentations then run on the composite. If segmentation looks
    unreliable (mask covers <min_fg or >max_fg of the frame) the image is left
    unchanged, so a bad mask never poisons a training example.
    """

    def __init__(self, background_dir: str, prob: float = 0.5,
                 boundary_blur: bool = True, min_fg: float = 0.05,
                 max_fg: float = 0.97, p: float = None):
        # The transform's own application probability IS the fraction `prob`.
        super().__init__(p=prob if p is None else p)
        self.background_dir = background_dir
        self.boundary_blur = boundary_blur
        self.min_fg = min_fg
        self.max_fg = max_fg
        self.backgrounds = _load_backgrounds(background_dir)

    def apply(self, img, **params):
        mask = segment_leaf(img)
        fg_frac = float((mask > 0).mean())
        if fg_frac < self.min_fg or fg_frac > self.max_fg:
            return img  # segmentation unreliable — keep the original
        bg = self.backgrounds[np.random.randint(len(self.backgrounds))]
        return composite(img, mask, bg, boundary_blur=self.boundary_blur)

    def get_transform_init_args_names(self):
        return ("background_dir", "boundary_blur", "min_fg", "max_fg")
