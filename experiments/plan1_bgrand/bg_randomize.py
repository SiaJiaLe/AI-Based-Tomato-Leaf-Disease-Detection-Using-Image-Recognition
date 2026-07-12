"""Background-randomization transform (train-only).

The one new variable Plan 1 adds on top of the EfficientNetB0 Stack-ON recipe.
For a fraction `p` of training images it (a) segments the leaf, (b) composites
it onto a random generic background, so the network can no longer lean on the
backdrop as a shortcut. The remaining `1 - p` pass through untouched, so the
model still sees the originals (no image copying needed).

Two segmentation routes:
  * "hsv_threshold" — classical foreground-vs-uniform-background (corner-colour
    distance + Otsu + largest component). Fast, no downloads, but degrades when
    the original backdrop is itself textured/shadowed.
  * "pretrained" — rembg (U^2-Net). Robust on textured backgrounds; needs the
    `rembg` package and a one-time model download.

Both routes erode the mask inward by `erode_px` before compositing, so the
soft edge sits INSIDE the true leaf boundary and no original background bleeds
through as a halo (the artifact that hurt the first classical run). Segmentation
keys on the leaf, not on green, so brown/yellow lesions stay inside the mask.
"""
import hashlib
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


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground blob."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _background_color(img: np.ndarray) -> np.ndarray:
    """Median colour of the four image corners — the assumed backdrop."""
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
    """Classical mask via background-colour distance + Otsu (0/255)."""
    bg = _background_color(img).astype(np.float32)
    dist = np.linalg.norm(img.astype(np.float32) - bg, axis=2)
    dist = (dist / (dist.max() + 1e-6) * 255.0).astype(np.uint8)
    _, mask = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return _largest_component(mask)


def make_rembg_session(model: str = "u2net"):
    """Build a rembg segmentation session (triggers a one-time model download)."""
    from rembg import new_session
    return new_session(model)


def segment_leaf_rembg(img: np.ndarray, session) -> np.ndarray:
    """Pretrained (U^2-Net) mask via rembg — robust on textured backgrounds."""
    from rembg import remove
    out = remove(img, only_mask=True, session=session)
    if out.ndim == 3:
        out = out[..., 0]
    _, mask = cv2.threshold(out.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return _largest_component(mask)


def composite(img: np.ndarray, mask: np.ndarray, bg_img: np.ndarray,
              boundary_blur: bool = True, erode_px: int = 3) -> np.ndarray:
    """Alpha-blend the masked leaf onto a resized background.

    The mask is eroded inward by `erode_px` FIRST, so the soft transition sits
    strictly inside the leaf and no original backdrop leaks in as a halo.
    """
    h, w = img.shape[:2]
    bg = cv2.resize(bg_img, (w, h), interpolation=cv2.INTER_LINEAR)
    bright = np.random.uniform(0.75, 1.25)  # mild lighting variation
    bg = np.clip(bg.astype(np.float32) * bright, 0, 255)

    m = mask.astype(np.float32) / 255.0
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        m = cv2.erode(m, k)
    if boundary_blur:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=2.0)
    m = m[..., None]
    out = img.astype(np.float32) * m + bg * (1.0 - m)
    return np.clip(out, 0, 255).astype(np.uint8)


class BackgroundRandomize(A.ImageOnlyTransform):
    """With probability `prob`, replace the leaf's backdrop with a random
    generic background. Placed at the FRONT of the train pipeline. If
    segmentation looks unreliable (mask <min_fg or >max_fg of the frame) the
    image is left unchanged so a bad mask never poisons a training example.
    """

    def __init__(self, background_dir: str, prob: float = 0.5,
                 boundary_blur: bool = True, segmentation: str = "hsv_threshold",
                 erode_px: int = 3, rembg_model: str = "u2net",
                 mask_cache_dir: str = None,
                 min_fg: float = 0.05, max_fg: float = 0.97, p: float = None):
        super().__init__(p=prob if p is None else p)
        self.background_dir = background_dir
        self.boundary_blur = boundary_blur
        self.segmentation = segmentation
        self.erode_px = erode_px
        self.rembg_model = rembg_model
        self.mask_cache_dir = mask_cache_dir
        self.min_fg = min_fg
        self.max_fg = max_fg
        self._session = None  # rembg session; built lazily, per worker process
        self.backgrounds = _load_backgrounds(background_dir)
        if mask_cache_dir:
            os.makedirs(mask_cache_dir, exist_ok=True)

    def _get_session(self):
        if self._session is None:
            self._session = make_rembg_session(self.rembg_model)
        return self._session

    def _segment(self, img):
        if self.segmentation == "pretrained":
            return segment_leaf_rembg(img, self._get_session())
        return segment_leaf(img)

    def _mask(self, img):
        """Segment `img`, caching the result to disk keyed by image content.

        The mask for a given source image is identical every epoch, so we only
        pay the (expensive, CPU) U^2-Net pass once per image across the whole
        run instead of once per image per epoch. The random background is still
        chosen fresh in composite(), so per-epoch variety is preserved.
        """
        if not self.mask_cache_dir:
            return self._segment(img)
        key = hashlib.md5(np.ascontiguousarray(img).tobytes()).hexdigest()
        path = os.path.join(self.mask_cache_dir, f"{self.segmentation}_{key}.png")
        if os.path.exists(path):
            cached = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if cached is not None and cached.shape == img.shape[:2]:
                return cached
        mask = self._segment(img)
        # Encode to PNG in memory (imwrite would key format off the temp file's
        # extension), then atomic-rename so parallel workers never read a partial.
        ok, buf = cv2.imencode(".png", mask)
        if ok:
            tmp = f"{path}.tmp{os.getpid()}"
            with open(tmp, "wb") as fh:
                fh.write(buf.tobytes())
            os.replace(tmp, path)
        return mask

    def apply(self, img, **params):
        mask = self._mask(img)
        fg_frac = float((mask > 0).mean())
        if fg_frac < self.min_fg or fg_frac > self.max_fg:
            return img  # segmentation unreliable — keep the original
        bg = self.backgrounds[np.random.randint(len(self.backgrounds))]
        return composite(img, mask, bg, boundary_blur=self.boundary_blur, erode_px=self.erode_px)

    def __getstate__(self):
        # onnxruntime sessions are not picklable; drop it so DataLoader workers
        # can receive the transform, then rebuild lazily inside each worker.
        state = self.__dict__.copy()
        state["_session"] = None
        return state

    def get_transform_init_args_names(self):
        return ("background_dir", "boundary_blur", "segmentation", "erode_px",
                "rembg_model", "mask_cache_dir", "min_fg", "max_fg")
