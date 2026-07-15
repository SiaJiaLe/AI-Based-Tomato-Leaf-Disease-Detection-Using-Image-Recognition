"""Generate generic, domain-neutral background textures for Plan 1.

Writes ~60 procedural textures (soil, foliage, grass, wood, sky, gravel) into
data/backgrounds_generic_synthetic/. These are the backdrops composited behind segmented
leaves during training. They are deliberately generic and are NOT PlantVillage
images and NOT the real-world test images — so the model learns background
*invariance* without leaking the test domain.

Swap these for real CC0 photos later by dropping them into the same folder
(the run reads whatever image files are present).

    python -m experiments.plan1_bgrand.make_synthetic_backgrounds
    python -m experiments.plan1_bgrand.make_synthetic_backgrounds --count 90 --size 512
"""
import argparse
import os

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DIR = os.path.join(REPO_ROOT, "data", "backgrounds_generic_synthetic")

# (name, base RGB colour, colour jitter) — coarse natural-scene palettes.
PALETTES = [
    ("soil",    (110, 78, 52),   35),
    ("foliage", (60, 110, 45),   40),
    ("grass",   (95, 140, 60),   35),
    ("wood",    (140, 100, 62),  30),
    ("sky",     (150, 180, 210), 30),
    ("gravel",  (130, 130, 125), 25),
]


def _value_noise(h: int, w: int, cell: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth [0,1] value noise: upscale a coarse random grid + blur."""
    gh, gw = max(2, h // cell), max(2, w // cell)
    low = rng.random((gh, gw), dtype=np.float32)
    up = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), sigmaX=cell / 3.0)
    return np.clip(up, 0.0, 1.0)


def _texture(name: str, base, jitter: int, size: int, variant: int,
             rng: np.random.Generator) -> np.ndarray:
    base = np.array(base, dtype=np.float32)
    # Multi-scale mottling.
    n = (_value_noise(size, size, 64, rng) * 0.6
         + _value_noise(size, size, 16, rng) * 0.4)
    img = base[None, None, :] + (n[..., None] - 0.5) * 2.0 * jitter

    # A gentle directional gradient so no two tiles look flat-identical.
    gg = np.linspace(-1, 1, size, dtype=np.float32)
    grad = (gg[:, None] if variant % 2 == 0 else gg[None, :])
    img += grad[..., None] * (jitter * 0.4)

    if name in ("grass", "wood"):
        # Add faint streaks (vertical for grass, horizontal for wood).
        streak = _value_noise(size, size, 3, rng)
        streak = cv2.GaussianBlur(streak, (1, 31) if name == "grass" else (31, 1), 0)
        img += (streak[..., None] - 0.5) * jitter * 1.2

    img += rng.normal(0, 4, img.shape).astype(np.float32)  # fine grain
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_DIR, help="Output directory.")
    parser.add_argument("--count", type=int, default=60, help="Total textures to generate.")
    parser.add_argument("--size", type=int, default=512, help="Texture size (px).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    per = max(1, args.count // len(PALETTES))

    written = 0
    for name, base, jitter in PALETTES:
        for v in range(per):
            img = _texture(name, base, jitter, args.size, v, rng)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            path = os.path.join(args.out, f"{name}_{v:02d}.png")
            cv2.imwrite(path, bgr)
            written += 1
    print(f"Wrote {written} synthetic background textures to {args.out}", flush=True)
    print("Swap in real CC0 textures later by dropping them into the same folder.", flush=True)


if __name__ == "__main__":
    main()
