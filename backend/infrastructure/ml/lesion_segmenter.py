"""Lightweight colour/texture-based lesion segmenter for severity estimation.

This is a classical CV heuristic — it does NOT use a trained model.
It estimates the proportion of leaf area showing disease symptoms by
comparing green (healthy leaf) pixels against brown/yellow (lesion) pixels
in HSV colour space.
"""
import cv2
import numpy as np
from PIL import Image


class LesionSegmenter:
    def estimate_affected_ratio(self, image: Image.Image) -> float:
        """Returns a 0.0–1.0 ratio of estimated lesion area to total leaf area."""
        img_hsv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2HSV)

        # Leaf area: green and yellow-green hues
        leaf_mask = cv2.inRange(img_hsv, (20, 30, 30), (90, 255, 255))

        # Lesion area: brown/yellow/dark patches within the leaf
        lesion_mask_brown = cv2.inRange(img_hsv, (0, 40, 20), (30, 255, 200))
        lesion_mask_yellow = cv2.inRange(img_hsv, (15, 50, 80), (35, 255, 255))
        lesion_combined = cv2.bitwise_or(lesion_mask_brown, lesion_mask_yellow)

        # Only count lesions inside the detected leaf area
        lesion_on_leaf = cv2.bitwise_and(lesion_combined, leaf_mask)

        leaf_pixels = int(np.count_nonzero(leaf_mask))
        lesion_pixels = int(np.count_nonzero(lesion_on_leaf))

        if leaf_pixels == 0:
            return 0.0
        return min(round(lesion_pixels / leaf_pixels, 4), 1.0)
