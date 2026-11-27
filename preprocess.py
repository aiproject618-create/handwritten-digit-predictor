"""
preprocess.py
-------------
Turns a user's drawing / photo of a digit into the 8x8 format the model was
trained on (the scikit-learn `digits` dataset: 8x8 pixels, values 0-16,
white digit on black background, digit centred).

Steps:
  1. Convert to grayscale
  2. Make the digit light on a dark background (auto-detect or forced)
  3. Remove background noise (Otsu threshold)
  4. Crop tightly around the digit
  5. Scale the digit to the full height of the box (keeps the aspect ratio)
     and centre it - exactly how the training images are framed
  6. Shrink to 8x8 and scale pixel values to 0-16
"""

import cv2
import numpy as np
from PIL import Image


def image_to_gray(img):
    """PIL image -> 2D float array (0-255). Transparent pixels become white."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return np.array(img.convert("L"), dtype=np.float32)


def preprocess_gray(gray, invert=None):
    """
    gray   : 2D array, values 0-255
    invert : True  -> input is a dark digit on light paper (flip it)
             False -> input is already a light digit on dark background
             None  -> decide automatically from the border pixels

    Returns (features, small_image) or None if no digit was found.
        features    : array of shape (1, 64), values 0-16
        small_image : array of shape (8, 8), values 0-16
    """
    g = gray.astype(np.float32)

    if invert is None:
        border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
        invert = np.median(border) > 127
    if invert:
        g = 255.0 - g

    g = g - g.min()
    if g.max() < 20:  # blank canvas
        return None
    g = g / g.max()

    # Otsu threshold separates ink from background / paper noise
    g8 = (g * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(g8, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = np.where(mask > 0, g, 0.0)

    ys, xs = np.nonzero(cleaned)
    if len(ys) < 10:
        return None

    # Tight crop around the digit
    digit = cleaned[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # In the training dataset every digit is scaled to fill the FULL HEIGHT
    # of the 8x8 box (width follows the aspect ratio, digit is centred).
    # We do the same here on a 64x64 canvas (= 8 x 8 blocks of 8 pixels).
    size = 64
    h, w = digit.shape
    scale = size / h
    new_h, new_w = size, max(1, int(round(w * scale)))
    if new_w > size:  # very wide shape: fit the width instead
        scale = size / w
        new_h, new_w = max(1, int(round(h * scale))), size
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(digit, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((size, size), dtype=np.float32)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    # Shrink to 8x8, scale to the 0-16 range used by the dataset
    small = cv2.resize(canvas, (8, 8), interpolation=cv2.INTER_AREA)
    if small.max() > 0:
        small = small / small.max() * 16.0
    return small.reshape(1, 64), small
