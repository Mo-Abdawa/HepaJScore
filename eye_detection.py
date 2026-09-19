"""
HepaJScore — Eye detection module
=================================
Mandatory eye detection for jaundice assessment.

Detection strategy (in order):
  1. MediaPipe Face Mesh — precise sclera landmarks (best)
  2. OpenCV Haar cascade — fallback if MediaPipe unavailable
  3. Heuristic skin/eye structure check — last resort

If NO eye is detected, the caller MUST refuse to compute a J-score.

Returns:
  {
    "detected": bool,
    "method":   "mediapipe" | "opencv" | "none",
    "boxes":    [(x1, y1, x2, y2), ...],   # in original image coords
    "sclera_pixels": ndarray | None,        # boolean mask of sclera region
    "confidence":   float,                  # 0.0 .. 1.0
    "reason":       str,                    # human-readable
  }
"""
from __future__ import annotations

import os
import numpy as np
from PIL import Image, ImageOps

# Optional MediaPipe — best detector
try:
    import mediapipe as mp
    _HAS_MP = True
except Exception:
    mp = None
    _HAS_MP = False

# Optional OpenCV — fallback
try:
    import cv2
    _HAS_CV = True
except Exception:
    cv2 = None
    _HAS_CV = False


# MediaPipe Face Mesh — left & right eye landmark indices for the sclera region.
# Source: MediaPipe Face Mesh canonical topology (478 landmarks model).
# These are the OUTER eye contour (sclera-bordering) points.
LEFT_EYE_OUTLINE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_OUTLINE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# Iris landmarks (for excluding iris from the sclera mask)
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def _empty_result(reason: str) -> dict:
    return {
        "detected": False, "method": "none", "boxes": [],
        "sclera_pixels": None, "confidence": 0.0, "reason": reason,
    }


def _bbox_from_points(points: np.ndarray, pad_frac: float = 0.10,
                       img_w: int = 0, img_h: int = 0) -> tuple:
    """Bounding box around a set of points, padded by pad_frac."""
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    w = x_max - x_min
    h = y_max - y_min
    pad_x = w * pad_frac
    pad_y = h * pad_frac
    return (
        max(0, int(x_min - pad_x)),
        max(0, int(y_min - pad_y)),
        min(img_w, int(x_max + pad_x)),
        min(img_h, int(y_max + pad_y)),
    )


def detect_with_mediapipe(rgb: np.ndarray) -> dict:
    """Detect eyes using MediaPipe Face Mesh. Returns the result dict."""
    if not _HAS_MP:
        return _empty_result("mediapipe-unavailable")
    h, w = rgb.shape[:2]
    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,     # adds iris landmarks
            min_detection_confidence=0.5,
        ) as fm:
            res = fm.process(rgb)
            if not res.multi_face_landmarks:
                return _empty_result("mediapipe-no-face")

            lms = res.multi_face_landmarks[0].landmark
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms])

            left_outline = pts[LEFT_EYE_OUTLINE]
            right_outline = pts[RIGHT_EYE_OUTLINE]

            # Sanity: eye outline must have non-degenerate area
            def _area(p):
                # Shoelace formula
                x = p[:, 0]; y = p[:, 1]
                return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

            if _area(left_outline) < 30 and _area(right_outline) < 30:
                return _empty_result("mediapipe-degenerate-eyes")

            # Build the sclera mask = inside-eye-outline minus iris disk
            mask = np.zeros((h, w), dtype=bool)
            for outline, iris_idx in (
                (left_outline,  LEFT_IRIS),
                (right_outline, RIGHT_IRIS),
            ):
                # Rasterize the polygon by sampling
                from PIL import ImageDraw
                im = Image.new("L", (w, h), 0)
                ImageDraw.Draw(im).polygon(
                    [(float(x), float(y)) for x, y in outline], fill=1)
                eye_inside = np.asarray(im, dtype=bool)
                # Remove iris area (a disk around iris centroid)
                if len(pts) > max(iris_idx):
                    iris_pts = pts[iris_idx]
                    icx, icy = iris_pts.mean(axis=0)
                    # Iris radius ~ max distance from centroid to iris landmarks
                    ir = float(np.linalg.norm(iris_pts - [icx, icy], axis=1).max())
                    yy, xx = np.mgrid[0:h, 0:w]
                    iris_disk = (xx - icx) ** 2 + (yy - icy) ** 2 <= (ir * 1.05) ** 2
                    eye_inside &= ~iris_disk
                mask |= eye_inside

            boxes = [_bbox_from_points(left_outline,  img_w=w, img_h=h),
                     _bbox_from_points(right_outline, img_w=w, img_h=h)]
            return {
                "detected": True, "method": "mediapipe",
                "boxes": boxes, "sclera_pixels": mask,
                "confidence": 0.95, "reason": "ok",
            }
    except Exception as e:
        return _empty_result(f"mediapipe-error:{e!s}")


def detect_with_opencv(rgb: np.ndarray) -> dict:
    """Fallback: detect eyes via OpenCV Haar cascades."""
    if not _HAS_CV:
        return _empty_result("opencv-unavailable")
    h, w = rgb.shape[:2]
    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hc = getattr(cv2.data, "haarcascades", None)
        if not hc:
            return _empty_result("opencv-no-cascades")
        face_c = cv2.CascadeClassifier(os.path.join(hc, "haarcascade_frontalface_default.xml"))
        eye_c  = cv2.CascadeClassifier(os.path.join(hc, "haarcascade_eye.xml"))
        if face_c.empty() or eye_c.empty():
            return _empty_result("opencv-empty-cascade")

        eyes_xyxy = []

        # First try: face → eyes within face
        faces = face_c.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        for (fx, fy, fw, fh) in faces:
            roi = gray[fy:fy + fh, fx:fx + fw]
            eyes = eye_c.detectMultiScale(roi, 1.1, 8, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                eyes_xyxy.append((fx + ex, fy + ey, fx + ex + ew, fy + ey + eh))

        # If no face, try direct eye detection (close-up shots)
        if not eyes_xyxy:
            eyes = eye_c.detectMultiScale(gray, 1.1, 8, minSize=(30, 30))
            for (ex, ey, ew, eh) in eyes:
                eyes_xyxy.append((ex, ey, ex + ew, ey + eh))

        if not eyes_xyxy:
            return _empty_result("opencv-no-eyes")

        # Build an approximate sclera mask: oval inside each eye box minus central iris disk
        mask = np.zeros((h, w), dtype=bool)
        from PIL import ImageDraw
        for (x1, y1, x2, y2) in eyes_xyxy:
            box_w = x2 - x1; box_h = y2 - y1
            # Eye-shape ellipse
            im = Image.new("L", (w, h), 0)
            ImageDraw.Draw(im).ellipse([x1, y1, x2, y2], fill=1)
            eye_inside = np.asarray(im, dtype=bool)
            # Iris disk in the middle, ~40% of eye-box height
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            ir = min(box_w, box_h) * 0.30
            yy, xx = np.mgrid[0:h, 0:w]
            iris_disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= ir ** 2
            mask |= (eye_inside & ~iris_disk)

        return {
            "detected": True, "method": "opencv",
            "boxes": eyes_xyxy, "sclera_pixels": mask,
            "confidence": 0.6, "reason": "ok",
        }
    except Exception as e:
        return _empty_result(f"opencv-error:{e!s}")


def detect_eyes(image_path: str) -> dict:
    """
    Public entry point: try MediaPipe, fall back to OpenCV.
    Returns a result dict (see module docstring).
    """
    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            # Limit size for speed (detection works fine on smaller images)
            MAX_SIDE = 1024
            w, h = im.size
            if max(w, h) > MAX_SIDE:
                scale = MAX_SIDE / max(w, h)
                im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            rgb = np.asarray(im)
    except Exception as e:
        return _empty_result(f"image-read-error:{e!s}")

    res = detect_with_mediapipe(rgb)
    if res["detected"]:
        return res
    res = detect_with_opencv(rgb)
    if res["detected"]:
        return res
    return _empty_result("no-eye-detected")


def sclera_yellow_fraction(image_path: str, det_result: dict) -> float:
    """
    Compute the fraction of yellow pixels INSIDE the detected sclera region.
    Returns 0.0 if detection failed or mask is empty.
    """
    if not det_result.get("detected") or det_result.get("sclera_pixels") is None:
        return 0.0
    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            MAX_SIDE = 1024
            w, h = im.size
            if max(w, h) > MAX_SIDE:
                scale = MAX_SIDE / max(w, h)
                im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            arr = np.asarray(im).astype("float32") / 255.0

        mask = det_result["sclera_pixels"]
        if mask.shape != arr.shape[:2]:
            return 0.0

        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        cmax = arr.max(axis=-1); cmin = arr.min(axis=-1); delta = cmax - cmin + 1e-6
        h_deg = np.where(cmax == r, (60 * ((g - b) / delta) % 360),
                 np.where(cmax == g, 60 * (((b - r) / delta) + 2),
                          60 * (((r - g) / delta) + 3)))
        s = np.where(cmax == 0, 0, delta / cmax)
        v = cmax
        yellow_mask = (h_deg >= 35.0) & (h_deg <= 65.0) & (s >= 0.15) & (v >= 0.40) & mask
        total_sclera = int(mask.sum())
        if total_sclera < 20:
            return 0.0
        return float(yellow_mask.sum()) / float(total_sclera)
    except Exception:
        return 0.0


# Debug helper
def detector_available() -> dict:
    return {"mediapipe": _HAS_MP, "opencv": _HAS_CV}
