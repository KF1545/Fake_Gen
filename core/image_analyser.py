"""
image_analyser.py
Analyses an uploaded image and decides which pipeline to use.
Face detected  → PCA preprocessing + GAN
No face        → GAN directly
"""

import cv2
import numpy as np
from PIL import Image


# OpenCV ships with a pre-trained face detector — no training needed
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def pil_to_cv2(pil_image):
    """Convert PIL Image → OpenCV numpy array (BGR)."""
    rgb   = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def analyse_image(pil_image):
    """
    Analyse an uploaded PIL image.

    Returns a dict:
        {
            "has_face"  : bool,
            "face_count": int,
            "face_boxes": list of (x, y, w, h),
            "image_type": "face" | "scenery" | "object",
            "size"      : (width, height),
        }
    """
    cv_img = pil_to_cv2(pil_image)
    gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    face_boxes = list(faces) if len(faces) > 0 else []
    has_face   = len(face_boxes) > 0

    # Simple image type classification
    if has_face:
        image_type = "face"
    else:
        # Use colour variance as a rough scenery vs object heuristic
        hsv       = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        sat_mean  = hsv[:, :, 1].mean()
        image_type = "scenery" if sat_mean > 50 else "object"

    result = {
        "has_face":   has_face,
        "face_count": len(face_boxes),
        "face_boxes": face_boxes,
        "image_type": image_type,
        "size":       pil_image.size,
    }

    print(f"Image analysis → type: {image_type}, "
          f"faces: {len(face_boxes)}, size: {pil_image.size}")
    return result


def crop_face(pil_image, face_box, padding=0.2):
    """
    Crop the largest face region from the image,
    with a small padding around it.
    Returns a PIL Image of just the face.
    """
    x, y, w, h = face_box
    W, H       = pil_image.size

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(W, x + w + pad_x)
    y2 = min(H, y + h + pad_y)

    return pil_image.crop((x1, y1, x2, y2))


def draw_face_boxes(pil_image, face_boxes):
    """
    Draw rectangles around detected faces.
    Useful for debugging and for your presentation/report.
    Returns a PIL Image with boxes drawn.
    """
    cv_img = pil_to_cv2(pil_image)
    for (x, y, w, h) in face_boxes:
        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ──────────────────────────────────────────────────────────────
# QUICK TEST
# python core/image_analyser.py
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Create a dummy solid-colour image for testing
    dummy = Image.fromarray(
        np.zeros((256, 256, 3), dtype=np.uint8), "RGB"
    )
    result = analyse_image(dummy)
    print("Test result:", result)
    print("✅ Image analyser OK")
