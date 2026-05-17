"""
backend/app.py — Flask API
Runs on your laptop during the demo.
Receives image + instruction from the frontend,
applies OpenCV transformations + PCA + GAN, returns the result.
"""

import os
import sys
import io
import base64

import cv2
import torch
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms

# ── Import our modules ────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from models.generator        import Generator
from models.pca_module       import FacePCA
from core.image_analyser     import analyse_image
from core.instruction_parser import parse_instruction

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
GENERATOR_WEIGHTS = os.path.join(ROOT, "checkpoints", "generator_final.pth")
PCA_MODEL_PATH    = os.path.join(ROOT, "checkpoints", "pca_model.pkl")
IMAGE_SIZE        = 256
DEVICE            = torch.device("cpu")

# ══════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)

to_tensor = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def tensor_to_pil(tensor):
    img = tensor.squeeze(0).detach().cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(img)

def pil_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_pil(b64_string):
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

# ══════════════════════════════════════════════════════════════
# OPENCV TRANSFORMATIONS
# This is the core transformation engine.
# Works on any image type — faces, scenery, objects, animals.
# No GPU needed. No training needed. Works right now.
# ══════════════════════════════════════════════════════════════
def apply_opencv_transformation(pil_image, instruction):
    """
    Apply real image transformations using OpenCV.
    Returns a modified PIL Image.
    """
    img = np.array(pil_image.convert("RGB"))
    text = instruction.lower()
    applied = []

    # ── COLOUR TINTS ─────────────────────────────────────────
    if any(w in text for w in ["blue", "cold", "cool"]):
        img[:, :, 0] = np.clip(img[:, :, 0] * 1.4, 0, 255)  # boost blue
        img[:, :, 2] = np.clip(img[:, :, 2] * 0.8, 0, 255)  # reduce red
        applied.append("blue tint")

    elif any(w in text for w in ["warm", "orange", "golden", "sunset"]):
        img[:, :, 2] = np.clip(img[:, :, 2] * 1.4, 0, 255)  # boost red
        img[:, :, 1] = np.clip(img[:, :, 1] * 1.1, 0, 255)  # boost green
        img[:, :, 0] = np.clip(img[:, :, 0] * 0.8, 0, 255)  # reduce blue
        applied.append("warm tint")

    elif any(w in text for w in ["green", "nature", "forest"]):
        img[:, :, 1] = np.clip(img[:, :, 1] * 1.4, 0, 255)  # boost green
        applied.append("green tint")

    elif any(w in text for w in ["red", "dramatic"]):
        img[:, :, 2] = np.clip(img[:, :, 2] * 1.5, 0, 255)  # boost red
        applied.append("red tint")

    elif any(w in text for w in ["purple", "violet"]):
        img[:, :, 0] = np.clip(img[:, :, 0] * 1.3, 0, 255)  # boost blue
        img[:, :, 2] = np.clip(img[:, :, 2] * 1.3, 0, 255)  # boost red
        applied.append("purple tint")

    # ── BRIGHTNESS & CONTRAST ─────────────────────────────────
    if any(w in text for w in ["bright", "brighter", "lighten", "sunny"]):
        img = np.clip(img * 1.4, 0, 255).astype(np.uint8)
        applied.append("brighten")

    elif any(w in text for w in ["dark", "darker", "dim", "night", "shadow"]):
        img = np.clip(img * 0.55, 0, 255).astype(np.uint8)
        applied.append("darken")

    if any(w in text for w in ["contrast", "vivid", "vibrant", "pop"]):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        applied.append("enhance contrast")

    # ── BLACK & WHITE / GREYSCALE ─────────────────────────────
    if any(w in text for w in ["black and white", "grayscale",
                                "grey", "gray", "monochrome", "bw"]):
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img  = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
        applied.append("black and white")

    # ── VINTAGE / SEPIA ───────────────────────────────────────
    elif any(w in text for w in ["vintage", "sepia", "retro",
                                  "old photo", "aged"]):
        grey  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sepia = np.zeros_like(img)
        sepia[:, :, 0] = np.clip(grey * 1.08, 0, 255)   # R
        sepia[:, :, 1] = np.clip(grey * 0.86, 0, 255)   # G
        sepia[:, :, 2] = np.clip(grey * 0.68, 0, 255)   # B
        img = sepia
        applied.append("vintage/sepia")

    # ── SKETCH / PENCIL DRAWING ───────────────────────────────
    if any(w in text for w in ["sketch", "pencil", "drawing",
                                "hand drawn", "line art"]):
        grey   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        inv    = cv2.bitwise_not(grey)
        blur   = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(grey, cv2.bitwise_not(blur), scale=256)
        img    = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        applied.append("sketch")

    # ── CARTOON ───────────────────────────────────────────────
    elif any(w in text for w in ["cartoon", "anime", "comic",
                                  "illustrated", "animated"]):
        grey  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur  = cv2.medianBlur(grey, 5)
        edges = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(img, 9, 300, 300)
        img   = cv2.bitwise_and(color, color, mask=edges)
        applied.append("cartoon")

    # ── OIL PAINTING ──────────────────────────────────────────
    elif any(w in text for w in ["painting", "oil painting",
                                  "watercolor", "artistic", "impressionist"]):
        img = cv2.xphoto.oilPainting(img, 7, 1) if hasattr(
            cv2, 'xphoto') else cv2.bilateralFilter(img, 15, 80, 80)
        applied.append("painting effect")

    # ── BLUR ──────────────────────────────────────────────────
    if any(w in text for w in ["blur", "soft", "dreamy", "smooth"]):
        img = cv2.GaussianBlur(img, (21, 21), 0)
        applied.append("blur")

    # ── SHARPEN ───────────────────────────────────────────────
    elif any(w in text for w in ["sharp", "sharpen", "crisp", "clear"]):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        applied.append("sharpen")

    # ── EMBOSS ────────────────────────────────────────────────
    if any(w in text for w in ["emboss", "relief", "3d effect"]):
        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]])
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        emboss = cv2.filter2D(grey, -1, kernel) + 128
        img = cv2.cvtColor(emboss.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        applied.append("emboss")

    # ── NEON / GLOW ───────────────────────────────────────────
    if any(w in text for w in ["neon", "glow", "electric", "cyberpunk"]):
        blur  = cv2.GaussianBlur(img, (0, 0), 3)
        edges = cv2.Canny(blur, 100, 200)
        neon  = np.zeros_like(img)
        neon[:, :, 0] = edges   # R channel — red neon
        neon[:, :, 1] = edges   # G channel
        img = cv2.addWeighted(img, 0.7, neon, 0.9, 0)
        applied.append("neon glow")

    # ── MIRROR / FLIP ─────────────────────────────────────────
    if any(w in text for w in ["mirror", "flip", "reverse"]):
        img = cv2.flip(img, 1)
        applied.append("mirror")

    # ── ROTATE ────────────────────────────────────────────────
    if "rotate" in text or "turn" in text:
        angle = 90
        if "180" in text:
            angle = 180
        elif "270" in text or "left" in text:
            angle = 270
        h, w = img.shape[:2]
        M   = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        img = cv2.warpAffine(img, M, (w, h))
        applied.append(f"rotate {angle}°")

    # ── PIXELATE ──────────────────────────────────────────────
    if any(w in text for w in ["pixelate", "pixel", "mosaic", "8bit"]):
        h, w   = img.shape[:2]
        small  = cv2.resize(img, (w // 20, h // 20),
                            interpolation=cv2.INTER_LINEAR)
        img    = cv2.resize(small, (w, h),
                            interpolation=cv2.INTER_NEAREST)
        applied.append("pixelate")

    # ── INVERT / NEGATIVE ─────────────────────────────────────
    if any(w in text for w in ["invert", "negative", "negate"]):
        img = cv2.bitwise_not(img)
        applied.append("invert")

    # ── SNOW EFFECT ───────────────────────────────────────────
    if any(w in text for w in ["snow", "snowy", "winter"]):
        snow_layer = np.random.randint(
            0, 256, img.shape, dtype=np.uint8)
        snow_layer = cv2.threshold(
            snow_layer, 200, 255, cv2.THRESH_BINARY)[1]
        img = cv2.add(img, snow_layer)
        applied.append("snow")

    # ── RAIN EFFECT ───────────────────────────────────────────
    if any(w in text for w in ["rain", "rainy", "storm"]):
        rain_layer = np.zeros_like(img)
        for _ in range(1000):
            x1 = np.random.randint(0, img.shape[1])
            y1 = np.random.randint(0, img.shape[0])
            x2 = x1 + np.random.randint(-2, 2)
            y2 = y1 + np.random.randint(10, 20)
            cv2.line(rain_layer, (x1, y1),
                     (min(x2, img.shape[1]-1),
                      min(y2, img.shape[0]-1)),
                     (200, 200, 200), 1)
        img = cv2.add(img, rain_layer)
        img = np.clip(img * 0.75, 0, 255).astype(np.uint8)
        applied.append("rain")

    # ── SUNSET / GOLDEN HOUR ──────────────────────────────────
    if any(w in text for w in ["sunset", "golden hour", "dusk"]):
        overlay        = np.zeros_like(img)
        overlay[:, :, 2] = 120   # red
        overlay[:, :, 1] = 60    # green
        img = cv2.addWeighted(img, 0.75, overlay, 0.35, 0)
        applied.append("sunset")

    # ── FOG / MIST ────────────────────────────────────────────
    if any(w in text for w in ["fog", "mist", "foggy", "misty", "haze"]):
        fog  = np.full_like(img, 200)
        img  = cv2.addWeighted(img, 0.55, fog, 0.45, 0)
        applied.append("fog")

    # ── VIGNETTE ──────────────────────────────────────────────
    if any(w in text for w in ["vignette", "fade edges", "cinematic"]):
        rows, cols = img.shape[:2]
        kernel_x   = cv2.getGaussianKernel(cols, cols * 0.5)
        kernel_y   = cv2.getGaussianKernel(rows, rows * 0.5)
        kernel     = kernel_y * kernel_x.T
        mask       = kernel / kernel.max()
        for i in range(3):
            img[:, :, i] = (img[:, :, i] * mask).astype(np.uint8)
        applied.append("vignette")

    if applied:
        print(f"OpenCV transformations applied: {applied}")
    else:
        print("No OpenCV transformation matched — returning original")

    return Image.fromarray(img.astype(np.uint8), "RGB")


# ══════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════
generator = None
pca_model  = None

def load_models():
    global generator, pca_model

    if os.path.exists(GENERATOR_WEIGHTS):
        try:
            generator = Generator()
            generator.load_state_dict(
                torch.load(GENERATOR_WEIGHTS, map_location=DEVICE)
            )
            generator.eval()
            print(f"✅ Generator loaded from {GENERATOR_WEIGHTS}")
        except Exception as e:
            print(f"⚠️  Generator weights incompatible: {e}")
            print("    Running with OpenCV transformations only")
            generator = None
    else:
        print("⚠️  No generator weights found")
        print("    Running with OpenCV transformations only")

    if os.path.exists(PCA_MODEL_PATH):
        pca_model = FacePCA.load(PCA_MODEL_PATH)
        print(f"✅ PCA model loaded")
    else:
        print("⚠️  No PCA model found — PCA step will be skipped")


# ══════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════
def run_pipeline(pil_image, instruction):
    """
    Full pipeline:
    1. Analyse image
    2. Parse instruction
    3. Apply OpenCV transformation  ← always runs, handles most edits
    4. Apply PCA if face detected    ← runs if face + PCA model loaded
    5. Apply GAN if weights loaded   ← runs if generator weights exist
    6. Return result
    """
    # Step 1 — Analyse
    analysis   = analyse_image(pil_image)
    image_type = analysis["image_type"]

    # Step 2 — Parse instruction
    parsed = parse_instruction(instruction, image_type)

    # Step 3 — OpenCV transformation (always runs)
    print("Applying OpenCV transformations...")
    result_image = apply_opencv_transformation(pil_image, instruction)

    # Step 4 — PCA modification (face images only)
    if (parsed["use_pca"] and
            pca_model is not None and
            pca_model.is_fitted):
        try:
            print("Applying PCA face modification...")
            components = pca_model.compress(pil_image)
            modified   = pca_model.modify_feature(
                components, instruction, strength=parsed["strength"]
            )
            pca_result   = pca_model.components_to_pil(modified)
            pca_rgb      = pca_result.resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
            ).convert("RGB")
            # Blend PCA result with OpenCV result
            result_image = Image.blend(
                result_image.resize((IMAGE_SIZE, IMAGE_SIZE)),
                pca_rgb, alpha=0.4
            )
            print("PCA modification applied ✓")
        except Exception as e:
            print(f"PCA step skipped: {e}")

    # Step 5 — GAN enhancement (if weights loaded)
    if generator is not None:
        try:
            print("Running GAN Generator...")
            tensor = to_tensor(result_image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = generator(tensor)
            result_image = tensor_to_pil(output)
            print("GAN generation complete ✓")
        except Exception as e:
            print(f"GAN step skipped: {e}")

    return result_image, analysis, parsed


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":           "running",
        "generator_loaded": generator is not None,
        "pca_loaded":       pca_model is not None,
        "opencv":           True,
    })


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body received"}), 400
    if "image" not in data:
        return jsonify({"error": "Missing image"}), 400
    if "instruction" not in data:
        return jsonify({"error": "Missing instruction"}), 400

    instruction = data["instruction"].strip()
    if not instruction:
        return jsonify({"error": "Instruction cannot be empty"}), 400

    try:
        pil_image = base64_to_pil(data["image"])
        print(f"\nRequest | Size: {pil_image.size} | "
              f"Instruction: '{instruction}'")

        result_image, analysis, parsed = run_pipeline(
            pil_image, instruction
        )

        return jsonify({
            "result": pil_to_base64(result_image),
            "info": {
                "image_type":          analysis["image_type"],
                "faces_detected":      analysis["face_count"],
                "transformation_type": parsed["transformation_type"],
                "label":               parsed["label"],
                "pca_used":            parsed["use_pca"],
                "opencv_used":         True,
                "gan_used":            generator is not None,
            },
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/transformations", methods=["GET"])
def list_transformations():
    """Return all supported transformations."""
    supported = [
        "make it vintage / sepia",
        "black and white / grayscale",
        "make it a sketch / pencil drawing",
        "cartoon / anime effect",
        "make it bright / dark",
        "add snow / rain",
        "sunset / golden hour",
        "fog / mist effect",
        "neon / glow effect",
        "blue / warm / green tint",
        "enhance contrast / vivid",
        "blur / sharpen",
        "pixelate / mosaic",
        "mirror / flip",
        "rotate",
        "vignette / cinematic",
        "invert / negative",
        "emboss / relief",
    ]
    return jsonify({"supported": supported, "count": len(supported)})


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  DeepFake App — Backend Server")
    print("=" * 55)
    load_models()
    print("\nServer starting on http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
