"""
backend/app.py — Flask API
Runs on your laptop during the demo.
Receives image + instruction from the frontend,
runs through PCA and/or GAN, returns the result.
"""

import os
import sys
import io
import base64
import pickle

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
DEVICE            = torch.device("cpu")   # CPU only — no GPU needed for demo

# ══════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)   # Allow requests from the Streamlit frontend

# ── Image transforms ──────────────────────────────────────────
to_tensor = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

def tensor_to_pil(tensor):
    """Convert a Generator output tensor back to a PIL Image."""
    # Denormalise from [-1,1] → [0,1] → [0,255]
    img = tensor.squeeze(0).detach().cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img = transforms.ToPILImage()(img)
    return img

def pil_to_base64(pil_image):
    """Encode a PIL Image to a base64 string for JSON transport."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_pil(b64_string):
    """Decode a base64 string back to a PIL Image."""
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

# ══════════════════════════════════════════════════════════════
# MODEL LOADING  (done once at startup)
# ══════════════════════════════════════════════════════════════
generator = None
pca_model  = None

def load_models():
    global generator, pca_model

    # Load GAN Generator
    if os.path.exists(GENERATOR_WEIGHTS):
        generator = Generator()
        generator.load_state_dict(
            torch.load(GENERATOR_WEIGHTS, map_location=DEVICE)
        )
        generator.eval()
        print(f"✅ Generator loaded from {GENERATOR_WEIGHTS}")
    else:
        print(f"⚠️  Generator weights not found at {GENERATOR_WEIGHTS}")
        print("    Train the model on Colab first, then download generator_final.pth")

    # Load PCA model
    if os.path.exists(PCA_MODEL_PATH):
        pca_model = FacePCA.load(PCA_MODEL_PATH)
        print(f"✅ PCA model loaded from {PCA_MODEL_PATH}")
    else:
        print(f"⚠️  PCA model not found at {PCA_MODEL_PATH}")
        print("    PCA features will be skipped until model is trained")


# ══════════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════════
def run_pipeline(pil_image, instruction):
    """
    Full pipeline:
      1. Analyse image (face? scenery? object?)
      2. Parse instruction
      3. Apply PCA if face + applicable
      4. Run GAN Generator
      5. Return result PIL image
    """
    # Step 1 — Analyse
    analysis = analyse_image(pil_image)
    image_type = analysis["image_type"]

    # Step 2 — Parse instruction
    parsed = parse_instruction(instruction, image_type)

    # Step 3 — PCA modification (face images only)
    processed_image = pil_image.copy()

    if parsed["use_pca"] and pca_model is not None and pca_model.is_fitted:
        print("Applying PCA feature modification …")
        try:
            # Compress face image to components
            components = pca_model.compress(pil_image)

            # Modify components based on instruction
            modified_components = pca_model.modify_feature(
                components,
                instruction,
                strength=parsed["strength"],
            )

            # Reconstruct modified face (64×64 grayscale)
            reconstructed = pca_model.components_to_pil(modified_components)

            # Upscale and convert back to RGB for GAN input
            processed_image = reconstructed.resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS
            ).convert("RGB")

            print("PCA modification applied ✓")

        except Exception as e:
            print(f"PCA step failed ({e}), proceeding with original image")
            processed_image = pil_image

    # Step 4 — GAN generation
    if generator is None:
        print("Generator not loaded — returning PCA result only")
        return processed_image, analysis, parsed

    print("Running GAN Generator …")
    input_tensor = to_tensor(processed_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output_tensor = generator(input_tensor)

    result_image = tensor_to_pil(output_tensor)
    print("GAN generation complete ✓")

    return result_image, analysis, parsed


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """Health check — call this to confirm the server is running."""
    return jsonify({
        "status":            "running",
        "generator_loaded":  generator is not None,
        "pca_loaded":        pca_model is not None and pca_model.is_fitted,
    })


@app.route("/generate", methods=["POST"])
def generate():
    """
    Main endpoint.
    Expects JSON:  { "image": "<base64>", "instruction": "<text>" }
    Returns JSON:  { "result": "<base64>", "info": {...} }
    """
    data = request.get_json()

    # Validate input
    if not data:
        return jsonify({"error": "No JSON body received"}), 400
    if "image" not in data:
        return jsonify({"error": "Missing 'image' field"}), 400
    if "instruction" not in data:
        return jsonify({"error": "Missing 'instruction' field"}), 400

    instruction = data["instruction"].strip()
    if not instruction:
        return jsonify({"error": "Instruction cannot be empty"}), 400

    try:
        # Decode image
        pil_image = base64_to_pil(data["image"])
        print(f"\nRequest received | "
              f"Image: {pil_image.size} | Instruction: '{instruction}'")

        # Run pipeline
        result_image, analysis, parsed = run_pipeline(pil_image, instruction)

        # Encode result
        result_b64 = pil_to_base64(result_image)

        return jsonify({
            "result": result_b64,
            "info": {
                "image_type":          analysis["image_type"],
                "faces_detected":      analysis["face_count"],
                "transformation_type": parsed["transformation_type"],
                "label":               parsed["label"],
                "pca_used":            parsed["use_pca"],
                "strength":            parsed["strength"],
            },
        })

    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/transformations", methods=["GET"])
def list_transformations():
    """Return a list of all supported transformation types."""
    from core.instruction_parser import get_supported_transformations
    return jsonify(get_supported_transformations())


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
