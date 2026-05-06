"""
PCA Module — deepfake-app
Handles face feature extraction and modification using Principal Component Analysis.
Used when a face is detected in the uploaded image.
When no face is found, the GAN pipeline takes over directly.
"""

import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import pickle
import os


class FacePCA:
    def __init__(self, n_components=100):
        """
        n_components: how many principal components to keep.
        100 captures roughly 90% of facial variation and trains fast.
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False
        self.image_size = (64, 64)   # work at 64x64 for speed

        # Maps instruction keywords → (component indices, direction)
        # These are approximate — real indices depend on your dataset.
        # After training PCA you can refine these by visualising components.
        self.feature_map = {
            "smile":      {"components": [5, 7, 12],  "sign":  1},
            "frown":      {"components": [5, 7, 12],  "sign": -1},
            "older":      {"components": [2, 15, 23], "sign":  1},
            "younger":    {"components": [2, 15, 23], "sign": -1},
            "dark hair":  {"components": [8, 10, 19], "sign":  1},
            "light hair": {"components": [8, 10, 19], "sign": -1},
            "glasses":    {"components": [4, 11],     "sign":  1},
            "no glasses": {"components": [4, 11],     "sign": -1},
            "bright":     {"components": [0, 1],      "sign":  1},
            "dark":       {"components": [0, 1],      "sign": -1},
        }

    # ──────────────────────────────────────────────
    # DATA LOADING
    # ──────────────────────────────────────────────

    def preprocess_image(self, image_input):
        """
        Accept a file path OR a PIL Image.
        Returns a flat normalised numpy array ready for PCA.
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("L")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("L")
        else:
            raise ValueError("image_input must be a file path or PIL Image")

        img = img.resize(self.image_size)
        arr = np.array(img, dtype=np.float32) / 255.0   # normalise 0–1
        return arr.flatten()

    def load_dataset(self, folder, max_images=5000):
        """
        Load up to max_images images from folder.
        Returns a 2D numpy array  [n_images, n_pixels].
        """
        all_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:max_images]

        print(f"Loading {len(all_files)} images from {folder} ...")
        images = []

        for i, filename in enumerate(all_files):
            path = os.path.join(folder, filename)
            try:
                images.append(self.preprocess_image(path))
            except Exception:
                pass   # skip corrupted files silently

            if i % 500 == 0 and i > 0:
                print(f"  {i}/{len(all_files)} loaded …")

        print(f"Dataset ready: {len(images)} images")
        return np.array(images)

    # ──────────────────────────────────────────────
    # TRAINING
    # ──────────────────────────────────────────────

    def fit(self, images):
        """
        Train PCA on a numpy array of flattened images.
        Call this once before compress / reconstruct.
        """
        print(f"Training PCA with {self.n_components} components …")
        self.pca.fit(images)
        self.is_fitted = True
        variance_captured = sum(self.pca.explained_variance_ratio_) * 100
        print(f"Done. Variance captured: {variance_captured:.1f}%")

    # ──────────────────────────────────────────────
    # COMPRESSION & RECONSTRUCTION
    # ──────────────────────────────────────────────

    def compress(self, image_input):
        """
        Compress one image down to its principal components.
        Returns a 1D array of length n_components.
        """
        self._assert_fitted()
        arr = self.preprocess_image(image_input)
        return self.pca.transform([arr])[0]

    def reconstruct(self, components):
        """
        Rebuild a 64×64 numpy image from principal components.
        Pixel values are clipped to [0, 1].
        """
        self._assert_fitted()
        flat = self.pca.inverse_transform([components])[0]
        flat = np.clip(flat, 0.0, 1.0)
        return flat.reshape(self.image_size)

    def components_to_pil(self, components):
        """Convert components → reconstructed PIL Image (grayscale)."""
        arr = self.reconstruct(components)
        return Image.fromarray((arr * 255).astype(np.uint8), mode="L")

    # ──────────────────────────────────────────────
    # FEATURE MODIFICATION
    # ──────────────────────────────────────────────

    def modify_feature(self, components, instruction, strength=2.5):
        """
        Modify principal components based on a text instruction.

        components : 1D numpy array from compress()
        instruction: free text e.g. "make me smile"
        strength   : how strongly to apply the change (1.0 – 5.0)

        Returns modified components ready for reconstruct().
        """
        modified = components.copy()
        instruction_lower = instruction.lower()
        applied = []

        for keyword, config in self.feature_map.items():
            if keyword in instruction_lower:
                for idx in config["components"]:
                    modified[idx] += config["sign"] * strength
                applied.append(keyword)

        if applied:
            print(f"PCA modifications applied: {applied}")
        else:
            print("No matching PCA feature found for instruction. "
                  "Passing components unchanged to GAN.")

        return modified

    # ──────────────────────────────────────────────
    # PERSISTENCE
    # ──────────────────────────────────────────────

    def save(self, path):
        """Save the trained PCA object to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"PCA model saved → {path}")

    @staticmethod
    def load(path):
        """Load a previously saved FacePCA object."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"PCA model loaded ← {path}")
        return obj

    # ──────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────

    def _assert_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "PCA is not fitted yet. Call fit() first "
                "or load a saved model with FacePCA.load()."
            )

    def visualise_eigenfaces(self, n=10):
        """
        Return the first n eigenfaces as a list of PIL Images.
        Useful for your report / presentation.
        """
        self._assert_fitted()
        eigenfaces = []
        for i in range(min(n, self.n_components)):
            face = self.pca.components_[i].reshape(self.image_size)
            face = (face - face.min()) / (face.max() - face.min() + 1e-8)
            eigenfaces.append(
                Image.fromarray((face * 255).astype(np.uint8), mode="L")
            )
        return eigenfaces


# ──────────────────────────────────────────────────────────────
# QUICK TEST  — run this file directly to verify everything works
# python models/pca_module.py
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=== FacePCA quick test ===")

    # Create dummy dataset (100 random 64x64 grayscale images)
    dummy = np.random.rand(100, 64 * 64).astype(np.float32)

    pca = FacePCA(n_components=50)
    pca.fit(dummy)

    # Compress then reconstruct a single image
    components = pca.pca.transform([dummy[0]])[0]
    reconstructed = pca.reconstruct(components)
    print(f"Reconstructed shape : {reconstructed.shape}")   # (64, 64)

    # Modify with instruction
    modified = pca.modify_feature(components, "make me smile", strength=2.0)
    print(f"Modified components shape: {modified.shape}")   # (50,)

    # Convert to PIL
    img = pca.components_to_pil(modified)
    print(f"PIL image size: {img.size}")   # (64, 64)

    # Save and reload
    pca.save("checkpoints/pca_test.pkl")
    loaded = FacePCA.load("checkpoints/pca_test.pkl")
    print("Save / load: OK")

    print("\n✅ All PCA tests passed!")
