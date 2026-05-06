"""
instruction_parser.py
Reads the user's text instruction and decides:
  1. What TYPE of transformation is needed
  2. Which keywords map to PCA feature modifications (face only)
  3. What prompt to pass to the GAN
"""


# ══════════════════════════════════════════════════════════════
# KEYWORD MAPS
# Each category maps keywords → transformation metadata
# ══════════════════════════════════════════════════════════════

TRANSFORMATION_RULES = [

    # ── Face attribute changes (uses PCA + GAN) ───────────────
    {
        "type":     "face_attribute",
        "keywords": ["smile", "smiling", "happy", "laugh"],
        "pca_key":  "smile",
        "label":    "Add smile",
    },
    {
        "type":     "face_attribute",
        "keywords": ["frown", "sad", "angry", "serious"],
        "pca_key":  "frown",
        "label":    "Change expression",
    },
    {
        "type":     "face_attribute",
        "keywords": ["older", "old", "age", "aged", "wrinkle"],
        "pca_key":  "older",
        "label":    "Age progression",
    },
    {
        "type":     "face_attribute",
        "keywords": ["younger", "young", "youthful"],
        "pca_key":  "younger",
        "label":    "Age regression",
    },
    {
        "type":     "face_attribute",
        "keywords": ["dark hair", "black hair", "brown hair", "darker hair"],
        "pca_key":  "dark hair",
        "label":    "Darken hair",
    },
    {
        "type":     "face_attribute",
        "keywords": ["blonde", "light hair", "bright hair", "white hair"],
        "pca_key":  "light hair",
        "label":    "Lighten hair",
    },
    {
        "type":     "face_attribute",
        "keywords": ["glasses", "spectacles", "eyeglasses"],
        "pca_key":  "glasses",
        "label":    "Add glasses",
    },
    {
        "type":     "face_attribute",
        "keywords": ["no glasses", "remove glasses", "without glasses"],
        "pca_key":  "no glasses",
        "label":    "Remove glasses",
    },

    # ── Object addition (GAN only) ────────────────────────────
    {
        "type":     "object_add",
        "keywords": ["tattoo", "add tattoo", "tattoo on"],
        "pca_key":  None,
        "label":    "Add tattoo",
    },
    {
        "type":     "object_add",
        "keywords": ["beard", "add beard", "facial hair"],
        "pca_key":  None,
        "label":    "Add beard",
    },
    {
        "type":     "object_add",
        "keywords": ["hat", "add hat", "cap", "add cap"],
        "pca_key":  None,
        "label":    "Add hat",
    },
    {
        "type":     "object_add",
        "keywords": ["makeup", "lipstick", "add makeup"],
        "pca_key":  None,
        "label":    "Add makeup",
    },

    # ── Object removal (GAN only) ─────────────────────────────
    {
        "type":     "object_remove",
        "keywords": ["remove", "erase", "delete", "without"],
        "pca_key":  None,
        "label":    "Remove object",
    },

    # ── Background / scenery change (GAN only) ────────────────
    {
        "type":     "background_change",
        "keywords": ["background", "change background", "replace background",
                     "put me in", "place me in"],
        "pca_key":  None,
        "label":    "Change background",
    },
    {
        "type":     "background_change",
        "keywords": ["beach", "forest", "city", "space", "mountains",
                     "desert", "snow", "jungle", "underwater"],
        "pca_key":  None,
        "label":    "Change scene",
    },

    # ── Style transfer (GAN only) ─────────────────────────────
    {
        "type":     "style_transfer",
        "keywords": ["painting", "cartoon", "sketch", "oil painting",
                     "watercolor", "anime", "drawing", "artistic"],
        "pca_key":  None,
        "label":    "Style transfer",
    },

    # ── Lighting / atmosphere (GAN only) ─────────────────────
    {
        "type":     "lighting",
        "keywords": ["night", "dark", "evening", "nighttime"],
        "pca_key":  "dark",
        "label":    "Night effect",
    },
    {
        "type":     "lighting",
        "keywords": ["bright", "brighter", "lighter", "sunny", "day"],
        "pca_key":  "bright",
        "label":    "Brighten image",
    },
    {
        "type":     "lighting",
        "keywords": ["vintage", "retro", "old photo", "sepia"],
        "pca_key":  None,
        "label":    "Vintage effect",
    },

    # ── Colour changes (GAN only) ─────────────────────────────
    {
        "type":     "color_change",
        "keywords": ["color", "colour", "recolor", "change color",
                     "blue", "red", "green", "yellow", "pink", "orange"],
        "pca_key":  None,
        "label":    "Colour change",
    },
]


# ══════════════════════════════════════════════════════════════
# PARSER
# ══════════════════════════════════════════════════════════════

def parse_instruction(instruction, image_type="face"):
    """
    Parse a free-text instruction into a structured action dict.

    instruction : str  — what the user typed
    image_type  : str  — 'face' | 'scenery' | 'object'
                         (from image_analyser.analyse_image)

    Returns:
        {
            "original_instruction": str,
            "transformation_type" : str,
            "label"               : str,
            "pca_key"             : str | None,
            "use_pca"             : bool,
            "gan_prompt"          : str,
            "strength"            : float,
        }
    """
    text   = instruction.lower().strip()
    result = None

    # Find the first matching rule
    for rule in TRANSFORMATION_RULES:
        for keyword in rule["keywords"]:
            if keyword in text:
                result = rule.copy()
                break
        if result:
            break

    # Nothing matched — treat as a generic GAN instruction
    if result is None:
        result = {
            "type":    "generic",
            "pca_key": None,
            "label":   "Custom transformation",
        }

    # PCA is only useful when there is a face AND a pca_key is defined
    use_pca = (image_type == "face") and (result["pca_key"] is not None)

    # Parse a strength modifier from the instruction
    # e.g. "slightly older" → 1.5,  "much older" → 4.0
    strength = 2.5   # default
    if any(w in text for w in ["slightly", "a bit", "little", "subtle"]):
        strength = 1.5
    elif any(w in text for w in ["very", "extremely", "a lot", "much",
                                  "completely", "fully"]):
        strength = 4.0
    elif any(w in text for w in ["moderately", "somewhat"]):
        strength = 2.5

    parsed = {
        "original_instruction": instruction,
        "transformation_type":  result["type"],
        "label":                result["label"],
        "pca_key":              result.get("pca_key"),
        "use_pca":              use_pca,
        "gan_prompt":           instruction,   # passed directly to GAN
        "strength":             strength,
    }

    print(f"Instruction parsed → type: {parsed['transformation_type']}, "
          f"use_pca: {use_pca}, strength: {strength}")
    return parsed


def get_supported_transformations():
    """Return a human-readable list of supported transformations."""
    seen   = set()
    result = []
    for rule in TRANSFORMATION_RULES:
        if rule["label"] not in seen:
            seen.add(rule["label"])
            result.append({
                "label":    rule["label"],
                "type":     rule["type"],
                "examples": rule["keywords"][:3],
            })
    return result


# ──────────────────────────────────────────────────────────────
# QUICK TEST
# python core/instruction_parser.py
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("make me smile",                "face"),
        ("add a tattoo on my arm",       "face"),
        ("change background to beach",   "scenery"),
        ("make it look like a painting", "scenery"),
        ("remove the car",               "object"),
        ("make me look older",           "face"),
        ("very bright",                  "scenery"),
        ("something completely random",  "face"),
    ]

    print("=== Instruction Parser Tests ===\n")
    for instruction, image_type in tests:
        parsed = parse_instruction(instruction, image_type)
        print(f"Input     : '{instruction}'  (image_type={image_type})")
        print(f"Type      : {parsed['transformation_type']}")
        print(f"Label     : {parsed['label']}")
        print(f"Use PCA   : {parsed['use_pca']}")
        print(f"Strength  : {parsed['strength']}")
        print()

    print("Supported transformations:")
    for t in get_supported_transformations():
        print(f"  [{t['type']}] {t['label']} "
              f"— e.g. {', '.join(t['examples'])}")

    print("\n✅ Instruction parser OK")
