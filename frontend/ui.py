"""
frontend/ui.py — Streamlit Frontend
Run with:  streamlit run frontend/ui.py

This is the interface your users (and your professor) will see.
It communicates with the Flask backend at localhost:5000.
"""

import streamlit as st
import requests
import base64
import io
from PIL import Image

# ══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FAke_Gen AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = "http://localhost:5000"

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def pil_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_pil(b64_string):
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes))

def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return None

def call_generate(pil_image, instruction):
    payload = {
        "image":       pil_to_base64(pil_image),
        "instruction": instruction,
    }
    response = requests.post(
        f"{BACKEND_URL}/generate",
        json=payload,
        timeout=60,
    )
    return response.json()

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Dark theme with accent colour */
    :root {
        --accent: #00e5ff;
    }
    .main { background-color: #0d0d0d; }
    h1 { color: var(--accent) !important; letter-spacing: 2px; }
    h3 { color: #aaaaaa; font-weight: 400; }
    .stButton > button {
        background: var(--accent);
        color: #000;
        font-weight: 700;
        border-radius: 4px;
        border: none;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    .result-box {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .info-chip {
        display: inline-block;
        background: #222;
        border: 1px solid #444;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.8rem;
        color: #aaa;
        margin: 2px;
    }
    .status-ok  { color: #00e676; font-weight: bold; }
    .status-err { color: #ff5252; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Backend status
    health = check_backend()
    if health:
        st.markdown('<span class="status-ok">● Backend online</span>',
                    unsafe_allow_html=True)
        st.markdown(
            f"Generator: {'✅' if health.get('generator_loaded') else '⚠️ not loaded'}")
        st.markdown(
            f"PCA model: {'✅' if health.get('pca_loaded') else '⚠️ not loaded'}")
    else:
        st.markdown('<span class="status-err">● Backend offline</span>',
                    unsafe_allow_html=True)
        st.warning("Start the backend:\n```\npython backend/app.py\n```")

    st.markdown("---")
    st.markdown("### 💡 Example Instructions")
    examples = [
        "make me smile",
        "make me look older",
        "make it cyberpunk style",
        "add a tattoo on my arm",
        "change background to beach",
        "make it oil painting",
        "remove the glasses",
        "make it night time",
        "make it anime style",
        "make it vintage",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state["instruction_input"] = ex

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This app uses **Pix2Pix GAN** and **PCA eigenfaces** "
        "to apply AI-powered transformations to any image."
    )

# ══════════════════════════════════════════════════════════════
# MAIN PAGE
# ══════════════════════════════════════════════════════════════
st.markdown("# 🎭 Fake_Gen AI")
st.markdown("### Upload an image. Describe your change(s). Get the result.")
st.markdown("---")

# ── Upload + Instruction ──────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image, caption="Original Image", use_column_width=True)

    st.markdown("#### ✏️ Your Instruction")
    instruction = st.text_input(
        "What change do you want?",
        value=st.session_state.get("instruction_input", ""),
        placeholder="e.g. make me smile, add a tattoo, change background to beach …",
        label_visibility="collapsed",
        key="instruction_input",
    )

    generate_clicked = st.button("🚀 Generate", disabled=(not uploaded_file))

# ── Result ────────────────────────────────────────────────────
with col_right:
    st.markdown("#### 🖼️ Result")

    if "result_image" in st.session_state:
        result_img = st.session_state["result_image"]
        info       = st.session_state.get("result_info", {})

        st.image(result_img, caption="Generated Image", use_column_width=True)

        # Info chips
        chips = [
            f"🏷️ {info.get('label', 'N/A')}",
            f"📦 {info.get('image_type', 'N/A')} image",
            f"👤 {info.get('faces_detected', 0)} face(s) detected",
            f"🧮 PCA: {'used' if info.get('pca_used') else 'skipped'}",
            f"💪 Strength: {info.get('strength', 'N/A')}",
        ]
        st.markdown(" ".join(
            f'<span class="info-chip">{c}</span>' for c in chips
        ), unsafe_allow_html=True)

        # Download button
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Result",
            data=buf.getvalue(),
            file_name="generated.png",
            mime="image/png",
        )

    else:
        st.markdown(
            '<div class="result-box" style="text-align:center;'
            'padding:4rem 1rem;color:#555;">'
            'Result will appear here after generation.'
            '</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════
# GENERATE LOGIC
# ══════════════════════════════════════════════════════════════
if generate_clicked:
    if not instruction.strip():
        st.error("Please enter an instruction before generating.")
    elif health is None:
        st.error("Backend is offline. Start it with: python backend/app.py")
    else:
        with st.spinner("Generating … this may take a few seconds"):
            try:
                response = call_generate(original_image, instruction)

                if "error" in response:
                    st.error(f"Error: {response['error']}")
                else:
                    result_pil = base64_to_pil(response["result"])
                    st.session_state["result_image"] = result_pil
                    st.session_state["result_info"]  = response.get("info", {})
                    st.rerun()

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot reach backend. "
                    "Make sure you ran: python backend/app.py"
                )
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ══════════════════════════════════════════════════════════════
# HISTORY (session only)
# ══════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state["history"] = []

if generate_clicked and "result_image" in st.session_state:
    st.session_state["history"].append({
        "instruction": instruction,
        "result":      st.session_state["result_image"],
    })

if st.session_state["history"]:
    st.markdown("---")
    st.markdown("#### 🕓 This Session's History")
    h_cols = st.columns(min(len(st.session_state["history"]), 5))
    for i, item in enumerate(reversed(st.session_state["history"][-5:])):
        with h_cols[i]:
            st.image(item["result"], caption=item["instruction"],
                     use_column_width=True)
