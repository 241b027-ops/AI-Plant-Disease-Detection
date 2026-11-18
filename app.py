import json
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import tensorflow as tf
from huggingface_hub import hf_hub_download
# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="LeafLens | Plant Disease AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_mobilenetv2_scratch.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"
VALIDATOR_PATH = BASE_DIR / "plant_validator.keras"
IMG_SIZE = (224, 224)
VALIDATOR_UNSUPPORTED_THRESHOLD = 0.70
VALIDATOR_REVIEW_THRESHOLD = 0.40

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --leaf:#237a4b; --leaf-dark:#155b36; --mint:#eaf5ed; --ink:#17251d; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: #f6f8f5; color:var(--ink); }
.block-container { padding-top: 1.7rem; padding-bottom: 3rem; max-width: 1250px; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); }
.hero {
  background: linear-gradient(120deg,#123e2a 0%,#237a4b 65%,#54a66d 100%);
  padding: 2.1rem 2.3rem; border-radius: 22px; color:white; margin-bottom:1.3rem;
}
.hero h1 { color:white !important; font-size:2.25rem; margin:0.2rem 0 0.55rem 0; }
.hero p { color:#e1f1e5; font-size:1rem; margin:0; max-width:680px; }
.eyebrow { color:#bce7c8; text-transform:uppercase; letter-spacing:2px; font-size:.72rem; font-weight:700; }
.metric-card {
  background:white; border:1px solid #e5ece6; border-radius:16px;
  padding:1.05rem 1.2rem; min-height:112px;
}
.metric-label { color:#718078; font-size:.82rem; font-weight:600; }
.metric-value { color:#173b28; font-family:'Manrope'; font-size:1.55rem; font-weight:800; margin-top:.25rem; }
.panel {
  background:white; border:1px solid #e5ece6; border-radius:18px;
  padding:1.2rem 1.35rem; margin-bottom:1rem;
}
.section-kicker { color:#348052; text-transform:uppercase; font-size:.72rem; font-weight:800; letter-spacing:1.4px; }
.result-name { font-family:'Manrope'; font-size:1.55rem; font-weight:800; color:#173b28; line-height:1.25; }
.muted { color:#748078; font-size:.9rem; }
div.stButton > button[kind="primary"] {
  background:#237a4b; border:0; border-radius:11px; font-weight:700;
  padding:.65rem 1.2rem;
}
div.stButton > button[kind="primary"]:hover { background:#155b36; border:0; }
[data-testid="stSidebar"] { background:#eef5ef; border-right:1px solid #e0e9e1; }
[data-testid="stSidebar"] .stRadio label { font-weight:600; }
.stProgress > div > div > div > div { background:#2c8a53; }
div[data-testid="stFileUploader"] { background:#fbfdfb; border:1px dashed #9cbea5; border-radius:14px; padding:.5rem; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Model and class loading
# ------------------------------------------------------------
def preprocess(x):
    # The notebook embeds this exact normalization in the model.
    return (x / 127.5) - 1.0

@st.cache_resource(show_spinner="Loading your trained MobileNetV2 model...")
def load_model():
    if not MODEL_PATH.exists():
        return None

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"preprocess": preprocess},
        safe_mode=False
    )

    
@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        return []
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)
    return list(names)

model = load_model()

@st.cache_resource(show_spinner="Loading the plant image validator...")
def load_validator():
    if not VALIDATOR_PATH.exists():
        return None
    return tf.keras.models.load_model(VALIDATOR_PATH, compile=False)

validator = load_validator()
class_names = load_class_names()

# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
def go_to_detection():
    st.session_state["navigation"] = "Disease Detection"

with st.sidebar:
    st.markdown("## 🌿 LeafLens")
    st.caption("Plant Disease Intelligence")
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Disease Detection", "Prediction History", "About the Model"],
        label_visibility="visible",
        key="navigation",
    )
    st.divider()
    if model is not None:
        st.success("Model loaded")
        st.caption(f"{len(class_names)} disease classes" if class_names else "Class labels missing")
    else:
        st.warning("Model file not found")
    if validator is not None:
        st.success("Plant validator loaded")
    else:
        st.warning("Plant validator not installed")
    st.markdown("---")
    st.caption("Built with Streamlit + TensorFlow")
    st.caption("MobileNetV2 • Trained from scratch")

def hero(title, subtitle, eyebrow="PLANT HEALTH • AI ASSISTANT"):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def metric(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="muted">{note}</div></div>',
        unsafe_allow_html=True,
    )

def check_supported_leaf(pil_image):
    """
    Returns (status, unsupported_score).
    The separate validator is trained with:
      class 0 = supported_leaf
      class 1 = unsupported (other plants and non-leaf images)
    """
    if validator is None:
        return "not_installed", None

    image = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)
    score = float(np.asarray(validator.predict(batch, verbose=0)).reshape(-1)[0])

    if score >= VALIDATOR_UNSUPPORTED_THRESHOLD:
        return "unsupported", score
    if score >= VALIDATOR_REVIEW_THRESHOLD:
        return "review", score
    return "supported", score


def make_prediction(pil_image):
    if model is None:
        raise RuntimeError("Model file is missing.")
    if not class_names:
        raise RuntimeError("class_names.json is missing or empty.")
    image = pil_image.convert("RGB").resize(IMG_SIZE)
    # Pass pixels in 0..255 range. The notebook's model performs [-1, 1]
    # normalization internally through its preprocess Lambda layer.
    arr = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(arr, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    if len(probabilities) != len(class_names):
        raise ValueError(
            f"Model returns {len(probabilities)} outputs, but class_names.json "
            f"contains {len(class_names)} labels. Regenerate class_names.json "
            "from the same PlantVillage dataset and class sorting used in training."
        )
    order = np.argsort(probabilities)[::-1]
    return image, probabilities, order

def save_history(image, label, confidence):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "disease": label,
        "confidence": float(confidence),
        "image": image.copy(),
    })

# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------
if page == "Dashboard":
    hero(
        "A healthier harvest starts with a leaf.",
        "An AI-powered plant disease screening interface. Upload a leaf image to explore a prediction from your trained MobileNetV2 model."
    )
    history = st.session_state.get("history", [])
    total = len(history)
    latest = history[0]["disease"] if total else "—"
    c1, c2, c3 = st.columns(3)
    with c1: metric("MODEL ARCHITECTURE", "MobileNetV2", "Hand-built • trained from scratch")
    with c2: metric("INPUT SIZE", "224 × 224", "RGB leaf images")
    with c3: metric("YOUR SCANS", str(total), "Predictions in this session")
    st.write("")
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="panel"><div class="section-kicker">GET STARTED</div><h3>Check a plant leaf</h3><p class="muted">Open Disease Detection, upload a clear image of a single leaf, and run the model.</p></div>', unsafe_allow_html=True)
        st.button("Start disease detection →", type="primary", use_container_width=True, on_click=go_to_detection)
    with right:
        st.markdown('<div class="panel"><div class="section-kicker">LATEST RESULT</div>', unsafe_allow_html=True)
        if total:
            st.markdown(f'<div class="result-name">{latest}</div><p class="muted">Confidence: {history[0]["confidence"]:.1%}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<h3>No scans yet</h3><p class="muted">Your latest prediction will appear here.</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("### How it works")
    steps = st.columns(3)
    for col, number, title, desc in zip(
        steps, ["01", "02", "03"],
        ["Upload a leaf", "AI analysis", "Review results"],
        ["Choose a JPG, JPEG or PNG image.", "The model scores each trained class.", "See the top prediction and alternatives."]
    ):
        with col:
            st.markdown(f'<div class="metric-card"><div class="section-kicker">{number}</div><h3>{title}</h3><p class="muted">{desc}</p></div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# Disease detection
# ------------------------------------------------------------
elif page == "Disease Detection":
    hero("Disease detection", "Upload a plant leaf photograph and let the model analyze its visual patterns.", "AI SCANNER")
    if model is None:
        st.error("The trained model file is not available. Place `best_mobilenetv2_scratch.keras` beside app.py. See README.md.")
        st.stop()
    if not class_names:
        st.error("Class labels are missing. Run `python prepare_classes.py --dataset path/to/raw/color` and place the generated class_names.json beside app.py.")
        st.stop()

    col_upload, col_preview = st.columns([1.05, .95], gap="large")
    with col_upload:
        st.markdown('<div class="panel"><div class="section-kicker">UPLOAD IMAGE</div><h3>Choose a leaf photo</h3><p class="muted">For best results, use a well-lit, in-focus image with one leaf clearly visible.</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])
        camera = st.camera_input("Or take a photo")
        st.markdown('</div>', unsafe_allow_html=True)
    source = camera if camera is not None else uploaded
    with col_preview:
        st.markdown('<div class="panel"><div class="section-kicker">IMAGE PREVIEW</div>', unsafe_allow_html=True)
        if source is not None:
            try:
                pil = Image.open(source)
                st.image(pil, caption="Selected leaf image", use_container_width=True)
            except Exception:
                st.error("This image could not be opened. Please choose another image.")
                pil = None
        else:
            pil = None
            st.markdown('<div style="padding:3.2rem 1rem;text-align:center;background:#f3f8f3;border-radius:14px;color:#789080">🌱<br><br>Your leaf image will appear here</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if pil is not None:
        if st.button("🔍  Analyze leaf", type="primary", use_container_width=True):
            try:
                with st.spinner("Checking the image and analyzing leaf patterns..."):
                    validation_status, unsupported_score = check_supported_leaf(pil)

                    if validation_status == "unsupported":
                        st.session_state["last_prediction"] = {
                            "status": "unsupported",
                            "image": pil.convert("RGB").copy(),
                            "unsupported_score": unsupported_score,
                        }
                    else:
                        processed, probs, order = make_prediction(pil)
                        best_idx = int(order[0])
                        best_name = class_names[best_idx]
                        confidence = float(probs[best_idx])
                        save_history(processed, best_name, confidence)
                        st.session_state["last_prediction"] = {
                            "status": validation_status,
                            "name": best_name, "confidence": confidence,
                            "probs": probs.tolist(), "order": order.tolist(),
                            "image": processed,
                            "unsupported_score": unsupported_score,
                        }
            except Exception as e:
                st.error(f"Prediction could not be completed: {e}")

    result = st.session_state.get("last_prediction")
    if result:
        st.divider()
        st.markdown("## Analysis result")

        if result.get("status") == "unsupported":
            st.image(result["image"], caption="Image checked", use_container_width=True)
            st.error("This image looks outside the supported leaf-image category.")
            st.write(
                "The disease classifier was not run. Please upload a clear leaf image "
                "from a plant category represented in the training dataset."
            )
            st.caption(
                f"Validator unsupported score: {result['unsupported_score']:.1%}. "
                "This is a model screening result, not a guarantee."
            )
            st.stop()

        if result.get("status") == "not_installed":
            st.warning(
                "No separate plant validator is installed. The disease model can still "
                "predict a class, but this app cannot reliably reject unsupported plants "
                "or non-leaf photos. Install and train plant_validator.keras as explained "
                "in README.md."
            )
        elif result.get("status") == "review":
            st.warning(
                "The validator is unsure whether this image belongs to a supported leaf "
                "category. Treat the disease prediction below as unreliable and retake "
                "the photo or use a supported plant."
            )

        r1, r2 = st.columns([1, 1.4], gap="large")
        with r1:
            st.image(result["image"], caption="Analyzed image", use_container_width=True)
        with r2:
            st.markdown('<div class="panel"><div class="section-kicker">TOP PREDICTION</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-name">{result["name"].replace("___", " — ").replace("_", " ")}</div>', unsafe_allow_html=True)
            st.metric("Model confidence", f'{result["confidence"]:.2%}')
            st.progress(float(result["confidence"]))
            if result["confidence"] < 0.60:
                st.warning("The model is not highly confident. Try a clearer image or consult an agricultural expert.")
            else:
                st.success("A relatively strong model prediction. This is still an AI screening result, not a confirmed diagnosis.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Top 3 predicted classes")
        top = result["order"][:3]
        top_df = pd.DataFrame({
            "Disease class": [class_names[int(i)].replace("___", " — ").replace("_", " ") for i in top],
            "Probability": [float(result["probs"][int(i)]) for i in top],
        })
        st.bar_chart(top_df.set_index("Disease class"), horizontal=True)
        with st.expander("View all class probabilities"):
            all_df = pd.DataFrame({
                "Class": [n.replace("___", " — ").replace("_", " ") for n in class_names],
                "Probability": result["probs"],
            }).sort_values("Probability", ascending=False)
            st.dataframe(all_df.style.format({"Probability": "{:.2%}"}), use_container_width=True, hide_index=True)

# ------------------------------------------------------------
# Prediction history
# ------------------------------------------------------------
elif page == "Prediction History":
    hero("Your scan history", "Review the predictions made during this browser session.", "RECENT ACTIVITY")
    history = st.session_state.get("history", [])
    if not history:
        st.info("No predictions yet. Visit Disease Detection to analyze your first leaf.")
    else:
        st.markdown(f"**{len(history)} scan(s) in this session**")
        for item in history:
            with st.container(border=True):
                a, b, c = st.columns([1, 3, 1])
                with a: st.image(item["image"], use_container_width=True)
                with b:
                    st.markdown(f"**{item['disease'].replace('___', ' — ').replace('_', ' ')}**")
                    st.caption(item["time"])
                with c:
                    st.metric("Confidence", f"{item['confidence']:.1%}")
        export_df = pd.DataFrame([{
            "Date and time": x["time"],
            "Predicted class": x["disease"],
            "Confidence": x["confidence"],
        } for x in history])
        st.download_button(
            "Download scan history (CSV)",
            export_df.to_csv(index=False).encode("utf-8"),
            file_name="plant_disease_scan_history.csv",
            mime="text/csv",
        )
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.pop("last_prediction", None)
            st.rerun()

# ------------------------------------------------------------
# About the model
# ------------------------------------------------------------
elif page == "About the Model":
    hero("Inside the intelligence", "A closer look at the architecture and training approach behind this plant disease classifier.", "MODEL CARD")
    st.markdown("### Model overview")
    st.markdown("""
    This project uses a **MobileNetV2-style convolutional neural network implemented manually** in the notebook. It is trained from random initialization; it does not use ImageNet weights or `keras.applications.MobileNetV2`.
    """)
    a, b, c, d = st.columns(4)
    with a: metric("BACKBONE", "MobileNetV2", "Hand-built")
    with b: metric("INPUT", "224 × 224", "RGB")
    with c: metric("OPTIMIZER", "Adam", "Learning rate 0.0005")
    with d: metric("OUTPUT", "Softmax", "One score per class")
    st.markdown("### Key architecture components")
    st.markdown("""
    - **Depthwise-separable convolutions:** spatial filtering followed by pointwise channel mixing.
    - **Inverted residual blocks:** expansion, depthwise convolution and projection.
    - **Linear bottlenecks:** projection layers do not apply an activation.
    - **ReLU6:** used in the activated convolution blocks.
    - **Residual connections:** used when stride is 1 and input/output channels match.
    - **Global average pooling + Dropout (0.3) + Dense softmax classifier.**
    """)
    st.markdown("### Optional input validation")
    st.markdown(
        "A separate binary image validator can be installed before the disease classifier. "
        "It is trained to distinguish supported leaf photos from unsupported plants and "
        "non-leaf images. It does not change or retrain the original MobileNetV2 disease model."
    )
    if validator is None:
        st.info("Validator file not found: add `plant_validator.keras` after training it.")
    else:
        st.success("Separate plant validator is loaded.")

    st.markdown("### Training configuration")
    config = pd.DataFrame([
        ["Dataset", "PlantVillage (raw/color)"],
        ["Train / validation / test", "70% / 15% / 15% stratified split"],
        ["Batch size", "32"],
        ["Maximum epochs", "15"],
        ["Loss", "Sparse categorical crossentropy"],
        ["Class imbalance", "Balanced class weights"],
        ["Callbacks", "EarlyStopping, ReduceLROnPlateau, ModelCheckpoint"],
        ["Augmentation", "Flip, rotation, zoom, contrast, translation"],
    ], columns=["Setting", "Notebook configuration"])
    st.dataframe(config, use_container_width=True, hide_index=True)
    st.warning("Important: predictions depend on image quality and the training dataset. The app is a project demonstration and should not replace diagnosis by an agricultural specialist.")
