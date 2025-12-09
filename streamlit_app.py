import os
import io
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx
import pytesseract
from pdf2image import convert_from_bytes
import cv2
from fpdf import FPDF
import smtplib
from email.message import EmailMessage
from datetime import datetime

# ----------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------
st.set_page_config(
    page_title="AI Duplicate Document Detector",
    page_icon="📁",
    layout="wide"
)

# ----------------------------------------------------------------
# CUSTOM CSS (OFF-WHITE + BROWN THEME)
# ----------------------------------------------------------------
def load_css():
    st.markdown("""
    <style>

    body {
        background-color: #f7f3ef !important;
    }

    .main {
        padding: 2rem 4rem;
        background-color: #f7f3ef;
    }

    /* Title */
    .app-title {
        font-size: 48px;
        font-weight: 900;
        color: #5d3a21;
        text-align: center;
        margin-bottom: -5px;
        font-family: 'Segoe UI';
    }

    /* Subtitle */
    .app-subtitle {
        text-align: center;
        font-size: 18px;
        color: #8b5e3c;
        margin-bottom: 40px;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        padding: 25px;
        border-radius: 18px;
        border: 2px solid rgba(93, 58, 33, 0.12);
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        backdrop-filter: blur(8px);
        margin-bottom: 25px;
    }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background-color: #5d3a21;
        color: white;
        border-radius: 10px;
        height: 48px;
        font-size: 18px;
        border: none;
        font-weight: 600;
    }

    .stButton>button:hover {
        background-color: #7a4a29;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #e8dfd6;
        border-right: 2px solid #cbb9a5;
    }

    .css-1d391kg, .css-1lcbmhc {
        color: #5d3a21 !important;
    }

    </style>
    """, unsafe_allow_html=True)

load_css()


# ----------------------------------------------------------------
# SMTP SETTINGS
# ----------------------------------------------------------------
SMTP = st.secrets.get("smtp", {})

# ----------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# ----------------------------------------------------------------
# TEXT EXTRACTION FUNCTIONS
# ----------------------------------------------------------------
def extract_text_from_image_bytes(file_bytes):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        return pytesseract.image_to_string(gray)
    except:
        return ""


def extract_text_from_pdf_bytes(file_bytes):
    text = ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

        if text.strip():
            return text
    except:
        pass

    # OCR fallback
    try:
        pages = convert_from_bytes(file_bytes)
        for page in pages:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                page.save(tmp.name, "JPEG")
                with open(tmp.name, "rb") as f:
                    text += extract_text_from_image_bytes(f.read())
                os.unlink(tmp.name)
    except:
        text += "\n[OCR not available on server]"
    return text


def extract_text_from_docx_bytes(file_bytes):
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name

    try:
        document = docx.Document(path)
        text = "\n".join([p.text for p in document.paragraphs])
    except:
        text = ""
    finally:
        os.unlink(path)

    return text


def extract_text_from_bytes(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(data)
    if name.endswith(".docx"):
        return extract_text_from_docx_bytes(data)
    if name.endswith((".jpg", ".jpeg", ".png")):
        return extract_text_from_image_bytes(data)

    return ""


# ----------------------------------------------------------------
# EMBEDDING FUNCTIONS
# ----------------------------------------------------------------
def create_embedding(text):
    text = (text or "").strip()
    if not text:
        return np.zeros((384,))
    text = text.encode("ascii", "ignore").decode()[:5000]
    return model.encode(text)


def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ----------------------------------------------------------------
# UI HEADER
# ----------------------------------------------------------------
st.markdown("<h1 class='app-title'>📁 AI Duplicate Document Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>Smart • Fast • Accurate — Compare Documents Using AI</p>", unsafe_allow_html=True)

# ----------------------------------------------------------------
# UPLOAD SECTION
# ----------------------------------------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("📤 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, JPG, PNG files",
    type=["pdf", "docx", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------
# SETTINGS SECTION
# ----------------------------------------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

st.subheader("⚙️ Settings")

similarity_threshold = st.slider(
    "Duplicate threshold (%)",
    0, 100, 90
)

include_text = st.checkbox("Include extracted text in report")

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------
# COMPARE BUTTON
# ----------------------------------------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

st.subheader("🚀 Start Comparison")
start_btn = st.button("🔍 Compare Files Now")

st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------
# MAIN LOGIC
# ----------------------------------------------------------------
if start_btn:
    if not uploaded_files or len(uploaded_files) < 2:
        st.error("Please upload at least two files.")
    else:
        with st.spinner("Extracting text & computing embeddings..."):
            texts = {}
            embeddings = {}

            for file in uploaded_files:
                file.seek(0)
                txt = extract_text_from_bytes(file)
                emb = create_embedding(txt)

                texts[file.name] = txt
                embeddings[file.name] = emb

        # Pairwise comparison
        results = []
        names = [f.name for f in uploaded_files]

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                sim = cosine_similarity(embeddings[a], embeddings[b])
                percent = round(sim * 100, 2)
                results.append({
                    "File A": a,
                    "File B": b,
                    "Similarity %": percent,
                    "Duplicate": percent >= similarity_threshold
                })

        df = pd.DataFrame(results)

        st.success("Comparison Complete!")
        st.dataframe(df)

        # Extracted text section
        st.subheader("📄 Extracted Text (Preview)")
        for name, txt in texts.items():
            with st.expander(name):
                st.text(txt[:8000])


# ----------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------
st.markdown("""
<hr>
<p style='text-align:center; color:#5d3a21; font-size:16px;'>
Developed by <b>Urooj Fatima</b> • Powered by SentenceTransformers & Streamlit
</p>
""", unsafe_allow_html=True)
