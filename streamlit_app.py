import os
import io
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

# ----------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------
st.set_page_config(
    page_title="AI Duplicate Document Detector",
    page_icon="📁",
    layout="wide"
)

# ----------------------------------------------------------------
# CUSTOM CSS (BEAUTIFUL THEME)
# ----------------------------------------------------------------
def load_css():
    st.markdown("""
    <style>
    /* General Background */
    body { background-color: #f7f3ef !important; font-family: 'Segoe UI', sans-serif; }

    /* Header */
    .app-title {
        font-size: 56px;
        font-weight: 900;
        color: #5d3a21;
        text-align: center;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 0;
    }
    .app-subtitle {
        text-align: center;
        font-size: 20px;
        color: #8b5e3c;
        margin-bottom: 40px;
    }

    /* Glass Cards */
    .glass-card {
        background: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.9));
        padding: 30px;
        border-radius: 20px;
        border: 2px solid rgba(93, 58, 33, 0.12);
        box-shadow: 0px 8px 20px rgba(0,0,0,0.1);
        margin-bottom: 30px;
        transition: transform 0.2s ease-in-out;
    }
    .glass-card:hover { transform: translateY(-4px); }

    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #5d3a21, #7a4a29);
        color: white;
        border-radius: 12px;
        height: 50px;
        font-size: 18px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #7a4a29, #5d3a21);
        transform: scale(1.02);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #e8dfd6; border-right: 2px solid #cbb9a5; }
    .css-1d391kg, .css-1lcbmhc { color: #5d3a21 !important; }

    /* Table styling */
    .dataframe tbody tr:hover { background-color: rgba(245, 222, 179, 0.2) !important; }

    /* Badges */
    .duplicate { color: white; font-weight: bold; background-color: #4CAF50; padding: 3px 8px; border-radius: 8px; }
    .not-duplicate { color: white; font-weight: bold; background-color: #f44336; padding: 3px 8px; border-radius: 8px; }

    /* Footer */
    footer { text-align:center; color:#5d3a21; font-size:15px; margin-top:50px; }
    </style>
    """, unsafe_allow_html=True)

load_css()

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
    if img is None: return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try: return pytesseract.image_to_string(gray)
    except: return ""

def extract_text_from_pdf_bytes(file_bytes):
    text = ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        if text.strip(): return text
    except: pass

    try:
        pages = convert_from_bytes(file_bytes)
        for page in pages:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                page.save(tmp.name, "JPEG")
                with open(tmp.name, "rb") as f:
                    text += extract_text_from_image_bytes(f.read())
                os.unlink(tmp.name)
    except: text += "\n[OCR not available on server]"
    return text

def extract_text_from_docx_bytes(file_bytes):
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    try:
        doc = docx.Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])
    except: text = ""
    finally: os.unlink(path)
    return text

def extract_text_from_bytes(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".pdf"): return extract_text_from_pdf_bytes(data)
    if name.endswith(".docx"): return extract_text_from_docx_bytes(data)
    if name.endswith((".jpg", ".jpeg", ".png")): return extract_text_from_image_bytes(data)
    return ""

# ----------------------------------------------------------------
# EMBEDDING FUNCTIONS
# ----------------------------------------------------------------
def create_embedding(text):
    text = (text or "").strip()
    if not text: return np.zeros((384,))
    text = text.encode("ascii","ignore").decode()[:5000]
    return model.encode(text)

def cosine_similarity(a,b):
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0: return 0.0
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

# ----------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------
st.markdown("<h1 class='app-title'>📁 AI Duplicate Document Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='app-subtitle'>Smart • Fast • Accurate — Compare Documents Using AI</p>", unsafe_allow_html=True)

# ----------------------------------------------------------------
# UPLOAD
# ----------------------------------------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("📤 Upload Documents")
uploaded_files = st.file_uploader(
    "PDF, DOCX, JPG, PNG",
    type=["pdf","docx","jpg","jpeg","png"],
    accept_multiple_files=True
)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("⚙️ Settings")
similarity_threshold = st.slider("Duplicate Threshold (%)", 0, 100, 90)
include_text = st.checkbox("Include extracted text in report")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------
# COMPARISON
# ----------------------------------------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("🚀 Start Comparison")

if st.button("🔍 Compare Files Now"):
    if not uploaded_files or len(uploaded_files)<2:
        st.error("Upload at least 2 files.")
    else:
        with st.spinner("Processing files..."):
            texts, embeddings = {}, {}
            for f in uploaded_files:
                f.seek(0)
                txt = extract_text_from_bytes(f)
                emb = create_embedding(txt)
                texts[f.name] = txt
                embeddings[f.name] = emb

        # Pairwise comparison
        results = []
        names = list(texts.keys())
        for i in range(len(names)):
            for j in range(i+1,len(names)):
                f1, f2 = names[i], names[j]
                sim = cosine_similarity(embeddings[f1], embeddings[f2])
                sim_pct = round(sim*100,2)
                results.append({
                    "File A": f1,
                    "File B": f2,
                    "Similarity %": sim_pct,
                    "Duplicate": f"<span class='duplicate'>✅</span>" if sim_pct>=similarity_threshold else f"<span class='not-duplicate'>❌</span>"
                })
        df = pd.DataFrame(results)
        st.success("✅ Comparison Complete!")
        st.markdown(df.to_html(escape=False), unsafe_allow_html=True)

        # Text preview
        if include_text:
            st.subheader("📄 Extracted Text Preview")
            for name, txt in texts.items():
                with st.expander(name):
                    st.text(txt[:8000])

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------
st.markdown("""
<footer>
Developed by <b>Urooj Fatima</b> • Powered by SentenceTransformers & Streamlit
</footer>
""", unsafe_allow_html=True)
