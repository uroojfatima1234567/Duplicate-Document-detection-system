import os, io, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import docx
import re
import pytesseract
import cv2
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Duplicate Document Detector",
    page_icon="📁",
    layout="wide",
)

# --------------------------------------------------
# SESSION STATE (LOGIN)
# --------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --------------------------------------------------
# CUSTOM CSS (PREMIUM UI)
# --------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#f7efe7,#f1e6da);
}

.hero {
    text-align:center;
    padding:40px;
}

.hero h1 {
    font-size:50px;
    font-weight:900;
    color:#4b2e19;
}

.hero p {
    font-size:18px;
    color:#7a5336;
}

.card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(10px);
    border-radius:20px;
    padding:30px;
    box-shadow:0px 10px 30px rgba(0,0,0,.1);
    margin-bottom:30px;
    border:1px solid rgba(120,80,50,0.2);
}

.stButton>button {
    width:100%;
    background: linear-gradient(135deg,#6b4026,#8c5a3c);
    color:white;
    border-radius:14px;
    height:48px;
    font-size:18px;
    font-weight:600;
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(135deg,#8c5a3c,#6b4026);
}

footer {
    text-align:center;
    margin-top:40px;
    color:#6b4026;
    font-size:15px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------
def login_page():
    st.markdown("<div class='hero'><h1>🔐 Secure Login</h1><p>AI Duplicate Document Detection System</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email == "admin@gmail.com" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid credentials ❌")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD STATE-OF-THE-ART MODEL
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

model = load_model()

# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return text.strip()

# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# --------------------------------------------------
# TEXT EXTRACTION
# --------------------------------------------------
def extract_text(uploaded_file):
    data = uploaded_file.read()
    fname = uploaded_file.name.lower()

    if fname.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            text = ""
            for p in reader.pages:
                text += p.extract_text() or " "
            return clean_text(text)
        except:
            return ""

    if fname.endswith(".docx"):
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(data)
                doc = docx.Document(tmp.name)
            os.unlink(tmp.name)
            return clean_text("\n".join([p.text for p in doc.paragraphs]))
        except:
            return ""

    if fname.endswith((".jpg", ".jpeg", ".png")):
        try:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return clean_text(pytesseract.image_to_string(gray))
        except:
            return ""

    return ""

# --------------------------------------------------
# EMBEDDING FUNCTION (Chunking + Mean Pool)
# --------------------------------------------------
def embed(text):
    text = text.strip()
    if not text:
        return np.zeros((1024,))  # embedding size of mxbai

    chunks = chunk_text(text)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return np.mean(embeddings, axis=0)

# --------------------------------------------------
# COSINE SIMILARITY
# --------------------------------------------------
def similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
def main_app():

    st.sidebar.success("✅ Logged In")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.markdown("""
    <div class='hero'>
        <h1>📁 AI Duplicate Document Detector</h1>
        <p>Smart • Fast • Accurate AI-based document similarity system</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📤 Upload Documents")
    files = st.file_uploader(
        "PDF, DOCX, JPG, PNG",
        type=["pdf", "docx", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Settings
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Settings")
    threshold = st.slider("Duplicate Threshold (%)", 0, 100, 90)
    show_text = st.checkbox("Include extracted text in report")
    st.markdown("</div>", unsafe_allow_html=True)

    # Compare
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🚀 Compare Documents")

    if st.button("🔍 Start Comparison"):

        if not files or len(files) < 2:
            st.error("Upload at least two files.")
            return

        texts, embeds = {}, {}

        with st.spinner("Analyzing documents..."):
            for i, f in enumerate(files):
                f.seek(0)
                txt = extract_text(f)
                unique_name = f"{f.name}_{i}"  # ensures same-name files are unique
                texts[unique_name] = txt
                embeds[unique_name] = embed(txt)

        results = []
        names = list(texts.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                score = round(similarity(embeds[names[i]], embeds[names[j]]) * 100, 2)
                results.append({
                    "File A": names[i],
                    "File B": names[j],
                    "Similarity (%)": score,
                    "Duplicate": "✅ Yes" if score >= threshold else "❌ No"
                })

        df = pd.DataFrame(results)
        st.success("✅ Comparison Completed")
        st.dataframe(df, use_container_width=True)

        if show_text:
            for k, v in texts.items():
                with st.expander(k):
                    st.text(v[:6000])

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<footer>Developed by <b>Urooj Fatima</b> • AI & Streamlit</footer>", unsafe_allow_html=True)

# --------------------------------------------------
# RUN
# --------------------------------------------------
if not st.session_state.logged_in:
    login_page()
else:
    main_app()
