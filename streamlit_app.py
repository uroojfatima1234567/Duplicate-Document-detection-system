import os, io, tempfile
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx
import pytesseract
from pdf2image import convert_from_bytes
import cv2

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
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------------------------------------------
# TEXT EXTRACTION
# --------------------------------------------------
def extract_text(uploaded_file):
    data = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return " ".join(p.extract_text() or "" for p in reader.pages)

    if name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            doc = docx.Document(tmp.name)
        os.unlink(tmp.name)
        return "\n".join(p.text for p in doc.paragraphs)

    if name.endswith((".jpg",".jpeg",".png")):
        img = cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
        if img is not None:
            return pytesseract.image_to_string(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

    return ""

# --------------------------------------------------
# EMBEDDING
# --------------------------------------------------
def embed(text):
    return model.encode(text[:5000]) if text.strip() else np.zeros((384,))

def similarity(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a) and np.linalg.norm(b) else 0

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
        type=["pdf","docx","jpg","jpeg","png"],
        accept_multiple_files=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Settings
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Settings")
    threshold = st.slider("Duplicate Threshold (%)",0,100,90)
    show_text = st.checkbox("Include extracted text in report")
    st.markdown("</div>", unsafe_allow_html=True)

    # Compare
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🚀 Compare Documents")

    if st.button("🔍 Start Comparison"):
        if not files or len(files) < 2:
            st.error("Upload at least two files.")
        else:
            texts, embeds = {}, {}
            with st.spinner("Analyzing documents..."):
                for f in files:
                    f.seek(0)
                    txt = extract_text(f)
                    texts[f.name] = txt
                    embeds[f.name] = embed(txt)

            results = []
            names = list(texts.keys())
            for i in range(len(names)):
                for j in range(i+1,len(names)):
                    score = round(similarity(embeds[names[i]],embeds[names[j]])*100,2)
                    results.append({
                        "File A": names[i],
                        "File B": names[j],
                        "Similarity (%)": score,
                        "Duplicate": "✅ Yes" if score>=threshold else "❌ No"
                    })

            df = pd.DataFrame(results)
            st.success("✅ Comparison Completed")
            st.dataframe(df,use_container_width=True)

            if show_text:
                for k,v in texts.items():
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
