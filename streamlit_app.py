import os, io, tempfile
import numpy as np
import streamlit as st
import pandas as pd
import docx
import re
import pytesseract
import cv2
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
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
# LOAD IMPROVED MODEL
# --------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

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
# TEXT CHUNKING (BEST ACCURACY)
# --------------------------------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# --------------------------------------------------
# TEXT EXTRACTION
# --------------------------------------------------
def extract_text(uploaded_file):
    data = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        full_text = ""
        for page in reader.pages:
            txt = page.extract_text() or ""
            full_text += "\n" + txt
        return clean_text(full_text)

    if name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            doc = docx.Document(tmp.name)
        os.unlink(tmp.name)
        return clean_text("\n".join(p.text for p in doc.paragraphs))

    if name.endswith((".jpg",".jpeg",".png")):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ocr_text = pytesseract.image_to_string(img)
            return clean_text(ocr_text)

    return ""

# --------------------------------------------------
# EMBEDDINGS FOR CHUNKS
# --------------------------------------------------
def get_embeddings(text):
    chunks = chunk_text(text)
    return model.encode(chunks)

# --------------------------------------------------
# HIGH-ACCURACY SIMILARITY
# --------------------------------------------------
def compute_similarity(emb1, emb2):
    sims = np.matmul(emb1, emb2.T)   # cosine similarity for all chunk pairs
    top_scores = np.sort(sims.flatten())[-5:]  # top 5 highest matches
    return float(np.mean(top_scores)) * 100    # convert to %

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
def main_app():

    st.title("📁 AI Duplicate Document Detector (Improved Model)")
    st.write("✔ MPNet Model • ✔ Chunking • ✔ Cleaned OCR • ✔ High Accuracy")

    files = st.file_uploader(
        "Upload Documents",
        type=["pdf","docx","jpg","jpeg","png"],
        accept_multiple_files=True
    )

    threshold = st.slider("Duplicate Threshold (%)", 0, 100, 85)

    if st.button("🔍 Compare Documents"):
        if not files or len(files) < 2:
            st.error("Please upload at least two files!")
            return

        texts, embeddings = {}, {}
        with st.spinner("Extracting & analyzing..."):
            for f in files:
                f.seek(0)
                txt = extract_text(f)
                texts[f.name] = txt
                embeddings[f.name] = get_embeddings(txt)

        results = []
        names = list(texts.keys())

        for i in range(len(names)):
            for j in range(i+1, len(names)):
                score = compute_similarity(embeddings[names[i]], embeddings[names[j]])
                results.append({
                    "File A": names[i],
                    "File B": names[j],
                    "Similarity (%)": round(score, 2),
                    "Duplicate": "✅ Yes" if score >= threshold else "❌ No"
                })

        df = pd.DataFrame(results)
        st.success("Comparison Completed!")
        st.dataframe(df, use_container_width=True)

# --------------------------------------------------
# RUN
# --------------------------------------------------
main_app()
