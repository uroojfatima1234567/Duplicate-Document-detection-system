import os, io, tempfile
import numpy as np
import streamlit as st
import pandas as pd
import docx
import re
import pytesseract
import cv2
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# LOAD MODEL
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
# TEXT CHUNKING
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
        txt = ""
        for p in reader.pages:
            t = p.extract_text() or ""
            txt += t + " "
        return clean_text(txt)

    if name.endswith(".docx"):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            doc = docx.Document(tmp.name)
        os.unlink(tmp.name)
        return clean_text("\n".join([p.text for p in doc.paragraphs]))

    if name.endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ocr_text = pytesseract.image_to_string(img)
        return clean_text(ocr_text)

    return ""

# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------
def get_embeddings(text):
    chunks = chunk_text(text)
    return model.encode(chunks)

# --------------------------------------------------
# SIMILARITY
# --------------------------------------------------
def compute_similarity(emb1, emb2):
    sims = np.matmul(emb1, emb2.T)
    top_scores = np.sort(sims.flatten())[-5:]
    return float(np.mean(top_scores)) * 100

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
def main_app():

    st.title("📁 AI Duplicate Document Detector (Improved + Same Name Fix)")

    files = st.file_uploader(
        "Upload documents",
        type=["pdf","docx","jpg","jpeg","png"],
        accept_multiple_files=True
    )

    threshold = st.slider("Duplicate Threshold (%)", 0, 100, 85)

    if st.button("🔍 Compare Documents"):

        if not files or len(files) < 2:
            st.error("Upload at least two files!")
            return

        texts = {}
        embeddings = {}
        name_counter = {}

        with st.spinner("Analyzing documents..."):

            for f in files:
                f.seek(0)

                base_name = f.name

                # Ensure uniqueness
                if base_name not in name_counter:
                    name_counter[base_name] = 1
                else:
                    name_counter[base_name] += 1

                unique_name = f"{base_name} ({name_counter[base_name]})"

                text = extract_text(f)
                texts[unique_name] = text
                embeddings[unique_name] = get_embeddings(text)

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
        st.success("Comparison complete!")
        st.dataframe(df, use_container_width=True)

# RUN APP
main_app()
