import os
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx
import pytesseract
from pdf2image import convert_from_path
import cv2

# ----------------- LOAD MODEL -----------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------- TEXT EXTRACTION -----------------
def extract_text_from_image_bytes(file_bytes):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    except:
        pages = convert_from_path(uploaded_file)
        for page in pages:
            text += pytesseract.image_to_string(page)
        return text

def extract_text_from_docx(file_bytes):
    with open("temp.docx", "wb") as f:
        f.write(file_bytes)
    document = docx.Document("temp.docx")
    return "\n".join([p.text for p in document.paragraphs])

def extract_text(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file.read())
    elif filename.endswith((".jpg", ".jpeg", ".png")):
        return extract_text_from_image_bytes(uploaded_file.read())
    return ""

# --------------- EMBEDDINGS ----------------
def create_embedding(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.encode("ascii", "ignore").decode()
    text = text.strip()
    if text == "":
        return np.zeros((384,))
    text = text[:5000]
    return model.encode(text)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- STREAMLIT UI ----------------
st.title("📁 AI Duplicate Document Detector")
st.write("Upload multiple files to check for duplicates using AI (SentenceTransformer).")

uploaded_files = st.file_uploader(
    "Upload Files",
    type=["pdf", "docx", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if st.button("🔍 Compare Files"):
    if not uploaded_files or len(uploaded_files) < 2:
        st.error("Please upload at least two files!")
    else:
        texts = {}
        embeddings = {}

        st.write("### 🔄 Extracting text from files...")
        progress = st.progress(0)

        # Extract text and embeddings
        for i, file in enumerate(uploaded_files):
            st.write(f"Processing: **{file.name}**")
            text = extract_text(file)
            texts[file.name] = text
            embeddings[file.name] = create_embedding(text)
            progress.progress((i + 1) / len(uploaded_files))

        st.write("### 🧠 Comparing Files...")
        results = ""

        # Compare pairwise
        for i in range(len(uploaded_files)):
            for j in range(i + 1, len(uploaded_files)):
                f1 = uploaded_files[i].name
                f2 = uploaded_files[j].name
                sim = cosine_similarity(embeddings[f1], embeddings[f2])
                percent = round(sim * 100, 2)

                results += f"### 🆚 {f1} VS {f2}\n"

                if percent >= 90:
                    results += f"🟢 **DUPLICATE — {percent}% similar**\n\n"
                else:
                    results += f"🔴 Not Duplicate — {percent}% similar\n\n"

        st.write(results)
