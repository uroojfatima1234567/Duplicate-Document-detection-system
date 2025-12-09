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

# ----------------- Page config -----------------
st.set_page_config(page_title="AI Duplicate Document Detector", page_icon="📁", layout="wide")

# ---------- Secrets / config ----------
# For email sending, set these in Streamlit secrets (or environment variables)
# Example .streamlit/secrets.toml:
# [smtp]
# server = "smtp.gmail.com"
# port = 587
# username = "youremail@gmail.com"
# password = "app_password"

SMTP = st.secrets.get("smtp", {})  # may be empty dict if not set

# --------------- Model (cached) ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------- Helpers: Text extraction ----------------
def extract_text_from_image_bytes(file_bytes):
    # read image from bytes (cv2)
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        return pytesseract.image_to_string(gray)
    except Exception:
        return ""

def extract_text_from_pdf_bytes(file_bytes):
    text = ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        # if text found return it; otherwise fallback to image-based OCR (needs poppler/tesseract)
        if text.strip():
            return text
    except Exception:
        pass

    # fallback to image conversion (may need poppler installed on server)
    try:
        pages = convert_from_bytes(file_bytes)
        for page in pages:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                page.save(tmp.name, "JPEG")
                with open(tmp.name, "rb") as f:
                    file_bytes_img = f.read()
                text += extract_text_from_image_bytes(file_bytes_img)
                os.unlink(tmp.name)
    except Exception:
        # If poppler or tesseract not present on server, indicate OCR unavailable
        text += "\n\n[OCR fallback unavailable on server — poppler/tesseract may be missing]"
    return text

def extract_text_from_docx_bytes(file_bytes):
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        document = docx.Document(tmp_path)
        text = "\n".join([p.text for p in document.paragraphs])
    except Exception:
        text = ""
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass
    return text

def extract_text_from_bytes(uploaded_file):
    name = uploaded_file.name.lower()
    b = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(b)
    if name.endswith(".docx"):
        return extract_text_from_docx_bytes(b)
    if name.endswith((".jpg", ".jpeg", ".png")):
        return extract_text_from_image_bytes(b)
    return ""

# --------------- Embedding & similarity -------------
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
    # handle zero vectors
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---------------- UI: Header ----------------
with st.container():
    cols = st.columns([0.12, 1, 0.8])
    with cols[0]:
        st.image("https://raw.githubusercontent.com/uroojfatima1234567/placeholder/main/logo.png", width=70)  # replace with your logo URL or local file
    with cols[1]:
        st.markdown("<h1 style='margin:0'>📁 AI Duplicate Document Detector</h1>", unsafe_allow_html=True)
        st.markdown("Upload documents (PDF/DOCX/PNG/JPG). The app extracts text, creates embeddings, and compares files pairwise.")
    with cols[2]:
        st.metric(label="Model", value="all-MiniLM-L6-v2")
        st.caption("Developed by Urooj Fatima")

st.divider()

# ---------------- Upload + manage files panel ----------------
st.sidebar.header("Upload & Manage Files")
uploaded = st.sidebar.file_uploader("Upload files (pdf, docx, jpg, png) — multiple allowed", accept_multiple_files=True, type=["pdf","docx","jpg","jpeg","png"])

# keep a session-state list of files (so user can rename/delete before comparing)
if "files" not in st.session_state:
    st.session_state["files"] = []  # each item: {"name":..., "file": UploadedFile, "modified_name":...}

# add newly uploaded files to session_state
if uploaded:
    for u in uploaded:
        # avoid duplicates by filename+size
        key = (u.name, len(u.getvalue()))
        exists = False
        for f in st.session_state["files"]:
            if (f["orig_key"] == key):
                exists = True
                break
        if not exists:
            st.session_state["files"].append({"orig_key": key, "name": u.name, "file": u, "modified_name": u.name})

# file manager UI (rename / delete)
st.sidebar.markdown("### Current files")
if st.session_state["files"]:
    for idx, f in enumerate(st.session_state["files"]):
        cols = st.sidebar.columns([0.6, 0.25, 0.15])
        with cols[0]:
            st.text_input(label=f"Rename #{idx+1}", key=f"rename_{idx}", value=f["modified_name"], on_change=lambda i=idx: st.session_state["files"].__setitem__(i, {**st.session_state['files'][i], "modified_name": st.session_state[f"rename_{i}"]}))
        with cols[1]:
            st.caption(f["name"])
        with cols[2]:
            if st.sidebar.button("Delete", key=f"del_{idx}"):
                st.session_state["files"].pop(idx)
                st.experimental_rerun()
else:
    st.sidebar.info("No files uploaded yet.")

st.sidebar.markdown("---")
st.sidebar.markdown("🔧 Options")
similarity_threshold = st.sidebar.slider("Duplicate threshold (%)", 0, 100, 90)
include_extracted_text_in_report = st.sidebar.checkbox("Include extracted text in report", value=False)

st.sidebar.markdown("---")
st.sidebar.write("📬 Email report (optional)")
send_email_to = st.sidebar.text_input("Send report to (email)", value="")
st.sidebar.caption("To enable sending, add SMTP credentials to Streamlit secrets under [smtp]")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: For OCR to work on server, Tesseract and Poppler must be installed. If not installed, OCR fallback may be unavailable.")

# ---------------- Main actions ----------------
st.header("Files & Actions")

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Uploaded files")
    if st.session_state["files"]:
        df_display = pd.DataFrame([{"Uploaded name": f["name"], "Use name": f["modified_name"]} for f in st.session_state["files"]])
        st.table(df_display)
    else:
        st.info("No files in session. Upload files via the sidebar.")

with col2:
    if st.button("🔍 Compare files now", type="primary"):
        if len(st.session_state["files"]) < 2:
            st.warning("Please upload at least two files to compare.")
        else:
            # run comparison
            with st.spinner("Extracting text & creating embeddings..."):
                results = []
                texts = {}
                embeddings = {}
                total = len(st.session_state["files"])
                for i, f in enumerate(st.session_state["files"], start=1):
                    st.write(f"Processing **{f['modified_name']}**")
                    uploaded_file = f["file"]
                    # ensure read pointer at start
                    uploaded_file.seek(0)
                    text = extract_text_from_bytes(uploaded_file)
                    texts[f["modified_name"]] = text
                    embeddings[f["modified_name"]] = create_embedding(text)
                    st.progress(i/total)

            # pairwise comparison
            st.success("Text extraction done. Comparing...")
            pairs = []
            files_names = [f["modified_name"] for f in st.session_state["files"]]
            for i in range(len(files_names)):
                for j in range(i+1, len(files_names)):
                    a = files_names[i]
                    b = files_names[j]
                    sim = cosine_similarity(embeddings[a], embeddings[b])
                    percent = round(sim * 100, 2)
                    is_dup = percent >= similarity_threshold
                    pairs.append({"file_a": a, "file_b": b, "similarity": percent, "duplicate": is_dup})

            results_df = pd.DataFrame(pairs).sort_values("similarity", ascending=False).reset_index(drop=True)
            st.markdown("### 🔎 Comparison results")
            st.dataframe(results_df.style.applymap(lambda v: "background-color: #b9f6ca" if isinstance(v,bool) and v else None, subset=["duplicate"]))

            # Expanders for extracted text (7)
            st.markdown("### 📄 Extracted texts")
            for name, txt in texts.items():
                with st.expander(f"{name} — show extracted text"):
                    if txt.strip():
                        st.code(txt[:10000], language="text")  # show first 10k chars
                        if len(txt) > 10000:
                            st.caption("Large text truncated in preview; full text can be included in downloadable report.")
                    else:
                        st.info("No text was extracted (maybe image OCR failed on server).")

            # Export (4)
            st.markdown("### 💾 Export report")
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            base_name = f"dup-report-{timestamp}"
            # JSON
            report_json = {
                "generated_at_utc": timestamp,
                "files": files_names,
                "pairs": pairs,
                "texts_included": include_extracted_text_in_report,
                "texts": texts if include_extracted_text_in_report else None
            }
            json_bytes = json.dumps(report_json, indent=2).encode("utf-8")
            st.download_button(label="Download JSON report", data=json_bytes, file_name=f"{base_name}.json", mime="application/json")

            # TXT
            txt_buf = io.StringIO()
            txt_buf.write(f"Duplicate Detection Report — generated {timestamp} UTC\n\n")
            txt_buf.write("Pairs:\n")
            for p in pairs:
                txt_buf.write(f"{p['file_a']}  VS  {p['file_b']}  =>  {p['similarity']}% {'DUPLICATE' if p['duplicate'] else 'NOT DUPLICATE'}\n")
            if include_extracted_text_in_report:
                txt_buf.write("\n\nExtracted texts:\n")
                for k, v in texts.items():
                    txt_buf.write(f"\n---- {k} ----\n")
                    txt_buf.write(v + "\n")
            st.download_button("Download TXT report", data=txt_buf.getvalue().encode("utf-8"), file_name=f"{base_name}.txt", mime="text/plain")

            # PDF via fpdf (simple)
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 8, "Duplicate Detection Report", ln=True)
            pdf.cell(0, 6, f"Generated: {timestamp} UTC", ln=True)
            pdf.ln(4)
            for p in pairs:
                pdf.multi_cell(0, 7, f"{p['file_a']}  VS  {p['file_b']}  =>  {p['similarity']}%  {'DUPLICATE' if p['duplicate'] else 'NOT DUPLICATE'}")
            if include_extracted_text_in_report:
                for k, v in texts.items():
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 12)
                    pdf.multi_cell(0, 7, f"---- {k} ----")
                    pdf.set_font("Arial", size=10)
                    # limit text written to keep PDF reasonable
                    pdf.multi_cell(0, 6, v[:20000] + ("\n\n[truncated]" if len(v) > 20000 else ""))
            pdf_bytes = pdf.output(dest='S').encode('latin-1', errors='ignore')
            st.download_button("Download PDF report", data=pdf_bytes, file_name=f"{base_name}.pdf", mime="application/pdf")

            # Email (10)
            if send_email_to:
                if not SMTP:
                    st.warning("SMTP credentials missing from Streamlit secrets — can't send email.")
                else:
                    if st.button("Send report by email"):
                        try:
                            msg = EmailMessage()
                            msg["Subject"] = f"Duplicate Report {timestamp}"
                            msg["From"] = SMTP.get("username")
                            msg["To"] = send_email_to
                            msg.set_content("Attached is the duplicate detection report.")

                            msg.add_attachment(json_bytes, maintype="application", subtype="json", filename=f"{base_name}.json")
                            msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=f"{base_name}.pdf")

                            server = smtplib.SMTP(SMTP.get("server"), int(SMTP.get("port", 587)))
                            server.starttls()
                            server.login(SMTP.get("username"), SMTP.get("password"))
                            server.send_message(msg)
                            server.quit()
                            st.success(f"Email sent to {send_email_to}")
                        except Exception as e:
                            st.error(f"Failed to send email: {e}")

# ---------------- Footer (9) ----------------
st.markdown("""---""")
with st.container():
    c1, c2 = st.columns([1,3])
    with c1:
        st.image("https://raw.githubusercontent.com/uroojfatima1234567/placeholder/main/logo_small.png", width=80)  # replace or remove
    with c2:
        st.write("Developed by **Urooj Fatima** • Duplicate Document Detector • Built with 🤖 SentenceTransformers & Streamlit")
        st.caption("Note: For full OCR support on deployed servers, tesseract & poppler must be installed. Locally, set pytesseract.pytesseract.tesseract_cmd if needed.")
