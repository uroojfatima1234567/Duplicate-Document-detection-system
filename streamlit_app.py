# streamlit_app.py
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

# ---------------- Page config ----------------
st.set_page_config(page_title="AI Duplicate Detector — Dark", page_icon="📁", layout="wide")
# Custom CSS for dark premium theme (glass + neon)
st.markdown(
    """
    <style>
    :root{
      --bg:#0f1115;
      --card:#121316;
      --muted:#9aa3b2;
      --accent:#32d1ff;
      --accent-2:#6a5cff;
      --glass: rgba(255,255,255,0.03);
      --success:#39d98a;
      --danger:#ff6b6b;
    }
    html, body, [class*="stApp"]{
      background: linear-gradient(180deg, #07080a 0%, #0f1115 100%);
      color: #dbe7ff;
    }
    .stToolbar {display:none}
    .app-card{
      background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
      border-radius:14px;
      padding:18px;
      box-shadow: 0 6px 24px rgba(0,0,0,0.6);
      border: 1px solid rgba(255,255,255,0.03);
    }
    .muted { color: var(--muted); }
    .neon { color: var(--accent); text-shadow: 0 0 12px rgba(50,209,255,0.12); }
    .btn-primary {
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      color: #06121a !important;
      border-radius: 10px;
      padding: 8px 18px;
      font-weight: 600;
      box-shadow: 0 6px 18px rgba(106,92,255,0.12);
    }
    .small {
      font-size:12px;
      color:var(--muted);
    }
    .result-dup { color: var(--success); font-weight:700; }
    .result-no { color: var(--danger); font-weight:700; }
    .logo-round { border-radius:12px; border:1px solid rgba(255,255,255,0.03); }
    .sidebar .stButton>button { border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Secrets / config ----------
SMTP = st.secrets.get("smtp", {})  # expects server, port, username, password (optional)

# --------------- Model (cached) ----------------
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")
model = load_model()

# --------------- Helpers: Text extraction ----------------
def extract_text_from_image_bytes(file_bytes):
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
        if text.strip():
            return text
    except Exception:
        pass
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
    # reset cursor for upstream reuse if needed
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
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---------------- Session state setup ----------------
if "files" not in st.session_state:
    st.session_state["files"] = []  # {"orig_key": (name,size), "name": original, "file":UploadedFile, "modified_name":str}

# ---------------- Header (neon) ----------------
with st.container():
    left, center, right = st.columns([0.12, 1, 0.28])
    with left:
        # Replace with your logo path or hosted image URL if you have one
        st.image("https://raw.githubusercontent.com/uroojfatima1234567/placeholder/main/logo_small.png", width=72, clamp=False, output_format="auto")
    with center:
        st.markdown("<div style='margin-bottom:4px'><h1 class='neon' style='margin:0'>📁 AI Duplicate Document Detector</h1></div>", unsafe_allow_html=True)
        st.markdown("<div class='small muted'>Dark / Premium • Compare documents using SentenceTransformers</div>", unsafe_allow_html=True)
    with right:
        st.metric(label="Model", value="all-MiniLM-L6-v2")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Upload & Settings")
    uploaded = st.file_uploader("Upload files (pdf, docx, jpg, png) — multiple", accept_multiple_files=True, type=["pdf","docx","jpg","jpeg","png"])
    if uploaded:
        for u in uploaded:
            key = (u.name, len(u.getvalue()))
            if not any(f["orig_key"] == key for f in st.session_state["files"]):
                st.session_state["files"].append({"orig_key": key, "name": u.name, "file": u, "modified_name": u.name})
    st.markdown("---")
    st.write("**Files in session**")
    if st.session_state["files"]:
        for idx, f in enumerate(st.session_state["files"]):
            cols = st.columns([0.62, 0.28, 0.1])
            with cols[0]:
                new_name = st.text_input(label=f"Rename #{idx+1}", key=f"rename_{idx}", value=f["modified_name"])
                # update modified_name when typed
                st.session_state["files"][idx]["modified_name"] = new_name
            with cols[1]:
                st.caption(f["name"])
            with cols[2]:
                if st.button("🗑", key=f"del_{idx}"):
                    st.session_state["files"].pop(idx)
                    st.experimental_rerun()
    else:
        st.caption("No files uploaded.")

    st.markdown("---")
    similarity_threshold = st.slider("Duplicate threshold (%)", 0, 100, 90)
    include_extracted_text_in_report = st.checkbox("Include extracted text in report", value=False)
    send_email_to = st.text_input("Send report to (email)", value="")
    st.caption("To enable email, add SMTP under Streamlit secrets as [smtp] (server, port, username, password).")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Main layout ----------------
st.markdown("<div class='app-card'>", unsafe_allow_html=True)
st.subheader("Files & Actions")
cols = st.columns([2, 1])
with cols[0]:
    if st.session_state["files"]:
        display_df = pd.DataFrame([{"Uploaded name": f["name"], "Use as": f["modified_name"]} for f in st.session_state["files"]])
        st.table(display_df)
    else:
        st.info("Upload files using the sidebar to begin.")

with cols[1]:
    compare_btn = st.button("🔍 Compare files now", key="compare", help="Run pairwise comparison")
    st.markdown("<div class='small muted'>Tip: first run extraction (heavy operations may take time)</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Compare action ----------------
if compare_btn:
    if len(st.session_state["files"]) < 2:
        st.warning("Please upload at least two files to compare.")
    else:
        # extraction + embeddings
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        total = len(st.session_state["files"])
        texts = {}
        embeddings = {}
        status_placeholder.info("🔄 Extracting text and building embeddings...")
        for i, f in enumerate(st.session_state["files"], start=1):
            name = f["modified_name"]
            status_placeholder.info(f"🔎 Processing: {name}")
            uploaded_file = f["file"]
            uploaded_file.seek(0)
            try:
                text = extract_text_from_bytes(uploaded_file)
            except Exception as e:
                text = ""
            texts[name] = text
            embeddings[name] = create_embedding(text)
            progress_bar.progress(min(i/total, 1.0))
        status_placeholder.success("✅ Extraction complete. Comparing now...")
        # pairwise compare
        pairs = []
        names = [f["modified_name"] for f in st.session_state["files"]]
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                a, b = names[i], names[j]
                sim = cosine_similarity(embeddings[a], embeddings[b])
                percent = round(sim * 100, 2)
                dup = percent >= similarity_threshold
                pairs.append({"file_a": a, "file_b": b, "similarity": percent, "duplicate": dup})

        results_df = pd.DataFrame(pairs).sort_values("similarity", ascending=False).reset_index(drop=True)
        # results card
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.markdown("### 🔎 Results")
        # highlight duplicate rows with color using st.dataframe (pandas style)
        def highlight_dup(val):
            if isinstance(val, bool) and val:
                return 'background-color: rgba(57,217,138,0.12); color: var(--success); font-weight:700;'
            return ''
        try:
            styled = results_df.style.applymap(lambda v: 'color: var(--success); font-weight:bold;' if (isinstance(v,bool) and v) else '', subset=['duplicate'])
            st.dataframe(results_df, use_container_width=True)
        except Exception:
            st.dataframe(results_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Expanders for extracted text
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.markdown("### 📄 Extracted Texts")
        for name, txt in texts.items():
            with st.expander(f"{name} — preview"):
                if txt.strip():
                    # show first 12k chars for preview
                    st.code(txt[:12000], language="text")
                    if len(txt) > 12000:
                        st.caption("Preview truncated — full text can be included in downloads.")
                else:
                    st.info("No text extracted (OCR/pdftotext may be unavailable).")
        st.markdown("</div>", unsafe_allow_html=True)

        # Export options (JSON, TXT, PDF)
        st.markdown("<div class='app-card'>", unsafe_allow_html=True)
        st.markdown("### 💾 Export & Email")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_name = f"dup-report-{timestamp}"
        report_json = {
            "generated_at_utc": timestamp,
            "files": names,
            "pairs": pairs,
            "texts_included": include_extracted_text_in_report,
            "texts": texts if include_extracted_text_in_report else None
        }
        json_bytes = json.dumps(report_json, indent=2).encode("utf-8")
        st.download_button("📥 Download JSON", data=json_bytes, file_name=f"{base_name}.json", mime="application/json")
        # txt
        txt_buf = io.StringIO()
        txt_buf.write(f"Duplicate Detection Report — {timestamp} UTC\n\n")
        for p in pairs:
            txt_buf.write(f"{p['file_a']}  vs  {p['file_b']}  =>  {p['similarity']}%  {'DUPLICATE' if p['duplicate'] else 'NOT DUPLICATE'}\n")
        if include_extracted_text_in_report:
            txt_buf.write("\n--- Extracted Texts ---\n")
            for k, v in texts.items():
                txt_buf.write(f"\n---- {k} ----\n")
                txt_buf.write(v + "\n")
        st.download_button("📥 Download TXT", data=txt_buf.getvalue().encode("utf-8"), file_name=f"{base_name}.txt", mime="text/plain")

        # PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, "Duplicate Detection Report", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 6, f"Generated: {timestamp} UTC", ln=True)
        pdf.ln(4)
        for p in pairs:
            pdf.multi_cell(0, 7, f"{p['file_a']}  vs  {p['file_b']}  =>  {p['similarity']}%  {'DUPLICATE' if p['duplicate'] else 'NOT DUPLICATE'}")
        if include_extracted_text_in_report:
            for k, v in texts.items():
                pdf.add_page()
                pdf.set_font("Arial", "B", 12)
                pdf.multi_cell(0, 7, f"---- {k} ----")
                pdf.set_font("Arial", size=10)
                pdf.multi_cell(0, 6, v[:20000] + ("\n\n[truncated]" if len(v) > 20000 else ""))
        pdf_bytes = pdf.output(dest='S').encode('latin-1', errors='ignore')
        st.download_button("📥 Download PDF", data=pdf_bytes, file_name=f"{base_name}.pdf", mime="application/pdf")

        # Email send
        if send_email_to:
            if not SMTP:
                st.warning("SMTP not set in Streamlit secrets. Add [smtp] credentials to send email.")
            else:
                if st.button("✉️ Send report via Email"):
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

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
st.markdown("<div class='app-card'>", unsafe_allow_html=True)
foot_col1, foot_col2 = st.columns([0.12, 1])
with foot_col1:
    st.image("https://raw.githubusercontent.com/uroojfatima1234567/placeholder/main/logo_small.png", width=64)
with foot_col2:
    st.markdown("**Developed by Urooj Fatima** • Built with 🤖 SentenceTransformers & Streamlit")
    st.caption("Note: For OCR on servers, Tesseract & Poppler may need to be installed. Locally set pytesseract.pytesseract.tesseract_cmd if needed.")
st.markdown("</div>", unsafe_allow_html=True)
