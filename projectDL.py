import os
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx
import pytesseract
from pdf2image import convert_from_path
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading

# ----------------- TESSERACT PATH -----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------- LOAD MODEL -----------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------- TEXT EXTRACTION -----------------
def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

def extract_text_from_pdf(path):
    text = ""
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    except:
        pages = convert_from_path(path)
        for i, page in enumerate(pages):
            img_path = f"page_{i}.jpg"
            page.save(img_path, "JPEG")
            text += extract_text_from_image(img_path)
            os.remove(img_path)
        return text

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(path):
    ext = path.lower().split(".")[-1]
    if ext == "pdf":
        return extract_text_from_pdf(path)
    elif ext == "docx":
        return extract_text_from_docx(path)
    elif ext in ["jpg", "jpeg", "png"]:
        return extract_text_from_image(path)
    return ""

# ---------------- EMBEDDINGS ----------------
def create_embedding(text):
    # ensure string
    if not isinstance(text, str):
        text = str(text)

    # remove illegal unicode
    text = text.encode("ascii", "ignore").decode()

    text = text.strip()

    # empty text return zero vector
    if text == "":
        return np.zeros((384,))

    # limit max length (fix tokenizer crash)
    text = text[:5000]

    return model.encode(text)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- FILE COMPARISON ----------------
def compare_files(file_paths, output_text, progress):
    texts, embeddings = {}, {}

    unique_files = [(f"{file_paths[i]}__copy{i}", file_paths[i]) for i in range(len(file_paths))]

    total_steps = len(unique_files) + len(unique_files)*(len(unique_files)-1)//2
    step_count = 0

    output_text.configure(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, "🔄 Extracting text from files...\n\n")
    output_text.update()

    # Extract text and embeddings
    for uid, real_path in unique_files:
        t = extract_text(real_path)
        texts[uid] = t
        embeddings[uid] = create_embedding(t)

        output_text.insert(tk.END, f"📄 Processed: {real_path}\n")
        step_count += 1
        progress['value'] = (step_count / total_steps) * 100
        progress_label.config(text=f"{int(progress['value'])}% Completed")
        root.update_idletasks()

    output_text.insert(tk.END, "\n🔍 Comparing files...\n\n")

    # Compare files
    for i in range(len(unique_files)):
        for j in range(i + 1, len(unique_files)):
            uid1, f1 = unique_files[i]
            uid2, f2 = unique_files[j]

            sim = cosine_similarity(embeddings[uid1], embeddings[uid2])
            percent = round(sim * 100, 2)

            output_text.insert(tk.END, f"\n🆚 Comparing:\n   {f1}\n   {f2}\n")
            if percent >= 90:
                output_text.insert(tk.END, f"➡ Similarity: {percent}%\n", "duplicate")
                output_text.insert(tk.END, "✅ DUPLICATE FILES\n", "duplicate")
            else:
                output_text.insert(tk.END, "❌ NOT Duplicate\n", "not_duplicate")

            step_count += 1
            progress['value'] = (step_count / total_steps) * 100
            progress_label.config(text=f"{int(progress['value'])}% Completed")
            root.update_idletasks()

    output_text.configure(state=tk.DISABLED)
    progress_label.config(text="✔ Completed")

# ---------------- GUI FUNCTIONS ----------------
def select_files():
    files = filedialog.askopenfilenames(title="Select Files")
    for f in files:
        file_listbox.insert(tk.END, f)

def remove_selected_files():
    selected = file_listbox.curselection()
    for index in reversed(selected):
        file_listbox.delete(index)

def start_comparison_thread():
    files = file_listbox.get(0, tk.END)

    if not files:
        messagebox.showwarning("Warning", "Please select files first!")
        return

    # ⭐ FIX: Prevent error when only one file is selected
    if len(files) < 2:
        messagebox.showerror("Error", "Please select at least TWO files to compare!")
        return

    threading.Thread(target=compare_files, args=(files, output_text, progress), daemon=True).start()

# ---------------- GUI DESIGN ----------------
root = tk.Tk()
root.title("📁 AI File Similarity Checker")
root.geometry("950x780")
root.configure(bg="#2c2f33")
root.resizable(False, False)

header = tk.Label(root, text="AI Powered Duplicate File Detector", bg="#23272a", fg="white",
                  font=("Calibri", 22, "bold"), pady=15)
header.pack(fill="x")

frame_top = tk.Frame(root, bg="#2c2f33")
frame_top.pack(pady=15)
button_style = {"font": ("Calibri", 12, "bold"), "bd": 0, "width": 16, "height": 1}

select_btn = tk.Button(frame_top, text="📂 Select Files", command=select_files,
                       bg="#7289da", fg="white", **button_style)
select_btn.pack(side=tk.LEFT, padx=10)

remove_btn = tk.Button(frame_top, text="🗑 Remove Selected", command=remove_selected_files,
                       bg="#f04747", fg="white", **button_style)
remove_btn.pack(side=tk.LEFT, padx=10)

compare_btn = tk.Button(frame_top, text="⚡ Compare Files", command=start_comparison_thread,
                        bg="#43b581", fg="white", **button_style)
compare_btn.pack(side=tk.LEFT, padx=10)

file_frame = tk.LabelFrame(root, text="Selected Files", bg="#2c2f33", fg="white",
                           font=("Calibri", 14, "bold"))
file_frame.pack(pady=10)

file_listbox = tk.Listbox(file_frame, width=110, height=7, bg="#23272a", fg="white",
                          selectmode=tk.MULTIPLE, font=("Consolas", 11))
file_listbox.pack(padx=10, pady=10)

progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=850, mode='determinate')
progress.pack(pady=6)
progress_label = tk.Label(root, text="0% Completed", bg="#2c2f33", fg="white", font=("Calibri", 12))
progress_label.pack()

output_text = scrolledtext.ScrolledText(root, width=110, height=25, bg="#1e2124", fg="white",
                                        insertbackground="white", font=("Consolas", 11))
output_text.pack(pady=15)
output_text.tag_config("duplicate", foreground="#43b581")
output_text.tag_config("not_duplicate", foreground="#f04747")
output_text.configure(state=tk.DISABLED)

footer = tk.Label(root, text="Developed by Urooj Fatima", bg="#23272a", fg="gray",
                  font=("Calibri", 10), pady=8)
footer.pack(fill="x")

root.mainloop()
