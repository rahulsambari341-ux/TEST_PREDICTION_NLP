import streamlit as st
import fitz  # PyMuPDF
import re
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ===============================
# Load FAST summarization model
# ===============================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "sshleifer/distilbart-cnn-12-6"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "sshleifer/distilbart-cnn-12-6"
    )
    return tokenizer, model

tokenizer, model = load_model()


# ===============================
# PDF Text Extraction
# ===============================
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# ===============================
# ERROR DETECTION
# ===============================
def detect_errors(text):
    errors = []

    if "\n" in text:
        errors.append("Line breaks found inside sentences")

    if re.search(r"\s{2,}", text):
        errors.append("Multiple extra spaces detected")

    if re.search(r"-\s*\n", text):
        errors.append("Broken hyphenated words detected")

    if re.search(r"[^\x00-\x7F]+", text):
        errors.append("Hidden / non-ASCII characters detected")

    if len(text.strip()) < 300:
        errors.append("Very less text content detected")

    return errors


# ===============================
# TEXT CLEANING
# ===============================
def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"-\s+", "", text)              # fix broken words
    text = re.sub(r"[^\x00-\x7F]+", " ", text)    # remove hidden chars
    text = re.sub(r"\s+", " ", text)
    return text


# ===============================
# CHUNKING
# ===============================
def chunk_text(text, chunk_size=700):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


# ===============================
# SUMMARIZATION
# ===============================
def summarize_text(chunks):
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        ids = model.generate(
            inputs["input_ids"],
            max_length=120,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer.decode(
            ids[0],
            skip_special_tokens=True
        )
        summaries.append(summary)

    return " ".join(summaries)


# ===============================
# ERROR HIGHLIGHTING
# ===============================
def highlight_errors(text):
    highlighted = text

    # Extra spaces
    highlighted = re.sub(
        r"( {2,})",
        r"<mark style='background-color:#ffcccc'>\1</mark>",
        highlighted
    )

    # Broken hyphenated words
    highlighted = re.sub(
        r"(\w+-\s+\w+)",
        r"<mark style='background-color:#cce5ff'>\1</mark>",
        highlighted
    )

    # Line breaks
    highlighted = re.sub(
        r"(\n)",
        r"<mark style='background-color:#fff3cd'>⏎</mark>\n",
        highlighted
    )

    # Hidden characters
    highlighted = re.sub(
        r"([^\x00-\x7F]+)",
        r"<mark style='background-color:#e0ccff'>\1</mark>",
        highlighted
    )

    return highlighted


# ===============================
# SAVE OUTPUT FILES
# ===============================
def save_outputs(summary, errors):
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    with open("outputs/detected_errors.txt", "w", encoding="utf-8") as f:
        if errors:
            for e in errors:
                f.write("- " + e + "\n")
        else:
            f.write("No major errors detected in PDF.")


# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Smart PDF Summarizer", layout="centered")

st.title("📄 Smart PDF Summarization System")
st.write(
    "Upload a PDF. The system detects text errors, "
    "highlights their exact locations, cleans the text, "
    "and generates a fast summary."
)

pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf:
    if st.button("Analyze & Summarize"):
        with st.spinner("Processing PDF..."):
            raw_text = extract_text(pdf)
            errors = detect_errors(raw_text)
            cleaned_text = clean_text(raw_text)
            chunks = chunk_text(cleaned_text)
            summary = summarize_text(chunks)
            save_outputs(summary, errors)

        st.success("Processing Complete!")

        # -----------------------------
        # Errors List
        # -----------------------------
        st.subheader("⚠️ Detected PDF Issues")
        if errors:
            for e in errors:
                st.write("•", e)
        else:
            st.write("✅ No major errors detected in PDF")

        # -----------------------------
        # Highlighted Error Locations
        # -----------------------------
        st.subheader("🧩 Highlighted Error Locations (Raw Extracted Text)")
        highlighted_text = highlight_errors(raw_text)

        st.markdown(
            f"<div style='white-space: pre-wrap; font-size:14px;'>{highlighted_text}</div>",
            unsafe_allow_html=True
        )

        # -----------------------------
        # Summary
        # -----------------------------
        st.subheader("📌 Clean Summary")
        st.write(summary)

        st.info(
            "Summary and detected errors are saved "
            "inside the `outputs/` folder."
        )
