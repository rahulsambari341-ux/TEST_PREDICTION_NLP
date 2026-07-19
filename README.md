Text-Extraction
📄 Smart PDF Summarizer with Error Detection & Highlighting
🔍 Project Overview
Smart PDF Summarizer is a real-world NLP application that allows users to upload PDF documents, automatically detects common PDF text extraction errors, highlights the exact error locations, cleans the text, and generates a fast abstractive summary using a Transformer-based model.

The project is built using Python, Streamlit, PyMuPDF, and Hugging Face Transformers and is designed to handle noisy PDFs effectively.

🎯 Key Features
📤 Upload PDF files via web UI
⚠️ Automatic detection of PDF text errors
🎯 Highlights exact error locations in extracted text
🧹 Text cleaning and preprocessing
⚡ Fast abstractive summarization using DistilBART
📝 Saves summary and detected errors to files
🌐 Interactive Streamlit web application
🧠 Errors Automatically Detected
The system identifies and reports the following issues:

Line breaks inside sentences
Multiple extra spaces
Broken hyphenated words
Hidden / non-ASCII characters
Very low text content
All detected errors are visually highlighted in the UI to show their exact positions.

🛠️ Technologies Used
Python
Streamlit (Web UI)
PyMuPDF (PDF text extraction)
Hugging Face Transformers
DistilBART (sshleifer/distilbart-cnn-12-6)
Regular Expressions (Regex)
Project Structure Smart_PDF_Summarizer/ ├── app.py ├── README.md ├── requirements.txt ├── sample_pdfs/ │ ├── clean_sample.pdf │ └── error_sample.pdf └── outputs/ ├── summary.txt └── detected_errors.txt

📁 Project Structure
