# 📘 Multi-Modal RAG QA System  
A Retrieval-Augmented Generation (RAG) system that extracts **text**, **tables**, and **images** from PDFs, applies **OCR**, generates **embeddings**, indexes them using **FAISS**, and answers user questions using **Groq LLaMA 3.3** models.  
Built as part of the **Wasserstoff Gen-AI Internship Qualification Task**.

---

## 🚀 Features

### 🔍 **1. Multi-Modal PDF Ingestion**
- Extracts **text**, **tables**, and **images**  
- Supports scanned PDFs (OCR enabled)  
- Handles noisy PDFs gracefully  

### 🧠 **2. OCR Pipeline**
- Primary: **Paddle OCR** (optional, auto-disabled if unavailable)  
- Fallback: **Tesseract OCR** (installed via apt)  

### 🧩 **3. Chunking**
- Splits extracted content into overlapping, context-preserving chunks  
- Separate chunk logic for:
  - text blocks  
  - table data  
  - OCR outputs  

### 🔢 **4. Embeddings**
- Uses **Sentence Transformers** (`all-mpnet-base-v2`)  
- Generates dense vector embeddings for all chunks  

### 🔎 **5. Vector Search (FAISS)**
- Fast approximate nearest neighbor (ANN) search  
- Retrieves top-k relevant chunks based on cosine similarity  

### 🤖 **6. Groq LLM Answering**
- Uses **Groq LLaMA-3.3-70B-Versatile**  
- Produces grounded answers **only from document context**  
- Includes **page-level citations**  

### 🌐 **7. Streamlit UI**
- Upload PDF  
- See extraction status  
- Ask questions  
- Get citation-backed answers  

---

## 🏗️ System Architecture
# 📘 Multi-Modal RAG QA System  
A Retrieval-Augmented Generation (RAG) system that extracts **text**, **tables**, and **images** from PDFs, applies **OCR**, generates **embeddings**, indexes them using **FAISS**, and answers user questions using **Groq LLaMA 3.3** models.  
Built as part of the **Wasserstoff Gen-AI Internship Qualification Task**.

---

## 🚀 Features

### 🔍 **1. Multi-Modal PDF Ingestion**
- Extracts **text**, **tables**, and **images**  
- Supports scanned PDFs (OCR enabled)  
- Handles noisy PDFs gracefully  

### 🧠 **2. OCR Pipeline**
- Primary: **Paddle OCR** (optional, auto-disabled if unavailable)  
- Fallback: **Tesseract OCR** (installed via apt)  

### 🧩 **3. Chunking**
- Splits extracted content into overlapping, context-preserving chunks  
- Separate chunk logic for:
  - text blocks  
  - table data  
  - OCR outputs  

### 🔢 **4. Embeddings**
- Uses **Sentence Transformers** (`all-mpnet-base-v2`)  
- Generates dense vector embeddings for all chunks  

### 🔎 **5. Vector Search (FAISS)**
- Fast approximate nearest neighbor (ANN) search  
- Retrieves top-k relevant chunks based on cosine similarity  

### 🤖 **6. Groq LLM Answering**
- Uses **Groq LLaMA-3.3-70B-Versatile**  
- Produces grounded answers **only from document context**  
- Includes **page-level citations**  

### 🌐 **7. Streamlit UI**
- Upload PDF  
- See extraction status  
- Ask questions  
- Get citation-backed answers  

---

## 🏗️ System Architecture
PDF → Extract Text/Tables/Images → OCR → Chunking → Embeddings → FAISS Index
↓
Query Embedding
↓
Retrieve Relevant Chunks
↓
Groq LLM (LLaMA-3.3-70B) Generates Answer

## 📦 Folder Structure
Multi-Modal-RAG-QA/
│
├── app/
│ └── streamlit_app.py # Main Streamlit UI
│
├── multi_modal_rag/
│ ├── ingestion/
│ │ ├── pdf_ingest.py # PDF extraction pipeline
│ │ ├── ocr.py # OCR (Paddle + Tesseract fallback)
│ │ └── table_extractor.py # Table extraction
│ │
│ ├── chunking/
│ │ └── chunker.py # Chunking logic
│ │
│ ├── embeddings/
│ │ └── embedder.py # SentenceTransformer embeddings
│ │
│ ├── index/
│ │ └── indexer.py # FAISS index for retrieval
│ │
│ └── llm/
│ └── generator.py # Groq LLaMA answer generator
│
├── requirements.txt
├── apt.txt # System dependencies for HF Spaces
├── README.md
└── .streamlit/config.toml

---

## 🌐 Live Demo (Hugging Face Spaces)

🔗 **[Add your Spaces link here once deployed]**

---

## 🛠️ Installation & Running Locally

### 1️⃣ Clone the repo
```bash
git clone https://github.com/<harshupadhyay14>/Multi-Modal-RAG-QA.git
cd Multi-Modal-RAG-QA

2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit UI
streamlit run app/streamlit_app.py
