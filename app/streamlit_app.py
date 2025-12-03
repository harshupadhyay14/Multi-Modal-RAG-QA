import sys
import os
import io
import streamlit as st
from PIL import Image

# -----------------------------------------
# FIX PYTHON PATH
# -----------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# -----------------------------------------
# IMPORT PROJECT MODULES
# -----------------------------------------
from multi_modal_rag.ingestion.pdf_ingest import extract_pdf
from multi_modal_rag.ingestion.ocr import ocr_try_best
from multi_modal_rag.chunking.chunker import chunk_item
from multi_modal_rag.embeddings.embedder import embed_texts
from multi_modal_rag.index.indexer import FaissIndexer
from multi_modal_rag.llm.generator import generate_answer


# -----------------------------------------
# STREAMLIT CONFIG
# -----------------------------------------
st.set_page_config(page_title="Multi-Modal RAG QA", layout="wide")
st.title("📘 Multi-Modal RAG QA System")


uploaded = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded is not None:

    
    # SAVE PDF CORRECTLY
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
         tmp.write(uploaded.read())
         temp_path = tmp.name


    # INGEST PDF USING PyMuPDF
 
    items = extract_pdf(temp_path)

 
    # OCR for IMAGE ITEMS (convert bytes → PIL)
    for it in items:
        if it["type"] == "image":
            try:
                pil_img = Image.open(io.BytesIO(it["content"]))
                it["metadata"]["ocr_text"] = ocr_try_best(pil_img)
            except Exception:
                it["metadata"]["ocr_text"] = ""

    # CHUNKING + STABLE META ID MAPPING
    chunks = []
    metas = []
    chunk_map = {}

    for it in items:
        for c in chunk_item(it):
            cid = len(chunk_map)
            chunk_map[cid] = c["text"]

            chunks.append(c["text"])
            metas.append({
                "id": cid,
                "page": c["page"]
            })

    # EMBEDDINGS + FAISS INDEX
    embs = embed_texts(chunks)
    index = FaissIndexer(dim=embs.shape[1])
    index.add(embs, metas)

    st.success("PDF processed successfully!")


    # QUESTION
    question = st.text_input("Ask a question about the PDF")

    if st.button("Answer") and question.strip():

        # 1) Embed Question
        q_emb = embed_texts([question])[0]

        # 2) Retrieve Top-K
        results = index.search(q_emb, top_k=5)

        # 3) Build Context For LLM
        context_items = []
        for meta, score in results:
            cid = meta["id"]
            context_items.append({
                "page": meta["page"],
                "text": chunk_map[cid]
            })

        # Debug (optional)
        # st.write("Context used:", context_items)

        # 4) Generate Answer via OpenAI v1 API
        answer = generate_answer(context_items, question)

        st.markdown("### 🎯 Answer")
        st.write(answer)
