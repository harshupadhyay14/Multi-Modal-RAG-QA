# app/gradio_app.py
import sys
import os

# Ensure project root is on sys.path so "multi_modal_rag.*" imports resolve
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import gradio as gr
from multi_modal_rag.ingestion.pdf_ingest import extract_pdf
# use the safe high-level OCR function (see step 2)
from multi_modal_rag.ingestion.ocr import ocr_try_best, ocr_tesseract
from multi_modal_rag.chunking.chunker import chunk_item
from multi_modal_rag.embeddings.embedder import embed_texts
from multi_modal_rag.index.indexer import FaissIndexer
from multi_modal_rag.llm.generator import generate_answer


# ----------- MAIN PIPELINE FUNCTION -------------- #

def process_pdf(file):
    """Ingest PDF → OCR → Chunk → Embed → Build FAISS index."""

    items = extract_pdf(file.name)
    
    # OCR for images
    for it in items:
        if it["type"] == "image":
            try:
                text = ocr_try_best(it["content"])
            except Exception:
                text = ocr_tesseract(it["content"])
            it["metadata"]["ocr_text"] = text

    # Chunking
    chunks = []
    metas = []
    for it in items:
        cs = chunk_item(it)
        for c in cs:
            chunks.append(c["text"])
            metas.append({
                "id": c["id"],
                "page": c["page"],
                "type": c["type"]
            })

    # Embeddings
    embeddings = embed_texts(chunks)
    index = FaissIndexer(dim=embeddings.shape[1])
    index.add(embeddings, metas)

    return chunks, metas, index


# ----------- QA FUNCTION AFTER PDF IS LOADED -------------- #

def answer_question(question, state):
    """Retrieve → Build context → Generate answer."""
    
    chunks, metas, index = state

    # Embed query
    q_emb = embed_texts([question])[0]

    # Retrieve top 5 chunks
    results = index.search(q_emb, top_k=5)

    context_items = []
    for meta, _ in results:
        idx = metas.index(meta)
        context_items.append({
            "page": meta["page"],
            "text": chunks[idx]
        })

    # Generate answer using LLM
    answer = generate_answer(context_items, question)

    # Build formatted retrieved snippets (for display)
    retrieved_text = ""
    for c in context_items:
        retrieved_text += f"(Page {c['page']}) {c['text'][:400]}...\n\n"

    return answer, retrieved_text


# ----------- GRADIO UI -------------- #

def load_pdf_ui(pdf_file):
    chunks, metas, index = process_pdf(pdf_file)
    return (chunks, metas, index), "PDF processed successfully! Ask questions now."


with gr.Blocks(title="Multi-Modal RAG QA System") as demo:

    gr.Markdown("## 📘 Multi-Modal RAG QA System (Gradio Version)")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        load_btn = gr.Button("Process PDF")

    status = gr.Textbox(label="Status")

    state = gr.State()

    load_btn.click(
        fn=load_pdf_ui,
        inputs=pdf_input,
        outputs=[state, status]
    )

    question = gr.Textbox(label="Ask a question")
    answer_btn = gr.Button("Ask")

    answer_box = gr.Textbox(label="Answer")
    retrieved_box = gr.Textbox(label="Retrieved Chunks (Context)", lines=12)

    answer_btn.click(
        fn=answer_question,
        inputs=[question, state],
        outputs=[answer_box, retrieved_box]
    )


demo.launch()
