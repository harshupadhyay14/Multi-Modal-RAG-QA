"""
Improved PDF ingestion using PyMuPDF (fitz) + pdfplumber.
Safe, stable, and fully compatible with your RAG system.
"""
import logging
# Suppress all MuPDF warnings
logging.getLogger("fitz").setLevel(logging.ERROR)
logging.getLogger("pymupdf").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

import fitz  # PyMuPDF
import pdfplumber
import warnings
from .table_extractor import extract_tables_from_pdf, table_to_tsv_string

# Suppress pdfminer warnings
warnings.filterwarnings("ignore", message="Could get FontBBox")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Cannot set gray")
warnings.filterwarnings("ignore", message="invalid float")
warnings.filterwarnings("ignore", message=".*stroke color.*")
warnings.filterwarnings("ignore", message=".*non-stroke.*")
warnings.filterwarnings("ignore")


def extract_pdf(filepath, save_images=False):
    """
    Extract text, images, and tables from a PDF.
    Returns a list of items in the format:
    {
        'type': 'text' | 'image' | 'table',
        'content': <text or bytes or table_text>,
        'page': page_number,
        'id': unique_id,
        'metadata': { ... }
    }
    """

    items = []

    # 1) OPEN DOCUMENT
    try:
        doc = fitz.open(filepath)
    except Exception:
        return []  # return empty if PDF is corrupted

    num_pages = len(doc)

    # 2) TEXT EXTRACTION (PyMuPDF)
    for page_number in range(num_pages):
        page = doc[page_number]
        blocks = page.get_text("blocks")

        for i, block in enumerate(blocks):
            text = block[4]
            if text and text.strip():
                items.append({
                    "type": "text",
                    "content": text,
                    "page": page_number + 1,
                    "id": f"text_{page_number+1}_{i}",
                    "metadata": {}
                })

    # 3) IMAGE EXTRACTION (SAFE)
    for page_number in range(num_pages):
        page = doc[page_number]
        image_list = page.get_images()

        for i, img in enumerate(image_list):
            xref = img[0]

            try:
                pix = fitz.Pixmap(doc, xref)

                # Skip invalid pixmaps
                if pix.width == 0 or pix.height == 0:
                    continue

                # CMYK -> RGB
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_bytes = pix.tobytes("png")

            except Exception:
                continue  # skip corrupted image safely

            items.append({
                "type": "image",
                "content": img_bytes,
                "page": page_number + 1,
                "id": f"img_{page_number+1}_{i}",
                "metadata": {}
            })

            if save_images:
                with open(f"image_{page_number+1}_{i}.png", "wb") as f:
                    f.write(img_bytes)

    doc.close()

    # 4) TABLE EXTRACTION (pdfplumber)
    try:
        tables = extract_tables_from_pdf(filepath)
        for t_index, (page_num, df) in enumerate(tables):
            tsv = table_to_tsv_string(df)

            items.append({
                "type": "table",
                "content": tsv,
                "page": page_num,
                "id": f"table_{page_num}_{t_index}",
                "metadata": {}
            })
    except Exception:
        pass

    return items
