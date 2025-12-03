import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multi_modal_rag.ingestion.pdf_ingest import extract_pdf

items = extract_pdf(r"C:\Users\ASUS\Multi Modal Rag Project Scaffold\scripts\sample.pdf")
print("Extracted:", len(items))