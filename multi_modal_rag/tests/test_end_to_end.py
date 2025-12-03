from ingestion.pdf_ingest import extract_pdf

def test_extract():
    items = extract_pdf("sample.pdf")
    assert isinstance(items, list)
