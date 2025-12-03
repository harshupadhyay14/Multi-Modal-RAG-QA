# chunking/chunker.py
import re

TARGET_WORDS = 240

def chunk_text(text):
    words = text.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= TARGET_WORDS:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def chunk_item(item):
    out = []
    if item['type'] == 'text':
        chunks = chunk_text(item['content'])
        for idx, c in enumerate(chunks):
            out.append({
                'id': f"{item['id']}_chunk{idx}",
                'text': c,
                'page': item['page'],
                'type': 'text'
            })

    elif item['type'] == 'table':
        out.append({
            'id': f"{item['id']}_table",
            'text': "TABLE:\n" + item['content'],
            'page': item['page'],
            'type': 'table'
        })

    elif item['type'] == 'image':
        ocr_text = item['metadata'].get('ocr_text', '')
        out.append({
            'id': f"{item['id']}_img",
            'text': ocr_text or f"Image on page {item['page']}",
            'page': item['page'],
            'type': 'image'
        })
    return out
