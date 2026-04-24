from pathlib import Path
from typing import List, Dict, Tuple
import re
from io import BytesIO
from pypdf import PdfReader

def load_document(data_dir: str) -> List[Dict]:
    # Returns list of dictionary (source, page, text): supports only .pdf and .txt
    
    docs = []
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    
    for p in sorted(root.glob("**/*")):
        if p.is_dir():
            continue
        if p.suffix.lower() == ".txt":
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({"source": str(p), "page": 0, "text": text})
        elif p.suffix.lower() == ".pdf":
            reader = PdfReader(str(p))
            for i, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                docs.append({"source": str(p), "page": i, "text": txt})
                
    return docs


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    # normalize whaitspace a bit before chunking
    
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def make_chunks(docs: List[Dict], chunk_size: int, overlap: int) -> List[Dict]:
   # Output chunk rows with metadata: {chunk_id, source, page, chunk_idx, text}
    
    out = []
    cid = 0
    for d in docs:
        parts = chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap)
        for j, ch in enumerate(parts):
            out.append({
                "chunk_id": cid,
                "source": d["source"],
                "page": d["page"],
                "chunk_idx": j,
                "text": ch,
            })
            cid += 1
    return out


def make_text_doc(text: str, source: str = "inline_text") -> List[Dict]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    return [{"source": source, "page": 0, "text": cleaned}]


def parse_uploaded_file(filename: str, content: bytes) -> List[Dict]:
    suffix = Path(filename or "").suffix.lower()

    if suffix == ".txt":
        text = content.decode("utf-8", errors="ignore")
        return make_text_doc(text, source=filename)

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        out: List[Dict] = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            if txt.strip():
                out.append({"source": filename, "page": i, "text": txt})
        return out

    raise ValueError("Unsupported file type. Only .txt and .pdf are allowed.")


# if __name__ == "__main__":
#     print("Ingest module loaded successfully")