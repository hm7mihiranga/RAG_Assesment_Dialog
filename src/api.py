import logging

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from transformers.utils import logging as hf_logging

from src.config import settings
from src.ingest import make_chunks, make_text_doc, parse_uploaded_file
from src.rag import (
    FaissStore,
    answer_with_gemini,
    build_faiss_store,
    load_store,
    rerank_chunks,
    retrieve,
    save_vector,
)
from sentence_transformers import CrossEncoder, SentenceTransformer

app = FastAPI(title="Document Q&A API (FAISS + Gemini)")

# Keep startup logs quiet for expected HuggingFace checkpoint key mismatches.
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

embedder = SentenceTransformer(settings.embed_model)
reranker: Optional[CrossEncoder] = None
store: Optional[FaissStore] = None
all_chunks: List[Dict[str, Any]] = []


def _init_store() -> None:
    global store, all_chunks
    try:
        store = load_store(settings.artifacts_dir)
        all_chunks = store.meta.to_dict("records")
    except FileNotFoundError:
        store = None
        all_chunks = []


_init_store()


def _get_reranker() -> CrossEncoder:
    global reranker
    if reranker is None:
        reranker = CrossEncoder(settings.rerank_model)
    return reranker


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]


class IngestResponse(BaseModel):
    docs_added: int
    chunks_added: int
    total_chunks: int


@app.get("/")
def root():
    return {
        "service": "Document Q&A API",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(all_chunks)}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: Request,
    file: Optional[UploadFile] = File(default=None),
    text: Optional[str] = Form(default=None),
    source: Optional[str] = Form(default=None),
):
    global store, all_chunks

    docs: List[Dict[str, Any]] = []
    request_source = source or "inline_text"
    request_text = text

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
        request_text = payload.get("text")
        request_source = payload.get("source", request_source)

    if request_text:
        docs.extend(make_text_doc(request_text, source=request_source))

    if file is not None:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file must have a filename")
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        try:
            docs.extend(parse_uploaded_file(file.filename, file_bytes))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not docs:
        raise HTTPException(
            status_code=400,
            detail="Provide either text (JSON/form) or a .txt/.pdf file",
        )

    new_chunks = make_chunks(docs, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    if not new_chunks:
        raise HTTPException(status_code=400, detail="No usable text could be extracted")

    base_chunk_id = len(all_chunks)
    for idx, chunk in enumerate(new_chunks):
        chunk["chunk_id"] = base_chunk_id + idx

    all_chunks.extend(new_chunks)
    store, _ = build_faiss_store(all_chunks, settings.embed_model, model=embedder)
    save_vector(store, settings.artifacts_dir)

    return IngestResponse(
        docs_added=len(docs),
        chunks_added=len(new_chunks),
        total_chunks=len(all_chunks),
    )

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if store is None or not all_chunks:
        raise HTTPException(status_code=400, detail="No documents ingested yet. Call /ingest first.")

    if not settings.google_api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is missing in environment")

    k = req.top_k or settings.top_k
    candidate_k = max(k, k * settings.retrieval_pool_multiplier)
    retrieved = retrieve(req.question, store=store, model=embedder, top_k=candidate_k)
    reranked = rerank_chunks(
        query=req.question,
        candidates=retrieved,
        top_k=k,
        reranker_model_name=settings.rerank_model,
        reranker=_get_reranker(),
        min_score=settings.rerank_min_score,
    )

    try:
        result = answer_with_gemini(
            query=req.question,
            retrieved=reranked,
            google_api_key=settings.google_api_key,
            gemini_model=settings.gemini_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc

    return result