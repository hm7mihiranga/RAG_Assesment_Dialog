import logging

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from transformers.utils import logging as hf_logging
from collections import OrderedDict, deque
from uuid import uuid4
from pathlib import Path

from src.config import settings
from src.ingest import make_chunks, make_text_doc, parse_uploaded_file
from src.rag import (
    FaissStore,
    answer_with_gemini,
    build_faiss_store,
    build_retrieval_query,
    load_store,
    rerank_chunks,
    retrieve,
    save_vector,
)
from sentence_transformers import CrossEncoder, SentenceTransformer

app = FastAPI(title="Document Q&A API (FAISS + Gemini)")

# HF Error Explanation, disable downloading progress bars, and access the sentence_transformers
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# This for Memory Store
conversation_memory: OrderedDict[str, deque[Dict[str, str]]] = OrderedDict()

embedder = SentenceTransformer(settings.embed_model)
reranker: Optional[CrossEncoder] = None
store: Optional[FaissStore] = None
all_chunks: List[Dict[str, Any]] = []


def _touch_session(conversation_id: Optional[str]) -> str:
    sid = (conversation_id or str(uuid4())).strip()

    if sid in conversation_memory:
        conversation_memory.move_to_end(sid)
        return sid

    if len(conversation_memory) >= settings.memory_max_sessions:
        conversation_memory.popitem(last=False)

    conversation_memory[sid] = deque(maxlen=settings.memory_turns_per_session)
    return sid



def _get_history(sid: str) -> List[Dict[str, str]]:
    return list(conversation_memory.get(sid, []))



def _append_history(sid: str, question: str, answer: str) -> None:
    if sid not in conversation_memory:
        _touch_session(sid)
    conversation_memory[sid].append({"question": question, "answer": answer})
    conversation_memory.move_to_end(sid)
    
    

def _history_to_temp_chunks(history: List[Dict[str, str]], max_turns: int) -> List[Dict[str, Any]]:
    if not history:
        return []
    
    recent = history[-max(1, max_turns):]
    out: List[Dict[str, Any]] = []
    for i, turn in enumerate(recent, 1):
        user_text = (turn.get("question") or "").strip()
        if not user_text:
            continue
        out.append({
            "source": "chat_memory",
            "page": 0,
            "chunk_id": -i,
            "chunk_idx": i,
            "score":0.0,
            "text":f"User prior turn {i}: {user_text}",
        })
    return out


def _delete_artifacts() -> None:
    out = Path(settings.artifacts_dir)
    for name in ("faiss.index", "meta.parquet"):
        p = out / name
        if p.exists():
            p.unlink()
            

def _reset_runtime_state(clear_artifacts: bool = False) -> None:
    global store, all_chunks
    store = None
    all_chunks = []
    conversation_memory.clear()
    if clear_artifacts:
        _delete_artifacts()


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
    conversation_id: Optional[str] = None


class AskResponse(BaseModel):
    conversation_id: str
    query: str
    answer: str
    sources: List[Dict[str, Any]]


class IngestResponse(BaseModel):
    docs_added: int
    chunks_added: int
    total_chunks: int
    
class SessionCloseResponse(BaseModel):
    status: str
    cleared_chunks: int
    cleared_sessions: int


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
    return {"status": "ok", "chunks": len(all_chunks), "sessions": len(conversation_memory)}


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


    for idx, chunk in enumerate(new_chunks):
        chunk["chunk_id"] = idx
        
    try:
        new_store, _ = build_faiss_store(new_chunks, settings.embed_model, model=embedder)
        save_vector(new_store, settings.artifacts_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build vector store: {exc}") from exc

    store = new_store
    all_chunks = new_chunks
    
    conversation_memory.clear()

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
    
    session_id = _touch_session(req.conversation_id)
    history = _get_history(session_id)

    k = req.top_k or settings.top_k
    candidate_k = max(k, k * settings.retrieval_pool_multiplier)
    
    retrieval_query = build_retrieval_query(
        question=req.question,
        conversation_history=history,
        max_turns=settings.memory_query_turns,
    )
    
    retrieved = retrieve(retrieval_query, store=store, model=embedder, top_k=candidate_k)
    
    temp_memory_chunks = _history_to_temp_chunks(
        history=history,
        max_turns=max(1, getattr(settings, "memory_context_turns", 3)),
    )
    candidates = retrieved + temp_memory_chunks
    
    reranked = rerank_chunks(
        query=req.question,
        candidates=candidates,
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
            conversation_history=history,
            history_turns=settings.memory_prompt_turns,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc
    
    _append_history(session_id, req.question, result["answer"])
    result["conversation_id"] = session_id

    return result

@app.post("/session/close", response_model=SessionCloseResponse)
def close_session():
    cleared_chunks = len(all_chunks)
    cleared_sessions = len(conversation_memory)
    _reset_runtime_state(clear_artifacts=True)
    return SessionCloseResponse(
        status="Cleared all session data and artifacts",
        cleared_chunks=cleared_chunks,
        cleared_sessions=cleared_sessions,
    )