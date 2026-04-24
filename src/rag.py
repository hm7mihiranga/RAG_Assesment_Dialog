
import os
import numpy as np
import pandas as pd
import faiss

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from sentence_transformers import CrossEncoder, SentenceTransformer
from google import genai


@dataclass
class FaissStore:
    index: faiss.Index
    meta: pd.DataFrame
    
def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

def build_faiss_store(chunks: List[Dict[str, Any]], embed_model_name: str, model: Optional[SentenceTransformer] = None, normalize: bool = True,) -> Tuple[FaissStore, SentenceTransformer]:
    if not chunks:
        raise ValueError("No chunks provided for vector store build")

    active_model = model or SentenceTransformer(embed_model_name)
    texts = [c["text"] for c in chunks]

    emb = active_model.encode(texts, batch_size=64, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)

    if normalize:
        emb = _l2_normalize(emb)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    meta = pd.DataFrame(chunks)
    return FaissStore(index=index, meta=meta), active_model

def save_vector(store: FaissStore, artifacts_dir: str):
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(store.index, str(out / "faiss.index"))
    store.meta.to_parquet(out / "meta.parquet", index=False)
            
            
def load_store(artifacts_dir: str) -> FaissStore:
    out = Path(artifacts_dir)
    index_path = out / "faiss.index"
    meta_path = out / "meta.parquet"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Vector store artifacts not found in '{artifacts_dir}'. Run ingestion first."
        )

    index = faiss.read_index(str(index_path))
    meta = pd.read_parquet(meta_path)
    return FaissStore(index=index, meta=meta)


def retrieve(query: str, store: FaissStore, model: SentenceTransformer, top_k: int = 5, normalize: bool = True,) -> List[Dict[str, Any]]:
    q = model.encode([query])
    q = np.asarray(q, dtype="float32")

    if normalize:
        q = _l2_normalize(q)

    capped_top_k = max(1, min(top_k, len(store.meta)))
    scores, ids = store.index.search(q, capped_top_k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results = []
    for score, idx in zip(scores, ids):
        if idx == -1:
            continue
        row = store.meta.iloc[idx].to_dict()
        row["score"] = float(score)
        results.append(row)

    return results


def rerank_chunks( query: str, candidates: List[Dict[str, Any]], top_k: int, reranker_model_name: str, reranker: Optional[CrossEncoder] = None, min_score: Optional[float] = None,) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    active_reranker = reranker or CrossEncoder(reranker_model_name)
    pairs = [(query, c.get("text", "")) for c in candidates]
    scores = active_reranker.predict(pairs)

    ranked: List[Dict[str, Any]] = []
    for candidate, score in zip(candidates, scores):
        row = dict(candidate)
        row["rerank_score"] = float(score)
        ranked.append(row)

    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    if min_score is not None:
        ranked = [r for r in ranked if r["rerank_score"] >= min_score]

    return ranked[:max(1, top_k)]

def _format_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, ch in enumerate(chunks, 1):
        rerank_part = ""
        if "rerank_score" in ch:
            rerank_part = f" rerank={ch.get('rerank_score', 0.0):.3f}"
        parts.append(
            f"[{i}] source={ch.get('source', 'unknown')} page={ch.get('page', 0)} vector_score={ch.get('score', 0.0):.3f}{rerank_part}\n{ch.get('text', '')}"
        )
    return "\n\n".join(parts)


# Conversational Memory Implementation
def build_retrieval_query(question: str, conversation_history: Optional[List[Dict[str, str]]] = None, max_turns: int = 3) -> str:
    if not conversation_history:
        return question
    
    recent = conversation_history[-max(1, max_turns):]
    prev_questions = [
        t.get("question", "").strip()
        for t in recent
        if t.get("question", "").strip()
    ]
    if not prev_questions:
        return question
    
    return("Previous user questions:\n" + "\n".join(f"- {q}" for q in prev_questions) + f"\nCurrent question:\n{question}")

def _format_conversation_history(conversation_history: Optional[List[Dict[str, str]]] = None, max_turns: int = 4) -> str:
    if not conversation_history:
        return "No prior conversation history."
    
    recent = conversation_history[-max(1, max_turns):]
    lines = []
    for i, turn in enumerate(recent, 1):
        q = turn.get("question", "").strip()
        a = turn.get("answer", "").strip()
        lines.append(f"Turns {i} Question: {q}")
        lines.append(f"Turns {i} Answer: {a}")
        
    return "\n".join(lines)



def answer_with_gemini(query: str, retrieved: List[Dict[str, Any]], google_api_key: str, gemini_model: str = "gemini-3.1-flash-lite-preview", conversation_history: Optional[List[Dict[str, str]]] = None, history_turns: int = 4,) -> Dict[str, Any]:
    if not retrieved:
        return {
            "query": query,
            "answer": "I don't know",
            "sources": [],
        }

    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is required to call Gemini")
    
    history_text = _format_conversation_history(
        conversation_history=conversation_history,
        max_turns=history_turns,
    )

    client = genai.Client(api_key=google_api_key)

    context = _format_context(retrieved)

    prompt = f"""
                You are a strict document question-answering assistant.
                Use ONLY the provided context.
                If the answer is not explicitly contained in the context, reply exactly: I don't know
                Do not use outside knowledge.
                Add citations like [1], [2] for statements you make.
                
                Conversation history:
                {history_text}

                Context:
                {context}

                Question:
                {query}

                Answer (with citations):
                """.strip()

    response = client.models.generate_content(
        model=gemini_model,
        contents=prompt,
    )
    text = (getattr(response, "text", "") or "").strip() or "I don't know"

    return {
        "query": query,
        "answer": text,
        "sources": [
            {
                "source": c["source"],
                "page": c.get("page", 0),
                "chunk_id": int(c["chunk_id"]),
                "score": float(c["score"]),
                "rerank_score": float(c["rerank_score"]) if "rerank_score" in c else None,
            }
            for c in retrieved
        ],
    }


# Backward-compatible aliases for older imports handling
def retrive(query: str, store: FaissStore, model: SentenceTransformer, top_k: int = 5, normalize: bool = True) -> List[Dict[str, Any]]:
    return retrieve(query=query, store=store, model=model, top_k=top_k, normalize=normalize)


def answere_with_gemini(query: str, retrieved: List[Dict[str, Any]], google_api_key: str, gemini_model: str = "gemini-3.1-flash-lite-preview") -> Dict[str, Any]:
    return answer_with_gemini(
        query=query,
        retrieved=retrieved,
        google_api_key=google_api_key,
        gemini_model=gemini_model,
    )



