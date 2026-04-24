from pathlib import Path

from pydantic import BaseModel
from dotenv import load_dotenv
import os

project_root = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=project_root / ".env", override=True)

def _clean(v: str | None) -> str:
    return (v or "").strip().strip('"').strip("'")

class Settings(BaseModel):
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    data_dir: str = "data/docs"
    artifacts_dir: str = "artifacts"

    chunk_size: int = 900          
    chunk_overlap: int = 150
    top_k: int = 5
    retrieval_pool_multiplier: int = 4

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_min_score: float | None = None
    gemini_model: str = "gemini-3.1-flash-lite-preview"
    
    # Conversational Memory Implementation
    memory_max_sessions: int = 200
    memory_turns_per_session: int = 8
    memory_query_turns: int = 3
    memory_prompt_turns: int = 4
    memory_context_turns: int = 3

settings = Settings()

# if __name__ == "__main__":
#     print("Configuration loaded successfully:")
    