from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

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
    gemini_model: str = "gemini-3-flash-preview"
    
    # Conversational Memory Implementation
    memory_max_sessions: int = 200
    memory_turns_per_session: int = 8
    memory_query_turns: int = 3
    memory_prompt_turns: int = 4

settings = Settings()

if __name__ == "__main__":
    print("Configuration loaded successfully:")
    