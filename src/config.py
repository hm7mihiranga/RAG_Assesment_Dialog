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

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    gemini_model: str = "gemini-3-flash-preview"

settings = Settings()

if __name__ == "__main__":
    print("Configuration loaded successfully:")
    