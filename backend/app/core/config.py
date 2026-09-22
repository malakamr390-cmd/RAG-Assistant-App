from pathlib import Path
from pydantic_settings import BaseSettings

# Points to the project root (three levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

class Settings(BaseSettings):
    vector_store_path: str = str(PROJECT_ROOT / "data" / "vector_store")
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "lecture_chunks"
    llm_model: str = "llama3.2"
    top_k: int = 3
    cors_origins: list[str] = ["http://localhost:8501", "http://localhost:7860"]

    class Config:
        env_file = ".env"

settings = Settings()