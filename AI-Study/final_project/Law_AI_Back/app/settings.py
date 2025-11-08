# app/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # ---- Ollama ----
    OLLAMA_HOST: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "llama3.1"
    EMBED_MODEL: str = "nomic-embed-text"  # Ollama에 pull한 임베딩 모델 태그

    # ---- Chroma ----
    # 통합 디비 경로 (네가 알려준 최신 경로)
    VECTOR_DIR: str = r"D:\Study25WJ\final_project\Law_AI_DB\chroma_dbs\lawai_unified"
    CHROMA_COLLECTION: str = "law_docs"

    # ---- API / CORS ----
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8001
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
