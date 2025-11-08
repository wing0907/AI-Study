"""
데이터 변경 시 실행:
  python -m app.scripts.ingest
"""
from app.settings import settings
from app.retriever import build_or_load_vectorstore

if __name__ == "__main__":
    vs = build_or_load_vectorstore(settings.DATA_DIR, settings.VECTOR_DIR)
    print("✅ Vectorstore ready at:", settings.VECTOR_DIR)
