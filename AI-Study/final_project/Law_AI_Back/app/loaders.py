from pathlib import Path
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader

def load_documents(path: str):
    """
    data/ 디렉터리 내 문서를 LangChain Documents로 로드
    확장자 필요시 추가 가능: .docx, .csv, .md 등
    """
    docs = []
    p = Path(path)
    for f in p.rglob("*"):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        try:
            if ext == ".pdf":
                docs.extend(PyPDFLoader(str(f)).load())
            elif ext in [".txt", ".md", ".html", ".csv", ".docx"]:
                docs.extend(UnstructuredFileLoader(str(f)).load())
            else:
                # 필요시 확장
                pass
        except Exception as e:
            print(f"[load skip] {f}: {e}")
    return docs
