# app/routers/search.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.chains import build_chain

router = APIRouter()

class SearchIn(BaseModel):
    query: str
    k: int | None = 5

class SearchOut(BaseModel):
    answer: str
    sources: list[dict]

@router.post("/search", response_model=SearchOut)
def search(body: SearchIn):
    try:
        run = build_chain(k=body.k or 5)
        result = run(body.query)
        return SearchOut(**result)
    except Exception as e:
        # 500 대신 원인 파악 가능한 메시지로 던져줌
        raise HTTPException(status_code=500, detail=f"search_failed: {e}")
