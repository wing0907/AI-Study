# app/chains.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from app.settings import settings
from app.retriever import get_retriever


def _format_docs_for_context(docs: List[Document]) -> str:
    """LLM 컨텍스트로 넣을 요약 텍스트 생성 (본문 + 간단 메타)"""
    parts: List[str] = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("file_path") or meta.get("url") or ""
        page = meta.get("page") or meta.get("page_number") or ""
        header = f"[{i}] {src}{f' (p.{page})' if page != '' else ''}"
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n".join(parts)


def _format_sources_meta(docs: List[Document]) -> List[Dict[str, Any]]:
    """프론트에 내려줄 출처 배열(파일명/페이지/기타 메타 포함)"""
    out: List[Dict[str, Any]] = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        out.append({
            "index": i,
            "snippet": d.page_content[:500],   # 너무 길면 일부만
            "source": meta.get("source") or meta.get("file_path") or meta.get("url"),
            "page": meta.get("page") or meta.get("page_number"),
            "id": meta.get("id") or meta.get("doc_id"),
            "metadata": meta,
        })
    return out


def build_chain(k: int = 5):
    """
    RAG 체인(LLM에게 컨텍스트+질문을 주어 답변만 생성)
    - 라우터에서 chain.invoke({"question": "..."} ) 형태로 사용 가능
    - 답변 텍스트만 필요할 때 사용
    """
    retriever = get_retriever(k=k)

    system = (
        "당신은 한국 법률 문헌을 바탕으로 답하는 법률 AI 비서입니다. "
        "반드시 제공된 컨텍스트 내부에서만 근거를 사용하여 한국어로 간결하게 답하세요. "
        "컨텍스트에 없는 내용은 추측하지 말고 '제공된 자료에서 확인되지 않습니다'라고 답하세요."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        (
            "human",
            "질문: {question}\n\n"
            "다음은 검색된 관련 근거입니다:\n\n{context}\n\n"
            "위 근거만 활용해서 답변하세요."
        ),
    ])

    llm = ChatOllama(model=settings.LLM_MODEL, base_url=settings.OLLAMA_HOST)

    # retriever → docs → context 텍스트 → 프롬프트 → LLM → 문자열
    chain = (
        {
            "docs": retriever,                           # question을 그대로 retriever에 전달
            "question": RunnablePassthrough(),          # question 그대로 프롬프트로
        }
        | {
            "context": lambda x: _format_docs_for_context(x["docs"]),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def answer_with_sources(question: str, k: int = 5) -> Dict[str, Any]:
    """
    API에서 많이 쓰는 유틸:
    - retriever로 문서 가져와 컨텍스트 만들고
    - LLM으로 답변 생성
    - 함께 사용한 출처 메타도 반환
    """
    retriever = get_retriever(k=k)

    # 1) 관련 문서
    docs: List[Document] = retriever.get_relevant_documents(question)

    # 2) 컨텍스트 문자열
    context = _format_docs_for_context(docs)

    # 3) LLM 호출 (ChatOllama)
    llm = ChatOllama(model=settings.LLM_MODEL, base_url=settings.OLLAMA_HOST)

    system = (
        "당신은 한국 법률 문헌을 바탕으로 답하는 법률 AI 비서입니다. "
        "반드시 제공된 컨텍스트 내부에서만 근거를 사용하여 한국어로 간결하게 답하세요. "
        "컨텍스트에 없는 내용은 추측하지 말고 '제공된 자료에서 확인되지 않습니다'라고 답하세요."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human",
         "질문: {question}\n\n"
         "다음은 검색된 관련 근거입니다:\n\n{context}\n\n"
         "위 근거만 활용해서 답변하세요."),
    ])

    chain = prompt | llm | StrOutputParser()
    answer: str = chain.invoke({"question": question, "context": context})

    return {
        "answer": answer,
        "sources": _format_sources_meta(docs),
    }
