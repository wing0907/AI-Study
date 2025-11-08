# app/tools/chroma_merge.py
import os
from pathlib import Path
import chromadb

# ---------- 설정 ----------
# 소스 DB 루트들(= 현재 각 카테고리 폴더들)
SOURCE_ROOTS = [
    r"D:\Study25WJ\final_project\Law_AI_DB\chroma_dbs\apeals",
    r"D:\Study25WJ\final_project\Law_AI_DB\chroma_dbs\civil",
    r"D:\Study25WJ\final_project\Law_AI_DB\chroma_dbs\constitution",
    r"D:\Study25WJ\final_project\Law_AI_DB\chroma_dbs\constitutional",
    r"D:\Study25WJ\final_project\Law_AI_DB\chroma_dbs\criminal",
    r"D:\Study25WJ\final_project\Law_AI_DB\chroma_dbs\interpretations",
    r"D:\Study25WJ\final_project\Law_AI_DB\chroma_dbs\precedents",
]

# 타겟(통합) DB 루트
TARGET_ROOT = r"D:\Study25WJ\final_project\Law_AI_DB\chroma_dbs\lawai_unified"

# 타겟 컬렉션 이름(한 곳으로 몰아넣을 이름)
TARGET_COLLECTION = "law_docs"

# ---------- 유틸 ----------
def list_collections_with_counts(client: chromadb.PersistentClient):
    out = []
    for c in client.list_collections():
        try:
            out.append((c.name, c.count()))
        except Exception as e:
            out.append((c.name, f"error: {e}"))
    return out

def pull_all_docs(col):
    # 모든 id만 뽑은 뒤, batch로 documents/embeddings/metadata를 가져온다
    ids = col.get(include=[]).get("ids", [])
    print(f"  - ids: {len(ids)}")
    B = 1000
    for i in range(0, len(ids), B):
        chunk_ids = ids[i:i+B]
        data = col.get(ids=chunk_ids, include=["documents", "embeddings", "metadatas"])
        yield data

def ensure_unique_ids(ids, prefix):
    # 서로 다른 소스에서 같은 id가 충돌하지 않도록 prefix를 붙인다.
    return [f"{prefix}::{i}" for i in ids]

# ---------- 메인 ----------
def main():
    Path(TARGET_ROOT).mkdir(parents=True, exist_ok=True)
    tgt_client = chromadb.PersistentClient(path=TARGET_ROOT)
    tgt = tgt_client.get_or_create_collection(TARGET_COLLECTION)

    total_added = 0

    for src_root in SOURCE_ROOTS:
        if not Path(src_root, "chroma.sqlite3").exists() and not Path(src_root, "chroma.sqlite").exists():
            print(f"[skip] no chroma db at {src_root}")
            continue
        print(f"[merge] from {src_root}")
        src_client = chromadb.PersistentClient(path=src_root)
        cols = src_client.list_collections()
        if not cols:
            print("  - no collections")
            continue

        for col in cols:
            print(f"  - source collection: {col.name} ({col.count()})")
            for data in pull_all_docs(col):
                ids = data.get("ids", [])
                docs = data.get("documents", [])
                metas = data.get("metadatas", [])
                embs = data.get("embeddings", [])

                if not ids:
                    continue
                # id 충돌 방지 (소스 루트명 + 컬렉션명 prefix)
                prefix = Path(src_root).name + "::" + col.name
                new_ids = ensure_unique_ids(ids, prefix)

                # precomputed embeddings가 있으니 그대로 추가(재임베딩 불필요)
                tgt.add(
                    ids=new_ids,
                    documents=docs,
                    metadatas=metas,
                    embeddings=embs
                )
                total_added += len(new_ids)
                print(f"    + added {len(new_ids)} (total {total_added})")

    print("[done] merged to", TARGET_ROOT, "collection =", TARGET_COLLECTION)

    # 요약
    print("[target collections]", list_collections_with_counts(tgt_client))

if __name__ == "__main__":
    main()
