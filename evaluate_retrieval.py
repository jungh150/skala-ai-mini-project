"""
Retrieval 성능 평가 스크립트
10-Retriever-Evaluation.ipynb 구조 기반

평가 방식:
1. arXiv 논문 청크에서 LLM이 질문 자동 생성 (Ground Truth)
2. 여러 Retriever를 동일한 평가셋으로 비교
3. Hit Rate@K (K=1,3,5), MRR 측정

사용 방법: python evaluate_retrieval.py
"""
import os
import json
import random
import pickle
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from typing import List, Dict, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

# ── 설정 ────────────────────────────────────────────────────────
CACHE_DIR = "data/cache"
EVAL_CACHE_PATH = "data/eval_dataset.json"
N_EVAL_SAMPLES = 20       # 평가 질문 수
MIN_CHUNK_LENGTH = 100    # 최소 청크 길이
K_VALUES = [1, 3, 5]      # Hit Rate@K
RANDOM_SEED = 42


# ── 1. 문서 로드 (기존 RAG 인덱스 활용) ──────────────────────────
def load_chunks() -> List[Document]:
    """기존 캐시된 arXiv 논문 청크 로드"""
    bm25_cache = "data/bm25_cache.pkl"
    if not os.path.exists(bm25_cache):
        print("캐시 없음 — app.py를 먼저 실행하여 인덱스를 생성하세요.")
        exit(1)

    with open(bm25_cache, "rb") as f:
        cache = pickle.load(f)

    docs = cache["docs"]
    # 각 청크에 고유 ID 부여
    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = i

    print(f"[로드] 총 {len(docs)}개 청크")
    return docs


# ── 2. LLM 기반 질문 자동 생성 ───────────────────────────────────
def generate_eval_dataset(chunks: List[Document]) -> List[Dict]:
    """청크 내용을 기반으로 LLM이 질문 자동 생성"""

    # 캐시 존재 시 재사용
    if os.path.exists(EVAL_CACHE_PATH):
        print(f"[질문 생성] 캐시 로드: {EVAL_CACHE_PATH}")
        with open(EVAL_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # 너무 짧은 청크 제외
    eligible = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LENGTH]

    random.seed(RANDOM_SEED)
    sampled = random.sample(eligible, min(N_EVAL_SAMPLES, len(eligible)))

    print(f"[질문 생성] 총 {len(sampled)}개 질문 생성 중...")

    prompt = PromptTemplate.from_template("""
다음은 반도체 기술 논문(PIM, CXL)의 일부 내용입니다.

<content>
{content}
</content>

위 내용을 읽고, 이 내용으로 답할 수 있는 자연스러운 질문 1개를 생성하세요.

조건:
- 질문은 반드시 위 내용에서 답을 찾을 수 있어야 합니다
- 구체적인 기술 용어나 수치를 포함하세요
- 반드시 아래 JSON 형식으로만 응답하세요

{{"question": "질문 내용"}}
""")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | JsonOutputParser()

    eval_dataset = []
    for i, chunk in enumerate(sampled):
        try:
            result = chain.invoke({"content": chunk.page_content[:800]})
            eval_dataset.append({
                "question": result["question"],
                "chunk_id": chunk.metadata["chunk_id"],
                "chunk_content": chunk.page_content[:200],
                "source_title": chunk.metadata.get("title", "Unknown"),
            })
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(sampled)} 완료")
        except Exception as e:
            print(f"  청크 {chunk.metadata.get('chunk_id', i)} 생성 실패: {e}")

    # 캐시 저장
    os.makedirs("data", exist_ok=True)
    with open(EVAL_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, ensure_ascii=False, indent=2)

    print(f"[질문 생성] 총 {len(eval_dataset)}개 완료 → {EVAL_CACHE_PATH} 저장")
    return eval_dataset


# ── 3. Retriever 구성 ────────────────────────────────────────────
def build_retrievers(chunks: List[Document], k: int = 5) -> Dict:
    """여러 Retriever를 동일한 청크로 구성"""

    print("[Retriever] 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # FAISS 인덱스 로드 or 생성
    faiss_path = "data/faiss_index"
    if os.path.exists(faiss_path):
        print("[Retriever] FAISS 인덱스 로드...")
        vectorstore = FAISS.load_local(
            faiss_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("[Retriever] FAISS 인덱스 생성...")
        vectorstore = FAISS.from_documents(chunks, embeddings)

    # BM25
    print("[Retriever] BM25 구성...")
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k

    retrievers = {
        "FAISS (Similarity)": vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        ),
        "FAISS (MMR)": vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "fetch_k": k * 3}
        ),
        "BM25": bm25,
        "Hybrid (FAISS+BM25)": EnsembleRetriever(
            retrievers=[
                vectorstore.as_retriever(search_kwargs={"k": k}),
                bm25
            ],
            weights=[0.6, 0.4]
        ),
    }

    print(f"[Retriever] {len(retrievers)}개 구성 완료: {list(retrievers.keys())}")
    return retrievers


# ── 4. 평가 함수 ─────────────────────────────────────────────────
def evaluate_retriever(
    retriever,
    eval_dataset: List[Dict],
    chunks: List[Document],
    k_values: List[int]
) -> Dict:
    """단일 Retriever 평가"""

    # chunk_id → chunk 매핑
    chunk_map = {doc.metadata["chunk_id"]: doc for doc in chunks}

    hit_counts = {k: 0 for k in k_values}
    rr_sum = 0.0

    for item in eval_dataset:
        question = item["question"]
        correct_id = item["chunk_id"]
        correct_chunk = chunk_map.get(correct_id)
        if not correct_chunk:
            continue

        try:
            retrieved = retriever.invoke(question)
        except Exception:
            continue

        # 정답 청크가 몇 번째에 나왔는지 확인
        rank = None
        for i, doc in enumerate(retrieved, 1):
            # chunk_id 또는 내용 유사도로 매칭
            if doc.metadata.get("chunk_id") == correct_id:
                rank = i
                break
            # chunk_id 없는 경우 내용 앞 50자로 매칭
            if doc.page_content[:50] == correct_chunk.page_content[:50]:
                rank = i
                break

        # Hit Rate@K
        for k in k_values:
            if rank is not None and rank <= k:
                hit_counts[k] += 1

        # MRR
        if rank is not None:
            rr_sum += 1.0 / rank

    n = len(eval_dataset)
    result = {f"Hit@{k}": hit_counts[k] / n for k in k_values}
    result["MRR"] = rr_sum / n
    return result


# ── 5. 메인 실행 ─────────────────────────────────────────────────
def evaluate():
    print("=" * 65)
    print("Retrieval 성능 평가 (10-Retriever-Evaluation 구조 기반)")
    print("=" * 65)

    # 청크 로드
    chunks = load_chunks()

    # 평가 데이터셋 생성 (LLM 자동 생성)
    eval_dataset = generate_eval_dataset(chunks)
    print(f"\n[평가] 질문 수: {len(eval_dataset)}개")
    print("[평가] 샘플 질문:")
    for item in eval_dataset[:3]:
        print(f"  - {item['question']}")
    print()

    # Retriever 구성
    retrievers = build_retrievers(chunks, k=max(K_VALUES))

    # 각 Retriever 평가
    results = {}
    for name, retriever in retrievers.items():
        print(f"[평가 중] {name}...")
        scores = evaluate_retriever(retriever, eval_dataset, chunks, K_VALUES)
        results[name] = scores
        hit_str = "  ".join(f"Hit@{k}={scores[f'Hit@{k}']:.3f}" for k in K_VALUES)
        print(f"  → {hit_str}  MRR={scores['MRR']:.3f}")

    # 결과 테이블
    results_df = pd.DataFrame(results).T
    results_df = results_df[[f"Hit@{k}" for k in K_VALUES] + ["MRR"]]
    results_df = results_df.sort_values("MRR", ascending=False)

    print("\n" + "=" * 65)
    print("최종 평가 결과")
    print("=" * 65)
    print(results_df.to_string(float_format="{:.4f}".format))

    best_mrr = results_df["MRR"].idxmax()
    best_hit1 = results_df["Hit@1"].idxmax()
    best_hit_max = results_df[f"Hit@{max(K_VALUES)}"].idxmax()

    print(f"\n- 최고 MRR:       {best_mrr} ({results_df.loc[best_mrr, 'MRR']:.4f})")
    print(f"- 최고 Hit@1:     {best_hit1} ({results_df.loc[best_hit1, 'Hit@1']:.4f})")
    print(f"- 최고 Hit@{max(K_VALUES)}: {best_hit_max} ({results_df.loc[best_hit_max, f'Hit@{max(K_VALUES)}']:.4f})")
    print("=" * 65)

    return results_df


if __name__ == "__main__":
    evaluate()
