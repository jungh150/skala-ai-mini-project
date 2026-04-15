"""
evaluate_embedding.py
─────────────────────────────────────────────────────────────────────
임베딩 모델 비교 평가 스크립트

평가 대상 (Retriever: FAISS Similarity 고정):
  1. multilingual-e5-large   — HuggingFace (한/영 혼재 범용, max 512 tokens)
  2. bge-m3                  — HuggingFace (다국어 SOTA, max 8192 tokens)
  3. ko-sroberta-multitask   — HuggingFace (한국어 특화, max 512 tokens)
  4. text-embedding-3-small  — OpenAI API  (max 8191 tokens)
  5. jina-embeddings-v3      — HuggingFace (다국어, max 8192 tokens)
  6. voyage-3-large          — Voyage AI API

평가 지표:
  - Hit Rate@K (K=1, 3, 5)
  - MRR (Mean Reciprocal Rank)

API 키 설정 (.env):
  OPENAI_API_KEY  = sk-...   (text-embedding-3-small)
  VOYAGE_API_KEY  = pa-...   (voyage-3-large)

API 키 미설정 모델은 자동으로 건너뜁니다.

사전 조건:
  app.py를 먼저 실행하여 data/bm25_cache.pkl 생성 필요
  평가셋은 evaluate_retriever.py와 data/eval_dataset.json을 공유합니다.

사용 방법:
  uv run python evaluate_embedding.py
  uv run python evaluate_embedding.py --reset-eval   # 평가셋 캐시 초기화
"""

import os
import json
import random
import pickle
import hashlib
import argparse
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from typing import List, Dict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# ── 설정 ──────────────────────────────────────────────────────────────────────
EVAL_CACHE_PATH = "data/eval_dataset.json"
N_EVAL_SAMPLES  = 20
MIN_CHUNK_LEN   = 100
K_VALUES        = [1, 3, 5]
RANDOM_SEED     = 42
RETRIEVER_K     = max(K_VALUES) + 5   # 10

# ── 임베딩 모델 정의 ──────────────────────────────────────────────────────────

EMBEDDING_MODELS: List[Dict] = [
    {
        "name":     "multilingual-e5-large",
        "model_id": "intfloat/multilingual-e5-large",
        "backend":  "huggingface",
        "note":     "한/영 혼재 문서에 강함, 범용성 높음 (max 512 tokens)",
    },
    {
        "name":     "bge-m3",
        "model_id": "BAAI/bge-m3",
        "backend":  "huggingface",
        "note":     "다국어 SOTA, 긴 문서 처리 우수 (max 8192 tokens)",
    },
    {
        "name":     "ko-sroberta-multitask",
        "model_id": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        "backend":  "huggingface",
        "note":     "한국어 특화, 영문 혼재 시 성능 저하 가능 (max 512 tokens)",
    },
    {
        "name":     "text-embedding-3-small",
        "model_id": "text-embedding-3-small",
        "backend":  "openai",
        "note":     "OpenAI 제공, 안정적 성능 (max 8191 tokens) — OPENAI_API_KEY 필요",
    },
    {
        "name":     "jina-embeddings-v3",
        "model_id": "jinaai/jina-embeddings-v3",
        "backend":  "huggingface",
        "note":     "Jina AI, 다국어 지원 (max 8192 tokens) — trust_remote_code=True 필요",
    },
    {
        "name":     "voyage-3-large",
        "model_id": "voyage-3-large",
        "backend":  "voyageai",
        "note":     "Voyage AI API, 고성능 범용 임베딩 — VOYAGE_API_KEY 필요",
    },
]


# ── 1. 청크 로드 ──────────────────────────────────────────────────────────────

def load_chunks() -> List[Document]:
    """
    기존 캐시된 arXiv 논문 청크를 로드합니다.

    chunk_id를 url + page + 내용 앞 80자의 MD5 해시로 부여합니다.
    enumerate 인덱스 대신 해시를 사용하는 이유:
      캐시 재생성 시 청크 순서가 달라지면 eval_dataset.json의
      정답 chunk_id가 실제 청크와 어긋나는 문제를 방지하기 위함입니다.
    """
    bm25_cache = "data/bm25_cache.pkl"
    if not os.path.exists(bm25_cache):
        print("캐시 없음 — app.py를 먼저 실행하여 인덱스를 생성하세요.")
        exit(1)

    with open(bm25_cache, "rb") as f:
        cache = pickle.load(f)

    docs = cache["docs"]
    for doc in docs:
        raw = (
            doc.metadata.get("url", "")
            + str(doc.metadata.get("page", ""))
            + doc.page_content[:80]
        )
        doc.metadata["chunk_id"] = hashlib.md5(raw.encode()).hexdigest()[:12]

    print(f"[로드] 총 {len(docs)}개 청크")
    return docs


# ── 2. LLM 기반 평가 질문 자동 생성 ──────────────────────────────────────────

def generate_eval_dataset(chunks: List[Document]) -> List[Dict]:
    """
    청크 내용을 기반으로 LLM이 평가 질문을 자동 생성합니다.
    data/eval_dataset.json에 캐시되며, evaluate_retriever.py와 공유됩니다.
    """
    if os.path.exists(EVAL_CACHE_PATH):
        print(f"[질문 생성] 캐시 로드: {EVAL_CACHE_PATH}")
        with open(EVAL_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    eligible = [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_LEN]
    random.seed(RANDOM_SEED)
    sampled = random.sample(eligible, min(N_EVAL_SAMPLES, len(eligible)))

    print(f"[질문 생성] {len(sampled)}개 질문 생성 중...")

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

    llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | JsonOutputParser()

    eval_dataset = []
    for i, chunk in enumerate(sampled):
        try:
            result = chain.invoke({"content": chunk.page_content[:800]})
            eval_dataset.append({
                "question":      result["question"],
                "chunk_id":      chunk.metadata["chunk_id"],
                "chunk_content": chunk.page_content[:200],
                "source_title":  chunk.metadata.get("title", "Unknown"),
            })
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(sampled)} 완료")
        except Exception as e:
            print(f"  청크 {chunk.metadata.get('chunk_id', i)} 생성 실패: {e}")

    os.makedirs("data", exist_ok=True)
    with open(EVAL_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, ensure_ascii=False, indent=2)

    print(f"[질문 생성] {len(eval_dataset)}개 완료 → {EVAL_CACHE_PATH}")
    return eval_dataset


# ── 3. 임베딩 모델 로더 ───────────────────────────────────────────────────────

def load_embedding_model(model_cfg: Dict) -> Optional[Embeddings]:
    """
    임베딩 모델 설정을 받아 LangChain Embeddings 인스턴스를 반환합니다.
    API 키 미설정 또는 패키지 미설치 시 None을 반환하고 해당 모델을 건너뜁니다.
    """
    name     = model_cfg["name"]
    backend  = model_cfg["backend"]
    model_id = model_cfg["model_id"]

    try:
        if backend == "huggingface":
            kwargs = {
                "model_name":    model_id,
                "model_kwargs":  {"device": "cpu"},
                "encode_kwargs": {"normalize_embeddings": True},
            }
            if "jina" in model_id:
                # jina-embeddings-v3는 커스텀 모델 코드를 사용하므로 필요
                kwargs["model_kwargs"]["trust_remote_code"] = True
            return HuggingFaceEmbeddings(**kwargs)

        elif backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                print(f"  [{name}] OPENAI_API_KEY 없음 → 건너뜀")
                return None
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model_id, api_key=api_key)

        elif backend == "voyageai":
            api_key = os.getenv("VOYAGE_API_KEY", "")
            if not api_key:
                print(f"  [{name}] VOYAGE_API_KEY 없음 → 건너뜀")
                return None
            try:
                from langchain_voyageai import VoyageAIEmbeddings
            except ImportError:
                print(f"  [{name}] langchain-voyageai 미설치 → 건너뜀")
                print("        설치: uv add langchain-voyageai")
                return None
            return VoyageAIEmbeddings(model=model_id, voyage_api_key=api_key)

        else:
            print(f"  [{name}] 알 수 없는 backend: {backend} → 건너뜀")
            return None

    except Exception as e:
        print(f"  [{name}] 로드 실패: {e} → 건너뜀")
        return None


# ── 4. FAISS Retriever 구성 ───────────────────────────────────────────────────

def build_faiss_retriever(
    chunks: List[Document],
    embeddings: Embeddings,
    k: int = RETRIEVER_K,
    cache_suffix: str = "",
) -> Optional[object]:
    """
    지정된 임베딩 모델로 FAISS 인덱스를 구성하고 Similarity Retriever를 반환합니다.
    모델마다 별도 캐시 디렉토리(data/faiss_emb_{cache_suffix})를 사용합니다.
    이미 생성된 인덱스는 재사용하여 임베딩 비용을 절약합니다.

    Retriever 기법을 FAISS Similarity로 고정하는 이유:
      임베딩 모델 간 순수 성능 차이를 측정하기 위해
      Retrieval 기법 변수를 통제해야 하기 때문입니다.
      MMR이나 Hybrid를 사용하면 Retrieval 기법 차이가 섞여
      어느 게 임베딩 성능인지 알 수 없게 됩니다.
    """
    faiss_cache = f"data/faiss_emb_{cache_suffix}"
    try:
        if os.path.exists(faiss_cache):
            print(f"    FAISS 캐시 로드: {faiss_cache}")
            vectorstore = FAISS.load_local(
                faiss_cache, embeddings, allow_dangerous_deserialization=True
            )
        else:
            print(f"    FAISS 인덱스 생성 중 (캐시: {faiss_cache})...")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local(faiss_cache)
        return vectorstore.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        print(f"    FAISS 인덱스 생성 실패: {e}")
        return None


# ── 5. 평가 함수 ──────────────────────────────────────────────────────────────

def _match_rank(
    retrieved: List[Document],
    correct_id: str,
    correct_chunk: Document,
) -> Optional[int]:
    """
    검색 결과에서 정답 청크의 순위(1-based)를 반환합니다.

    매칭 우선순위:
      1. chunk_id 일치 (MD5 해시 기반, 가장 정확)
      2. 내용 앞 50자 일치 (chunk_id 없는 경우 fallback)
    """
    prefix = correct_chunk.page_content[:50]
    for rank, doc in enumerate(retrieved, 1):
        if doc.metadata.get("chunk_id") == correct_id:
            return rank
        if doc.page_content[:50] == prefix:
            return rank
    return None


def evaluate_retriever(
    retriever,
    eval_dataset: List[Dict],
    chunks: List[Document],
    k_values: List[int] = K_VALUES,
) -> Dict:
    """
    단일 Retriever를 평가하여 Hit Rate@K와 MRR을 반환합니다.

    Returns:
        {"Hit@1": float, "Hit@3": float, "Hit@5": float, "MRR": float, "n_valid": int}
    """
    chunk_map   = {doc.metadata["chunk_id"]: doc for doc in chunks}
    hit_counts  = {k: 0 for k in k_values}
    rr_sum      = 0.0
    valid_count = 0

    for item in eval_dataset:
        correct_id  = item["chunk_id"]
        correct_doc = chunk_map.get(correct_id)
        if not correct_doc:
            continue

        try:
            retrieved = retriever.invoke(item["question"])
        except Exception as e:
            print(f"    retriever.invoke 실패: {e}")
            continue

        valid_count += 1
        rank = _match_rank(retrieved, correct_id, correct_doc)

        for k in k_values:
            if rank is not None and rank <= k:
                hit_counts[k] += 1
        if rank is not None:
            rr_sum += 1.0 / rank

    n      = max(valid_count, 1)
    result = {f"Hit@{k}": round(hit_counts[k] / n, 4) for k in k_values}
    result["MRR"]     = round(rr_sum / n, 4)
    result["n_valid"] = valid_count
    return result


# ── 6. 결과 출력 ──────────────────────────────────────────────────────────────

def _make_result_df(results: Dict) -> pd.DataFrame:
    rows = {}
    for name, scores in results.items():
        row            = {f"Hit@{k}": scores.get(f"Hit@{k}") for k in K_VALUES}
        row["MRR"]     = scores.get("MRR")
        row["n_valid"] = scores.get("n_valid", 0)
        row["status"]  = scores.get("status", "")
        rows[name]     = row

    df   = pd.DataFrame(rows).T
    cols = [f"Hit@{k}" for k in K_VALUES] + ["MRR", "n_valid", "status"]
    df   = df[cols]
    for col in [f"Hit@{k}" for k in K_VALUES] + ["MRR"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("MRR", ascending=False, na_position="last")


def _print_result_table(df: pd.DataFrame) -> None:
    print(f"\n{'=' * 65}")
    print("임베딩 모델 비교 결과 (Retriever: FAISS Similarity 고정)")
    print("=" * 65)
    display = df.copy()
    for col in [f"Hit@{k}" for k in K_VALUES] + ["MRR"]:
        display[col] = display[col].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
        )
    print(display.to_string())
    print("=" * 65)

    valid = df.dropna(subset=["MRR"])
    if valid.empty:
        return
    for metric in [f"Hit@{k}" for k in K_VALUES] + ["MRR"]:
        best_name  = valid[metric].idxmax()
        best_score = valid.loc[best_name, metric]
        print(f"  최고 {metric:<8}: {best_name} ({best_score:.4f})")

    print("\n[설계서 기준 평가]")
    print("  목표: Hit@3 ≥ 0.75, MRR ≥ 0.50")
    for name, row in valid.iterrows():
        hit3    = row["Hit@3"]
        mrr     = row["MRR"]
        hit3_ok = "✅" if hit3 >= 0.75 else "❌"
        mrr_ok  = "✅" if mrr  >= 0.50 else "❌"
        print(f"  {name:<35} Hit@3={hit3:.3f}{hit3_ok}  MRR={mrr:.3f}{mrr_ok}")


# ── 7. 메인 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="임베딩 모델 비교 평가 (Retriever: FAISS Similarity 고정)",
    )
    parser.add_argument(
        "--reset-eval",
        action="store_true",
        help="평가 질문 캐시(eval_dataset.json)를 삭제하고 재생성",
    )
    args = parser.parse_args()

    if args.reset_eval and os.path.exists(EVAL_CACHE_PATH):
        os.remove(EVAL_CACHE_PATH)
        print(f"[초기화] {EVAL_CACHE_PATH} 삭제 완료")

    print("=" * 65)
    print("임베딩 모델 비교 평가")
    print(f"Retriever: FAISS Similarity 고정  |  평가 질문 수: {N_EVAL_SAMPLES}  |  K: {K_VALUES}")
    print("=" * 65)

    chunks       = load_chunks()
    eval_dataset = generate_eval_dataset(chunks)

    print(f"\n[평가셋] 질문 수: {len(eval_dataset)}개")
    print("[평가셋] 샘플 질문 (3개):")
    for item in eval_dataset[:3]:
        print(f"  Q: {item['question']}")
        print(f"     출처: {item['source_title']}")

    results = {}
    print()

    for model_cfg in EMBEDDING_MODELS:
        name = model_cfg["name"]
        print(f"[모델 로드] {name}")
        print(f"  ℹ  {model_cfg['note']}")

        embeddings = load_embedding_model(model_cfg)
        if embeddings is None:
            results[name] = {
                f"Hit@{k}": None for k in K_VALUES
            } | {"MRR": None, "n_valid": 0, "status": "건너뜀 (API 키 없음 또는 로드 실패)"}
            continue

        cache_suffix = name.replace("/", "_").replace("-", "_").lower()
        retriever    = build_faiss_retriever(
            chunks, embeddings, k=RETRIEVER_K, cache_suffix=cache_suffix
        )
        if retriever is None:
            results[name] = {
                f"Hit@{k}": None for k in K_VALUES
            } | {"MRR": None, "n_valid": 0, "status": "건너뜀 (FAISS 인덱스 생성 실패)"}
            continue

        print(f"  [평가 중] {name}...")
        scores           = evaluate_retriever(retriever, eval_dataset, chunks, K_VALUES)
        scores["status"] = "완료"
        results[name]    = scores

        hit_str = "  ".join(f"Hit@{k}={scores[f'Hit@{k}']:.3f}" for k in K_VALUES)
        print(f"  → {hit_str}  MRR={scores['MRR']:.3f}  (유효 질문={scores['n_valid']})")
        print()

    df = _make_result_df(results)
    _print_result_table(df)

    os.makedirs("data", exist_ok=True)
    out_path = "data/embedding_eval_results.csv"
    df.to_csv(out_path, encoding="utf-8-sig")
    print(f"\n[저장] {out_path}")
    print("\n완료.")


if __name__ == "__main__":
    main()
