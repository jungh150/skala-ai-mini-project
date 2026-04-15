"""
RAG Agent: 로컬 문서(arXiv 논문)에서 관련 청크 검색
- Embedding: bge-m3 (BAAI/bge-m3)
- Vector DB: FAISS
- Retrieval: Hybrid (Dense + BM25 Sparse)
- Query Rewrite 후 재검색 (max 2회)

【Dense 점수 정규화 수정】
기존 코드: normalized = 1 - (score / max_dense)  → 역변환 적용
문제: HuggingFaceEmbeddings(normalize_embeddings=True) + FAISS IndexFlatL2 조합에서
     similarity_search_with_score()가 반환하는 score는 L2 거리(낮을수록 유사)이므로
     역변환(1 - score/max)이 맞다.
     그러나 normalize_embeddings=True이면 벡터가 단위구로 정규화되어
     L2 거리와 코사인 유사도가 단조 관계를 갖는다.
     실제로 FAISS 기본 인덱스(IndexFlatL2)는 L2 거리를 반환하므로
     낮은 score = 유사도 높음 → 역변환 방향이 맞다.

     다만 기존 코드의 정규화식 `1 - (score / max_dense)`에서
     max_dense 자체가 가장 유사도가 낮은(거리가 먼) 결과의 score이므로
     정규화 범위가 [0, 1]로 보장되지 않는 경우가 있다.

     수정: min-max 정규화를 적용하여 Dense 점수를 [0, 1]로 안정적으로 변환한다.
          정규화 후 유사할수록(L2 거리 작을수록) 높은 값이 되도록 역변환을 유지한다.
"""
import os
import pickle
import hashlib
from typing import List, Dict, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI

from agents.state import AgentState

# arXiv 논문 URL 목록
ARXIV_PDFS = [
    {
        "url": "https://arxiv.org/pdf/2012.03112",
        "title": "A Modern Primer on Processing-In-Memory",
        "category": "PIM",
        "source_type": "논문"
    },
    {
        "url": "https://arxiv.org/pdf/2105.03814",
        "title": "Benchmarking a Real PIM Architecture — UPMEM",
        "category": "PIM",
        "source_type": "논문"
    },
    {
        "url": "https://arxiv.org/pdf/2502.07578",
        "title": "PIM Is All You Need: CXL+PIM으로 GPU 없이 LLM 추론",
        "category": "PIM",
        "source_type": "논문"
    },
    {
        "url": "https://arxiv.org/pdf/2511.14400",
        "title": "PIM or CXL-PIM? 아키텍처 트레이드오프 비교",
        "category": "PIM",
        "source_type": "논문"
    },
    {
        "url": "https://arxiv.org/pdf/2412.20249",
        "title": "Next-Gen Computing Systems with CXL — 종합 서베이",
        "category": "CXL",
        "source_type": "논문"
    },
    {
        "url": "https://arxiv.org/pdf/2306.11227",
        "title": "An Introduction to the CXL Interconnect",
        "category": "CXL",
        "source_type": "논문"
    },
    {
        "url": "https://arxiv.org/pdf/2310.09385",
        "title": "PIM-GPT: Hybrid PIM Accelerator for Autoregressive Transformers",
        "category": "PIM",
        "source_type": "논문"
    },
]

CACHE_DIR        = "data/cache"
FAISS_INDEX_PATH = "data/faiss_index"
BM25_CACHE_PATH  = "data/bm25_cache.pkl"


class RAGAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore: Optional[FAISS] = None
        self.bm25:        Optional[BM25Okapi] = None
        self.all_docs:    List[Document] = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "],
        )
        self._initialized = False

    def _download_pdf(self, url: str, title: str) -> Optional[str]:
        """PDF 다운로드 및 캐시"""
        os.makedirs(CACHE_DIR, exist_ok=True)
        url_hash   = hashlib.md5(url.encode()).hexdigest()[:8]
        safe_title = "".join(c for c in title if c.isalnum() or c in " -_")[:40]
        cache_path = os.path.join(CACHE_DIR, f"{url_hash}_{safe_title}.pdf")

        if os.path.exists(cache_path):
            print(f"  [캐시 사용] {title}")
            return cache_path

        import requests
        print(f"  [다운로드] {title} ...")
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            with open(cache_path, "wb") as f:
                f.write(resp.content)
            print(f"  [완료] {title} ({len(resp.content) // 1024}KB)")
            return cache_path
        except Exception as e:
            print(f"  [실패] {title}: {e}")
            return None

    def initialize(self):
        """FAISS 인덱스 및 BM25 초기화 (캐시 활용)"""
        if self._initialized:
            return

        os.makedirs("data", exist_ok=True)

        # 캐시 존재 시 로드
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(BM25_CACHE_PATH):
            print("[RAG] 기존 인덱스 로드 중...")
            try:
                self.vectorstore = FAISS.load_local(
                    FAISS_INDEX_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                with open(BM25_CACHE_PATH, "rb") as f:
                    cache = pickle.load(f)
                    self.bm25     = cache["bm25"]
                    self.all_docs = cache["docs"]
                print(f"[RAG] 인덱스 로드 완료: {len(self.all_docs)}개 청크")
                self._initialized = True
                return
            except Exception as e:
                print(f"[RAG] 캐시 로드 실패, 재생성: {e}")

        # PDF 다운로드 및 인덱싱
        print("[RAG] 문서 인덱싱 시작...")
        all_chunks = []

        for paper in ARXIV_PDFS:
            pdf_path = self._download_pdf(paper["url"], paper["title"])
            if not pdf_path:
                continue
            try:
                loader = PyPDFLoader(pdf_path)
                pages  = loader.load()
                chunks = self.text_splitter.split_documents(pages)
                for chunk in chunks:
                    chunk.metadata.update({
                        "title":       paper["title"],
                        "category":    paper["category"],
                        "source_type": paper["source_type"],
                        "url":         paper["url"],
                    })
                all_chunks.extend(chunks)
                print(f"  [인덱싱] {paper['title']}: {len(chunks)}청크")
            except Exception as e:
                print(f"  [오류] {paper['title']}: {e}")

        if not all_chunks:
            print("[RAG] 경고: 인덱싱된 문서 없음")
            self._initialized = True
            return

        self.all_docs = all_chunks

        # FAISS 인덱스 생성
        print("[RAG] FAISS 인덱스 생성 중...")
        self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
        self.vectorstore.save_local(FAISS_INDEX_PATH)

        # BM25 인덱스 생성
        print("[RAG] BM25 인덱스 생성 중...")
        tokenized_corpus = [doc.page_content.split() for doc in all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "docs": all_chunks}, f)

        print(f"[RAG] 초기화 완료: 총 {len(all_chunks)}개 청크")
        self._initialized = True

    def _hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Hybrid Retrieval: Dense(FAISS) + Sparse(BM25) → 점수 통합

        【Dense 점수 정규화 수정】
        FAISS.similarity_search_with_score()는 IndexFlatL2 기준으로
        L2 거리(낮을수록 유사)를 반환한다.

        기존 코드: normalized = 1 - (score / max_dense)
          - max_dense 단일값으로 나누면 분포가 [0, 1] 범위를 보장하지 않음
          - min이 0이 아닌 경우 정규화 결과가 왜곡될 수 있음

        수정: min-max 정규화 적용
          normalized = 1 - (score - min_dense) / (max_dense - min_dense + ε)
          - L2 거리이므로 역변환(1 - x)으로 유사도 방향을 맞춤
          - [0, 1] 범위 보장
        """
        dense_docs: List[tuple] = []
        bm25_docs:  List[tuple] = []

        # Dense Retrieval (FAISS)
        if self.vectorstore:
            try:
                raw = self.vectorstore.similarity_search_with_score(query, k=k * 2)
                dense_docs = [(doc, score) for doc, score in raw]
            except Exception:
                dense_docs = []

        # Sparse Retrieval (BM25)
        if self.bm25 and self.all_docs:
            tokenized_query = query.split()
            bm25_scores     = self.bm25.get_scores(tokenized_query)
            top_indices     = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:k * 2]
            bm25_docs = [(self.all_docs[i], bm25_scores[i]) for i in top_indices]

        doc_scores: Dict[str, float] = {}
        doc_map:    Dict[str, Document] = {}

        # ── Dense: min-max 정규화 후 역변환 (L2 거리 → 유사도) ────────────────
        if dense_docs:
            scores     = [s for _, s in dense_docs]
            min_dense  = min(scores)
            max_dense  = max(scores)
            denom      = max_dense - min_dense + 1e-8

            for doc, score in dense_docs:
                key        = doc.page_content[:100]
                # L2 거리: 작을수록 유사 → 1 - 정규화값으로 역변환
                normalized = 1.0 - (score - min_dense) / denom
                doc_scores[key] = doc_scores.get(key, 0.0) + normalized * 0.6
                doc_map[key]    = doc

        # ── BM25: max 정규화 (높을수록 관련성 높음) ───────────────────────────
        if bm25_docs:
            max_bm25 = max(s for _, s in bm25_docs) if bm25_docs else 1.0
            for doc, score in bm25_docs:
                key        = doc.page_content[:100]
                normalized = score / (max_bm25 + 1e-8)
                doc_scores[key] = doc_scores.get(key, 0.0) + normalized * 0.4
                doc_map[key]    = doc

        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        return [doc_map[key] for key in sorted_keys[:k]]

    def _rewrite_query(self, query: str, attempt: int) -> str:
        """쿼리 재작성"""
        prompt = (
            f"다음 검색 쿼리를 HBM4, PIM, CXL 반도체 기술 문서 검색에 더 적합하게 재작성하십시오.\n"
            f"시도 횟수: {attempt}\n"
            f"원본 쿼리: {query}\n"
            f"재작성된 쿼리 (한 줄만):"
        )
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def search(self, query: str, retry_count: int = 0) -> List[Dict]:
        """RAG 검색 수행 (재검색 포함)"""
        self.initialize()

        if not self.vectorstore and not self.bm25:
            return [{"content": "RAG 문서 없음 (인덱스 미구성)", "source": "N/A", "fallback": True}]

        current_query = query
        for attempt in range(retry_count + 1):
            if attempt > 0:
                current_query = self._rewrite_query(query, attempt)
                print(f"  [RAG] 쿼리 재작성 (시도 {attempt}): {current_query}")

            docs = self._hybrid_search(current_query, k=5)

            if docs:
                results = []
                for doc in docs:
                    results.append({
                        "content":     doc.page_content,
                        "title":       doc.metadata.get("title", "Unknown"),
                        "category":    doc.metadata.get("category", "Unknown"),
                        "source_type": doc.metadata.get("source_type", "논문"),
                        "url":         doc.metadata.get("url", ""),
                        "page":        doc.metadata.get("page", 0),
                    })
                return results

        return [{"content": "검색 결과 없음", "source": "N/A", "fallback": True}]


def run_rag_agent(state: AgentState) -> AgentState:
    """RAG Agent 노드 실행"""
    print("\n[RAG Agent] 문서 검색 시작...")
    agent = _get_rag_agent()

    queries     = state.get("search_queries", {}).get("rag") or _generate_rag_queries(state["query"])
    all_results = []

    for q in queries:
        print(f"  [RAG] 검색 쿼리: {q}")
        results = agent.search(q, retry_count=min(state.get("retry_count", 0), 2))
        all_results.extend(results)

    # 중복 제거
    seen   = set()
    unique = []
    for r in all_results:
        key = r["content"][:80]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"[RAG Agent] 검색 완료: {len(unique)}개 청크")
    state["rag_results"] = unique[:15]
    state["rag_done"]    = True
    return state


def _generate_rag_queries(user_query: str) -> List[str]:
    """RAG용 다각화 쿼리 생성"""
    return [
        "HBM4 High Bandwidth Memory architecture development",
        "Processing-In-Memory PIM technology DRAM",
        "CXL Compute Express Link memory interconnect",
        "Samsung Micron HBM memory competition",
        "PIM GPU-free LLM inference acceleration",
    ]


# 싱글턴 패턴
_rag_agent_instance: Optional[RAGAgent] = None


def _get_rag_agent() -> RAGAgent:
    global _rag_agent_instance
    if _rag_agent_instance is None:
        _rag_agent_instance = RAGAgent()
    return _rag_agent_instance
