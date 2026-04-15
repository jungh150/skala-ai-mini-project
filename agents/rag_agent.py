"""
rag_agent.py
RAG Agent — arXiv 논문 기반 로컬 문서 검색

구성:
    Embedding  : BAAI/bge-m3 (다국어 SOTA, max 8192 tokens)
    Vector DB  : FAISS (IndexFlatL2)
    Retrieval  : Hybrid — Dense (FAISS, 가중치 0.6) + Sparse (BM25, 가중치 0.4)
    재검색     : Query Rewrite 후 최대 2회

Dense 점수 정규화:
    FAISS IndexFlatL2는 L2 거리를 반환한다(낮을수록 유사).
    min-max 정규화 후 역변환하여 유사도 방향을 BM25 점수와 통일한다.
        normalized = 1 - (score - min) / (max - min + ε)
    이를 통해 [0, 1] 범위를 보장하고 두 점수를 안정적으로 결합한다.
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

# 평가 대상 arXiv 논문 (PIM 5편, CXL 2편)
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
        self.vectorstore: Optional[FAISS]     = None
        self.bm25:        Optional[BM25Okapi] = None
        self.all_docs:    List[Document]      = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "],
        )
        self._initialized = False

    def _download_pdf(self, url: str, title: str) -> Optional[str]:
        """PDF를 다운로드하고 로컬에 캐시한다."""
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
        """FAISS 인덱스 및 BM25를 초기화한다. 캐시가 있으면 로드한다."""
        if self._initialized:
            return

        os.makedirs("data", exist_ok=True)

        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(BM25_CACHE_PATH):
            print("[RAG] 인덱스 로드 중...")
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
                print(f"  [인덱싱 실패] {paper['title']}: {e}")

        if not all_chunks:
            print("[RAG] 경고: 인덱싱된 문서 없음")
            self._initialized = True
            return

        self.all_docs = all_chunks

        print("[RAG] FAISS 인덱스 생성 중...")
        self.vectorstore = FAISS.from_documents(all_chunks, self.embeddings)
        self.vectorstore.save_local(FAISS_INDEX_PATH)

        print("[RAG] BM25 인덱스 생성 중...")
        tokenized_corpus = [doc.page_content.split() for doc in all_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        with open(BM25_CACHE_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "docs": all_chunks}, f)

        print(f"[RAG] 초기화 완료: 총 {len(all_chunks)}개 청크")
        self._initialized = True

    def _hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Hybrid Retrieval: Dense (FAISS) + Sparse (BM25) 점수 통합.

        Dense 점수 정규화:
            L2 거리를 min-max 정규화 후 역변환하여 [0, 1] 범위의 유사도로 변환한다.
            normalized = 1 - (score - min) / (max - min + ε)
        """
        dense_docs: List[tuple] = []
        bm25_docs:  List[tuple] = []

        if self.vectorstore:
            try:
                raw        = self.vectorstore.similarity_search_with_score(query, k=k * 2)
                dense_docs = [(doc, score) for doc, score in raw]
            except Exception:
                dense_docs = []

        if self.bm25 and self.all_docs:
            tokenized_query = query.split()
            bm25_scores     = self.bm25.get_scores(tokenized_query)
            top_indices     = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:k * 2]
            bm25_docs = [(self.all_docs[i], bm25_scores[i]) for i in top_indices]

        doc_scores: Dict[str, float]    = {}
        doc_map:    Dict[str, Document] = {}

        # Dense: min-max 정규화 + 역변환 (L2 거리 → 유사도)
        if dense_docs:
            scores    = [s for _, s in dense_docs]
            min_dense = min(scores)
            max_dense = max(scores)
            denom     = max_dense - min_dense + 1e-8

            for doc, score in dense_docs:
                key        = doc.page_content[:100]
                normalized = 1.0 - (score - min_dense) / denom
                doc_scores[key] = doc_scores.get(key, 0.0) + normalized * 0.6
                doc_map[key]    = doc

        # BM25: max 정규화
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
        """검색 결과가 부족할 때 쿼리를 재작성한다."""
        prompt = (
            f"다음 검색 쿼리를 HBM4, PIM, CXL 반도체 기술 문서 검색에 더 적합하게 재작성하십시오.\n"
            f"시도 횟수: {attempt}\n"
            f"원본 쿼리: {query}\n"
            f"재작성된 쿼리 (한 줄만):"
        )
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def search(self, query: str, retry_count: int = 0) -> List[Dict]:
        """Hybrid 검색을 수행한다. 결과가 없으면 Query Rewrite 후 재시도한다."""
        self.initialize()

        if not self.vectorstore and not self.bm25:
            return [{"content": "RAG 문서 없음 (인덱스 미구성)", "source": "N/A", "fallback": True}]

        current_query = query
        for attempt in range(retry_count + 1):
            if attempt > 0:
                current_query = self._rewrite_query(query, attempt)
                print(f"  [RAG] 쿼리 재작성 ({attempt}): {current_query}")

            docs = self._hybrid_search(current_query, k=5)

            if docs:
                return [
                    {
                        "content":     doc.page_content,
                        "title":       doc.metadata.get("title", "Unknown"),
                        "category":    doc.metadata.get("category", "Unknown"),
                        "source_type": doc.metadata.get("source_type", "논문"),
                        "url":         doc.metadata.get("url", ""),
                        "page":        doc.metadata.get("page", 0),
                    }
                    for doc in docs
                ]

        return [{"content": "검색 결과 없음", "source": "N/A", "fallback": True}]


def run_rag_agent(state: AgentState) -> AgentState:
    """RAG Agent 노드 진입점."""
    print("\n[RAG Agent] 문서 검색 시작...")
    agent = _get_rag_agent()

    queries     = state.get("search_queries", {}).get("rag") or _default_rag_queries()
    all_results = []

    for q in queries:
        print(f"  [RAG] 검색 쿼리: {q}")
        results = agent.search(q, retry_count=min(state.get("retry_count", 0), 2))
        all_results.extend(results)

    # 중복 제거 (내용 앞 80자 기준)
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


def _default_rag_queries() -> List[str]:
    """search_queries가 없을 때 사용하는 기본 쿼리셋."""
    return [
        "HBM4 High Bandwidth Memory architecture development",
        "Processing-In-Memory PIM technology DRAM",
        "CXL Compute Express Link memory interconnect",
        "Samsung Micron HBM memory competition",
        "PIM GPU-free LLM inference acceleration",
    ]


# 싱글턴 — 임베딩 모델을 매 호출마다 재로드하지 않기 위해 인스턴스를 재사용한다.
_rag_agent_instance: Optional[RAGAgent] = None


def _get_rag_agent() -> RAGAgent:
    global _rag_agent_instance
    if _rag_agent_instance is None:
        _rag_agent_instance = RAGAgent()
    return _rag_agent_instance
