"""
Web Search Agent: 최신 뉴스·IR·특허·기술 블로그 검색
- 확증 편향 방지: 다각도 쿼리 (SK Hynix / Samsung / Micron / 부정 / 중립)
- Tavily API 사용 (fallback: DuckDuckGo)
- 출처 다양성 강제 (뉴스/논문/특허/채용공고/IR)
"""
import os
import time
from typing import List, Dict, Optional

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI

from agents.state import AgentState

# 기업 공식 페이지 (WebBaseLoader)
OFFICIAL_PAGES = [
    {
        "url": "https://semiconductor.samsung.com/news-events/tech-blog/harnessing-the-ai-era-with-breakthrough-memory-solutions/",
        "company": "Samsung",
        "topic": "HBM4/PIM/CXL",
        "source_type": "공식 블로그"
    },
    {
        "url": "https://semiconductor.samsung.com/us/technologies/memory/pim/",
        "company": "Samsung",
        "topic": "PIM",
        "source_type": "공식 블로그"
    },
    {
        "url": "https://semiconductor.samsung.com/news-events/tech-blog/how-samsung-is-breaking-new-ground-in-dram-for-the-ai-era/",
        "company": "Samsung",
        "topic": "DRAM AI",
        "source_type": "공식 블로그"
    },
    {
        "url": "https://semiconductor.samsung.com/news-events/tech-blog/samsung-electronics-presents-vision-for-ai-memory-and-storage-at-fms-2025/",
        "company": "Samsung",
        "topic": "FMS 2025",
        "source_type": "공식 블로그"
    },
    {
        "url": "https://news.skhynix.com/sk-hynix-completes-worlds-first-hbm4-development-and-readies-mass-production/",
        "company": "SK Hynix",
        "topic": "HBM4",
        "source_type": "공식 뉴스"
    },
    {
        "url": "https://news.skhynix.com/sk-hynix-41st-anniversary-rise-to-ai-memory-leader/",
        "company": "SK Hynix",
        "topic": "PIM/CXL/AiMX",
        "source_type": "공식 뉴스"
    },
    {
        "url": "https://investors.micron.com/news-releases/news-release-details/micron-ships-hbm4-key-customers-power-next-gen-ai-platforms",
        "company": "Micron",
        "topic": "HBM4",
        "source_type": "IR"
    },
]

WEB_CACHE_DIR = "data/web_cache"


class WebSearchAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self._official_content: List[Dict] = []
        self._official_loaded = False

    def _load_official_pages(self) -> List[Dict]:
        """공식 페이지 WebBaseLoader로 로드 (캐시)"""
        if self._official_loaded:
            return self._official_content

        import os
        import pickle
        os.makedirs(WEB_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(WEB_CACHE_DIR, "official_pages.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self._official_content = pickle.load(f)
                print(f"[Web] 공식 페이지 캐시 로드: {len(self._official_content)}개")
                self._official_loaded = True
                return self._official_content
            except Exception:
                pass

        results = []
        for page_info in OFFICIAL_PAGES:
            try:
                print(f"  [Web] 로드 중: {page_info['company']} - {page_info['topic']}")
                loader = WebBaseLoader(page_info["url"])
                loader.requests_kwargs = {"timeout": 30, "headers": {"User-Agent": "Mozilla/5.0"}}
                docs = loader.load()
                content = " ".join([d.page_content[:2000] for d in docs])
                if content.strip():
                    results.append({
                        "content": content[:3000],
                        "company": page_info["company"],
                        "topic": page_info["topic"],
                        "source_type": page_info["source_type"],
                        "url": page_info["url"]
                    })
                    print(f"  [완료] {page_info['company']} ({len(content)}자)")
                time.sleep(1)  # 요청 간격
            except Exception as e:
                print(f"  [실패] {page_info['url']}: {e}")

        self._official_content = results
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)

        self._official_loaded = True
        return results

    def _search_tavily(self, queries: List[str]) -> List[Dict]:
        """Tavily API 검색"""
        try:
            from tavily import TavilyClient
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError("TAVILY_API_KEY not set")

            client = TavilyClient(api_key=api_key)
            results = []

            for q in queries:
                try:
                    response = client.search(
                        query=q,
                        max_results=3,
                        search_depth="basic"
                    )
                    for r in response.get("results", []):
                        results.append({
                            "content": r.get("content", ""),
                            "url": r.get("url", ""),
                            "title": r.get("title", ""),
                            "source_type": self._classify_source(r.get("url", "")),
                            "query": q
                        })
                except Exception as e:
                    print(f"  [Tavily 오류] {q}: {e}")

            return results
        except Exception:
            return []

    def _search_duckduckgo(self, queries: List[str]) -> List[Dict]:
        """DuckDuckGo 검색 (Tavily fallback)"""
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for q in queries[:3]:  # DuckDuckGo는 3개 쿼리로 제한
                    try:
                        for r in ddgs.text(q, max_results=3):
                            results.append({
                                "content": r.get("body", ""),
                                "url": r.get("href", ""),
                                "title": r.get("title", ""),
                                "source_type": self._classify_source(r.get("href", "")),
                                "query": q
                            })
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"  [DDG 오류] {q}: {e}")
            return results
        except Exception:
            return []

    def _classify_source(self, url: str) -> str:
        """URL 기반 출처 유형 분류"""
        url_lower = url.lower()
        if any(x in url_lower for x in ["arxiv", "ieee", "acm", "scholar"]):
            return "논문"
        elif any(x in url_lower for x in ["patent", "uspto", "kipris", "wipo"]):
            return "특허"
        elif any(x in url_lower for x in ["investors", "ir.", "earnings", "finance"]):
            return "IR"
        elif any(x in url_lower for x in ["linkedin", "job", "career", "recruit"]):
            return "채용공고"
        elif any(x in url_lower for x in ["semiconductor.samsung", "news.skhynix", "micron.com"]):
            return "공식 블로그/IR"
        else:
            return "뉴스"

    def _generate_diverse_queries(self, base_query: str) -> List[str]:
        """확증 편향 방지를 위한 다각화 쿼리 생성"""
        return [
            # SK Hynix 관점
            "SK Hynix HBM4 development mass production 2025",
            # Samsung 관점
            "Samsung HBM4 development strategy competition 2025",
            # Micron 관점
            "Micron HBM4 customers shipment AI 2025",
            # 부정 관점
            "HBM4 PIM CXL technical challenges limitations problems",
            # 중립 관점
            "HBM4 PIM CXL industry comparison analysis 2025",
            # PIM 특화
            "Processing-In-Memory semiconductor AI inference 2025",
            # CXL 특화
            "CXL memory compute express link adoption 2025",
        ]

    def search(self, query: str) -> List[Dict]:
        """웹 검색 수행 (공식 페이지 + 검색 API)"""
        all_results = []

        # 1. 공식 페이지 로드
        print("  [Web] 공식 페이지 로드 중...")
        official = self._load_official_pages()
        all_results.extend(official)

        # 2. 검색 API (Tavily 우선, DuckDuckGo fallback)
        queries = self._generate_diverse_queries(query)
        print(f"  [Web] 검색 API 쿼리 {len(queries)}개 실행...")

        tavily_results = self._search_tavily(queries)
        if tavily_results:
            print(f"  [Tavily] {len(tavily_results)}개 결과")
            all_results.extend(tavily_results)
        else:
            print("  [Tavily] 실패 → DuckDuckGo fallback")
            ddg_results = self._search_duckduckgo(queries)
            print(f"  [DuckDuckGo] {len(ddg_results)}개 결과")
            all_results.extend(ddg_results)

        # 3. 출처 다양성 확인
        source_types = set(r.get("source_type", "") for r in all_results)
        print(f"  [Web] 출처 유형: {source_types}")

        # 중복 제거
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get("url", r.get("content", "")[:50])
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        return unique_results[:20]


def run_web_search_agent(state: AgentState) -> AgentState:
    """Web Search Agent 노드 실행"""
    print("\n[Web Search Agent] 웹 검색 시작...")
    agent = WebSearchAgent()
    results = agent.search(state["query"])
    print(f"[Web Search Agent] 검색 완료: {len(results)}개 결과")
    state["web_results"] = results
    return state
