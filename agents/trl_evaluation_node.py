"""
TRL Evaluation Node: 수집된 RAG/Web 결과를 바탕으로 기술·회사별 TRL을 추정
- 검색은 수행하지 않고 평가만 담당
- 근거가 부족하면 missing_trl_info를 통해 Supervisor에 재검색 신호 전달
"""
from __future__ import annotations

from typing import Dict, List
from langchain_openai import ChatOpenAI

from agents.state import AgentState
from prompts.templates import TRL_EVALUATION_PROMPT


def _format_rag_results(rag_results: List[Dict]) -> str:
    if not rag_results:
        return "RAG 검색 결과 없음"

    formatted = []
    for i, r in enumerate(rag_results[:12], 1):
        title = r.get("title", "Unknown")
        url = r.get("url", "")
        category = r.get("category", "")
        source_type = r.get("source_type", "논문")
        content = r.get("content", "")[:700]
        formatted.append(
            f"[{i}] 제목: {title} | 기술: {category} | 출처유형: {source_type}\n"
            f"    URL: {url}\n"
            f"    내용: {content}\n"
        )
    return "\n".join(formatted)


def _format_web_results(web_results: List[Dict]) -> str:
    if not web_results:
        return "웹 검색 결과 없음"

    formatted = []
    for i, r in enumerate(web_results[:20], 1):
        company = r.get("company", "") or "미분류"
        topic = r.get("topic", "") or r.get("title", "")
        source_type = r.get("source_type", "뉴스")
        url = r.get("url", "")
        content = r.get("content", "")[:600]
        formatted.append(
            f"[{i}] 회사: {company} | 토픽: {topic} | 출처유형: {source_type}\n"
            f"    URL: {url}\n"
            f"    내용: {content}\n"
        )
    return "\n".join(formatted)


def _collect_structured_evidence(rag_results: List[Dict], web_results: List[Dict]) -> Dict[str, List[Dict]]:
    buckets: Dict[str, List[Dict]] = {
        "HBM4": [], "PIM": [], "CXL": [],
        "SK Hynix": [], "Samsung": [], "Micron": [],
    }

    for r in rag_results:
        text = f"{r.get('title','')} {r.get('content','')} {r.get('category','')}".lower()
        item = {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "source_type": r.get("source_type", "논문"),
            "snippet": r.get("content", "")[:240],
        }
        if "hbm4" in text or "hbm" in text:
            buckets["HBM4"].append(item)
        if "pim" in text or "processing-in-memory" in text:
            buckets["PIM"].append(item)
        if "cxl" in text or "compute express" in text:
            buckets["CXL"].append(item)

    for r in web_results:
        text = f"{r.get('title','')} {r.get('topic','')} {r.get('content','')} {r.get('company','')}".lower()
        item = {
            "title": r.get("title", r.get("topic", "")),
            "url": r.get("url", ""),
            "source_type": r.get("source_type", "뉴스"),
            "snippet": r.get("content", "")[:240],
        }
        if "hbm4" in text or "hbm" in text:
            buckets["HBM4"].append(item)
        if "pim" in text or "processing-in-memory" in text:
            buckets["PIM"].append(item)
        if "cxl" in text or "compute express" in text:
            buckets["CXL"].append(item)

        company = r.get("company", "")
        if company in ("SK Hynix", "Samsung", "Micron"):
            buckets[company].append(item)
        else:
            url = r.get("url", "").lower()
            if "skhynix" in url:
                buckets["SK Hynix"].append(item)
            elif "samsung" in url:
                buckets["Samsung"].append(item)
            elif "micron" in url:
                buckets["Micron"].append(item)

    return buckets


def _infer_missing_info(evidence: Dict[str, List[Dict]]) -> List[str]:
    missing: List[str] = []
    for tech in ["HBM4", "PIM", "CXL"]:
        if len(evidence.get(tech, [])) < 2:
            missing.append(f"{tech} 관련 근거 2건 미만")
    for company in ["Samsung", "Micron"]:
        if len(evidence.get(company, [])) < 2:
            missing.append(f"{company} 관련 공개 근거 부족")
    return missing


def run_trl_evaluation_node(state: AgentState) -> AgentState:
    print("\n[TRL Evaluation Node] TRL 평가 시작...")

    rag_results = state.get("rag_results", [])
    web_results = state.get("web_results", [])
    evidence = _collect_structured_evidence(rag_results, web_results)
    missing = _infer_missing_info(evidence)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = TRL_EVALUATION_PROMPT.format(
        rag_results=_format_rag_results(rag_results),
        web_results=_format_web_results(web_results),
    )
    response = llm.invoke(prompt)
    assessment = response.content.strip()

    state["trl_evidence"] = evidence
    state["trl_assessment"] = assessment
    state["missing_trl_info"] = missing
    state["trl_ready"] = len(missing) == 0

    print(f"[TRL Evaluation Node] 평가 완료 | missing: {len(missing)}건 | trl_ready={state['trl_ready']}")
    return state
