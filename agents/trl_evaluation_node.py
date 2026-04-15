"""
trl_evaluation_node.py
TRL Evaluation Node — 수집된 RAG/웹 결과 기반 기술·회사별 TRL 추정

역할:
    검색 결과를 구조화된 근거로 정리한 뒤 LLM이 TRL을 추정한다.
    근거가 부족한 항목은 missing_trl_info에 기록하여
    Supervisor가 재검색 여부를 판단할 수 있도록 신호를 전달한다.

Node로 분류하는 이유:
    검색 실행 없이 평가만 수행하므로 Agent가 아닌 독립 Node로 분류한다.
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
        formatted.append(
            f"[{i}] 제목: {r.get('title', 'Unknown')} | "
            f"기술: {r.get('category', '')} | "
            f"출처유형: {r.get('source_type', '논문')}\n"
            f"    URL: {r.get('url', '')}\n"
            f"    내용: {r.get('content', '')[:700]}\n"
        )
    return "\n".join(formatted)


def _format_web_results(web_results: List[Dict]) -> str:
    if not web_results:
        return "웹 검색 결과 없음"

    formatted = []
    for i, r in enumerate(web_results[:20], 1):
        company     = r.get("company", "") or "미분류"
        topic       = r.get("topic", "") or r.get("title", "")
        formatted.append(
            f"[{i}] 회사: {company} | 토픽: {topic} | "
            f"출처유형: {r.get('source_type', '뉴스')}\n"
            f"    URL: {r.get('url', '')}\n"
            f"    내용: {r.get('content', '')[:600]}\n"
        )
    return "\n".join(formatted)


def _collect_structured_evidence(
    rag_results: List[Dict], web_results: List[Dict]
) -> Dict[str, List[Dict]]:
    """검색 결과를 기술별·회사별 버킷으로 구조화한다."""
    buckets: Dict[str, List[Dict]] = {
        "HBM4": [], "PIM": [], "CXL": [],
        "SK Hynix": [], "Samsung": [], "Micron": [],
    }

    for r in rag_results:
        text = f"{r.get('title','')} {r.get('content','')} {r.get('category','')}".lower()
        item = {
            "title":       r.get("title", ""),
            "url":         r.get("url", ""),
            "source_type": r.get("source_type", "논문"),
            "snippet":     r.get("content", "")[:240],
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
            "title":       r.get("title", r.get("topic", "")),
            "url":         r.get("url", ""),
            "source_type": r.get("source_type", "뉴스"),
            "snippet":     r.get("content", "")[:240],
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
    """근거가 부족한 항목을 식별하여 반환한다."""
    missing: List[str] = []
    for tech in ["HBM4", "PIM", "CXL"]:
        if len(evidence.get(tech, [])) < 2:
            missing.append(f"{tech} 관련 근거 2건 미만")
    for company in ["Samsung", "Micron"]:
        if len(evidence.get(company, [])) < 2:
            missing.append(f"{company} 관련 공개 근거 부족")
    return missing


def run_trl_evaluation_node(state: AgentState) -> AgentState:
    """TRL Evaluation Node 진입점."""
    print("\n[TRL Evaluation Node] TRL 평가 시작...")

    rag_results = state.get("rag_results", [])
    web_results = state.get("web_results", [])
    evidence    = _collect_structured_evidence(rag_results, web_results)
    missing     = _infer_missing_info(evidence)

    llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = TRL_EVALUATION_PROMPT.format(
        rag_results=_format_rag_results(rag_results),
        web_results=_format_web_results(web_results),
    )
    assessment = llm.invoke(prompt).content.strip()

    state["trl_evidence"]    = evidence
    state["trl_assessment"]  = assessment
    state["missing_trl_info"] = missing
    state["trl_ready"]       = len(missing) == 0

    print(f"[TRL Evaluation Node] 완료 | missing: {len(missing)}건 | trl_ready={state['trl_ready']}")
    return state
