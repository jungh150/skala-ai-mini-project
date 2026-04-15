"""
Supervisor: 전체 워크플로우 조율
- T1 질의 분석 및 검색 계획 수립
- T2 RAG / T3 Web Search 개별 호출
- T4 수집 정보 통합·평가 및 재검색 분기
- TRL Evaluation Node 호출 및 재검색 분기
- T6 초안 검증 루프 (max 3회)
- T7 PDF 생성 종료 정책 (Formatting Node 복귀 후 최종 확인)

【retry_count 분리 설계】
- retry_count     : T4(검색 충분성) 단계의 재검색 횟수 (max 2)
- trl_retry_count : TRL 근거 부족으로 인한 재검색 횟수 (max 2)
두 카운터를 분리하여 T4에서 재검색 기회를 소진해도
TRL 단계에서 독립적으로 재검색을 시도할 수 있게 한다.

【company_ok 수정】
기존 코드는 web_results의 'company' 필드만 확인했으나,
Tavily/DuckDuckGo 검색 결과에는 'company' 필드가 없다.
URL 기반 회사 판별 로직을 추가하여 실질적인 회사 커버리지를 확인한다.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI

from agents.state import AgentState
from prompts.templates import SUPERVISOR_VALIDATION_PROMPT


DEFAULT_SEARCH_QUERIES = {
    "rag": [
        "HBM4 High Bandwidth Memory architecture development",
        "Processing-In-Memory PIM technology DRAM",
        "CXL Compute Express Link memory interconnect",
        "Samsung Micron HBM memory competition",
        "PIM GPU-free LLM inference acceleration",
    ],
    "web": [
        "SK Hynix HBM4 technology status",
        "Samsung HBM4 development roadmap",
        "Micron HBM4 key customers shipment",
        "HBM4 limitations challenges",
        "HBM4 industry comparison",
        "PIM memory commercialization challenges Samsung Micron",
        "CXL memory roadmap Samsung Micron SK Hynix",
    ],
}


def _extract_company_from_result(r: dict) -> str:
    """
    검색 결과에서 회사명을 추출한다.
    'company' 필드가 없는 경우(Tavily/DuckDuckGo 결과) URL로 판별한다.
    """
    company = r.get("company", "")
    if company:
        return company

    url = r.get("url", "").lower()
    title = (r.get("title", "") + r.get("topic", "")).lower()

    if "skhynix" in url or "sk hynix" in title or "skhynix" in title:
        return "SK Hynix"
    if "samsung" in url or "samsung" in title:
        return "Samsung"
    if "micron" in url or "micron" in title:
        return "Micron"
    return ""


def run_supervisor(state: AgentState) -> AgentState:
    print("\n[Supervisor] 상태 평가 중...")

    rag_results      = state.get("rag_results", [])
    web_results      = state.get("web_results", [])
    rag_done         = state.get("rag_done", False)
    web_done         = state.get("web_done", False)
    trl_assessment   = state.get("trl_assessment", "")
    draft            = state.get("draft", "")
    iteration_count  = state.get("iteration_count", 0)
    retry_count      = state.get("retry_count", 0)
    trl_retry_count  = state.get("trl_retry_count", 0)   # TRL 전용 재시도 카운터
    trl_ready        = state.get("trl_ready", False)
    missing_trl_info = state.get("missing_trl_info", [])
    final_report_path = state.get("final_report_path", "")

    # ── T7 종료: Formatting Node 복귀 후 PDF 생성 확인 ──────────────────────────
    # 설계서: formatting_node → supervisor → END
    if final_report_path:
        print(f"[Supervisor] PDF 생성 확인 완료 → END | 경로: {final_report_path}")
        state["next"] = "end"
        return state

    # ── T1: 최초 진입 — 검색 계획 수립 ────────────────────────────────────────
    if not state.get("search_queries"):
        state["search_queries"] = _build_search_plan(state.get("query", ""))

    # ── T2: RAG Agent 호출 ───────────────────────────────────────────────────
    if not rag_done:
        print("[Supervisor] → RAG Agent 실행")
        state["next"] = "rag"
        return state

    # ── T3: Web Search Agent 호출 ────────────────────────────────────────────
    if rag_done and not web_done:
        print("[Supervisor] → Web Search Agent 실행")
        state["next"] = "web_search"
        return state

    # ── T4: 검색 결과 충분성 평가 → TRL 평가 분기 ─────────────────────────────
    if rag_done and web_done and not trl_assessment:
        retrieval_ok, details = _check_retrieval_sufficiency(rag_results, web_results)
        state["retrieval_ready"]    = retrieval_ok
        state["validation_details"] = details

        if not retrieval_ok and retry_count < 2:
            print(f"[Supervisor] 검색 결과 부족 → RAG/Web 재검색 (T4 재시도 {retry_count + 1}/2)")
            state["retry_count"] = retry_count + 1
            state["rag_done"]    = False
            state["web_done"]    = False
            state["rag_results"] = []
            state["web_results"] = []
            state["next"]        = "rag"
            return state
        elif not retrieval_ok:
            print("[Supervisor] T4 재검색 max 도달 → 정보 부재 플래그와 함께 진행")
            state["fallback_flags"] = {
                **state.get("fallback_flags", {}),
                "insufficient_data": True,
            }

        print("[Supervisor] 검색 결과 충분 → TRL Evaluation")
        state["next"] = "trl"
        return state

    # ── TRL 평가 완료 후 재검색 분기 ──────────────────────────────────────────
    if trl_assessment and not draft:
        if not trl_ready and trl_retry_count < 2:
            # TRL 근거 부족 → 재검색 (trl_retry_count 사용, retry_count와 독립)
            print(
                f"[Supervisor] TRL 정보 부족 → RAG/Web 재검색 "
                f"(TRL 재시도 {trl_retry_count + 1}/2)"
            )
            state["trl_retry_count"] = trl_retry_count + 1
            state["feedback"]        = "TRL 평가에 필요한 근거 보강 필요: " + "; ".join(missing_trl_info)
            state["rag_done"]        = False
            state["web_done"]        = False
            state["rag_results"]     = []
            state["web_results"]     = []
            state["trl_assessment"]  = ""
            state["next"]            = "rag"
            return state
        elif not trl_ready:
            print("[Supervisor] TRL 재검색 max 도달 → 한계 명시 후 초안 생성")
            state["fallback_flags"] = {
                **state.get("fallback_flags", {}),
                "trl_insufficient": True,
            }

        print("[Supervisor] TRL 평가 완료 → 초안 생성")
        state["next"] = "draft"
        return state

    # ── T6: 초안 검증 루프 (max 3회) ─────────────────────────────────────────
    if draft:
        if iteration_count >= 3:
            print("[Supervisor] Max iteration 도달 → 최선 초안으로 확정 → 포맷팅")
            state["validation_pass"] = True
            state["next"]            = "format"
            return state

        validation_result, details = _validate_draft(draft)
        state["validation_details"] = details

        if validation_result:
            print("[Supervisor] 초안 검증 PASS → 포맷팅")
            state["validation_pass"] = True
            state["next"]            = "format"
        else:
            print(f"[Supervisor] 초안 검증 FAIL → 재작성 요청 (시도 {iteration_count}/3)")
            state["validation_pass"] = False
            state["feedback"]        = details
            state["next"]            = "draft"
        return state

    state["next"] = "draft"
    return state


def _build_search_plan(query: str) -> Dict[str, List[str]]:
    return {
        "rag": list(DEFAULT_SEARCH_QUERIES["rag"]),
        "web": list(DEFAULT_SEARCH_QUERIES["web"]),
    }


def _check_retrieval_sufficiency(
    rag_results: list, web_results: list
) -> Tuple[bool, str]:
    """
    T4 검색 충분성 검증.

    【company_ok 수정】
    기존: web_results의 'company' 필드만 확인 → Tavily/DuckDuckGo 결과 누락
    변경: URL과 title 기반으로도 회사를 추출하여 실질적인 커버리지를 확인
    """
    tech_coverage = {"HBM4": 0, "PIM": 0, "CXL": 0}
    source_types  = set()
    companies     = set()

    for r in rag_results:
        content    = r.get("content", "").lower()
        source_types.add(r.get("source_type", ""))
        if "hbm4" in content or "hbm" in content:
            tech_coverage["HBM4"] += 1
        if "pim" in content or "processing-in-memory" in content:
            tech_coverage["PIM"] += 1
        if "cxl" in content or "compute express" in content:
            tech_coverage["CXL"] += 1
        category = r.get("category", "")
        if category in tech_coverage:
            tech_coverage[category] += 1

    for r in web_results:
        content = r.get("content", "").lower()
        topic   = r.get("topic", "").lower()
        source_types.add(r.get("source_type", ""))

        if "hbm4" in content or "hbm4" in topic or "hbm" in topic:
            tech_coverage["HBM4"] += 1
        if "pim" in content or "pim" in topic:
            tech_coverage["PIM"] += 1
        if "cxl" in content or "cxl" in topic:
            tech_coverage["CXL"] += 1

        # URL·title 기반 회사 추출 (company 필드 없는 경우 대응)
        company = _extract_company_from_result(r)
        if company:
            companies.add(company)

    tech_ok    = all(v >= 2 for v in tech_coverage.values())
    company_ok = {"Samsung", "Micron"}.issubset(companies)
    source_ok  = len({s for s in source_types if s}) >= 3

    details = (
        f"기술 커버리지={tech_coverage}, 회사={sorted(companies)}, "
        f"출처유형={sorted(s for s in source_types if s)} | "
        f"tech_ok={tech_ok}, company_ok={company_ok}, source_ok={source_ok}"
    )
    print(f"  [Supervisor] {details}")
    return tech_ok and company_ok and source_ok, details


def _validate_draft(draft: str) -> tuple[bool, str]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = SUPERVISOR_VALIDATION_PROMPT.format(draft=draft[:7000])
    response = llm.invoke(prompt)
    result_text = response.content
    print(f"  [Supervisor 검증] {result_text[:300]}...")
    is_pass = "PASS" in result_text.upper() and "FAIL" not in result_text.upper()
    return is_pass, result_text
