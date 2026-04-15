"""
Supervisor: 전체 워크플로우 조율
- 검색 결과 평가 (Conditional Branch)
- 초안 검증 루프 (Loop: max 3회)
- 종료 정책
"""
from langchain_openai import ChatOpenAI

from agents.state import AgentState
from prompts.templates import SUPERVISOR_VALIDATION_PROMPT


def run_supervisor(state: AgentState) -> AgentState:
    """Supervisor 노드: 상태에 따라 다음 노드 결정"""
    print(f"\n[Supervisor] 상태 평가 중...")
    
    rag_results = state.get("rag_results", [])
    web_results = state.get("web_results", [])
    draft = state.get("draft", "")
    iteration_count = state.get("iteration_count", 0)
    retry_count = state.get("retry_count", 0)
    
    # 초기 상태: 검색 시작
    if not rag_results and not web_results:
        print("[Supervisor] → RAG + Web Search 병렬 실행")
        state["next"] = "search"
        return state
    
    # 검색 결과 평가 (T4)
    if rag_results and web_results and not draft:
        rag_ok = _check_retrieval_sufficiency(rag_results, web_results)
        
        if not rag_ok and retry_count < 2:
            print(f"[Supervisor] 검색 결과 부족 → 재검색 (시도 {retry_count + 1}/2)")
            state["retry_count"] = retry_count + 1
            state["next"] = "search"
            return state
        elif not rag_ok:
            print("[Supervisor] 검색 결과 부족하나 max retry 도달 → 초안 생성 진행 (정보 부재 명시)")
            state["fallback_flags"] = {"insufficient_data": True}
        
        print("[Supervisor] 검색 결과 충분 → 초안 생성")
        state["next"] = "draft"
        return state
    
    # 초안 검증 (T6)
    if draft:
        if iteration_count >= 3:
            print("[Supervisor] Max iteration 도달 → 최선 초안으로 확정")
            state["validation_pass"] = True
            state["next"] = "format"
            return state
        
        validation_result, details = _validate_draft(draft)
        state["validation_details"] = details
        
        if validation_result:
            print("[Supervisor] 초안 검증 PASS → 포맷팅")
            state["validation_pass"] = True
            state["next"] = "format"
        else:
            print(f"[Supervisor] 초안 검증 FAIL → 재작성 요청 (시도 {iteration_count}/3)")
            state["validation_pass"] = False
            state["feedback"] = details
            state["next"] = "draft"
        
        return state
    
    # 기본값
    state["next"] = "draft"
    return state


def _check_retrieval_sufficiency(rag_results, web_results) -> bool:
    """검색 결과 충분성 평가 (Rule 기반, LLM 불사용)"""
    # 기술별 최소 2개 출처
    tech_coverage = {"HBM4": 0, "PIM": 0, "CXL": 0}
    
    for r in rag_results:
        category = r.get("category", "")
        content = r.get("content", "").lower()
        if "hbm4" in content or "hbm" in content:
            tech_coverage["HBM4"] += 1
        if "pim" in content or "processing-in-memory" in content:
            tech_coverage["PIM"] += 1
        if "cxl" in content or "compute express" in content:
            tech_coverage["CXL"] += 1
        if category == "PIM":
            tech_coverage["PIM"] += 1
        if category == "CXL":
            tech_coverage["CXL"] += 1
    
    for r in web_results:
        content = r.get("content", "").lower()
        topic = r.get("topic", "").lower()
        if "hbm4" in content or "hbm4" in topic:
            tech_coverage["HBM4"] += 1
        if "pim" in content or "pim" in topic:
            tech_coverage["PIM"] += 1
        if "cxl" in content or "cxl" in topic:
            tech_coverage["CXL"] += 1
    
    print(f"  [Supervisor] 기술 커버리지: {tech_coverage}")
    
    # 경쟁사 커버리지
    companies = set()
    for r in web_results:
        company = r.get("company", "")
        if company:
            companies.add(company)
    
    # 최소 조건: 각 기술 2개 이상 출처, 최소 2개 회사 커버
    tech_ok = all(v >= 2 for v in tech_coverage.values())
    company_ok = len(companies) >= 2
    
    print(f"  [Supervisor] 회사 커버리지: {companies}")
    print(f"  [Supervisor] 기술 OK: {tech_ok}, 회사 OK: {company_ok}")
    
    return tech_ok and company_ok


def _validate_draft(draft: str) -> tuple[bool, str]:
    """초안 검증 (LLM 기반)"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = SUPERVISOR_VALIDATION_PROMPT.format(draft=draft[:4000])
    response = llm.invoke(prompt)
    result_text = response.content
    
    print(f"  [Supervisor 검증] {result_text[:300]}...")
    
    is_pass = "PASS" in result_text.upper() and "FAIL" not in result_text.upper()
    
    return is_pass, result_text


def route_from_supervisor(state: AgentState) -> str:
    """Supervisor 결정에 따른 라우팅"""
    next_node = state.get("next", "draft")
    print(f"[Router] → {next_node}")
    return next_node
