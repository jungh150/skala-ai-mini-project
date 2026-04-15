"""
SK Hynix R&D 기술 전략 분석 시스템
설계 문서와 일치하는 Supervisor 패턴 기반 LangGraph 워크플로우

【그래프 엣지 설계】
설계서 명시 흐름:
    formatting_node → supervisor  # PDF 생성 완료 반환
    supervisor      → END         # 생성 확인 후

따라서 format 노드는 supervisor로 복귀하고,
supervisor가 final_report_path 확인 후 END로 라우팅한다.
"""
import os
import sys
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.supervisor import run_supervisor
from agents.rag_agent import run_rag_agent
from agents.web_search_agent import run_web_search_agent
from agents.trl_evaluation_node import run_trl_evaluation_node
from agents.draft_agent import run_draft_agent
from agents.formatting_node import run_formatting_node

load_dotenv()


def supervisor_node(state: AgentState) -> AgentState:
    return run_supervisor(state)


def rag_node(state: AgentState) -> AgentState:
    return run_rag_agent(state)


def web_search_node(state: AgentState) -> AgentState:
    return run_web_search_agent(state)


def trl_node(state: AgentState) -> AgentState:
    return run_trl_evaluation_node(state)


def draft_node(state: AgentState) -> AgentState:
    return run_draft_agent(state)


def format_node(state: AgentState) -> AgentState:
    return run_formatting_node(state)


def route_supervisor(state: AgentState) -> str:
    """
    Supervisor의 state["next"] 값에 따라 다음 노드를 결정한다.

    "end" 라우팅은 Supervisor가 final_report_path 확인 후 직접 결정한다.
    Formatting Node가 END로 직행하지 않고 Supervisor를 경유하는 것이
    설계서의 의도("생성 확인 후 END")를 정확히 구현한다.
    """
    next_node = state.get("next", "draft")
    if next_node == "rag":
        return "rag"
    if next_node == "web_search":
        return "web_search"
    if next_node == "trl":
        return "trl"
    if next_node == "draft":
        return "draft"
    if next_node == "format":
        return "format"
    if next_node == "end":
        return END
    return "draft"


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    # ── 노드 등록 ──────────────────────────────────────────────────────────────
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("trl", trl_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("format", format_node)

    # ── 진입점 ────────────────────────────────────────────────────────────────
    workflow.set_entry_point("supervisor")

    # ── Supervisor → 조건부 분기 ───────────────────────────────────────────────
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "rag":        "rag",
            "web_search": "web_search",
            "trl":        "trl",
            "draft":      "draft",
            "format":     "format",
            END:          END,
        },
    )

    # ── 고정 엣지: 각 노드 → Supervisor 복귀 ─────────────────────────────────
    # 설계서 명시:
    #   rag_agent        → supervisor  (결과 반환)
    #   web_search_agent → supervisor  (결과 반환)
    #   trl_node         → supervisor  (평가 결과 반환)
    #   draft_agent      → supervisor  (초안 검증 요청)
    #   formatting_node  → supervisor  (PDF 생성 완료 반환) ← 핵심 수정
    workflow.add_edge("rag",        "supervisor")
    workflow.add_edge("web_search", "supervisor")
    workflow.add_edge("trl",        "supervisor")
    workflow.add_edge("draft",      "supervisor")
    workflow.add_edge("format",     "supervisor")   # format이 END가 아닌 supervisor로 복귀

    return workflow.compile()


def run_analysis(query: str = None) -> dict:
    if query is None:
        query = (
            "SK Hynix R&D 담당자를 위한 HBM4, PIM, CXL 기술에 대한 "
            "경쟁사(Samsung, Micron) 기술 성숙도와 위협 수준을 TRL 기반으로 분석하고, "
            "R&D 우선순위 대응 방향을 제시하는 기술 전략 분석 보고서를 작성하십시오."
        )

    print("=" * 70)
    print("SK Hynix R&D 기술 전략 분석 시스템")
    print("분석 대상: HBM4, PIM, CXL | 경쟁사: Samsung, Micron")
    print("=" * 70)

    initial_state: AgentState = {
        "query":            query,
        "search_queries":   {},
        "rag_results":      [],
        "web_results":      [],
        "rag_done":         False,
        "web_done":         False,
        "retrieval_ready":  False,
        "trl_evidence":     {},
        "trl_assessment":   "",
        "missing_trl_info": [],
        "trl_ready":        False,
        "draft":            "",
        "validation_pass":  False,
        "feedback":         "",
        "iteration_count":  0,
        "retry_count":      0,
        "trl_retry_count":  0,
        "final_report_path": "",
        "fallback_flags":   {},
        "next":             "",
        "validation_details": "",
    }

    graph = build_graph()
    final_state = None
    step_count = 0

    for step in graph.stream(initial_state, {"recursion_limit": 60}):
        step_count += 1
        node_name = list(step.keys())[0]
        print(f"\n--- Step {step_count}: {node_name} 완료 ---")
        final_state = step[node_name]
        if step_count >= 30:
            print("[시스템] Max steps 도달, 종료")
            break

    return final_state


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        sys.exit(1)
    run_analysis()
