"""
SK Hynix R&D 기술 전략 분석 시스템
Supervisor 패턴 기반 LangGraph 워크플로우

실행: python app.py
"""
import os
import sys
from dotenv import load_dotenv
from typing import Literal

from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.supervisor import run_supervisor, route_from_supervisor
from agents.rag_agent import run_rag_agent
from agents.web_search_agent import run_web_search_agent
from agents.draft_agent import run_draft_agent
from agents.formatting_node import run_formatting_node

load_dotenv()


# ─────────────────────────────────────────
# 노드 함수 정의
# ─────────────────────────────────────────

def search_node(state: AgentState) -> AgentState:
    """RAG + Web Search 병렬 실행 (순차 실행으로 구현)"""
    state = run_rag_agent(state)
    state = run_web_search_agent(state)
    return state


def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor 노드"""
    return run_supervisor(state)


def draft_node(state: AgentState) -> AgentState:
    """Draft Generation 노드"""
    return run_draft_agent(state)


def format_node(state: AgentState) -> AgentState:
    """Formatting 노드"""
    return run_formatting_node(state)


# ─────────────────────────────────────────
# 라우팅 함수
# ─────────────────────────────────────────

def route_supervisor(state: AgentState) -> str:
    """Supervisor 결정에 따른 라우팅"""
    next_node = state.get("next", "draft")
    
    if next_node == "search":
        return "search"
    elif next_node == "draft":
        return "draft"
    elif next_node == "format":
        return "format"
    elif next_node == "end":
        return END
    else:
        return "draft"


# ─────────────────────────────────────────
# 그래프 구성
# ─────────────────────────────────────────

def build_graph() -> StateGraph:
    """LangGraph 워크플로우 그래프 구성"""
    
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("search", search_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("format", format_node)
    
    # 엣지 정의
    # START → supervisor
    workflow.set_entry_point("supervisor")
    
    # supervisor → (search | draft | format | END)
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "search": "search",
            "draft": "draft",
            "format": "format",
            END: END,
        }
    )
    
    # search → supervisor (결과 반환)
    workflow.add_edge("search", "supervisor")
    
    # draft → supervisor (검증 요청)
    workflow.add_edge("draft", "supervisor")
    
    # format → END
    workflow.add_edge("format", END)
    
    return workflow.compile()


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────

def run_analysis(query: str = None) -> dict:
    """분석 실행"""
    
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
    print(f"쿼리: {query[:100]}...")
    print("=" * 70)
    
    # 초기 상태
    initial_state: AgentState = {
        "query": query,
        "rag_results": [],
        "web_results": [],
        "draft": "",
        "validation_pass": False,
        "feedback": "",
        "iteration_count": 0,
        "retry_count": 0,
        "fallback_flags": {},
        "next": "",
        "validation_details": "",
    }
    
    # 그래프 실행
    graph = build_graph()
    
    print("\n[시스템] 워크플로우 시작...")
    
    final_state = None
    step_count = 0
    max_steps = 20  # 무한루프 방지
    
    for step in graph.stream(initial_state, {"recursion_limit": 50}):
        step_count += 1
        node_name = list(step.keys())[0]
        print(f"\n--- Step {step_count}: {node_name} 완료 ---")
        
        if step_count >= max_steps:
            print("[시스템] Max steps 도달, 종료")
            break
        
        final_state = step[node_name]
    
    print("\n" + "=" * 70)
    print("[시스템] 워크플로우 완료")
    
    if final_state:
        draft = final_state.get("draft", "")
        validation = final_state.get("validation_pass", False)
        iterations = final_state.get("iteration_count", 0)
        
        print(f"  - 초안 길이: {len(draft)}자")
        print(f"  - 검증 통과: {validation}")
        print(f"  - 수정 횟수: {iterations}회")
        print(f"  - RAG 청크: {len(final_state.get('rag_results', []))}개")
        print(f"  - 웹 결과: {len(final_state.get('web_results', []))}개")
        
        # 출력 파일 확인
        import glob
        output_files = glob.glob("outputs/*.pdf") + glob.glob("outputs/*.txt")
        if output_files:
            latest = max(output_files, key=os.path.getctime)
            print(f"  - 출력 파일: {latest}")
    
    print("=" * 70)
    return final_state


if __name__ == "__main__":
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 오류: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 설정하거나 환경변수로 export 하십시오.")
        sys.exit(1)
    
    result = run_analysis()
