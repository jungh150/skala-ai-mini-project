"""
state.py
LangGraph AgentState 정의

워크플로우 전체에서 공유되는 상태를 TypedDict로 정의한다.
각 필드는 담당 Task(T1~T7)에 따라 구분된다.
"""
from typing import TypedDict, List, Dict


class AgentState(TypedDict):
    """워크플로우 전체 상태"""

    query: str                           # 사용자 질의

    # T1~T4: 검색 및 통합 평가
    search_queries: Dict[str, List[str]] # Supervisor가 수립한 검색 전략 (rag/web 구분)
    rag_results:    List[Dict]           # RAG 검색 결과 (청크 + 메타데이터)
    web_results:    List[Dict]           # 웹 검색 결과 (URL + 출처 유형)
    rag_done:       bool                 # RAG 노드 실행 완료 여부
    web_done:       bool                 # Web Search 노드 실행 완료 여부
    retrieval_ready: bool                # T4 Retrieval Sufficiency Check 통과 여부

    # TRL 평가 (T4와 T5 사이 독립 노드)
    trl_evidence:    Dict[str, List[Dict]] # 기술·회사별 구조화된 근거
    trl_assessment:  str                   # LLM TRL 추정 결과 원문
    missing_trl_info: List[str]            # TRL 판단에 부족한 정보 목록
    trl_ready:       bool                  # Draft 생성 가능 수준의 TRL 근거 확보 여부

    # T5~T7: 초안 생성 / 검증 / 포맷팅
    draft:           str                 # 보고서 초안
    validation_pass: bool                # Supervisor 검증 통과 여부
    feedback:        str                 # Supervisor → Draft Agent 피드백
    iteration_count: int                 # 초안 수정 횟수 (max 3)
    retry_count:     int                 # T4 검색 재시도 횟수 (max 2)
    trl_retry_count: int                 # TRL 근거 보강을 위한 재검색 횟수 (max 2)
    final_report_path: str               # Formatting Node가 생성한 PDF 경로
    fallback_flags:  Dict[str, bool]     # 정보 부재 항목 기록
    next:            str                 # Supervisor가 결정하는 다음 노드
    validation_details: str              # 검증 상세 내용 (디버깅용)
