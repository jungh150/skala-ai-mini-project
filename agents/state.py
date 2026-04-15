"""
LangGraph AgentState 정의
"""
from typing import TypedDict, List, Dict, Optional, Annotated
import operator


class AgentState(TypedDict):
    """전체 워크플로우 상태"""
    query: str                          # 사용자 질의
    rag_results: List[Dict]             # RAG 검색 결과 (청크 + 메타데이터)
    web_results: List[Dict]             # 웹 검색 결과 (URL + 출처 유형)
    draft: str                          # 보고서 초안
    validation_pass: bool               # 검증 통과 여부
    feedback: str                       # Supervisor → Draft Agent 피드백
    iteration_count: int                # 초안 수정 횟수 (max 3)
    retry_count: int                    # 검색 재시도 횟수 (max 2)
    fallback_flags: Dict[str, bool]     # 정보 부재 항목 기록
    next: str                           # Supervisor가 결정하는 다음 노드
    validation_details: str             # 검증 상세 내용
