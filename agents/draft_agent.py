"""
draft_agent.py
Draft Generation Agent — 보고서 초안 생성

프로세스:
    1. RAG 결과 + 웹 결과 + TRL 평가 결과를 GPT-4o에 입력하여 초안을 생성한다.
    2. Self-Reflection으로 자체 검증 후 필요 시 개선한다.
    3. Supervisor 피드백이 있으면 재작성 시 반영한다.

확증 편향 방지:
    프롬프트에 긍정·부정 근거 병기 원칙을 명시하고,
    웹 결과를 회사별로 분리 포맷하여 특정 회사 편향을 방지한다.
"""
from typing import List, Dict
from langchain_openai import ChatOpenAI

from agents.state import AgentState
from prompts.templates import DRAFT_GENERATION_SYSTEM_PROMPT, DRAFT_SELF_CHECK_PROMPT


def _format_rag_results(rag_results: List[Dict]) -> str:
    if not rag_results:
        return "RAG 검색 결과 없음"

    formatted = []
    for i, r in enumerate(rag_results[:12], 1):
        formatted.append(
            f"[{i}] 출처: {r.get('title', 'Unknown')} ({r.get('source_type', '논문')})\n"
            f"    URL: {r.get('url', '')}\n"
            f"    내용: {r.get('content', '')[:600]}\n"
        )
    return "\n".join(formatted)


def _format_web_results(web_results: List[Dict]) -> str:
    if not web_results:
        return "웹 검색 결과 없음"

    # 확증 편향 방지를 위해 회사별로 분리하여 포맷
    by_company = {"Samsung": [], "SK Hynix": [], "Micron": [], "기타": []}
    for r in web_results:
        company = r.get("company", "")
        if not company:
            url = r.get("url", "").lower()
            if "samsung" in url:
                company = "Samsung"
            elif "skhynix" in url or "sk hynix" in url:
                company = "SK Hynix"
            elif "micron" in url:
                company = "Micron"
            else:
                company = "기타"
        if company not in by_company:
            company = "기타"
        by_company[company].append(r)

    formatted = []
    for company, results in by_company.items():
        if not results:
            continue
        formatted.append(f"\n=== {company} ===")
        for i, r in enumerate(results[:4], 1):
            formatted.append(
                f"[{company}-{i}] {r.get('title', r.get('topic', ''))} "
                f"({r.get('source_type', '뉴스')})\n"
                f"    URL: {r.get('url', '')}\n"
                f"    내용: {r.get('content', '')[:500]}\n"
            )
    return "\n".join(formatted)


def run_draft_agent(state: AgentState) -> AgentState:
    """Draft Generation Agent 노드 진입점."""
    print(f"\n[Draft Agent] 보고서 초안 생성 시작 (시도 {state.get('iteration_count', 0) + 1}/3)...")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=8000)

    prompt = DRAFT_GENERATION_SYSTEM_PROMPT.format(
        rag_results=_format_rag_results(state.get("rag_results", [])),
        web_results=_format_web_results(state.get("web_results", [])),
        trl_assessment=state.get("trl_assessment", "TRL 평가 결과 없음"),
        missing_trl_info="; ".join(state.get("missing_trl_info", [])) or "없음",
        feedback=state.get("feedback", "없음"),
    )

    print("  [Draft Agent] 초안 생성 중...")
    draft = llm.invoke(prompt).content.strip()

    # Self-Reflection: 자체 검증 후 필요 시 개선
    print("  [Draft Agent] Self-Reflection 수행 중...")
    check_prompt = DRAFT_SELF_CHECK_PROMPT.format(draft=draft[:3500])
    check_result = llm.invoke(check_prompt).content
    print(f"  [Self-Check] {check_result[:200]}...")

    if "수정 불필요" not in check_result:
        print("  [Draft Agent] Self-Reflection 기반 개선 중...")
        improve_prompt = f"""다음 보고서를 아래 피드백에 따라 개선하십시오.

규칙:
- 보고서 본문만 출력할 것
- 메타 코멘트 절대 금지
- 기존 구조(SUMMARY, 1~4, REFERENCE)를 유지할 것
- TRL 평가는 반드시 제공된 사전 평가 결과를 따를 것

자체 검토 피드백:
{check_result}

현재 보고서:
{draft}

개선된 보고서:"""
        draft = llm.invoke(improve_prompt).content.strip()

    state["draft"]           = draft
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    print(f"[Draft Agent] 초안 생성 완료 ({len(draft)}자)")
    return state
