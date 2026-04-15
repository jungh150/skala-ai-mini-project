"""
Draft Generation Agent: RAG + Web Search 결과를 종합하여 보고서 초안 생성
- TRL 기반 경쟁사 분석 포함
- 확증 편향 방지 원칙 적용
- Self-Reflection으로 자체 검증
- TRL은 수집된 데이터 기반으로 GPT-4o가 직접 추정 (하드코딩 없음)
"""
from typing import List, Dict
from langchain_openai import ChatOpenAI

from agents.state import AgentState
from prompts.templates import DRAFT_GENERATION_SYSTEM_PROMPT, DRAFT_SELF_CHECK_PROMPT

TRL_ESTIMATION_PROMPT = """당신은 반도체 기술 TRL(Technology Readiness Level) 분석 전문가입니다.
아래 수집된 공개 정보를 분석하여 각 기술·회사별 TRL을 추정하십시오.

## TRL 9단계 판단 기준
| TRL | 정의 |
|-----|------|
| TRL 1 | 기초 원리 관찰, 아이디어/이론 수준 |
| TRL 2 | 기술 개념 정립, 적용 가능성 검토 |
| TRL 3 | 개념 검증, 실험실 수준 실증 |
| TRL 4 | 부품 검증, 실험실 환경 통합 |
| TRL 5 | 부품 검증(실환경), 유사 환경 통합 테스트 |
| TRL 6 | 시스템 시연, 실제 환경 유사 조건 시연 |
| TRL 7 | 시스템 시제품, 실제 운용 환경 시연 |
| TRL 8 | 시스템 완성, 양산 적합성 검증 완료 |
| TRL 9 | 실제 운용, 상용 양산 및 납품 |

## 공개 정보 신뢰도 등급
- **직접 확인 가능 (TRL 7~9)**: 양산 완료 발표, 고객사 출하 공시, IR 실적 발표
- **간접 추정 필요 (TRL 4~6)**: 특허 출원 패턴, 학회 발표 빈도, 채용공고 키워드, 파트너십 발표
- **연구 단계 (TRL 1~3)**: 논문, 학회 발표, 특허 출원

## 추정 원칙
1. 수집된 정보에 명시적 근거가 있을 때만 TRL 7~9 부여
2. 근거가 불충분하면 TRL 4~6으로 보수적 추정하고 한계 명시
3. 각 TRL에 사용한 근거 지표를 반드시 1~2개 명시
4. "~로 추정된다", "간접 지표 기반으로 판단된다" 표현 사용

## 분석 대상
- 기술: HBM4, PIM, CXL
- 회사: SK Hynix, Samsung, Micron

## 수집된 정보
### RAG 문서 (논문 기반)
{rag_results}

### 웹 검색 결과 (공식 발표/뉴스 기반)
{web_results}

## 출력 형식 (반드시 이 형식으로)
### HBM4
- SK Hynix: TRL X~Y | 근거: [근거 지표 1~2개] | 신뢰도: 직접확인/간접추정
- Samsung: TRL X~Y | 근거: [근거 지표 1~2개] | 신뢰도: 직접확인/간접추정
- Micron: TRL X~Y | 근거: [근거 지표 1~2개] | 신뢰도: 직접확인/간접추정

### PIM
- SK Hynix: TRL X~Y | 근거: [근거 지표 1~2개] | 신뢰도: 직접확인/간접추정
- Samsung: TRL X~Y | 근거: [근거 지표 1~2개] | 신뢰도: 직접확인/간접추정
- Micron: TRL X~Y | 근거: [근거 지표 1~2개] | 신뢰도: 직접확인/간접추정

### CXL
- SK Hynix: TRL X~Y | 근거: [근거 지표 1~2개] | 신뢰도: 직접확인/간접추정
- Samsung: TRL X~Y | 근거: [근거 지표 1~2개] | 신뢰도: 직접확인/간접추정
- Micron: TRL X~Y | 근거: [근거 지표 1~2개] | 신뢰도: 직접확인/간접추정
"""


def _format_rag_results(rag_results: List[Dict]) -> str:
    """RAG 결과 포맷팅"""
    if not rag_results:
        return "RAG 검색 결과 없음"

    formatted = []
    for i, r in enumerate(rag_results[:12], 1):
        title = r.get("title", "Unknown")
        url = r.get("url", "")
        source_type = r.get("source_type", "논문")
        content = r.get("content", "")[:600]
        formatted.append(
            f"[{i}] 출처: {title} ({source_type})\n"
            f"    URL: {url}\n"
            f"    내용: {content}\n"
        )
    return "\n".join(formatted)


def _format_web_results(web_results: List[Dict]) -> str:
    """웹 검색 결과 포맷팅"""
    if not web_results:
        return "웹 검색 결과 없음"

    by_company = {"Samsung": [], "SK Hynix": [], "Micron": [], "기타": []}

    for r in web_results:
        company = r.get("company", "")
        if not company:
            url = r.get("url", "").lower()
            if "samsung" in url:
                company = "Samsung"
            elif "skhynix" in url or "sk hynix" in url.lower():
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
            source_type = r.get("source_type", "뉴스")
            url = r.get("url", "")
            title = r.get("title", r.get("topic", ""))
            content = r.get("content", "")[:500]
            formatted.append(
                f"[{company}-{i}] {title} ({source_type})\n"
                f"    URL: {url}\n"
                f"    내용: {content}\n"
            )

    return "\n".join(formatted)


def _estimate_trl(llm, rag_formatted: str, web_formatted: str) -> str:
    """수집된 데이터 기반 TRL 추정 (하드코딩 없음)"""
    print("  [TRL 추정] 수집된 데이터 기반 TRL 분석 중...")
    prompt = TRL_ESTIMATION_PROMPT.format(
        rag_results=rag_formatted,
        web_results=web_formatted
    )
    response = llm.invoke(prompt)
    trl_result = response.content
    print(f"  [TRL 추정 완료]\n{trl_result[:400]}...")
    return trl_result


def run_draft_agent(state: AgentState) -> AgentState:
    """Draft Generation Agent 노드 실행"""
    print(f"\n[Draft Agent] 보고서 초안 생성 시작 (시도 {state.get('iteration_count', 0) + 1}/3)...")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=8000)

    rag_formatted = _format_rag_results(state.get("rag_results", []))
    web_formatted = _format_web_results(state.get("web_results", []))
    feedback = state.get("feedback", "없음")

    # Step 1: 수집된 데이터 기반 TRL 추정 (하드코딩 없음)
    trl_estimation = _estimate_trl(llm, rag_formatted, web_formatted)

    # Step 2: TRL 추정 결과를 포함하여 보고서 초안 생성
    prompt = DRAFT_GENERATION_SYSTEM_PROMPT.format(
        rag_results=rag_formatted,
        web_results=web_formatted,
        feedback=feedback
    )
    # TRL 추정 결과를 프롬프트에 주입
    prompt += f"\n\n## 사전 TRL 분석 결과 (수집된 데이터 기반 추정값 — 이 값을 보고서에 반영할 것)\n{trl_estimation}"

    print("  [Draft Agent] LLM으로 보고서 초안 생성 중...")
    response = llm.invoke(prompt)
    draft = response.content

    # Step 3: Self-Reflection
    print("  [Draft Agent] Self-Reflection 수행 중...")
    check_prompt = DRAFT_SELF_CHECK_PROMPT.format(draft=draft[:3000])
    check_response = llm.invoke(check_prompt)
    check_result = check_response.content
    print(f"  [Self-Check] {check_result[:200]}...")

    # Self-check 기반 개선
    if "수정 불필요" not in check_result:
        print("  [Draft Agent] Self-check 기반 보고서 개선 중...")
        improve_prompt = f"""다음 보고서를 아래 피드백에 따라 개선하십시오.

규칙:
- 보고서 본문만 출력할 것
- "보고서를 개선하기 위해", "위 보고서는" 같은 메타 코멘트 절대 포함 금지
- 개선된 보고서 내용만 그대로 출력

자체 검토 피드백:
{check_result}

현재 보고서:
{draft}

개선된 보고서:"""
        improved_response = llm.invoke(improve_prompt)
        draft = improved_response.content

    state["draft"] = draft
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    print(f"[Draft Agent] 초안 생성 완료 ({len(draft)}자)")
    return state
