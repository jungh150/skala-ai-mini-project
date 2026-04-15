# HBM4·PIM·CXL 경쟁사 기술 전략 분석 시스템

SK Hynix R&D 담당자가 경쟁사(Samsung, Micron)의 HBM4·PIM·CXL 기술 위협 수준을 TRL 기반으로 분석하고, R&D 우선순위 의사결정에 활용할 수 있는 AI 에이전트 기반 보고서 자동 생성 시스템

## Overview

- **Objective** : HBM4·PIM·CXL 경쟁사 기술 성숙도를 TRL 기반으로 분석하여 R&D 전략 보고서 자동 생성
- **Method** : Supervisor 패턴 Multi-Agent Workflow — RAG + Web Search → TRL 추정 → 초안 생성 → 품질 검증 루프 → PDF 출력
- **Tools** : LangGraph, LangChain, FAISS, BM25, bge-m3, Tavily, GPT-4o, ReportLab

## Features

- arXiv 논문 7편(PIM 5편·CXL 2편) `PyPDFLoader` 적재 후 FAISS+BM25 Hybrid 검색
- Samsung·SK Hynix·Micron 공식 페이지 `WebBaseLoader` 실시간 로드
- **확증 편향 방지** : SK Hynix·Samsung·Micron·부정·중립 5개 관점 분리 쿼리, 출처 유형별(뉴스/논문/특허/IR/채용공고) 균형 수집, 상반 근거 병기 원칙 프롬프트 명시
- **TRL 데이터 기반 추정** : 수집된 공개 정보(양산 발표·IR 공시·특허 패턴·채용공고)를 LLM이 분석하여 TRL 판단
- **TRL 4~6 한계 명시** : 영업 비밀 구간은 간접 지표 기반 추정임을 보고서에 명시
- Supervisor 품질 검증 루프 — 미달 시 최대 3회 재작성
- PDF 보고서 자동 생성 — 표지·TRL 비교표·위협 수준 포함

## Tech Stack

| Category   | Details                                        |
|------------|------------------------------------------------|
| Framework  | LangGraph, LangChain, Python 3.10+             |
| LLM        | GPT-4o (Draft/TRL추정), GPT-4o-mini (Supervisor/검증) |
| Retrieval  | FAISS + BM25 Hybrid                            |
| Embedding  | BAAI/bge-m3 (다국어 SOTA, max 8192 tokens)     |
| Web Search | Tavily API (fallback: DuckDuckGo)              |
| PDF        | ReportLab                                      |

## Embedding Model Performance

후보군 4종 + 추가 2종, FAISS Similarity Retriever 고정, LLM 자동 생성 질문 20개 기반 평가

선정 기준: 한/영 혼재 문서 처리, 긴 논문 청크 대응(최대 8192 토큰), 오픈소스 필수

| 임베딩 모델 | Hit@1 | Hit@3 | Hit@5 | MRR | 오픈소스 | 비고 |
|------------|-------|-------|-------|-----|---------|------|
| **BAAI/bge-m3** | **0.750** | **0.950** | **1.000** | **0.843** | ✅ | **선정** — 다국어 SOTA, max 8192 tokens |
| voyage-3-large | 0.750 | 0.900 | 0.900 | 0.833 | ❌ 유료 | 선정 기준(오픈소스) 미충족 |
| multilingual-e5-large | 0.750 | 0.850 | 0.850 | 0.807 | ✅ | max 512 tokens로 긴 청크 처리 불리 |
| text-embedding-3-small | 0.450 | 0.700 | 0.750 | 0.579 | ❌ 유료 | 선정 기준(오픈소스) 미충족 |
| ko-sroberta-multitask | 0.000 | 0.000 | 0.000 | 0.008 | ✅ | 한국어 전용 — 영문 논문 도메인 부적합 |
| jina-embeddings-v3 | - | - | - | - | ✅ | transformers 버전 호환성 문제로 평가 제외 |

> 목표: Hit@3 ≥ 0.75, MRR ≥ 0.50
>
> **최종 선정: `BAAI/bge-m3`** — 오픈소스 조건을 충족하는 후보 중 Hit@3·MRR 모두 최고, 긴 논문 청크(max 8192 tokens) 처리 가능

## Retrieval Performance

LLM 자동 생성 질문 20개 기반 평가 (10-Retriever-Evaluation 구조 동일)

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |
|-----------|-------|-------|-------|-----|
| **Hybrid (FAISS+BM25)** | 0.800 | 0.900 | 0.950 | **0.860** |
| FAISS Similarity (bge-m3) | 0.800 | 0.900 | 0.950 | 0.852 |
| FAISS MMR | 0.800 | 0.800 | 0.850 | 0.812 |
| BM25 단독 | 0.200 | 0.350 | 0.350 | 0.258 |

> **Hybrid 채택 근거**: bge-m3 의미 검색 + BM25 영어 기술 용어 보완으로 MRR 최고 달성

## Agents

- **Supervisor** : 워크플로우 조율, 검색 충분성 평가(Rule 기반), 초안 품질 검증(LLM, max 3회), 종료 정책
- **RAG Agent** : arXiv 논문 7편 FAISS+BM25 Hybrid 검색, Query Rewrite 재검색(max 2회)
- **Web Search Agent** : 공식 페이지 WebBaseLoader + Tavily 5개 관점 다각도 쿼리, 출처 다양성 강제
- **TRL Evaluation Node** : RAG+웹 결과를 바탕으로 기술·회사별 TRL 추정, 부족 근거 식별 후 재검색 신호 전달
- **Draft Generation Agent** : TRL 추정 결과 + RAG/웹 컨텍스트 → 보고서 초안 생성 → Self-Reflection 자체 검증
- **Formatting Node** : 검증된 초안 → PDF 변환 (LLM 판단 없음)

## Architecture

```
User Query
    └→ Supervisor
        ├→ RAG Agent ─────────────→ Supervisor
        ├→ Web Search Agent ───────→ Supervisor
        ├→ TRL Evaluation Node ────→ Supervisor
        ├→ Draft Generation Agent
        │      ↑ 수정 피드백 (max 3회)
        │      ↓ 초안 검증 요청
        │   Supervisor (검증)
        └→ Formatting Node → PDF → Supervisor → END
```

## Directory Structure

```
├── agents/
│   ├── state.py                  # AgentState 정의
│   ├── supervisor.py             # Supervisor 노드
│   ├── rag_agent.py              # RAG Agent (FAISS+BM25 Hybrid)
│   ├── web_search_agent.py       # Web Search Agent
│   ├── trl_evaluation_node.py    # TRL Evaluation Node
│   ├── draft_agent.py            # Draft Agent (TRL 기반 보고서 생성)
│   └── formatting_node.py        # PDF Formatting Node
├── prompts/
│   └── templates.py              # 전체 프롬프트 템플릿
├── data/                         # arXiv PDF 캐시 및 FAISS 인덱스 (실행 시 자동 생성)
├── outputs/                      # 생성된 PDF 보고서 (실행 시 자동 생성)
├── evaluate_retrieval.py         # Retriever 비교 평가 (Hit Rate@K, MRR)
├── evaluate_embedding.py         # 임베딩 모델 비교 평가 (Hit Rate@K, MRR)
├── app.py                        # 메인 실행 스크립트
├── pyproject.toml                # uv 의존성 관리
├── requirements.txt
└── README.md
```

## 실행 방법

```bash
# 1. 의존성 설치
uv sync

# 2. 환경변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY, TAVILY_API_KEY 입력

# 3. 실행 (첫 실행 시 arXiv PDF 자동 다운로드 및 인덱싱 — 약 5분 소요)
uv run python app.py

# 4. 임베딩 모델 비교 평가 (선택)
uv run python evaluate_embedding.py

# 5. Retriever 비교 평가 (선택)
uv run python evaluate_retrieval.py

# 평가셋 캐시 초기화 후 재생성 (문서 변경 시)
uv run python evaluate_embedding.py --reset-eval
```

## Contributors

- 조정윤 : Supervisor Agent 설계, Control Strategy (Loop/Branch/Termination), LangGraph Graph 구성
- 신정화 : RAG Agent 개발 (FAISS+BM25 Hybrid Retrieval, Query Rewrite, 인덱싱 파이프라인)
- 김도연 : Web Search Agent 개발 (Tavily/DuckDuckGo, 확증 편향 방지 전략, 출처 다양성 강제)
- 권서현 : Draft Generation Agent 개발 (TRL 기반 보고서 생성, Self-Reflection), PDF Formatting Node
