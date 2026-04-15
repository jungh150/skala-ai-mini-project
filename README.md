# 반도체 R&D 기술 전략 분석 멀티에이전트 시스템

HBM4, PIM, CXL 기술에 대한 경쟁사 위협 수준을 분석하고, R&D 우선순위 의사결정을 지원하기 위한 멀티에이전트 기반 기술 전략 분석 시스템입니다.

## Overview
- **Objective** : 경쟁사의 HBM4, PIM, CXL 기술 현황과 위협 수준을 근거 기반으로 빠르게 분석하고, 이를 R&D 우선순위 설정 및 전략적 의사결정에 활용할 수 있도록 지원합니다.
- **Method** : Supervisor 중심의 멀티에이전트 워크플로우를 기반으로, 로컬 문서 검색과 웹 검색 결과를 통합한 뒤 초안 생성 및 검증 루프를 거쳐 최종 PDF 보고서를 생성합니다.
- **Tools** : LangGraph, LangChain, Python, bge-m3, Hybrid Retrieval (Dense + Sparse), PDF Generator

## Features
- PDF, 보고서, 기사 등 로컬 문서에서 관련 정보를 추출하고 청크 단위로 검색
- 웹 검색을 통해 뉴스, 논문, 특허, 채용공고, 기업 공식 자료 등 외부 근거 수집
- RAG 검색 결과와 웹 검색 결과를 통합하여 교차 검증 기반 분석 수행
- HBM4, PIM, CXL에 대한 경쟁사 기술 동향 및 위협 수준 평가
- TRL 기반 기술 성숙도 추정 및 직접 확인이 어려운 구간에 대한 한계 명시
- 보고서 초안 자동 생성 및 검증-수정 루프를 통한 품질 개선
- 일관된 형식의 최종 PDF 보고서 생성
- 확증 편향 방지 전략 적용: 관점별 쿼리 분리, 출처 다양성 확보, 상반된 근거 병기, TRL 4~6 구간의 간접 추정 한계 명시

## Tech Stack

| Category   | Details                           |
|------------|-----------------------------------|
| Framework  | LangGraph, LangChain, Python      |
| Retrieval  | Hybrid Retrieval (Dense + Sparse) |
| Embedding  | bge-m3                            |
| Output     | PDF Generator                     |

## Agents

- **Supervisor** : 전체 워크플로우를 조율하고, 검색 결과 충분성 판단, 초안 품질 검증, 재시도 및 종료를 관리합니다.
- **RAG Agent** : 로컬 문서에서 관련 청크를 검색하고 출처 메타데이터와 함께 반환합니다.
- **Web Search Agent** : 웹에서 외부 정보를 수집하고 URL 및 출처 유형 메타데이터와 함께 반환합니다.
- **Draft Generation Agent** : 수집된 근거를 바탕으로 경쟁사 기술 분석 및 TRL 기반 평가가 포함된 보고서 초안을 생성합니다.
- **Formatting Node** : 검증이 완료된 최종 초안을 PDF 보고서로 변환합니다.

## Architecture
![Architecture](./assets/architecture.png)

## Directory Structure
```text
├── data/                  # PDF, 보고서, 기사 등 분석 대상 문서
├── agents/                # Supervisor, RAG Agent, Web Search Agent, Draft Generation Agent
├── prompts/               # 에이전트별 프롬프트 템플릿
├── outputs/               # 생성된 초안 및 최종 PDF 보고서
├── app.py                 # 워크플로우 실행 스크립트
└── README.md

## Contributors
- 조정윤: 
- 신정화: 
- 김도연: 
- 권서현: 
