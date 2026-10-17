# 증권사 리포트 질의응답: KG-RAG vs Naive RAG

증권사 종목분석 리포트를 대상으로, 지식그래프 기반 RAG(KG-RAG)가 기존 청크 기반 RAG(Naive RAG) 대비 어떤 질의 유형에서 유리한지 검증한 프로젝트입니다. (출처: 2025.03~06, 대학원 팀 프로젝트)

## 문제 정의

리포트는 서술형 해석과 수치 표가 섞여 있어, 고정 길이 청크 검색만으로는 "기업–산업–지표 간 관계"를 묻는 복합 질의에 한계가 있습니다. 문서에서 triple(subject–relation–object)을 추출해 그래프로 구조화하면 관계 추론 질의에서 이점이 있을 것이라는 가설을 세우고, 생성 모델을 동일하게 통제한 베이스라인과 비교했습니다.

## 데이터

- 네이버 증권에 게재된 증권사 종목분석 리포트 PDF
- 증권사 3사 × 산업군 3개 × 기업 9개, 총 27건 (표현 양식의 이질성을 포괄하도록 균형 구성)

## 파이프라인

```
PDF (pdfplumber)
  ├─ Naive RAG : 500자 청크(20자 중첩) → all-MiniLM-L6-v2 임베딩 → ChromaDB → top-k 청크 → Gemini
  └─ KG-RAG   : Gemini로 triple 추출 → Neo4j 저장 (+ triple 임베딩 → ChromaDB)
                → 질의와 유사한 top-5 triple 검색 → Neo4j 멀티홉 확장(2홉, 최대 50개) → Gemini
```

| 파일 | 역할 |
| --- | --- |
| `triple_extraction.py` | PDF 파싱, Gemini 기반 triple 추출, Neo4j 적재 |
| `kg-code-unified-ver2.py` | KG-RAG 검색(ChromaDB top-k → Neo4j 확장) 및 답변 생성 |
| `naive-rag.py` | 청크 기반 베이스라인 RAG |
| `eval.py` | 평가셋 기반 정답 매칭(숫자 단위 보정 포함) 및 정확도 집계 |

## 평가

- 복합 추론형 Closed-set QA 180문항: Accuracy
- 수치 조회형 Short-form QA 360문항: Recall@k
- 응답 생성 모델은 두 방식 모두 동일(Gemini)하게 두어 구조화 효과만 분리

## 결과

| 질의 유형 | 지표 | 결과 |
| --- | --- | --- |
| 복합 추론형 (180문항) | Accuracy | KG-RAG 66.02%, Naive RAG 대비 +6.75%p |
| 수치 조회형 (360문항) | Recall@k | Naive RAG 우위, KG-RAG 대비 +20.83%p |

- 관계 추론이 필요한 질의에서는 triple 구조화와 멀티홉 확장이 문서에 흩어진 정보를 연결해 주는 효과가 확인됨
- 단순 수치 조회에서는 표 데이터가 triple로 변환되며 왜곡되고, 멀티홉 확장이 노이즈를 더해 베이스라인이 우위
- 후속 방향: 질의 유형을 분류해 관계 추론은 KG-RAG, 수치 조회는 Naive RAG로 보내는 라우팅 구조

## 한계

- 리포트 내 그래프·차트 이미지는 처리하지 않음
- 서술형 응답의 정성 평가는 수행하지 않음
- 평가셋 규모(540문항)가 크지 않아 산업군별 일반화는 제한적

## 역할

팀 프로젝트로 진행했으며, 본인은 주제 기획·데이터 선정, 구성요소(파서·청킹·임베딩·LLM·Vector DB) 선정 실험, Naive RAG 베이스라인과 평가 체계 구축, 전체 파이프라인 구현에 참여했습니다. Triple 추출과 Neo4j 적재 코드는 팀원이 주도했고, 최종 코드 통합도 팀원이 담당했습니다.

## 실행

```bash
pip install -r requirements.txt
```

각 스크립트 상단의 `CONFIG`에 Gemini API 키, 데이터 폴더 경로, Neo4j 접속 정보를 입력한 뒤 실행합니다.

```bash
python triple_extraction.py        # triple 추출 및 Neo4j 적재
python kg-code-unified-ver2.py     # KG-RAG 질의응답
python naive-rag.py                # 베이스라인
python eval.py                     # 평가
```
