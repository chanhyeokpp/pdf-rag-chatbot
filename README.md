# RAG 실전 강의 — 저출생 단일 코퍼스 (초급자용)

‘저출생 추세 반전을 위한 대책’(관계부처 합동, 2024.6.19) 문서 하나로
**추출 → 검색 → 정밀화 → 챗봇 → 배포**까지 RAG 전 과정을 배우는 1일 실습 강의.

- **세션당 주피터 노트북 1개.** 각 노트북은 **단독 실행**됩니다(다른 .py import 없음, 헬퍼는 노트북 안에 포함).
- 초급자 대상으로 코드를 쉽게 쓰고 **한국어 주석**을 달았습니다.
- 실습은 각 노트북 안의 **"직접 해보기"** 로 통합돼 있습니다.

## 빠른 시작

```bash
# 1) 의존성 설치
poetry install            # 또는: pip install -r requirements.txt

# 2) 본인 API 키 설정 (수강생 각자 자신의 키 사용)
cp .env.example .env      # .env 에 OPENAI_API_KEY 입력

# 3) 모델 미리 받기 (bge-m3 / reranker / docling) — 강의 전 1회 필수 권장
#    합쳐서 약 5GB 다운로드 (디스크 여유 10GB 확보). 강의장에서 동시에 받으면 한 세션이 사라집니다!
poetry run python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')"
poetry run python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"

# 4) 환경 점검
poetry run jupyter notebook notebooks/00_smoke_test.ipynb

# 5) 챗봇 데모
poetry run streamlit run app/streamlit_app.py
```

> **예상 API 비용**: `REBUILD=False`(기본)로 전 노트북을 따라 하면 **$1~2 수준**입니다.
> `REBUILD=True` 로 VLM 추출·맥락 생성까지 직접 돌려도 **$10 이내**로 설계돼 있습니다.
> **디스크**: 모델 다운로드(bge-m3·리랭커·docling 합계 약 5GB) 포함 **10GB 여유**를 권장합니다.

## 커리큘럼 (세션당 노트북 1개)

| 노트북 | 주제 | 핵심 |
|--------|------|------|
| `00_smoke_test.ipynb` | 환경 점검 | 라이브러리·모델·API 키 확인 |
| `01_eval_set.ipynb` | RAG 개요 & 평가셋 | 정답 12문항(ground truth) |
| `02_extraction_base_rag.ipynb` | 추출 & Base RAG | PyMuPDF·4LLM·**docling**·VLM 비교 + FAISS |
| `03_hybrid_search.ipynb` | 하이브리드 검색 | BM25 + Ensemble |
| `04_contextual_rerank.ipynb` | 맥락 추가 & 재정렬 | Contextual + Cross-Encoder Rerank |
| `05_chatbot.ipynb` | 챗봇 | LCEL → LangGraph → ReAct |
| `app/streamlit_app.py` | 배포 (세션 7) | 노트북이 아니라 **이 파일을 직접 수정**하는 실습 |

> **세션 7 실습 안내** — `app/streamlit_app.py` 를 열어 파일 끝의 "직접 해보기"를 따라 하세요.
> 근거 표시·스트리밍 같은 고급 기능은 파일 상단의 `SHOW_SOURCES` / `STREAMING` 스위치로 켜고 끕니다.
> 수정 후 저장하면 브라우저 새로고침만으로 반영됩니다 (`streamlit run` 재시작 불필요).

## 직접 빌드 vs 캐시

수강생은 **자기 API 키로 인덱스를 직접 만들 수도 있고, 동봉된 캐시를 쓸 수도 있습니다.**
- **세션 02**: 추출 → `cache/splits.pkl` + `cache/faiss_index/` + `cache/splits_all.pkl` + `cache/idx_*` 저장
- **세션 04**: 맥락 생성 → `cache/contextual_all.pkl` + `cache/idx_ctx_*` 저장
- **세션 03·05 및 앱**: 위에서 저장한 결과를 불러서 사용

각 노트북 상단 `REBUILD` 스위치:
- `REBUILD=True` → 직접 빌드(VLM 추출 등, 시간·소액 과금)
- `REBUILD=False` → 동봉된 `cache/` 를 폴백으로 사용(빠름)

## PDF 추출 4종

| 방법 | 표 보존 | 비용 |
|------|---------|------|
| PyMuPDF | 약함 | 무료·빠름 |
| PyMuPDF4LLM | 보통 | 무료·빠름 |
| **docling** | 좋음 | 무료·로컬(AI 모델) |
| **VLM (GPT-4o)** | 매우 좋음 | 유료·느림 |

> 이 문서는 디지털 PDF라 텍스트가 깨끗해 네 방법 점수가 비슷합니다.
> docling·VLM의 진가는 **스캔본·복잡표** 문서에서 드러납니다.

## 폴더 구조

```
bcmd_rag_lecture/
├── notebooks/                 # 세션당 단독 노트북 1개 (00~05)
├── data/                      # 저출생 PDF
├── cache/                     # splits_all.pkl, contextual_all.pkl, idx_* / idx_ctx_*, faiss_index/
├── app/streamlit_app.py       # 배포용 챗봇 (단독 실행, 고급 기능 스위치 내장)
├── slides/                    # 강의 슬라이드(pptx)
├── .env.example
├── requirements.txt / pyproject.toml / poetry.lock
└── README.md
```

## 기술 스택
Python 3.12 또는 3.13 · LangChain 0.3.x · langgraph 0.2.76 · streamlit 1.58.0 ·
임베딩 `bge-m3` · 리랭커 `bge-reranker-v2-m3` · FAISS · BM25(+Kiwi) · docling · LLM `gpt-4o-mini`
