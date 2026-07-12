# PDF RAG Chatbot

<table>
  <tr>
    <td width="42%" align="center" valign="top">
      <img src="assets/bcmd-2026-program.jpg" alt="2026 BCMD 프로그램" width="360">
    </td>
    <td width="58%" valign="middle">
      <h2>문서를 이해하고 근거로 답하는 AI</h2>
      <p>
        한국어 정책 PDF를 대상으로 문서 추출부터 검색 품질 개선,
        대화형 AI 서비스 구현까지 연결한 RAG 프로젝트입니다.
      </p>
      <p>
        <strong>PDF Extraction</strong> → <strong>Hybrid Search</strong> →
        <strong>Contextual Retrieval</strong> → <strong>Reranking</strong> →
        <strong>AI Chatbot</strong>
      </p>
      <p>
        동일한 평가셋을 사용해 네 가지 PDF 추출 방식과 여러 검색 전략을 비교하고,
        LCEL·LangGraph·ReAct를 거쳐 Streamlit 웹 서비스로 완성합니다.
      </p>
      <p>
        <sub>2026 BCMD · LangChain/ChatGPT 기반 AI 웹 서비스 개발</sub>
      </p>
    </td>
  </tr>
</table>

실습 문서는 관계부처 합동 **「저출생 추세 반전을 위한 대책」(2024. 6. 19.)**입니다. 문서의 텍스트·표를 여러 방식으로 추출하고 같은 평가셋으로 검색 및 답변 성능을 비교합니다. 교재 PDF의 페이지 이미지는 저작권을 고려해 README에 수록하지 않았습니다.

## 구현 흐름

```text
PDF
 └─ 문서 추출: PyMuPDF · PyMuPDF4LLM · Docling · VLM
     └─ 청크 분할 및 임베딩: BAAI/bge-m3
         └─ 검색: FAISS · BM25 · Ensemble
             └─ 검색 개선: Contextual Retrieval · Reranking
                 └─ 챗봇: LCEL · LangGraph · ReAct
                     └─ 웹 UI: Streamlit
```

## 학습 내용

| 단계 | 내용 |
|---|---|
| 환경 점검 | Poetry 가상환경, 필수 라이브러리, API 키, 로컬 AI 모델 확인 |
| 평가 설계 | 문서에 근거한 질문·정답 12개를 만들고 일관된 기준으로 성능 측정 |
| PDF 추출 | PyMuPDF, PyMuPDF4LLM, Docling, GPT-4o VLM의 텍스트·표 추출 결과 비교 |
| Base RAG | 문서를 청크로 나누고 `bge-m3`로 임베딩한 뒤 FAISS 검색 구성 |
| Hybrid Search | 의미 기반 FAISS와 키워드 기반 BM25를 Ensemble로 결합 |
| 한국어 검색 | Kiwi 형태소 분석을 적용해 한국어 키워드 검색 품질 개선 |
| Contextual Retrieval | 청크에 문서 맥락을 추가해 잘린 표나 짧은 문장의 의미 보완 |
| Reranking | 넓게 검색한 후보를 `bge-reranker-v2-m3`로 다시 정렬 |
| RAG 챗봇 | LCEL 체인, 대화 기억을 갖는 LangGraph, 도구를 선택하는 ReAct 구현 |
| 서비스 구현 | Streamlit 기반 질의응답 UI 실행 및 근거 표시·스트리밍 확장 |

## 노트북 구성

| 파일 | 주제 |
|---|---|
| `notebooks/00_smoke_test.ipynb` | 개발 환경과 모델 준비 상태 확인 |
| `notebooks/01_eval_set.ipynb` | RAG 개요, 평가셋 구성, 검색 없는 베이스라인 비교 |
| `notebooks/02_extraction_base_rag.ipynb` | PDF 추출 방식 비교 및 Base RAG 구축 |
| `notebooks/03_hybrid_search.ipynb` | FAISS, BM25, Ensemble 기반 하이브리드 검색 |
| `notebooks/04_contextual_rerank.ipynb` | Contextual Retrieval과 Cross-Encoder Reranking |
| `notebooks/05_chatbot.ipynb` | LCEL, LangGraph, ReAct 기반 챗봇 구현 |
| `app/streamlit_app.py` | 최종 RAG 챗봇 웹 애플리케이션 |

## 기술 스택

- Python 3.12+
- Poetry, Jupyter Notebook
- LangChain, LangGraph, OpenAI API
- Hugging Face Sentence Transformers
- BAAI/bge-m3, BAAI/bge-reranker-v2-m3
- FAISS, BM25, Kiwi
- PyMuPDF, PyMuPDF4LLM, Docling
- Streamlit

## 프로젝트 구조

```text
pdf-rag-chatbot/
├── app/
│   └── streamlit_app.py
├── assets/
│   └── bcmd-2026-program.jpg
├── cache/                    # 추출 결과와 검색 인덱스
├── data/                     # 실습용 정책 PDF
├── notebooks/               # 단계별 실습 노트북 00~05
├── poetry.lock
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 시작하기

### 1. 저장소 복제

```bash
git clone https://github.com/chanhyeokpp/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. 의존성 설치

Python 3.12와 Poetry가 필요합니다.

```bash
poetry install
```

### 3. OpenAI API 키 설정

프로젝트 루트에 `.env` 파일을 만들고 본인의 키를 입력합니다.

```text
OPENAI_API_KEY=your_api_key_here
```

`.env`는 `.gitignore`에 포함되어 있으며 GitHub에 업로드하면 안 됩니다. ChatGPT 구독과 OpenAI API 사용 요금은 별도입니다.

### 4. VS Code 커널 선택

`notebooks/00_smoke_test.ipynb`를 열고 오른쪽 위 커널에서 Poetry가 만든 Python 3.12 가상환경을 선택합니다. 이후 셀을 위에서 아래로 실행합니다.

### 5. 로컬 모델 준비

최초 실행 시 임베딩·리랭커·Docling 모델이 사용자 캐시 폴더에 다운로드됩니다. 모델 파일은 저장소에 포함되지 않으며 디스크 여유 공간 10GB 이상을 권장합니다.

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=600 poetry run python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

`00_smoke_test.ipynb`의 모델 확인 셀을 실행해도 같은 준비를 할 수 있습니다.

### 6. Streamlit 앱 실행

```bash
poetry run streamlit run app/streamlit_app.py
```

브라우저에서 `http://localhost:8501`에 접속합니다. 앱 종료는 실행 중인 터미널에서 `Ctrl+C`를 누릅니다.

## 캐시 사용

저장소에는 실습에 필요한 추출 결과와 검색 인덱스가 `cache/`에 포함되어 있습니다.

- `REBUILD=False`: 동봉된 캐시를 불러와 빠르게 실습
- `REBUILD=True`: 문서 추출·인덱스·맥락 데이터를 직접 생성

`REBUILD=True`는 처리 시간이 길고 OpenAI API 비용이 발생할 수 있습니다. 기본 학습과 실행에는 `REBUILD=False`를 권장합니다.

## PDF 추출 방식 비교

| 방식 | 특징 |
|---|---|
| PyMuPDF | 빠르고 간단한 기본 텍스트 추출 |
| PyMuPDF4LLM | LLM 입력에 적합한 Markdown 구조 생성 |
| Docling | 문서 구조와 표 보존에 강한 로컬 추출 |
| GPT-4o VLM | 복잡한 시각 구조를 해석할 수 있지만 API 비용과 처리 시간이 필요 |

## 참고 및 주의사항

- 실습 PDF의 내용과 권리는 원저작자 및 배포 기관에 있습니다.
- 이 저장소는 학습 목적으로 문서를 활용하며, README에는 교재 PDF 페이지 이미지를 포함하지 않습니다.
- `.env`, API 키, 개인 설정 파일은 공개 저장소에 커밋하지 마세요.
- 로컬 AI 모델은 수 GB 규모이므로 Git 저장소가 아닌 Hugging Face 캐시에 저장됩니다.

## 행사

2026 BCMD(BlockChain Meetup Day) 프로그램의 **LangChain·ChatGPT 기반 AI 웹 서비스 개발** 실습을 바탕으로 구성했습니다.
