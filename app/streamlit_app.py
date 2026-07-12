"""저출생 정책 RAG 챗봇 — Streamlit 배포 (세션 5의 LangGraph 챗봇을 웹 화면으로).

실행:  poetry run streamlit run app/streamlit_app.py

- 이 파일 하나로 동작한다 (다른 .py 필요 없음). 사전계산된 cache/idx_ctx_docling 을 불러서 기동한다.
- [1]~[5] 구획 주석을 따라 위에서 아래로 읽으면 앱 전체 구조가 보인다.
- 고급 기능(근거 표시·스트리밍)은 [2]의 스위치로 켜고 끈다.
- 맨 아래 "직접 해보기"에 실습 과제가 있다.
"""

# ============================================================================
# [1] 준비 — 키 · 경로
# ============================================================================
import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()                                     # .env 의 OPENAI_API_KEY 를 환경변수로 등록

ROOT = Path(__file__).resolve().parent.parent     # app/ 의 상위 = 프로젝트 폴더
# 세션 4에서 가장 좋았던 조합: docling 추출 + Contextual + Rerank
CTX_DOCLING_DIR = ROOT / "cache" / "idx_ctx_docling"

# ============================================================================
# [2] 실습용 설정 — 조절판 (여기 값만 바꾸고 저장 → 브라우저 새로고침)
#     초급자는 아래 build_chatbot() 함수 속을 뒤질 필요 없이,
#     이 블록의 값만 바꾸면 됩니다. 오타로 앱이 깨지면 값을 원래대로 되돌리세요.
# ============================================================================
TITLE         = "저출생 정책 RAG 챗봇"          # 화면 상단 제목
SYSTEM_PROMPT = "너는 저출생 정책 도우미야. 아래 맥락으로만 답해."  # 봇의 성격·규칙
SEARCH_K      = 20     # 검색 후보 개수(넓게 검색). 5로 줄이면 빠르지만 놓칠 수 있음
RERANK_TOPN   = 5      # 리랭크 후 남길 청크 수. 2로 줄이면 근거가 짧아짐
SHOW_SOURCES  = True   # 답변 아래 근거 청크 표시 (True / False)
STREAMING     = True   # 답을 타이핑되듯 흘려보내기 (True / False)
SOURCE_PREVIEW_CHARS = 300   # 근거 청크 미리보기 글자 수 (도전 과제용)

# ============================================================================
# [3] 화면 상단
# ============================================================================
# Streamlit 은 스크립트를 위에서 아래로 실행하며 화면을 그린다
st.set_page_config(page_title=TITLE)
st.title(TITLE)
st.caption("‘저출생 추세 반전을 위한 대책’(2024.6.19) 문서 기반 · LangGraph 멀티턴")


# ============================================================================
# [4] 챗봇 조립 — 무거운 준비는 1회만
#     @st.cache_resource: Streamlit 은 사용자가 입력할 때마다 이 스크립트 전체를
#     다시 실행한다. 이 데코레이터가 없으면 매 질문마다 2GB 모델을 다시 로드한다.
#     아래 함수는 [2] 조절판의 값(SEARCH_K·RERANK_TOPN·SYSTEM_PROMPT)을 읽어 쓴다.
# ============================================================================
@st.cache_resource(show_spinner="인덱스·모델 불러오는 중...")
def build_chatbot():
    """검색 인덱스 + LLM 으로 LangGraph 챗봇을 만든다 (앱 시작 시 1회).

    세션 5에서 만든 LangGraph 챗봇과 완전히 같은 구조다:
    State(대화 이력 + 검색 결과) → retrieve 노드 → generate 노드.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_core.messages import SystemMessage
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langgraph.graph import StateGraph, START, END, MessagesState
    from langgraph.checkpoint.memory import MemorySaver

    # 인덱스를 만들 때와 같은 임베딩 설정이어야 검색이 제대로 된다 (정규화 포함)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",
                                       model_kwargs={"device": "cpu"},
                                       encode_kwargs={"normalize_embeddings": True})
    ctx_db = FAISS.load_local(str(CTX_DOCLING_DIR), embeddings,
                              allow_dangerous_deserialization=True)
    # "넓게 검색(k=SEARCH_K) → 리랭커로 압축(상위 RERANK_TOPN개)" — 세션 4 최고 조합
    reranker = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3"), top_n=RERANK_TOPN)
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ctx_db.as_retriever(search_kwargs={"k": SEARCH_K}))
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def join_docs(docs):
        # 청크 리스트를 구분선으로 이어 하나의 문자열로
        return "\n---\n".join(d.page_content for d in docs)

    class State(MessagesState):
        context: str                              # 검색 결과를 담아 두는 칸

    def retrieve(state):
        # 노드 1: 마지막 사용자 메시지로 검색해 context 에 저장
        docs = retriever.invoke(state["messages"][-1].content)
        return {"context": join_docs(docs)}

    def generate(state):
        # 노드 2: 검색 결과 + 지금까지의 대화 전체로 답변 생성
        system = SystemMessage(content=f"{SYSTEM_PROMPT}\n\n{state['context']}")
        return {"messages": [llm.invoke([system] + state["messages"])]}

    # START → retrieve → generate → END 로 배선하고, MemorySaver 로 대화를 기억하게 한다
    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile(checkpointer=MemorySaver())


# ============================================================================
# [5] 안전 가드 — 준비물이 없으면 친절한 안내를 띄우고 멈춘다
# ============================================================================
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY 가 없습니다. `.env` 를 설정하세요 (.env.example 참고).")
    st.stop()
if not CTX_DOCLING_DIR.exists():
    st.error("cache/idx_ctx_docling 가 없습니다. 먼저 노트북 04를 실행해 맥락 인덱스를 만드세요.")
    st.stop()

chatbot = build_chatbot()                 # 캐시 덕분에 실제 로드는 첫 실행 때 한 번만


# ============================================================================
# [6] 대화 상태와 채팅 루프 — 입력 → 검색·생성 → 말풍선
# ============================================================================
# st.session_state: 이 브라우저 탭에서만 유지되는 저장 공간 (새 탭 = 새 대화)
# thread_id 는 탭마다 발급해 LangGraph 기억이 사용자끼리 섞이지 않게 한다
if "history" not in st.session_state:
    st.session_state.history = []         # [(role, text, sources|None)]
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


def show_sources(sources):
    # 고급 기능 1 — 근거 청크를 접었다 펼 수 있는 상자(expander)로 표시
    with st.expander(f"근거 청크 {len(sources)}개"):
        for i, chunk in enumerate(sources, 1):
            st.markdown(f"**{i}.** {chunk[:SOURCE_PREVIEW_CHARS]} …")


# 화면을 다시 그릴 때마다 지난 대화를 위에서부터 다시 표시한다
for role, text, sources in st.session_state.history:
    with st.chat_message(role):
        st.write(text)
        if SHOW_SOURCES and sources:
            show_sources(sources)

# st.chat_input: 화면 아래 채팅 입력창. 사용자가 입력하면 그 문자열이 들어온다
if user_input := st.chat_input("저출생 대책에 대해 물어보세요 (예: 육아휴직 급여 상한은?)"):
    st.chat_message("user").write(user_input)                # 사용자 말풍선
    st.session_state.history.append(("user", user_input, None))

    config = {"configurable": {"thread_id": st.session_state.thread_id}}   # 내 대화방 지정
    with st.chat_message("assistant"):                       # 봇 말풍선
        if STREAMING:
            # 고급 기능 2 — generate 노드가 만드는 LLM 토큰을 낱개로 받아 흘려보낸다
            def token_stream():
                for chunk, meta in chatbot.stream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config, stream_mode="messages"):
                    if meta.get("langgraph_node") == "generate" and chunk.content:
                        yield chunk.content
            with st.spinner("검색 중..."):
                answer = st.write_stream(token_stream())
        else:
            # 기본 동작 — 답이 다 만들어진 뒤 한 번에 표시
            with st.spinner("검색·생성 중..."):
                out = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config)
                answer = out["messages"][-1].content         # 마지막 메시지 = 이번 답변
            st.write(answer)

        sources = None
        if SHOW_SOURCES:
            # retrieve 노드가 상태에 담아 둔 검색 결과를 꺼내 청크 단위로 나눈다
            state = chatbot.get_state(config).values
            sources = state["context"].split("\n---\n")
            show_sources(sources)

    st.session_state.history.append(("assistant", answer, sources))


# ============================================================================
# 직접 해보기 — 맨 위 [2] 조절판의 값만 바꾸고 저장하면 브라우저 새로고침으로 반영된다
#              (함수 속을 뒤질 필요 없다 / streamlit run 재시작도 필요 없다)
#
# 1. 쉬움   : [2] 조절판의 TITLE(제목)·SYSTEM_PROMPT(봇 성격)를 바꿔 보세요.
#             저장 후 브라우저를 새로고침하면 바로 바뀝니다.
# 2. 중간   : [2]의 SHOW_SOURCES / STREAMING 을 False 로 꺼 보고 차이를 느껴 보세요.
#             SEARCH_K 를 20→5, RERANK_TOPN 을 5→2 로 줄이면 품질·속도가 어떻게 달라질까요?
# 3. 도전   : 간단히는 [2]의 SOURCE_PREVIEW_CHARS(근거 미리보기 글자 수)를 바꿔 보세요.
#             더 나아가 [6]의 show_sources 함수를 직접 고쳐도 됩니다 — 청크 전체 표시,
#             표시 개수를 상위 3개로 줄이기, expander 제목에 질문 함께 적기 등.
#
# 막히면 세션 5 노트북(05_chatbot.ipynb)의 LangGraph 부분과 비교하며 읽어 보세요 —
# 이 앱의 [4]는 그 노트북과 같은 코드입니다.
# ============================================================================
