"""
부산외국어대학교 학사 안내 다국어 AI 챗봇.

db 폴더의 ChromaDB에서 관련 문서를 검색(RAG)한 뒤,
HuggingFace Inference API의 LLM으로 질문과 동일한 언어의 답변을 생성한다.

실행:
    streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "db"

EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
CHAT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
RETRIEVAL_K = 4

SYSTEM_PROMPT = """\
당신은 부산외국어대학교 학생들을 돕는 학사 안내 챗봇입니다.
아래 [참고 문서]에 있는 내용만 근거로 답변하세요.

규칙:
- Answer in the same language as the user's question. If the question is in English, reply in English. If in Korean, reply in Korean.
- [참고 문서]가 질문과 다른 언어로 되어 있다면, 내용을 질문의 언어로 번역하고 정리해서 답변하세요.
- [참고 문서]에서 답을 찾을 수 없으면, 모른다고 솔직히 말하고 학사지원팀에 문의하라고 안내하세요.
- 날짜, 기간, 금액 등 구체적인 수치는 문서에 있는 그대로 정확히 인용하세요.
- 답변은 간결하고 명확하게 작성하세요.

[참고 문서]
{context}
"""


@st.cache_resource(show_spinner=False)
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def source_names(docs):
    names = []
    for doc in docs:
        name = Path(doc.metadata.get("source", "")).name
        if name and name not in names:
            names.append(name)
    return names


@st.cache_resource(show_spinner=False)
def build_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    llm = HuggingFaceEndpoint(repo_id=CHAT_MODEL, task="text-generation", max_new_tokens=512, temperature=0.1)
    chat_model = ChatHuggingFace(llm=llm)
    return prompt | chat_model | StrOutputParser()


def answer_question(retriever, chain, question):
    docs = retriever.invoke(question)
    if not docs:
        return "관련된 내용을 찾을 수 없습니다.", []
    answer = chain.invoke({"context": format_docs(docs), "question": question})
    return answer, source_names(docs)


def main():
    load_dotenv()

    st.set_page_config(page_title="부산외대 학사 안내 챗봇", page_icon="🎓")
    st.title("🎓 부산외대 학사 안내 챗봇")
    st.caption("2026학년도 2학기 학사 안내 문서를 기반으로 답변합니다.")

    if not (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")):
        st.error(
            "HUGGINGFACEHUB_API_TOKEN 환경변수가 설정되어 있지 않습니다. "
            ".env 파일에 HuggingFace 토큰을 추가해주세요."
        )
        st.stop()

    if not DB_DIR.exists():
        st.error(f"'{DB_DIR}' 폴더가 없습니다. 먼저 `python ingest.py`를 실행해주세요.")
        st.stop()

    retriever = load_retriever()
    chain = build_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.text_input(
        "질문을 입력하세요",
        placeholder="학사 일정, 수강신청, 등록금 등 궁금한 점을 물어보세요.",
    )
    ask_clicked = st.button("질문", type="primary")

    if ask_clicked and question.strip():
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중입니다..."):
                answer, sources = answer_question(retriever, chain, question)
            if sources:
                answer += "\n\n---\n**참고 문서:** " + ", ".join(sources)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
