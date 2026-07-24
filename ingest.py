"""
data 폴더의 PDF를 읽어 청크로 분할하고, HuggingFace Embeddings(jhgan/ko-sroberta-multitask)로
임베딩을 생성한 뒤 ChromaDB(db 폴더)에 저장하는 스크립트.

실행:
    python ingest.py
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"


def load_documents():
    """data 폴더의 모든 PDF를 로드한다."""
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"'{DATA_DIR}' 폴더에서 PDF 파일을 찾을 수 없습니다.")

    documents = []
    for pdf_path in pdf_paths:
        print(f"로딩 중: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    print(f"총 {len(pdf_paths)}개 PDF에서 {len(documents)}페이지를 로드했습니다.")
    return documents


def split_documents(documents):
    """문서를 청크로 분할한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)}개의 청크로 분할했습니다.")
    return chunks


def build_vectorstore(chunks):
    """임베딩을 생성하고 ChromaDB에 저장한다."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
    )
    print(f"'{DB_DIR}' 에 벡터스토어 저장을 완료했습니다.")
    return vectorstore


def main():
    documents = load_documents()
    chunks = split_documents(documents)
    build_vectorstore(chunks)


if __name__ == "__main__":
    main()
