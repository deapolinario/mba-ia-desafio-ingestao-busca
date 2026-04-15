import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def _validate_env(*keys: str) -> None:
    for key in keys:
        if not os.getenv(key):
            raise RuntimeError(f"Variável de ambiente obrigatória não definida: {key}")


def _build_embeddings():
    if os.getenv("OPENAI_API_KEY"):
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)

    if os.getenv("GOOGLE_API_KEY"):
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    raise RuntimeError(
        "Nenhum provedor de embeddings configurado. "
        "Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env"
    )


def ingest_pdf() -> None:
    _validate_env("DATABASE_URL", "PG_VECTOR_COLLECTION_NAME", "PDF_PATH")

    pdf_path = Path(os.getenv("PDF_PATH"))
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path.resolve()}")

    print(f"Carregando PDF: {pdf_path.resolve()}")
    docs = PyPDFLoader(str(pdf_path)).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=False,
    ).split_documents(docs)

    if not splits:
        print("Nenhum conteúdo extraído do PDF. Encerrando.")
        raise SystemExit(0)

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
        )
        for d in splits
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

    print(f"Chunks gerados: {len(enriched)}. Gerando embeddings e persistindo...")

    embeddings = _build_embeddings()

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    store.add_documents(documents=enriched, ids=ids)

    print(f"Ingestão concluída: {len(enriched)} chunks armazenados.")


if __name__ == "__main__":
    ingest_pdf()
