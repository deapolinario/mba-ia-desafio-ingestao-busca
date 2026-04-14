import os
from typing import Callable
from dotenv import load_dotenv

from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def _build_embeddings():
    if os.getenv("OPENAI_API_KEY"):
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)

    if os.getenv("GOOGLE_API_KEY"):
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(model=model)

    raise RuntimeError(
        "Nenhum provedor configurado. Defina OPENAI_API_KEY ou GOOGLE_API_KEY no .env"
    )


def _build_llm() -> tuple:
    if os.getenv("OPENAI_API_KEY"):
        model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model), model

    if os.getenv("GOOGLE_API_KEY"):
        model = os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite")
        return ChatGoogleGenerativeAI(model=model), model

    raise RuntimeError(
        "Nenhum provedor de LLM configurado. Defina OPENAI_API_KEY ou GOOGLE_API_KEY no .env"
    )


def search_prompt() -> Callable[[str], str] | None:
    required = ["DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"]
    for key in required:
        if not os.getenv(key):
            print(f"Erro: variável de ambiente não definida: {key}")
            return None

    try:
        embeddings = _build_embeddings()
        llm, model_name = _build_llm()

        store = PGVector(
            embeddings=embeddings,
            collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
            connection=os.getenv("DATABASE_URL"),
            use_jsonb=True,
        )

        def chain(question: str) -> str:
            results = store.similarity_search_with_score(question, k=10)
            contexto = "\n\n".join(doc.page_content for doc, _score in results)
            prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)
            response = llm.invoke(prompt)

            usage = response.usage_metadata or {}
            input_tokens = usage.get("input_tokens", "—")
            output_tokens = usage.get("output_tokens", "—")
            total_tokens = usage.get("total_tokens", "—")
            print(
                f"[modelo: {model_name} | "
                f"tokens — entrada: {input_tokens}, "
                f"saída: {output_tokens}, "
                f"total: {total_tokens}]"
            )

            return response.content

        return chain

    except Exception as e:
        print(f"Erro ao inicializar o sistema de busca: {e}")
        return None
