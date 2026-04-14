# SDD — Software Design Document
## Ingestão e Busca Semântica com LangChain e PostgreSQL

**Versão:** 1.0  
**Data:** 2026-04-14  
**Autor:** Andre Apolinario  
**Referência:** [PRD.md](PRD.md)  
**Status:** Aprovado para implementação

---

## 1. Visão Geral da Arquitetura

O sistema é composto por três módulos Python independentes que se comunicam via banco de dados vetorial e variáveis de ambiente. A figura abaixo representa o fluxo de dados:

```
                ┌─────────────────────────────────────────┐
                │           docker-compose.yml             │
                │   PostgreSQL 17 + pgVector extension     │
                │   host: localhost  porta: 5432           │
                │   db: rag   user: postgres               │
                └──────────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                   │                    │
              ▼                   ▼                    ▼
       src/ingest.py        src/search.py         src/chat.py
    ──────────────────    ─────────────────    ─────────────────
    Carrega o PDF         Conecta ao store     Loop interativo
    Divide em chunks      Inicializa LLM       Lê pergunta
    Gera embeddings       Retorna chain()      Chama chain()
    Persiste no store     via LCEL             Exibe resposta
```

### Dependências entre módulos

```
chat.py  ──imports──►  search.py  (função search_prompt)
ingest.py              (independente, execução única)
```

---

## 2. Variáveis de Ambiente

Arquivo de referência: `.env.example`

| Variável | Obrigatória | Descrição | Exemplo |
|---|---|---|---|
| `OPENAI_API_KEY` | Condicional* | API Key da OpenAI | `sk-...` |
| `OPENAI_EMBEDDING_MODEL` | Não | Modelo de embedding OpenAI | `text-embedding-3-small` |
| `GOOGLE_API_KEY` | Condicional* | API Key da Google/Gemini | `AIza...` |
| `GOOGLE_EMBEDDING_MODEL` | Não | Modelo de embedding Gemini | `models/embedding-001` |
| `DATABASE_URL` | Sim | Connection string PostgreSQL | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Sim | Nome da coleção no pgVector | `documentos_rag` |
| `PDF_PATH` | Sim | Caminho para o arquivo PDF | `./document.pdf` |

> \* Pelo menos uma das chaves de API deve estar preenchida. O sistema detecta qual provedor usar pela presença das variáveis.

### Lógica de seleção de provedor

```
se OPENAI_API_KEY está definida  →  usa OpenAI (embeddings + LLM)
senão se GOOGLE_API_KEY está definida  →  usa Gemini (embeddings + LLM)
senão  →  raise RuntimeError
```

---

## 3. Spec: `docker-compose.yml`

**Status:** Já implementado. Nenhuma alteração necessária.

### Serviços

| Serviço | Imagem | Função |
|---|---|---|
| `postgres` | `pgvector/pgvector:pg17` | Banco principal com extensão pgVector |
| `bootstrap_vector_ext` | `pgvector/pgvector:pg17` | Executa `CREATE EXTENSION IF NOT EXISTS vector;` após o banco estar saudável |

### Pontos de atenção

- O healthcheck garante que o `bootstrap_vector_ext` só roda após o postgres estar aceitando conexões.
- O volume `postgres_data` persiste os dados entre reinicializações.
- A connection string esperada pelos scripts Python é:
  ```
  postgresql+psycopg://postgres:postgres@localhost:5432/rag
  ```

---

## 4. Spec: `src/ingest.py`

### Responsabilidade

Carregar o PDF, dividir em chunks, gerar embeddings e persistir no banco vetorial. Deve ser idempotente: re-execuções não duplicam dados.

### Variáveis de ambiente consumidas

`PDF_PATH`, `DATABASE_URL`, `PG_VECTOR_COLLECTION_NAME`, `OPENAI_API_KEY` ou `GOOGLE_API_KEY`

### Assinatura da função principal

```python
def ingest_pdf() -> None
```

### Algoritmo detalhado

```
1. load_dotenv()
2. Validar variáveis de ambiente obrigatórias
   └── [PDF_PATH, DATABASE_URL, PG_VECTOR_COLLECTION_NAME]
   └── raise RuntimeError se ausente
3. Verificar se o arquivo PDF existe no caminho informado
   └── raise FileNotFoundError se ausente
4. Carregar PDF com PyPDFLoader(PDF_PATH).load()
5. Dividir com RecursiveCharacterTextSplitter(
       chunk_size=1000,
       chunk_overlap=150,
       add_start_index=False
   ).split_documents(docs)
6. Se splits estiver vazio → raise SystemExit(0)
7. Enriquecer documentos: filtrar metadata com valores None ou ""
   └── Document(page_content=..., metadata={k:v para k,v em d.metadata.items() se v não em ("", None)})
8. Gerar IDs determinísticos: ids = [f"doc-{i}" para i em range(len(enriched))]
9. Instanciar embeddings:
   └── se OPENAI_API_KEY → OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL ou "text-embedding-3-small")
   └── senão             → GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL ou "models/embedding-001")
10. Instanciar PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )
11. store.add_documents(documents=enriched, ids=ids)
    └── IDs determinísticos garantem upsert (sem duplicação)
12. Imprimir: f"Ingestão concluída: {len(enriched)} chunks armazenados."
```

### Imports necessários

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector
# Condicional:
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
```

### Idempotência

Os IDs `doc-0`, `doc-1`, ..., `doc-N` são fixos para o mesmo PDF. O `add_documents` do `PGVector` com IDs explícitos faz upsert, impedindo duplicação em re-execuções.

---

## 5. Spec: `src/search.py`

### Responsabilidade

Inicializar e retornar uma chain LangChain (LCEL) que recebe uma pergunta e retorna uma resposta grounded no contexto do banco vetorial.

### Variáveis de ambiente consumidas

`DATABASE_URL`, `PG_VECTOR_COLLECTION_NAME`, `OPENAI_API_KEY` ou `GOOGLE_API_KEY`

### Assinatura da função principal

```python
def search_prompt() -> Runnable | None
```

Retorna `None` se a inicialização falhar (variáveis ausentes, banco inacessível).

### Template de Prompt

O template já existe no scaffold. Mantê-lo exatamente como está:

```python
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
```

### Algoritmo detalhado de `search_prompt()`

```
1. load_dotenv()
2. Validar variáveis obrigatórias: DATABASE_URL, PG_VECTOR_COLLECTION_NAME
   └── se ausente: imprimir erro e retornar None
3. Instanciar embeddings (mesma lógica do ingest.py)
4. Instanciar PGVector(
       embeddings=embeddings,
       collection_name=PG_VECTOR_COLLECTION_NAME,
       connection=DATABASE_URL,
       use_jsonb=True,
   )
5. Instanciar LLM:
   └── se OPENAI_API_KEY → ChatOpenAI(model="gpt-4o-mini") [ou model do env]
   └── senão             → ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
6. Construir chain usando LCEL:

   def build_chain(question: str) -> str:
       results = store.similarity_search_with_score(question, k=10)
       contexto = "\n\n".join([doc.page_content for doc, _score in results])
       prompt_text = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)
       response = llm.invoke(prompt_text)
       return response.content

   return build_chain

7. Em caso de Exception na inicialização: imprimir erro, retornar None
```

### Notas de design

- A função retorna um callable Python simples (`build_chain`), não um objeto LCEL complexo. Isso simplifica o uso em `chat.py` e facilita testes.
- O score retornado pelo `similarity_search_with_score` é ignorado na resposta ao usuário, mas pode ser logado para debug.
- O contexto é montado concatenando os `page_content` dos 10 chunks mais similares, separados por `\n\n`.

### Imports necessários

```python
import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_core.language_models import BaseChatModel
# Condicional:
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
```

---

## 6. Spec: `src/chat.py`

### Responsabilidade

Loop interativo no terminal que recebe perguntas do usuário, chama a chain de busca e exibe as respostas formatadas.

### Algoritmo detalhado de `main()`

```
1. chain = search_prompt()
2. Se chain é None:
   └── imprimir "Não foi possível iniciar o chat. Verifique os erros de inicialização."
   └── return
3. imprimir cabeçalho de boas-vindas
4. LOOP (while True):
   a. imprimir "\nFaça sua pergunta (Ctrl+C para sair):"
   b. question = input("PERGUNTA: ").strip()
   c. Se question vazio → continue (volta ao início do loop)
   d. resposta = chain(question)
   e. imprimir f"RESPOSTA: {resposta}"
5. Capturar KeyboardInterrupt:
   └── imprimir "\nEncerrando o chat. Até logo!"
   └── return
6. Capturar Exception genérica:
   └── imprimir f"Erro ao processar pergunta: {e}"
   └── continue (mantém o loop ativo)
```

### Formato de saída esperado

```
Bem-vindo ao chat RAG! Faça perguntas sobre o documento carregado.

Faça sua pergunta (Ctrl+C para sair):
PERGUNTA: Qual o faturamento da empresa?
RESPOSTA: O faturamento foi de 10 milhões de reais.

Faça sua pergunta (Ctrl+C para sair):
PERGUNTA: Qual a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

### Imports necessários

```python
from search import search_prompt
```

---

## 7. Spec: Configuração de Ambiente (`.env`)

### Arquivo `.env` a ser criado pelo desenvolvedor (baseado em `.env.example`)

```dotenv
# Escolha um provedor: preencha OpenAI OU Google
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# GOOGLE_API_KEY=AIza...
# GOOGLE_EMBEDDING_MODEL=models/embedding-001

# Banco de dados (padrão do docker-compose.yml)
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag

# Nome da coleção no pgVector (pode ser qualquer string)
PG_VECTOR_COLLECTION_NAME=documentos_rag

# Caminho para o PDF a ser ingerido
PDF_PATH=./document.pdf
```

### Validação esperada em cada script

Cada script deve validar suas variáveis na inicialização e falhar explicitamente com mensagem clara antes de tentar conectar ao banco ou API.

---

## 8. Diagrama de Sequência — Fluxo Completo

### Ingestão

```
Desenvolvedor          ingest.py           PyPDFLoader    RecursiveTextSplitter    PGVector (DB)
     │                     │                    │                  │                    │
     │  python src/ingest.py                    │                  │                    │
     │──────────────────►  │                    │                  │                    │
     │                     │  load(PDF_PATH)     │                  │                    │
     │                     │──────────────────►  │                  │                    │
     │                     │  [Document list]    │                  │                    │
     │                     │ ◄──────────────────  │                  │                    │
     │                     │  split_documents()  │                  │                    │
     │                     │─────────────────────────────────────►  │                    │
     │                     │  [chunks list]      │                  │                    │
     │                     │ ◄─────────────────────────────────────  │                    │
     │                     │  add_documents(chunks, ids)            │                    │
     │                     │───────────────────────────────────────────────────────────►  │
     │                     │  OK                 │                  │                    │
     │                     │ ◄───────────────────────────────────────────────────────────  │
     │  "Ingestão concluída"                     │                  │                    │
     │ ◄───────────────────  │                    │                  │                    │
```

### Chat

```
Usuário           chat.py          search.py           PGVector (DB)          LLM API
   │                 │                 │                     │                    │
   │  python src/chat.py               │                     │                    │
   │──────────────►  │                 │                     │                    │
   │                 │  search_prompt()│                     │                    │
   │                 │───────────────► │                     │                    │
   │                 │  chain (fn)     │                     │                    │
   │                 │ ◄───────────────│                     │                    │
   │  PERGUNTA: ...  │                 │                     │                    │
   │───────────────► │                 │                     │                    │
   │                 │  chain(question)│                     │                    │
   │                 │───────────────► │                     │                    │
   │                 │                 │  similarity_search  │                    │
   │                 │                 │────────────────────►│                    │
   │                 │                 │  [10 chunks]        │                    │
   │                 │                 │◄────────────────────│                    │
   │                 │                 │  llm.invoke(prompt) │                    │
   │                 │                 │─────────────────────────────────────────►│
   │                 │                 │  resposta           │                    │
   │                 │                 │◄─────────────────────────────────────────│
   │                 │  resposta       │                     │                    │
   │                 │◄───────────────│                     │                    │
   │  RESPOSTA: ...  │                 │                     │                    │
   │◄────────────────│                 │                     │                    │
```

---

## 9. Matriz de Rastreabilidade PRD → SDD

| Requisito PRD | Implementado em | Spec Seção |
|---|---|---|
| RF-01.1 — Carregar `document.pdf` | `ingest.py` passo 4 | §4 |
| RF-01.2 — Chunk 1000 / overlap 150 | `ingest.py` passo 5 | §4 |
| RF-01.3 — Gerar embeddings por chunk | `ingest.py` passo 9-10 | §4 |
| RF-01.4 — Persistir no pgVector | `ingest.py` passo 11 | §4 |
| RF-01.5 — Script isolado `ingest.py` | `src/ingest.py` | §4 |
| RF-01.6 — Idempotência | IDs determinísticos `doc-{i}` | §4 |
| RF-02.1 — Vetorizar pergunta | `search.py` passo 6a | §5 |
| RF-02.2 — k=10 resultados | `similarity_search_with_score(q, k=10)` | §5 |
| RF-02.3 — Montar contexto | `"\n\n".join(chunks)` | §5 |
| RF-02.4 — `src/search.py` isolado | `search_prompt()` | §5 |
| RF-03.1 — Prompt interativo | `input("PERGUNTA: ")` | §6 |
| RF-03.2 — Formato `RESPOSTA:` | `print(f"RESPOSTA: {resposta}")` | §6 |
| RF-03.3 — Loop contínuo | `while True` + `KeyboardInterrupt` | §6 |
| RF-03.4 — `src/chat.py` | `main()` em `chat.py` | §6 |
| RF-04.1 — Template de prompt | `PROMPT_TEMPLATE` em `search.py` | §5 |
| RF-04.2 — Grounding no contexto | REGRAS no prompt | §5 |
| RF-04.3 — Resposta padrão | REGRAS no prompt | §5 |
| RNF-01 — Docker Compose | `docker-compose.yml` | §3 |
| RNF-02 — Credenciais via env vars | `load_dotenv()` + validação | §2, §7 |
| RNF-03 — `.env.example` | `.env.example` | §7 |

---

## 10. Ordem de Implementação Recomendada

```
1. Verificar/ajustar .env  (§7)
2. docker compose up -d     (§3)
3. Implementar ingest.py    (§4)
4. Testar: python src/ingest.py
5. Implementar search.py    (§5)
6. Implementar chat.py      (§6)
7. Testar: python src/chat.py
```
