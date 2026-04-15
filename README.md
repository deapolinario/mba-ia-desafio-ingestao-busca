# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema de busca semântica baseado em RAG (Retrieval-Augmented Generation) que ingere um documento PDF em um banco vetorial PostgreSQL com pgVector e permite consultas em linguagem natural via terminal.

---

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- API Key da [OpenAI](https://platform.openai.com/) **ou** da [Google (Gemini)](https://aistudio.google.com/)

---

## Configuração do ambiente

### 1. Clone o repositório e crie o ambiente virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Configure as variáveis de ambiente

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp .env.example .env
```

Edite o `.env`:

```dotenv
# Escolha um provedor — preencha OpenAI OU Google
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# GOOGLE_API_KEY=AIza...
# GOOGLE_EMBEDDING_MODEL=models/embedding-001

# Banco de dados (padrão do docker-compose.yml)
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag

# Nome da coleção no pgVector
PG_VECTOR_COLLECTION_NAME=documentos_rag

# Caminho para o PDF a ser ingerido
PDF_PATH=document.pdf
```

> Se ambas as chaves estiverem preenchidas, o sistema usa OpenAI por padrão.

---

## Execução

### Passo 1 — Subir o banco de dados

```bash
docker compose up -d
```

Aguarde o container ficar `healthy` antes de continuar:

```bash
docker compose ps
```

### Passo 2 — Ingerir o PDF

```bash
python src/ingest.py
```

Saída esperada:

```
Carregando PDF: .../document.pdf
Chunks gerados: 67. Gerando embeddings e persistindo...
Ingestão concluída: 67 chunks armazenados.
```

> A ingestão é **idempotente** — pode ser re-executada sem duplicar dados.

### Passo 3 — Iniciar o chat

```bash
python src/chat.py
```

Saída esperada:

```
Bem-vindo ao chat Desafio_1! Faça perguntas sobre o documento carregado.
Pressione Ctrl+C para sair.

Faça sua pergunta:
PERGUNTA: Qual o faturamento da empresa?
[modelo: gpt-4o-mini | tokens — entrada: 1843, saída: 38, total: 1881]
RESPOSTA: O faturamento foi de 10 milhões de reais.

Faça sua pergunta:
PERGUNTA: Qual a capital da França?
[modelo: gpt-4o-mini | tokens — entrada: 1850, saída: 18, total: 1868]
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Para encerrar, pressione `Ctrl+C`.

---

## Estrutura do projeto

```
├── docker-compose.yml        # PostgreSQL 17 + extensão pgVector
├── requirements.txt          # Dependências Python
├── .env.example              # Template de variáveis de ambiente
├── src/
│   ├── ingest.py             # Pipeline de ingestão do PDF
│   ├── search.py             # Busca semântica e geração de resposta
│   └── chat.py               # Interface CLI interativa
├── document.pdf              # PDF a ser ingerido
└── docs/
    ├── feat.md               # Especificação do desafio
    ├── PRD.md                # Product Requirements Document
    └── SDD.md                # Software Design Document
```

---

## Stack tecnológica

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.10+ |
| Framework de IA | LangChain |
| Banco vetorial | PostgreSQL 17 + pgVector |
| Containerização | Docker + Docker Compose |
| Embeddings (OpenAI) | `text-embedding-3-small` |
| LLM (OpenAI) | `gpt-4o-mini` |
| Embeddings (Gemini) | `models/embedding-001` |
| LLM (Gemini) | `gemini-2.5-flash-lite` |

---

## Como funciona

```
document.pdf
     │
     ▼ PyPDFLoader
     │ RecursiveCharacterTextSplitter (chunk=1000, overlap=150)
     ▼
  chunks
     │
     ▼ OpenAIEmbeddings / GoogleGenerativeAIEmbeddings
     ▼
  vetores ──► PostgreSQL + pgVector
                    │
              (busca por similaridade)
                    │
              pergunta do usuário
                    │
                    ▼ k=10 chunks mais relevantes
                    │
              PROMPT_TEMPLATE (contexto + regras + pergunta)
                    │
                    ▼ ChatOpenAI / ChatGoogleGenerativeAI
                    │
              RESPOSTA
```
