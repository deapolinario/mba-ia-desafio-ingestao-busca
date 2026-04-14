# PRD — Ingestão e Busca Semântica com LangChain e PostgreSQL

**Versão:** 1.0  
**Data:** 2026-04-14  
**Autor:** Andre Apolinario  
**Status:** Draft

---

## 1. Visão Geral

Este documento descreve os requisitos de produto para um sistema de **busca semântica baseado em RAG (Retrieval-Augmented Generation)**. O sistema deve ser capaz de ingerir documentos PDF, armazenar suas representações vetoriais em um banco de dados PostgreSQL com extensão pgVector, e permitir que usuários façam perguntas em linguagem natural via CLI, recebendo respostas fundamentadas exclusivamente no conteúdo do documento.

---

## 2. Problema

Usuários que precisam extrair informações específicas de documentos extensos em PDF enfrentam dificuldade em localizar respostas de forma rápida e precisa. A leitura manual é lenta e propensa a erros. Não existe hoje uma interface que permita perguntar diretamente ao conteúdo do documento e obter respostas confiáveis, sem alucinações ou uso de conhecimento externo.

---

## 3. Objetivo do Produto

Construir um sistema CLI em Python que:

1. **Ingira** um arquivo PDF, quebrando-o em chunks semânticos e armazenando seus embeddings em um banco vetorial.
2. **Responda** perguntas do usuário com base exclusivamente no conteúdo do PDF, utilizando busca por similaridade e uma LLM para geração de respostas.

---

## 4. Usuários-alvo

| Perfil | Descrição |
|---|---|
| Usuário técnico | Desenvolvedor ou analista capaz de executar scripts Python e subir containers Docker via CLI. |
| Avaliador acadêmico | Avaliador do desafio técnico de pós-graduação que verificará a corretude e completude da implementação. |

---

## 5. Escopo

### 5.1 Dentro do escopo

- Ingestão de um único arquivo PDF (`document.pdf`).
- Chunking do PDF com parâmetros fixos (1000 caracteres, overlap de 150).
- Geração de embeddings via OpenAI ou Gemini (configurável por variável de ambiente).
- Armazenamento de vetores no PostgreSQL com pgVector via Docker.
- Interface de chat interativo no terminal (loop contínuo de perguntas e respostas).
- Busca por similaridade com `k=10` resultados.
- Prompt com regras de grounding estrito: responder apenas com base no contexto recuperado.
- Resposta padrão para perguntas fora do contexto.

### 5.2 Fora do escopo

- Interface gráfica (web ou desktop).
- Suporte a múltiplos PDFs simultâneos.
- Autenticação ou controle de acesso.
- Histórico persistente de conversas.
- Fine-tuning de modelos.
- Deploy em produção ou cloud.

---

## 6. Requisitos Funcionais

### RF-01 — Ingestão do PDF

| ID | Requisito |
|---|---|
| RF-01.1 | O sistema deve carregar o arquivo `document.pdf` da raiz do projeto. |
| RF-01.2 | O PDF deve ser dividido em chunks de **1000 caracteres** com **overlap de 150 caracteres** usando `RecursiveCharacterTextSplitter`. |
| RF-01.3 | Cada chunk deve ser convertido em um vetor de embedding usando o modelo configurado. |
| RF-01.4 | Os vetores devem ser persistidos no PostgreSQL com pgVector via `PGVector` do LangChain. |
| RF-01.5 | A ingestão deve ser executada por um script isolado: `src/ingest.py`. |
| RF-01.6 | A re-execução da ingestão não deve duplicar dados (idempotência desejável). |

### RF-02 — Busca Semântica

| ID | Requisito |
|---|---|
| RF-02.1 | O sistema deve vetorizar a pergunta do usuário usando o mesmo modelo de embeddings da ingestão. |
| RF-02.2 | O sistema deve buscar os **10 chunks mais relevantes** (`k=10`) usando `similarity_search_with_score`. |
| RF-02.3 | Os chunks recuperados devem ser concatenados e injetados no prompt como contexto. |
| RF-02.4 | A lógica de busca deve estar encapsulada em `src/search.py`. |

### RF-03 — Interface CLI (Chat)

| ID | Requisito |
|---|---|
| RF-03.1 | O sistema deve exibir um prompt interativo no terminal solicitando a pergunta do usuário. |
| RF-03.2 | O sistema deve exibir a resposta gerada pela LLM formatada como `RESPOSTA: <texto>`. |
| RF-03.3 | O chat deve funcionar em loop contínuo até o usuário interrompê-lo (ex.: `Ctrl+C`). |
| RF-03.4 | A interface deve estar implementada em `src/chat.py`. |

### RF-04 — Geração de Resposta com Grounding

| ID | Requisito |
|---|---|
| RF-04.1 | O prompt enviado à LLM deve seguir o template definido no feat.md, com as seções: `CONTEXTO`, `REGRAS`, `EXEMPLOS` e `PERGUNTA DO USUÁRIO`. |
| RF-04.2 | A LLM deve responder **somente** com base no contexto recuperado do banco vetorial. |
| RF-04.3 | Para perguntas sem resposta no contexto, o sistema deve retornar exatamente: `"Não tenho informações necessárias para responder sua pergunta."` |
| RF-04.4 | O sistema nunca deve inventar informações ou usar conhecimento externo ao PDF. |

---

## 7. Requisitos Não Funcionais

| ID | Requisito |
|---|---|
| RNF-01 | O banco de dados deve ser executado via Docker Compose, sem instalação local do PostgreSQL. |
| RNF-02 | As credenciais (API Keys) devem ser gerenciadas via variáveis de ambiente, nunca hardcoded. |
| RNF-03 | O projeto deve conter um arquivo `.env.example` como template de configuração. |
| RNF-04 | O projeto deve conter um `requirements.txt` com todas as dependências e versões fixadas. |
| RNF-05 | O tempo de resposta para uma pergunta deve ser aceitável para uso interativo (sem SLA formal neste desafio). |
| RNF-06 | O código deve ser organizado na estrutura de diretórios definida no feat.md. |

---

## 8. Restrições Técnicas

| Restrição | Detalhe |
|---|---|
| Linguagem | Python (obrigatório) |
| Framework de IA | LangChain (obrigatório) |
| Banco de dados | PostgreSQL + pgVector (obrigatório) |
| Containerização | Docker + Docker Compose (obrigatório) |
| Provedor de LLM | OpenAI (`gpt-5-nano`) **ou** Gemini (`gemini-2.5-flash-lite`) |
| Provedor de Embeddings | OpenAI (`text-embedding-3-small`) **ou** Gemini (`models/embedding-001`) |
| Chunking | `RecursiveCharacterTextSplitter` — chunk 1000, overlap 150 |
| Busca | `similarity_search_with_score(query, k=10)` |

---

## 9. Fluxos Principais

### Fluxo 1 — Ingestão

```
[Usuário executa src/ingest.py]
        │
        ▼
[Carrega document.pdf com PyPDFLoader]
        │
        ▼
[Divide em chunks (1000 chars, overlap 150)]
        │
        ▼
[Gera embeddings para cada chunk]
        │
        ▼
[Persiste vetores no PostgreSQL/pgVector]
        │
        ▼
[Ingestão concluída]
```

### Fluxo 2 — Chat / Busca

```
[Usuário executa src/chat.py]
        │
        ▼
[Exibe prompt: "Faça sua pergunta:"]
        │
        ▼
[Usuário digita a pergunta]
        │
        ▼
[Vetoriza a pergunta]
        │
        ▼
[Busca os k=10 chunks mais similares no pgVector]
        │
        ▼
[Monta o prompt com CONTEXTO + REGRAS + PERGUNTA]
        │
        ▼
[Chama a LLM]
        │
        ▼
[Exibe RESPOSTA ao usuário]
        │
        ▼
[Volta ao início do loop]
```

---

## 10. Critérios de Aceitação

| # | Critério |
|---|---|
| CA-01 | `docker compose up -d` sobe o banco sem erros. |
| CA-02 | `python src/ingest.py` processa o PDF e persiste os chunks no banco sem erros. |
| CA-03 | `python src/chat.py` inicia o chat interativo no terminal. |
| CA-04 | Pergunta sobre conteúdo presente no PDF retorna resposta correta e fundamentada. |
| CA-05 | Pergunta fora do contexto retorna exatamente a mensagem padrão definida. |
| CA-06 | As API Keys são lidas de variáveis de ambiente (não hardcoded). |
| CA-07 | O repositório é público no GitHub e contém `README.md` com instruções de execução. |

---

## 11. Estrutura de Arquivos Esperada

```
├── docker-compose.yml        # Configuração do PostgreSQL + pgVector
├── requirements.txt          # Dependências Python fixadas
├── .env.example              # Template das variáveis de ambiente
├── src/
│   ├── ingest.py             # Pipeline de ingestão do PDF
│   ├── search.py             # Lógica de busca semântica
│   └── chat.py               # CLI de interação com o usuário
├── document.pdf              # Documento PDF a ser ingerido
├── docs/
│   ├── feat.txt              # Especificação original do desafio
│   ├── feat.md               # Especificação formatada
│   └── PRD.md                # Este documento
└── README.md                 # Instruções de execução do projeto
```

---

## 12. Dependências e Pré-requisitos

- Docker e Docker Compose instalados.
- Python 3.10+.
- Ambiente virtual Python ativado (`venv`).
- API Key válida da OpenAI **ou** da Google (Gemini).
- Arquivo `document.pdf` presente na raiz do projeto.

---

## 13. Riscos

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| API Key inválida ou sem crédito | Média | Alto | Documentar claramente no README como configurar o `.env`. |
| PDF com conteúdo corrompido ou não extraível | Baixa | Alto | Validar a extração no início do script de ingestão. |
| Banco não inicializado antes da ingestão | Média | Alto | Documentar ordem de execução no README. |
| Respostas fora do contexto passando pelo filtro | Baixa | Médio | Garantir que o prompt siga estritamente o template com as REGRAS definidas. |

---

## 14. Próximos Passos (para o SDD)

A partir deste PRD, as seguintes specs devem ser detalhadas no **Software Design Document (SDD)**:

1. **Spec: Módulo de Ingestão (`ingest.py`)** — design do pipeline, tratamento de erros, idempotência.
2. **Spec: Módulo de Busca (`search.py`)** — interface da função, parâmetros, retorno esperado.
3. **Spec: Módulo de Chat (`chat.py`)** — loop de interação, formatação de saída, tratamento de interrupção.
4. **Spec: Configuração de Infraestrutura (`docker-compose.yml`)** — serviços, portas, variáveis de ambiente do container.
5. **Spec: Gerenciamento de Configuração (`.env`)** — variáveis necessárias, valores padrão, validação na inicialização.
6. **Spec: Template de Prompt** — estrutura formal do prompt, variáveis de substituição, comportamento esperado.
