# RAG + Semantic Cache Plugin

A lightweight, local-first API to augment LLM workflows with Retrieval-Augmented Generation (RAG) and a semantic response cache.

## Terminology

In this project, "cache" means semantic query-response caching (embedding similarity over previously answered queries).
This is different from the "CAG" term in some recent papers, where CAG usually refers to persistent KV-cache reuse inside the model runtime.
To avoid confusion, this README uses "semantic cache" for the implemented layer.

## Architecture

Query pipeline (in order):

1. **Exact cache** — identical question text, no Ollama call (~milliseconds)
2. **Semantic cache** — paraphrased questions via embedding similarity (>90% match)
3. **RAG** — retrieve relevant chunks from ingested docs
4. **LLM** — Ollama generation on cache miss

Corpus-aware invalidation:

- Every ingest bumps a **corpus version**
- Cache entries are tied to that version
- Stale cache entries are removed when docs change

## Getting Started

1. Set up your local virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Ensure you have [Ollama](https://ollama.com) installed and running locally with the required models:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

3. Start the API server:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints

### 1. Ingest text (chunked, append mode)
`POST /api/v1/documents`
```json
{
  "content": "The Eiffel Tower is located in Paris, France."
}
```

### 2. Ingest a file
`POST /api/v1/documents/file`
```json
{
  "path": "/absolute/path/to/README.md",
  "mode": "append"
}
```

### 3. Ingest a repository
`POST /api/v1/documents/repo`

Skips `.git`, `venv`, `node_modules`, `__pycache__`, and respects `.gitignore`.

```json
{
  "path": "/absolute/path/to/your-project",
  "mode": "replace"
}
```

- `append` — keep prior docs, add new chunks, invalidate cache
- `replace` — rebuild corpus from this ingest only

### 4. Query (Exact cache -> Semantic cache -> RAG -> LLM)
`POST /api/v1/generate`
```json
{
  "query": "Where is the Eiffel Tower situated?",
  "use_cache": true,
  "use_rag": true
}
```

Response sources:

| `source` | Meaning |
|----------|---------|
| `exact_cache` | Same question text as before |
| `semantic_cache` | Paraphrased match |
| `rag_llm` | Retrieved context + LLM |
| `llm_only` | LLM with no retrieved context |

Observability fields on each response:

- `cache_hit`, `cache_similarity`, `retrieved_chunk_count`
- `estimated_tokens_saved`, `corpus_version`, `latency_ms`

### 5. Metrics
`GET /api/v1/metrics`

Reset metrics: `DELETE /api/v1/tools/metrics`

## Tool Endpoints (Plugin Surface)

- `POST /api/v1/tools/ingest` — ingest text
- `POST /api/v1/tools/ingest/file` — ingest file
- `POST /api/v1/tools/ingest/repo` — ingest repository
- `POST /api/v1/tools/query` — run query pipeline
- `GET /api/v1/tools/cache/stats` — cache + corpus counters
- `DELETE /api/v1/tools/cache` — clear semantic cache
- `GET /api/v1/tools/metrics` — hit rate, latency, token savings

## Quick test

```bash
# Ingest this repo
curl -X POST http://127.0.0.1:8000/api/v1/documents/repo \
  -H "Content-Type: application/json" \
  -d '{"path":"/absolute/path/to/rag-cag-plugin","mode":"replace"}'

# First query (slow — LLM)
curl -X POST http://127.0.0.1:8000/api/v1/tools/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What does the semantic cache do?","use_cache":true,"use_rag":true}'

# Exact repeat (fast — exact_cache, no embedding)
curl -X POST http://127.0.0.1:8000/api/v1/tools/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What does the semantic cache do?","use_cache":true,"use_rag":true}'
```

## MCP Server (stdio)

Tools exposed:

- `query`
- `ingest_document`
- `ingest_file`
- `ingest_repo`
- `cache_stats`
- `clear_cache`
- `get_metrics`
- `health_check`

### Run

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

2. In another terminal, run the MCP server:
   ```bash
   python mcp_server.py
   ```

Optional environment variable:

- `RAG_CAG_API_BASE_URL` (default: `http://127.0.0.1:8000/api/v1`)

### Cursor MCP config example

```json
{
  "mcpServers": {
    "rag-semantic-cache": {
      "command": "python",
      "args": ["/absolute/path/to/rag-cag-plugin/mcp_server.py"]
    }
  }
}
```

## Configuration

See `.env.example` for all options including `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `INGEST_MAX_FILE_BYTES`.
# rag-cag-plugin
