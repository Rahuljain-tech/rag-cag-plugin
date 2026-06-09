from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import time
from typing import Any, Dict, List, Optional

from app.core.llm_client import OllamaClient
from app.cag.cache_service import SemanticCache
from app.rag.retrieval_service import VectorStore
from app.rag.ingest_service import (
    build_chunks_from_file,
    build_chunks_from_text,
    collect_repo_files,
)
from app.core.config import settings
from app.core.metrics_service import (
    MetricsService,
    tokens_saved_on_cache_hit,
    tokens_used_on_generation,
)

router = APIRouter()
llm_client = OllamaClient()
cache = SemanticCache()
vector_store = VectorStore()
metrics = MetricsService()

class QueryRequest(BaseModel):
    query: str
    use_cache: bool = True
    use_rag: bool = True

class QueryResponse(BaseModel):
    response: str
    source: str
    latency_ms: float
    cache_hit: bool = False
    cache_similarity: Optional[float] = None
    retrieved_chunk_count: int = 0
    estimated_tokens_saved: int = 0
    corpus_version: int = 1

class DocumentRequest(BaseModel):
    content: str

class FileIngestRequest(BaseModel):
    path: str
    mode: str = Field(default="append", pattern="^(append|replace)$")

class RepoIngestRequest(BaseModel):
    path: str
    mode: str = Field(default="replace", pattern="^(append|replace)$")
    extensions: Optional[List[str]] = None

class ToolIngestResponse(BaseModel):
    status: str
    message: str
    documents_count: int
    corpus_version: int
    chunks_ingested: int = 1
    cache_entries_invalidated: int = 0

class CacheStatsResponse(BaseModel):
    cache_entries: int
    cache_db_path: str
    documents_count: int
    cache_similarity_threshold: float
    corpus_version: int

class CacheClearResponse(BaseModel):
    status: str
    deleted_entries: int

class MetricsSummaryResponse(BaseModel):
    total_queries: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    rag_llm_queries: int
    llm_only_queries: int
    total_estimated_tokens_saved: int
    total_estimated_tokens_used: int
    estimated_token_savings_pct: float
    avg_latency_cache_ms: float
    avg_latency_rag_llm_ms: float
    avg_latency_llm_only_ms: float
    recent_queries: List[Dict[str, Any]]

class MetricsResetResponse(BaseModel):
    status: str

async def _ingest_chunks(
    chunks: List[str],
    *,
    mode: str,
    source_path: Optional[str] = None,
) -> ToolIngestResponse:
    if not chunks:
        raise ValueError("No content to ingest")

    corpus_version = vector_store.prepare_corpus(mode=mode)
    invalidated = cache.invalidate_for_corpus(corpus_version)

    for chunk in chunks:
        embedding = await llm_client.get_embedding(chunk)
        vector_store.add_document(
            chunk,
            embedding,
            source_path=source_path,
            corpus_version=corpus_version,
        )

    return ToolIngestResponse(
        status="success",
        message="Document ingested",
        documents_count=vector_store.count_documents(),
        corpus_version=corpus_version,
        chunks_ingested=len(chunks),
        cache_entries_invalidated=invalidated,
    )

async def _ingest_document(req: DocumentRequest, mode: str = "append") -> ToolIngestResponse:
    chunks = build_chunks_from_text(req.content)
    return await _ingest_chunks(chunks, mode=mode)

async def _ingest_file(req: FileIngestRequest) -> ToolIngestResponse:
    file_path = Path(req.path).expanduser().resolve()
    if not file_path.is_file():
        raise ValueError(f"File not found: {req.path}")

    chunks = build_chunks_from_file(file_path)
    if not chunks:
        raise ValueError(f"Could not read or chunk file: {req.path}")

    result = await _ingest_chunks(
        chunks,
        mode=req.mode,
        source_path=str(file_path),
    )
    result.message = f"Ingested file: {file_path.name}"
    return result

async def _ingest_repo(req: RepoIngestRequest) -> ToolIngestResponse:
    files = collect_repo_files(req.path, req.extensions)
    if not files:
        raise ValueError(f"No ingestible files found in: {req.path}")

    corpus_version = vector_store.prepare_corpus(mode=req.mode)
    invalidated = cache.invalidate_for_corpus(corpus_version)
    total_chunks = 0

    root = Path(req.path).expanduser().resolve()
    for file_path in files:
        chunks = build_chunks_from_file(file_path, root=root)
        for chunk in chunks:
            embedding = await llm_client.get_embedding(chunk)
            vector_store.add_document(
                chunk,
                embedding,
                source_path=str(file_path.relative_to(root)),
                corpus_version=corpus_version,
            )
            total_chunks += 1

    return ToolIngestResponse(
        status="success",
        message=f"Ingested {len(files)} files from repository",
        documents_count=vector_store.count_documents(),
        corpus_version=corpus_version,
        chunks_ingested=total_chunks,
        cache_entries_invalidated=invalidated,
    )

def _record_metrics(
    *,
    query: str,
    source: str,
    cache_hit: bool,
    cache_miss: bool,
    latency_ms: float,
    estimated_tokens_saved: int,
    estimated_tokens_used: int,
    cache_similarity: Optional[float] = None,
) -> None:
    metrics.record_query(
        query=query,
        source=source,
        cache_hit=cache_hit,
        cache_miss=cache_miss,
        latency_ms=latency_ms,
        estimated_tokens_saved=estimated_tokens_saved,
        estimated_tokens_used=estimated_tokens_used,
        cache_similarity=cache_similarity,
    )

def _cache_hit_response(
    *,
    req: QueryRequest,
    response: str,
    source: str,
    start_time: float,
    cache_similarity: Optional[float],
) -> QueryResponse:
    latency = round((time.time() - start_time) * 1000, 2)
    tokens_saved = tokens_saved_on_cache_hit(req.query, response)
    corpus_version = vector_store.get_corpus_version()

    _record_metrics(
        query=req.query,
        source=source,
        cache_hit=True,
        cache_miss=False,
        latency_ms=latency,
        estimated_tokens_saved=tokens_saved,
        estimated_tokens_used=0,
        cache_similarity=cache_similarity,
    )

    return QueryResponse(
        response=response,
        source=source,
        latency_ms=latency,
        cache_hit=True,
        cache_similarity=cache_similarity,
        retrieved_chunk_count=0,
        estimated_tokens_saved=tokens_saved,
        corpus_version=corpus_version,
    )

async def _generate_response(req: QueryRequest, background_tasks: BackgroundTasks) -> QueryResponse:
    """Generate using exact cache -> semantic cache -> RAG -> LLM."""
    start_time = time.time()
    corpus_version = vector_store.get_corpus_version()

    if req.use_cache:
        exact_match = cache.lookup_exact(req.query, corpus_version)
        if exact_match:
            return _cache_hit_response(
                req=req,
                response=exact_match,
                source="exact_cache",
                start_time=start_time,
                cache_similarity=1.0,
            )

    query_embedding = await llm_client.get_embedding(req.query)

    if req.use_cache:
        cached_response, similarity = cache.lookup(query_embedding, corpus_version)
        if cached_response:
            return _cache_hit_response(
                req=req,
                response=cached_response,
                source="semantic_cache",
                start_time=start_time,
                cache_similarity=round(similarity, 4),
            )

    context = ""
    retrieved_chunk_count = 0
    if req.use_rag:
        docs = vector_store.search(query_embedding, top_k=2)
        retrieved_chunk_count = len(docs)
        if docs:
            context = "\n---\n".join(docs)

    llm_response = await llm_client.generate_response(req.query, context)

    if req.use_cache:
        background_tasks.add_task(
            cache.add_to_cache,
            req.query,
            query_embedding,
            llm_response,
            corpus_version,
        )

    source = "rag_llm" if context else "llm_only"
    latency = round((time.time() - start_time) * 1000, 2)
    tokens_used = tokens_used_on_generation(req.query, context, llm_response)

    _record_metrics(
        query=req.query,
        source=source,
        cache_hit=False,
        cache_miss=req.use_cache,
        latency_ms=latency,
        estimated_tokens_saved=0,
        estimated_tokens_used=tokens_used,
        cache_similarity=None,
    )

    return QueryResponse(
        response=llm_response,
        source=source,
        latency_ms=latency,
        cache_hit=False,
        cache_similarity=None,
        retrieved_chunk_count=retrieved_chunk_count,
        estimated_tokens_saved=0,
        corpus_version=corpus_version,
    )

@router.post("/documents", response_model=ToolIngestResponse)
async def add_document(req: DocumentRequest):
    try:
        return await _ingest_document(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/file", response_model=ToolIngestResponse)
async def add_document_file(req: FileIngestRequest):
    try:
        return await _ingest_file(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/repo", response_model=ToolIngestResponse)
async def add_document_repo(req: RepoIngestRequest):
    try:
        return await _ingest_repo(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate", response_model=QueryResponse)
async def generate(req: QueryRequest, background_tasks: BackgroundTasks):
    try:
        return await _generate_response(req, background_tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/ingest", response_model=ToolIngestResponse)
async def tool_ingest(req: DocumentRequest):
    try:
        return await _ingest_document(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/ingest/file", response_model=ToolIngestResponse)
async def tool_ingest_file(req: FileIngestRequest):
    try:
        return await _ingest_file(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/ingest/repo", response_model=ToolIngestResponse)
async def tool_ingest_repo(req: RepoIngestRequest):
    try:
        return await _ingest_repo(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/query", response_model=QueryResponse)
async def tool_query(req: QueryRequest, background_tasks: BackgroundTasks):
    try:
        return await _generate_response(req, background_tasks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tools/cache/stats", response_model=CacheStatsResponse)
async def tool_cache_stats():
    try:
        corpus_version = vector_store.get_corpus_version()
        stats = cache.get_stats(corpus_version=corpus_version)
        return CacheStatsResponse(
            cache_entries=int(stats["entries"]),
            cache_db_path=str(stats["db_path"]),
            documents_count=vector_store.count_documents(),
            cache_similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD,
            corpus_version=corpus_version,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/tools/cache", response_model=CacheClearResponse)
async def tool_cache_clear():
    try:
        deleted_entries = cache.clear()
        return CacheClearResponse(status="success", deleted_entries=deleted_entries)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics", response_model=MetricsSummaryResponse)
async def get_metrics():
    try:
        return MetricsSummaryResponse(**metrics.get_summary())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tools/metrics", response_model=MetricsSummaryResponse)
async def tool_metrics():
    try:
        return MetricsSummaryResponse(**metrics.get_summary())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/tools/metrics", response_model=MetricsResetResponse)
async def tool_metrics_reset():
    try:
        metrics.reset()
        return MetricsResetResponse(status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
