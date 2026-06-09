from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="RAG + Semantic Cache Plugin API",
    description="A local-first semantic cache plus RAG middleware for lower latency and better LLM context.",
    version="1.3.0"
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "RAG + Semantic Cache Plugin API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "ingest_text": "POST /api/v1/documents",
            "ingest_file": "POST /api/v1/documents/file",
            "ingest_repo": "POST /api/v1/documents/repo",
            "query": "POST /api/v1/generate",
            "metrics": "GET /api/v1/metrics",
            "tool_query": "POST /api/v1/tools/query",
            "tool_metrics": "GET /api/v1/tools/metrics",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "RAG-CAG Plugin is running healthy!"}
