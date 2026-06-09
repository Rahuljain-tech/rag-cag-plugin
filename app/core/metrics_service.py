import json
import sqlite3
import time
from typing import Optional

from app.core.config import settings
from app.core.token_estimator import estimate_generation_tokens


class MetricsService:
    def __init__(self) -> None:
        self.db_path = settings.METRICS_DB_PATH
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics_totals (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    total_queries INTEGER NOT NULL DEFAULT 0,
                    cache_hits INTEGER NOT NULL DEFAULT 0,
                    cache_misses INTEGER NOT NULL DEFAULT 0,
                    rag_llm_queries INTEGER NOT NULL DEFAULT 0,
                    llm_only_queries INTEGER NOT NULL DEFAULT 0,
                    total_estimated_tokens_saved INTEGER NOT NULL DEFAULT 0,
                    total_estimated_tokens_used INTEGER NOT NULL DEFAULT 0,
                    total_latency_cache_ms REAL NOT NULL DEFAULT 0,
                    total_latency_rag_llm_ms REAL NOT NULL DEFAULT 0,
                    total_latency_llm_only_ms REAL NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recent_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_preview TEXT NOT NULL,
                    source TEXT NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    estimated_tokens_saved INTEGER NOT NULL,
                    cache_similarity REAL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO metrics_totals (id) VALUES (1)"
            )
            conn.commit()

    def record_query(
        self,
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
        preview = query[:120] + ("..." if len(query) > 120 else "")
        now = time.time()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE metrics_totals
                SET
                    total_queries = total_queries + 1,
                    cache_hits = cache_hits + ?,
                    cache_misses = cache_misses + ?,
                    rag_llm_queries = rag_llm_queries + ?,
                    llm_only_queries = llm_only_queries + ?,
                    total_estimated_tokens_saved = total_estimated_tokens_saved + ?,
                    total_estimated_tokens_used = total_estimated_tokens_used + ?,
                    total_latency_cache_ms = total_latency_cache_ms + ?,
                    total_latency_rag_llm_ms = total_latency_rag_llm_ms + ?,
                    total_latency_llm_only_ms = total_latency_llm_only_ms + ?
                WHERE id = 1
                """,
                (
                    1 if cache_hit else 0,
                    1 if cache_miss else 0,
                    1 if source == "rag_llm" else 0,
                    1 if source == "llm_only" else 0,
                    estimated_tokens_saved,
                    estimated_tokens_used,
                    latency_ms if cache_hit else 0,
                    latency_ms if source == "rag_llm" else 0,
                    latency_ms if source == "llm_only" else 0,
                ),
            )
            conn.execute(
                """
                INSERT INTO recent_queries (
                    query_preview, source, cache_hit, latency_ms,
                    estimated_tokens_saved, cache_similarity, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    preview,
                    source,
                    1 if cache_hit else 0,
                    latency_ms,
                    estimated_tokens_saved,
                    cache_similarity,
                    now,
                ),
            )
            conn.execute(
                """
                DELETE FROM recent_queries
                WHERE id NOT IN (
                    SELECT id FROM recent_queries
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (settings.METRICS_RECENT_LIMIT,),
            )
            conn.commit()

    def get_summary(self) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            totals = conn.execute("SELECT * FROM metrics_totals WHERE id = 1").fetchone()
            recent_rows = conn.execute(
                """
                SELECT query_preview, source, cache_hit, latency_ms,
                       estimated_tokens_saved, cache_similarity, created_at
                FROM recent_queries
                ORDER BY id DESC
                LIMIT ?
                """,
                (settings.METRICS_RECENT_LIMIT,),
            ).fetchall()

        if totals is None:
            return self._empty_summary()

        total_queries = int(totals["total_queries"])
        cache_hits = int(totals["cache_hits"])
        cache_misses = int(totals["cache_misses"])
        rag_llm_queries = int(totals["rag_llm_queries"])
        llm_only_queries = int(totals["llm_only_queries"])
        tokens_saved = int(totals["total_estimated_tokens_saved"])
        tokens_used = int(totals["total_estimated_tokens_used"])

        cache_attempts = cache_hits + cache_misses
        hit_rate = round(cache_hits / cache_attempts, 4) if cache_attempts else 0.0

        def avg_latency(total_ms: float, count: int) -> float:
            if count == 0:
                return 0.0
            return round(total_ms / count, 2)

        recent = [
            {
                "query_preview": row["query_preview"],
                "source": row["source"],
                "cache_hit": bool(row["cache_hit"]),
                "latency_ms": row["latency_ms"],
                "estimated_tokens_saved": row["estimated_tokens_saved"],
                "cache_similarity": row["cache_similarity"],
                "created_at": row["created_at"],
            }
            for row in recent_rows
        ]

        return {
            "total_queries": total_queries,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": hit_rate,
            "rag_llm_queries": rag_llm_queries,
            "llm_only_queries": llm_only_queries,
            "total_estimated_tokens_saved": tokens_saved,
            "total_estimated_tokens_used": tokens_used,
            "estimated_token_savings_pct": round(
                tokens_saved / (tokens_saved + tokens_used) * 100, 2
            )
            if (tokens_saved + tokens_used) > 0
            else 0.0,
            "avg_latency_cache_ms": avg_latency(
                float(totals["total_latency_cache_ms"]), cache_hits
            ),
            "avg_latency_rag_llm_ms": avg_latency(
                float(totals["total_latency_rag_llm_ms"]), rag_llm_queries
            ),
            "avg_latency_llm_only_ms": avg_latency(
                float(totals["total_latency_llm_only_ms"]), llm_only_queries
            ),
            "recent_queries": recent,
        }

    def reset(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM recent_queries")
            conn.execute(
                """
                UPDATE metrics_totals
                SET
                    total_queries = 0,
                    cache_hits = 0,
                    cache_misses = 0,
                    rag_llm_queries = 0,
                    llm_only_queries = 0,
                    total_estimated_tokens_saved = 0,
                    total_estimated_tokens_used = 0,
                    total_latency_cache_ms = 0,
                    total_latency_rag_llm_ms = 0,
                    total_latency_llm_only_ms = 0
                WHERE id = 1
                """
            )
            conn.commit()

    @staticmethod
    def _empty_summary() -> dict:
        return {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_rate": 0.0,
            "rag_llm_queries": 0,
            "llm_only_queries": 0,
            "total_estimated_tokens_saved": 0,
            "total_estimated_tokens_used": 0,
            "estimated_token_savings_pct": 0.0,
            "avg_latency_cache_ms": 0.0,
            "avg_latency_rag_llm_ms": 0.0,
            "avg_latency_llm_only_ms": 0.0,
            "recent_queries": [],
        }


def tokens_saved_on_cache_hit(query: str, response: str, context: str = "") -> int:
    return estimate_generation_tokens(query, context, response)


def tokens_used_on_generation(query: str, context: str, response: str) -> int:
    return estimate_generation_tokens(query, context, response)
