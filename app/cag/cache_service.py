import sqlite3
import json
import math
from typing import Dict, List, Optional, Tuple, Union

from app.core.config import settings
from app.rag.ingest_service import normalize_query

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm_a = math.sqrt(sum(x * x for x in v1))
    norm_b = math.sqrt(sum(y * y for y in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

class SemanticCache:
    def __init__(self):
        self.db_path = settings.CACHE_DB_PATH
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    query_normalized TEXT NOT NULL DEFAULT '',
                    embedding TEXT NOT NULL,
                    response TEXT NOT NULL,
                    corpus_version INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            self._ensure_column(conn, "semantic_cache", "query_normalized", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "semantic_cache", "corpus_version", "INTEGER NOT NULL DEFAULT 1")
            conn.execute(
                """
                UPDATE semantic_cache
                SET query_normalized = query
                WHERE query_normalized = '' OR query_normalized IS NULL
                """
            )
            conn.commit()

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {
            row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def lookup_exact(self, query: str, corpus_version: int) -> Optional[str]:
        normalized = normalize_query(query)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT response FROM semantic_cache
                WHERE query_normalized = ? AND corpus_version = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (normalized, corpus_version),
            ).fetchone()
            if row:
                return row[0]
        return None

    def lookup(
        self,
        query_embedding: List[float],
        corpus_version: int,
    ) -> Tuple[Optional[str], float]:
        """Returns cached response and best similarity score for the active corpus."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT query, embedding, response FROM semantic_cache
                WHERE corpus_version = ?
                """,
                (corpus_version,),
            )
            rows = cursor.fetchall()

            best_match = None
            highest_score = 0.0

            for row in rows:
                _cached_query, cached_emb_str, response = row
                cached_emb = json.loads(cached_emb_str)
                score = cosine_similarity(query_embedding, cached_emb)

                if score > highest_score:
                    highest_score = score
                    best_match = response

            if highest_score >= settings.CACHE_SIMILARITY_THRESHOLD:
                return best_match, highest_score
        return None, highest_score

    def get_cached_response(
        self,
        query_embedding: List[float],
        corpus_version: int,
    ) -> Optional[str]:
        response, _score = self.lookup(query_embedding, corpus_version)
        return response

    def add_to_cache(
        self,
        query: str,
        embedding: List[float],
        response: str,
        corpus_version: int,
    ) -> None:
        normalized = normalize_query(query)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO semantic_cache (
                    query, query_normalized, embedding, response, corpus_version
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (query, normalized, json.dumps(embedding), response, corpus_version),
            )
            conn.commit()

    def invalidate_for_corpus(self, corpus_version: int) -> int:
        """Remove cache entries that do not match the active corpus version."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM semantic_cache WHERE corpus_version != ?",
                (corpus_version,),
            )
            deleted_count = cursor.fetchone()[0]
            cursor.execute(
                "DELETE FROM semantic_cache WHERE corpus_version != ?",
                (corpus_version,),
            )
            conn.commit()
        return deleted_count

    def get_stats(self, corpus_version: Optional[int] = None) -> Dict[str, Union[int, str]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if corpus_version is None:
                cursor.execute("SELECT COUNT(*) FROM semantic_cache")
            else:
                cursor.execute(
                    "SELECT COUNT(*) FROM semantic_cache WHERE corpus_version = ?",
                    (corpus_version,),
                )
            row_count = cursor.fetchone()[0]
        return {"entries": row_count, "db_path": self.db_path}

    def clear(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM semantic_cache")
            deleted_count = cursor.fetchone()[0]
            cursor.execute("DELETE FROM semantic_cache")
            conn.commit()
        return deleted_count
