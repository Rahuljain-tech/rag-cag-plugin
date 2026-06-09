import sqlite3
import json
import math
import os
from typing import List, Optional

from app.core.config import settings

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm_a = math.sqrt(sum(x * x for x in v1))
    norm_b = math.sqrt(sum(y * y for y in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

class VectorStore:
    def __init__(self, db_path: str = "rag_store.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    source_path TEXT,
                    corpus_version INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS corpus_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.execute("INSERT OR IGNORE INTO corpus_meta (id, version) VALUES (1, 1)")
            self._ensure_column(conn, "documents", "source_path", "TEXT")
            self._ensure_column(conn, "documents", "corpus_version", "INTEGER NOT NULL DEFAULT 1")
            conn.commit()

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {
            row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def get_corpus_version(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT version FROM corpus_meta WHERE id = 1").fetchone()
            return int(row[0]) if row else 1

    def bump_corpus_version(self) -> int:
        return self.prepare_corpus(mode="append")

    def prepare_corpus(self, mode: str = "append") -> int:
        """Activate a new corpus version. Append copies prior docs; replace starts fresh."""
        with sqlite3.connect(self.db_path) as conn:
            old_row = conn.execute("SELECT version FROM corpus_meta WHERE id = 1").fetchone()
            old_version = int(old_row[0])
            new_version = old_version + 1
            conn.execute(
                "UPDATE corpus_meta SET version = ? WHERE id = 1",
                (new_version,),
            )

            if mode == "append":
                conn.execute(
                    """
                    INSERT INTO documents (content, embedding, source_path, corpus_version)
                    SELECT content, embedding, source_path, ?
                    FROM documents
                    WHERE corpus_version = ?
                    """,
                    (new_version, old_version),
                )

            conn.execute(
                "DELETE FROM documents WHERE corpus_version = ?",
                (old_version,),
            )
            conn.commit()
            return new_version

    def add_document(
        self,
        content: str,
        embedding: List[float],
        source_path: Optional[str] = None,
        corpus_version: Optional[int] = None,
    ) -> None:
        version = corpus_version if corpus_version is not None else self.get_corpus_version()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO documents (content, embedding, source_path, corpus_version)
                VALUES (?, ?, ?, ?)
                """,
                (content, json.dumps(embedding), source_path, version),
            )
            conn.commit()

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        if not os.path.exists(self.db_path):
            return []

        current_version = self.get_corpus_version()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    SELECT content, embedding FROM documents
                    WHERE corpus_version = ?
                    """,
                    (current_version,),
                )
            except sqlite3.OperationalError:
                return []
            rows = cursor.fetchall()

            results = []
            for row in rows:
                content, emb_str = row
                emb = json.loads(emb_str)
                score = cosine_similarity(query_embedding, emb)
                results.append((score, content))

            results.sort(key=lambda item: item[0], reverse=True)
            return [doc[1] for doc in results[:top_k]]

    def count_documents(self) -> int:
        if not os.path.exists(self.db_path):
            return 0

        current_version = self.get_corpus_version()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT COUNT(*) FROM documents WHERE corpus_version = ?",
                    (current_version,),
                )
                return cursor.fetchone()[0]
            except sqlite3.OperationalError:
                return 0
