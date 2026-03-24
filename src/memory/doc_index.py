"""
DocIndex – penyimpanan indeks dokumen .docx per sesi ke SQLite.

Agar bot bisa menjawab pertanyaan lanjutan tanpa file diunggah ulang,
hasil bedahan dokumen (judul bab + ringkasan + teks asli) disimpan di sini.

Schema tabel `doc_sections`:
    id           INTEGER PRIMARY KEY
    session_id   TEXT NOT NULL
    file_id      TEXT NOT NULL       – identifier file (biasanya nama file)
    bab_index    INTEGER NOT NULL    – nomor urut seksi
    bab_title    TEXT NOT NULL       – judul bab / heading
    level        INTEGER NOT NULL    – level heading (1, 2, 3, ...)
    content_text TEXT                – teks asli konten bab
    summary      TEXT                – ringkasan yang dibuat oleh LLM
    created_at   TEXT                – ISO timestamp saat disimpan

Schema tabel `doc_meta`:
    id           INTEGER PRIMARY KEY
    session_id   TEXT NOT NULL
    file_id      TEXT NOT NULL
    doc_title    TEXT                – judul dokumen
    total_sections INTEGER
    total_words  INTEGER
    indexed_at   TEXT
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)

# Lokasi database SQLite (di /tmp agar tidak di-commit ke repo)
_DB_PATH = os.path.join("/tmp", "advance_ai_doc_index.db")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    """Context manager yang memberikan koneksi SQLite thread-safe."""
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_tables() -> None:
    """Buat tabel jika belum ada."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS doc_sections (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL,
                file_id      TEXT NOT NULL,
                bab_index    INTEGER NOT NULL,
                bab_title    TEXT NOT NULL,
                level        INTEGER NOT NULL DEFAULT 1,
                content_text TEXT,
                summary      TEXT,
                created_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS doc_meta (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id     TEXT NOT NULL,
                file_id        TEXT NOT NULL,
                doc_title      TEXT,
                total_sections INTEGER DEFAULT 0,
                total_words    INTEGER DEFAULT 0,
                indexed_at     TEXT NOT NULL,
                UNIQUE(session_id, file_id)
            );

            CREATE INDEX IF NOT EXISTS idx_sections_session
                ON doc_sections(session_id, file_id);

            CREATE INDEX IF NOT EXISTS idx_meta_session
                ON doc_meta(session_id);
        """)


# ── Public API ────────────────────────────────────────────────────────────────

class DocIndex:
    """Interface untuk menyimpan & mengambil indeks dokumen per sesi."""

    def __init__(self) -> None:
        _ensure_tables()

    # ── Writes ────────────────────────────────────────────────────────────────

    def save_document(
        self,
        session_id: str,
        file_id: str,
        doc_title: str,
        sections: list[dict[str, Any]],
        total_words: int = 0,
    ) -> None:
        """
        Simpan seluruh seksi dokumen ke database.

        Args:
            session_id: ID sesi pengguna.
            file_id:    Identifier unik file (biasanya nama file).
            doc_title:  Judul dokumen.
            sections:   List dict dari docx_parser: [{index, title, level, content}]
            total_words: Total jumlah kata di dokumen.
        """
        now = _now_iso()
        with _get_conn() as conn:
            # Hapus data lama untuk file_id ini di sesi ini (jika ada)
            conn.execute(
                "DELETE FROM doc_sections WHERE session_id=? AND file_id=?",
                (session_id, file_id),
            )
            conn.execute(
                "DELETE FROM doc_meta WHERE session_id=? AND file_id=?",
                (session_id, file_id),
            )

            # Insert meta
            conn.execute(
                """
                INSERT INTO doc_meta (session_id, file_id, doc_title,
                    total_sections, total_words, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, file_id, doc_title, len(sections), total_words, now),
            )

            # Insert sections
            conn.executemany(
                """
                INSERT INTO doc_sections
                    (session_id, file_id, bab_index, bab_title,
                     level, content_text, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                [
                    (
                        session_id,
                        file_id,
                        s["index"],
                        s["title"],
                        s.get("level", 1),
                        s.get("content", ""),
                        now,
                    )
                    for s in sections
                ],
            )
        logger.info(
            "DocIndex: saved %d sections for session=%s file_id=%r",
            len(sections), session_id, file_id,
        )

    def save_summary(
        self,
        session_id: str,
        file_id: str,
        bab_index: int,
        summary: str,
    ) -> None:
        """Simpan ringkasan LLM untuk satu seksi bab."""
        with _get_conn() as conn:
            conn.execute(
                """
                UPDATE doc_sections SET summary=?
                WHERE session_id=? AND file_id=? AND bab_index=?
                """,
                (summary, session_id, file_id, bab_index),
            )

    # ── Reads ─────────────────────────────────────────────────────────────────

    def has_document(self, session_id: str) -> bool:
        """Kembalikan True jika sesi ini sudah memiliki dokumen yang terindeks."""
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM doc_meta WHERE session_id=? LIMIT 1",
                (session_id,),
            ).fetchone()
        return row is not None

    def get_latest_file_id(self, session_id: str) -> Optional[str]:
        """Kembalikan file_id dokumen yang paling baru diindeks untuk sesi ini."""
        with _get_conn() as conn:
            row = conn.execute(
                """
                SELECT file_id FROM doc_meta
                WHERE session_id=?
                ORDER BY indexed_at DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return row["file_id"] if row else None

    def get_doc_meta(
        self, session_id: str, file_id: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Ambil metadata dokumen. Jika file_id None, ambil yang paling baru."""
        if file_id is None:
            file_id = self.get_latest_file_id(session_id)
        if file_id is None:
            return None
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM doc_meta WHERE session_id=? AND file_id=?",
                (session_id, file_id),
            ).fetchone()
        return dict(row) if row else None

    def get_sections(
        self, session_id: str, file_id: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Ambil semua seksi untuk sesi + file_id (default: file paling baru)."""
        if file_id is None:
            file_id = self.get_latest_file_id(session_id)
        if file_id is None:
            return []
        with _get_conn() as conn:
            rows = conn.execute(
                """
                SELECT bab_index, bab_title, level, content_text, summary
                FROM doc_sections
                WHERE session_id=? AND file_id=?
                ORDER BY bab_index
                """,
                (session_id, file_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_sections(
        self,
        session_id: str,
        query: str,
        file_id: Optional[str] = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Cari seksi yang relevan berdasarkan kata kunci di judul dan konten.

        Implementasi: simple case-insensitive substring matching
        (cukup untuk kasus penggunaan tanpa vector DB).
        """
        if file_id is None:
            file_id = self.get_latest_file_id(session_id)
        if file_id is None:
            return []

        q_lower = query.lower()
        all_sections = self.get_sections(session_id, file_id)

        # Score = hits in title (weight 3) + hits in content (weight 1)
        scored: list[tuple[int, dict]] = []
        for sec in all_sections:
            title_score   = int(q_lower in sec["bab_title"].lower())
            content_score = int(q_lower in (sec["content_text"] or "").lower())
            score = title_score * 3 + content_score * 1
            if score > 0:
                scored.append((score, sec))

        # Fallback: return first N sections if no keyword match
        if not scored:
            return all_sections[:limit]

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:limit]]

    def clear_session(self, session_id: str) -> None:
        """Hapus semua data indeks untuk sesi ini (dipanggil saat /reset)."""
        with _get_conn() as conn:
            conn.execute(
                "DELETE FROM doc_sections WHERE session_id=?", (session_id,)
            )
            conn.execute(
                "DELETE FROM doc_meta WHERE session_id=?", (session_id,)
            )
        logger.info("DocIndex: cleared all data for session=%s", session_id)


# ── Module-level singleton ────────────────────────────────────────────────────

_doc_index: Optional[DocIndex] = None


def get_doc_index() -> DocIndex:
    """Lazy singleton – buat instance DocIndex pertama kali dipanggil."""
    global _doc_index
    if _doc_index is None:
        _doc_index = DocIndex()
    return _doc_index
