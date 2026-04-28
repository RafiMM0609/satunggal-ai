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

Schema tabel `qna_log`:
    id           INTEGER PRIMARY KEY
    session_id   TEXT NOT NULL
    file_id      TEXT NOT NULL       – identifier file terkait
    turn_index   INTEGER NOT NULL    – nomor urut pertanyaan dalam sesi
    question     TEXT NOT NULL       – pertanyaan dari pengguna
    answer       TEXT NOT NULL       – jawaban dari agent
    created_at   TEXT                – ISO timestamp saat disimpan
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
    """Buat tabel jika belum ada, dan jalankan migrasi kolom baru."""
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
                docx_path      TEXT,
                total_sections INTEGER DEFAULT 0,
                total_words    INTEGER DEFAULT 0,
                indexed_at     TEXT NOT NULL,
                UNIQUE(session_id, file_id)
            );

            CREATE TABLE IF NOT EXISTS doc_pending_edits (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL,
                file_id      TEXT NOT NULL,
                edit_order   INTEGER NOT NULL,
                instruction  TEXT NOT NULL,
                edit_ops_json TEXT NOT NULL,
                added_at     TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sections_session
                ON doc_sections(session_id, file_id);

            CREATE INDEX IF NOT EXISTS idx_meta_session
                ON doc_meta(session_id);

            CREATE INDEX IF NOT EXISTS idx_pending_edits_session
                ON doc_pending_edits(session_id, file_id, edit_order);

            CREATE TABLE IF NOT EXISTS qna_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT NOT NULL,
                file_id      TEXT NOT NULL,
                turn_index   INTEGER NOT NULL,
                question     TEXT NOT NULL,
                answer       TEXT NOT NULL,
                created_at   TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_qna_log_session
                ON qna_log(session_id, file_id, turn_index);
        """)

        # ── Schema migrations (safe for existing databases) ───────────────
        # Add columns introduced after initial deployment.
        existing_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(doc_meta)").fetchall()
        }
        if "docx_path" not in existing_cols:
            conn.execute("ALTER TABLE doc_meta ADD COLUMN docx_path TEXT")
            logger.info("DocIndex migration: added doc_meta.docx_path")

        existing_pending = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='doc_pending_edits'"
        ).fetchone()
        if existing_pending is None:
            conn.executescript("""
                CREATE TABLE doc_pending_edits (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id    TEXT NOT NULL,
                    file_id       TEXT NOT NULL,
                    edit_order    INTEGER NOT NULL,
                    instruction   TEXT NOT NULL,
                    edit_ops_json TEXT NOT NULL,
                    added_at      TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pending_edits_session
                    ON doc_pending_edits(session_id, file_id, edit_order);
            """)
            logger.info("DocIndex migration: created doc_pending_edits table")

        # ── Migration: qna_log table ───────────────────────────────────────
        existing_qna = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='qna_log'"
        ).fetchone()
        if existing_qna is None:
            conn.executescript("""
                CREATE TABLE qna_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   TEXT NOT NULL,
                    file_id      TEXT NOT NULL,
                    turn_index   INTEGER NOT NULL,
                    question     TEXT NOT NULL,
                    answer       TEXT NOT NULL,
                    created_at   TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_qna_log_session
                    ON qna_log(session_id, file_id, turn_index);
            """)
            logger.info("DocIndex migration: created qna_log table")


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
        docx_path: str = "",
    ) -> None:
        """
        Simpan seluruh seksi dokumen ke database.

        Args:
            session_id: ID sesi pengguna.
            file_id:    Identifier unik file (biasanya nama file).
            doc_title:  Judul dokumen.
            sections:   List dict dari docx_parser: [{index, title, level, content}]
            total_words: Total jumlah kata di dokumen.
            docx_path:  Path absolut ke file .docx asli (untuk edit lanjutan).
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
            conn.execute(
                "DELETE FROM doc_pending_edits WHERE session_id=? AND file_id=?",
                (session_id, file_id),
            )

            # Insert meta
            conn.execute(
                """
                INSERT INTO doc_meta (session_id, file_id, doc_title,
                    docx_path, total_sections, total_words, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, file_id, doc_title, docx_path, len(sections), total_words, now),
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

    def save_docx_path(self, session_id: str, docx_path: str) -> None:
        """Update docx_path di doc_meta untuk sesi (pakai file_id terbaru)."""
        file_id = self.get_latest_file_id(session_id)
        if file_id is None:
            return
        with _get_conn() as conn:
            conn.execute(
                "UPDATE doc_meta SET docx_path=? WHERE session_id=? AND file_id=?",
                (docx_path, session_id, file_id),
            )

    def get_docx_path(self, session_id: str) -> Optional[str]:
        """Ambil path file .docx asli untuk sesi ini (dari file_id terbaru)."""
        meta = self.get_doc_meta(session_id)
        if meta is None:
            return None
        return meta.get("docx_path") or None

    # ── Pending edits ─────────────────────────────────────────────────────────

    def add_pending_edit(
        self,
        session_id: str,
        instruction: str,
        edit_ops: list[dict[str, Any]],
    ) -> int:
        """
        Simpan satu set operasi edit ke antrian pending.

        Args:
            session_id:  ID sesi pengguna.
            instruction: Instruksi asli dari pengguna.
            edit_ops:    List operasi edit JSON (dari LLM).

        Returns:
            Nomor urut edit yang baru saja disimpan.
        """
        import json as _json

        file_id = self.get_latest_file_id(session_id)
        if file_id is None:
            raise ValueError(f"Tidak ada dokumen terindeks untuk sesi {session_id!r}")

        now = _now_iso()
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(edit_order), 0) FROM doc_pending_edits "
                "WHERE session_id=? AND file_id=?",
                (session_id, file_id),
            ).fetchone()
            next_order = (row[0] or 0) + 1

            conn.execute(
                """
                INSERT INTO doc_pending_edits
                    (session_id, file_id, edit_order, instruction, edit_ops_json, added_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, file_id, next_order, instruction, _json.dumps(edit_ops, ensure_ascii=False), now),
            )

        logger.info(
            "DocIndex: added pending edit #%d for session=%s file_id=%r ops=%d",
            next_order, session_id, file_id, len(edit_ops),
        )
        return next_order

    def get_pending_edits(
        self, session_id: str
    ) -> list[dict[str, Any]]:
        """Ambil semua pending edits untuk sesi, diurutkan berdasarkan edit_order."""
        import json as _json

        file_id = self.get_latest_file_id(session_id)
        if file_id is None:
            return []
        with _get_conn() as conn:
            rows = conn.execute(
                """
                SELECT edit_order, instruction, edit_ops_json
                FROM doc_pending_edits
                WHERE session_id=? AND file_id=?
                ORDER BY edit_order
                """,
                (session_id, file_id),
            ).fetchall()
        result = []
        for r in rows:
            try:
                ops = _json.loads(r["edit_ops_json"])
            except Exception as _exc:
                logger.warning(
                    "doc_index: failed to parse edit_ops_json for session=%s order=%s: %s",
                    session_id, r["edit_order"], _exc,
                )
                ops = []
            result.append({
                "edit_order":  r["edit_order"],
                "instruction": r["instruction"],
                "edit_ops":    ops,
            })
        return result

    def get_pending_edit_count(self, session_id: str) -> int:
        """Kembalikan jumlah pending edits untuk sesi ini."""
        file_id = self.get_latest_file_id(session_id)
        if file_id is None:
            return 0
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM doc_pending_edits WHERE session_id=? AND file_id=?",
                (session_id, file_id),
            ).fetchone()
        return row[0] if row else 0

    def clear_pending_edits(self, session_id: str) -> None:
        """Hapus semua pending edits untuk sesi ini setelah berhasil diterapkan."""
        file_id = self.get_latest_file_id(session_id)
        if file_id is None:
            return
        with _get_conn() as conn:
            conn.execute(
                "DELETE FROM doc_pending_edits WHERE session_id=? AND file_id=?",
                (session_id, file_id),
            )
        logger.info("DocIndex: cleared pending edits for session=%s", session_id)

    # ── QnA Log ───────────────────────────────────────────────────────────────

    def add_qna(
        self,
        session_id: str,
        question: str,
        answer: str,
        file_id: Optional[str] = None,
    ) -> None:
        """
        Catat satu pasangan Q&A ke tabel qna_log.

        Args:
            session_id: ID sesi pengguna.
            question:   Pertanyaan dari pengguna.
            answer:     Jawaban dari agent.
            file_id:    Identifier file; jika None, ambil file_id terbaru.
        """
        if file_id is None:
            file_id = self.get_latest_file_id(session_id) or "unknown"

        now = _now_iso()
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(turn_index), 0) FROM qna_log "
                "WHERE session_id=? AND file_id=?",
                (session_id, file_id),
            ).fetchone()
            next_turn = (row[0] or 0) + 1

            conn.execute(
                """
                INSERT INTO qna_log
                    (session_id, file_id, turn_index, question, answer, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, file_id, next_turn, question, answer, now),
            )
        logger.debug(
            "DocIndex: qna_log turn=%d session=%s file_id=%r",
            next_turn, session_id, file_id,
        )

    def get_qna_log(
        self,
        session_id: str,
        file_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Ambil semua pasangan Q&A untuk sesi ini, diurutkan berdasarkan turn_index.

        Returns:
            List of dicts: [{"turn_index": int, "question": str, "answer": str, "created_at": str}]
        """
        if file_id is None:
            file_id = self.get_latest_file_id(session_id)
        if file_id is None:
            return []

        with _get_conn() as conn:
            rows = conn.execute(
                """
                SELECT turn_index, question, answer, created_at
                FROM qna_log
                WHERE session_id=? AND file_id=?
                ORDER BY turn_index
                """,
                (session_id, file_id),
            ).fetchall()
        return [dict(r) for r in rows]

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
            conn.execute(
                "DELETE FROM doc_pending_edits WHERE session_id=?", (session_id,)
            )
            conn.execute(
                "DELETE FROM qna_log WHERE session_id=?", (session_id,)
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
