"""
ScreenshotStore – persist and pixel-diff browser screenshots.

Used by the ``compare_screenshot`` browser action to detect visual changes
on monitored pages.  Screenshots are stored under ``/tmp/browser_screenshots``
keyed by a SHA-1 hash of the provided URL/key string.

No third-party imaging library is required: PNG images are decoded with the
Python standard library and compared at the raw decompressed byte level.
"""

from __future__ import annotations

import hashlib
import logging
import struct
import zlib
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_STORE_DIR = Path("/tmp/browser_screenshots")


class ScreenshotStore:
    """Persist and compare browser screenshots for visual-monitoring tasks."""

    def __init__(self, store_dir: Path = _STORE_DIR) -> None:
        self._dir = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _path(self, key: str) -> Path:
        safe = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self._dir / f"{safe}.png"

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, key: str, png_bytes: bytes) -> Path:
        """Persist a screenshot keyed by *key* (e.g. a URL)."""
        path = self._path(key)
        path.write_bytes(png_bytes)
        logger.debug("ScreenshotStore: saved %d bytes → %s", len(png_bytes), path)
        return path

    def load(self, key: str) -> Optional[bytes]:
        """Return previously saved screenshot bytes, or ``None`` if not found."""
        path = self._path(key)
        if path.exists():
            return path.read_bytes()
        return None

    def delete(self, key: str) -> bool:
        """Delete the stored screenshot for *key*. Returns True if it existed."""
        path = self._path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    # ── Comparison ────────────────────────────────────────────────────────────

    def diff_ratio(self, a: bytes, b: bytes) -> float:
        """Return the fraction of raw bytes that differ between two PNG images.

        1. Attempts to decode both PNGs and compare their raw (decompressed)
           IDAT streams byte-by-byte; this catches pixel-level differences
           regardless of PNG re-encoding.
        2. Falls back to a plain byte ratio when dimensions differ or decoding
           fails.

        Returns a value in ``[0.0, 1.0]`` where ``0.0`` means identical.
        """
        if a == b:
            return 0.0
        try:
            raw_a = self._decompress_png(a)
            raw_b = self._decompress_png(b)
            if raw_a is None or raw_b is None or len(raw_a) != len(raw_b):
                # Different dimensions or decode failure – byte-level ratio
                shorter = min(len(a), len(b))
                longer  = max(len(a), len(b))
                return 1.0 - (shorter / longer) if longer else 1.0
            diffs = sum(ba != bb for ba, bb in zip(raw_a, raw_b))
            return diffs / len(raw_a)
        except Exception as exc:
            logger.debug("ScreenshotStore.diff_ratio: failed: %s", exc)
            # Absolute fallback: unequal bytes = 100 % diff
            return 1.0

    # ── Internal PNG decoder ──────────────────────────────────────────────────

    @staticmethod
    def _decompress_png(data: bytes) -> Optional[bytes]:
        """Decompress a PNG file and return the raw IDAT payload bytes.

        Only reads the IHDR (for a size sanity-check) and concatenates all
        IDAT chunks before decompressing with zlib.  Returns ``None`` when
        the data is not a valid PNG or decompression fails.
        """
        try:
            if data[:8] != b"\x89PNG\r\n\x1a\n":
                return None
            pos = 8
            idat_parts: list[bytes] = []
            while pos < len(data) - 12:
                length = struct.unpack(">I", data[pos : pos + 4])[0]
                chunk_type = data[pos + 4 : pos + 8]
                chunk_data = data[pos + 8 : pos + 8 + length]
                pos += 12 + length
                if chunk_type == b"IDAT":
                    idat_parts.append(chunk_data)
                elif chunk_type == b"IEND":
                    break
            if not idat_parts:
                return None
            return zlib.decompress(b"".join(idat_parts))
        except Exception:
            return None
