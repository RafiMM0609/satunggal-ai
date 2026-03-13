import logging
import importlib
from pathlib import Path

import pytest

from src.tools import code_search as cs


def test_skip_dir_logs(caplog, tmp_path: Path):
    caplog.set_level(logging.DEBUG)
    # create a skipped directory
    (tmp_path / "node_modules").mkdir()
    f = tmp_path / "node_modules" / "foo.go"
    f.write_text("package main\n")

    cs.build_ast_index(tmp_path)

    assert any("skip_dir" in rec.message for rec in caplog.records), "Expected skip_dir log"


def test_ext_not_supported_logs(caplog, tmp_path: Path):
    caplog.set_level(logging.DEBUG)
    f = tmp_path / "README.txt"
    f.write_text("some text")

    cs.build_ast_index(tmp_path)

    assert any("ext_not_supported" in rec.message for rec in caplog.records), "Expected ext_not_supported log"


def test_too_large_logs(caplog, tmp_path: Path, monkeypatch):
    caplog.set_level(logging.DEBUG)
    # Reduce threshold to force too_large
    monkeypatch.setattr(cs, "_MAX_FILE_BYTES", 10)
    f = tmp_path / "big.go"
    f.write_text("x" * 100)

    cs.build_ast_index(tmp_path)

    assert any("too_large" in rec.message for rec in caplog.records), "Expected too_large log"


def test_read_error_logs(caplog, tmp_path: Path, monkeypatch):
    caplog.set_level(logging.DEBUG)
    f = tmp_path / "bad.go"
    f.write_text("package main\n")

    original_read_text = Path.read_text

    def patched_read_text(self, *args, **kwargs):
        if Path(self) == f:
            raise OSError("unreadable")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", patched_read_text)

    try:
        cs.build_ast_index(tmp_path)
    finally:
        monkeypatch.setattr(Path, "read_text", original_read_text)

    assert any("read_error" in rec.message for rec in caplog.records), "Expected read_error log"
