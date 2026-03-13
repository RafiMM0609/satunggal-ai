import asyncio
from pathlib import Path

from src.tools.repo_qa import extract_api_endpoints


def _write(file_path: Path, content: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)


def test_extract_go_routes(tmp_path: Path):
    # Create sample Go file with various routing styles
    go_code = '''package main

import (
    "net/http"
    "github.com/gorilla/mux"
)

func main() {
    http.HandleFunc("/ping", pingHandler)

    r := mux.NewRouter()
    r.HandleFunc("/users", usersHandler).Methods("GET")

    // gin-like
    // r.GET("/items", itemsHandler)

    // raw string path
    r.HandleFunc(`/raw/path`, rawHandler)
}

func pingHandler(w http.ResponseWriter, r *http.Request) {}
func usersHandler(w http.ResponseWriter, r *http.Request) {}
func rawHandler(w http.ResponseWriter, r *http.Request) {}
'''

    f = tmp_path / "routes.go"
    _write(f, go_code)

    result = asyncio.get_event_loop().run_until_complete(
        extract_api_endpoints(tmp_path)
    )

    assert "/ping" in result
    assert "/users" in result
    assert "/raw/path" in result
