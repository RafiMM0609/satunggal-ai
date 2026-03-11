"""
Shared Git URL helpers used by all agents that interact with git repositories.

Centralising these functions here eliminates duplication across
developer/agent.py, developer_inspector/agent.py, technical_writer/agent.py,
and git_manager.py.

GitLab detection
────────────────
'gitlab.com' is always recognised as GitLab.
Self-hosted GitLab instances are recognised via the GITLAB_HOSTS setting
(comma-separated list of hostnames, e.g. ``repo.jesica.id,git.myco.io``).

Auth URL format
───────────────
GitHub : https://<PAT>@github.com/owner/repo.git
GitLab : https://oauth2:<PAT>@<host>/owner/repo.git
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse, urlunparse


@lru_cache(maxsize=1)
def _gitlab_host_set() -> frozenset[str]:
    """
    Return the set of hostnames that must be treated as GitLab.

    Always includes 'gitlab.com'.  Additional hosts come from the
    GITLAB_HOSTS setting (lazy-loaded to avoid import-time side-effects).
    """
    try:
        from config.settings import get_settings  # noqa: PLC0415
        extra = get_settings().gitlab_hosts
    except Exception:  # noqa: BLE001
        extra = ""

    hosts: set[str] = {"gitlab.com"}
    for h in extra.split(","):
        h = h.strip().lower()
        if h:
            hosts.add(h)
    return frozenset(hosts)


def is_gitlab_url(repo_url: str) -> bool:
    """
    Return True when *repo_url* points to a GitLab instance.

    Detects:
    - gitlab.com  (public SaaS)
    - Any hostname listed in the GITLAB_HOSTS setting (self-hosted)

    Examples::

        is_gitlab_url("https://gitlab.com/org/repo")          → True
        is_gitlab_url("https://repo.jesica.id/org/repo.git")  → True  # GITLAB_HOSTS=repo.jesica.id
        is_gitlab_url("https://github.com/org/repo")          → False
    """
    try:
        hostname = (urlparse(repo_url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return hostname in _gitlab_host_set()


def inject_pat_into_url(repo_url: str, pat: str) -> str:
    """
    Embed a PAT into an HTTPS clone URL.

    GitHub : https://<PAT>@github.com/owner/repo.git
    GitLab : https://oauth2:<PAT>@<host>/owner/repo.git
             (GitLab requires ``oauth2`` as the username for PAT auth,
              including self-hosted instances.)

    SSH URLs and empty PATs are returned unchanged.
    """
    if not pat:
        return repo_url
    parsed = urlparse(repo_url)
    if parsed.scheme not in ("http", "https"):
        return repo_url  # SSH – leave as-is

    port = f":{parsed.port}" if parsed.port and parsed.port not in (80, 443) else ""

    if is_gitlab_url(repo_url):
        # GitLab auth: oauth2:<PAT>@host
        netloc = f"oauth2:{pat}@{parsed.hostname}{port}"
    else:
        # GitHub (and other hosts): <PAT>@host
        netloc = f"{pat}@{parsed.hostname}{port}"

    return urlunparse(parsed._replace(netloc=netloc))


def repo_name_from_url(repo_url: str) -> str:
    """
    Build an ``owner-repo`` slug from a repository URL.

    Consistent across all agents so they share the same local clone directory
    for the same repository.

    Examples::

        repo_name_from_url("https://github.com/org/myrepo.git")          → "org-myrepo"
        repo_name_from_url("https://gitlab.com/team/project")            → "team-project"
        repo_name_from_url("https://repo.jesica.id/okai/poc-telkom.git") → "okai-poc-telkom"
    """
    clean = repo_url.rstrip("/").removesuffix(".git")
    parts = clean.split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}-{parts[-1]}"
    return parts[-1]
