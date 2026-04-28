"""
RepoWatcherJob – Fase 4: Automated Repository Watcher.

Memantau satu atau beberapa repository GitHub/GitLab secara berkala.
Jika ada commit baru sejak pemeriksaan terakhir, DeveloperInspectorAgent
dijalankan untuk melakukan code review dan hasilnya dikirim ke Telegram.

Konfigurasi via `.env` (semua prefix PROACTIVE_REPO_WATCHER_*):

    PROACTIVE_REPO_WATCHER_ENABLED=true
    PROACTIVE_REPO_WATCHER_CHAT_ID=<telegram_chat_id>      # wajib jika enabled
    PROACTIVE_REPO_WATCHER_REPOS=https://github.com/org/repo1,https://github.com/org/repo2
    PROACTIVE_REPO_WATCHER_INTERVAL=60                      # interval polling (menit)

Cara kerja:
    1. APScheduler menembak job setiap N menit (PROACTIVE_REPO_WATCHER_INTERVAL).
    2. Job menjalankan ``git ls-remote`` untuk mendapatkan HEAD commit terbaru
       dari setiap repo (tanpa perlu clone penuh).
    3. Jika hash berubah dibanding pemeriksaan terakhir, DeveloperInspectorAgent
       dipanggil via ``inspect_diff()`` (atau analisis melalui ``research_for_delegation``
       jika diff tidak tersedia).
    4. Laporan dikirim ke Telegram.

State commit terakhir disimpan di memori (per-proses). Saat bot restart,
watcher langsung mulai dari commit HEAD saat itu (tidak replay history lama).

Integrasi:
    Panggil ``start_repo_watcher_job(bot)`` sekali saat startup.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, TYPE_CHECKING

from apscheduler.triggers.interval import IntervalTrigger

if TYPE_CHECKING:
    from telegram import Bot

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_JOB_ID_PREFIX = "proactive_repo_watcher_"

# In-memory state: repo_url → last seen HEAD commit hash
# Reset on process restart (by design — we want to re-inspect on startup)
_last_seen: dict[str, str] = {}


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _get_remote_head(repo_url: str, timeout: int = 30) -> Optional[str]:
    """Dapatkan commit hash HEAD dari repo remote tanpa clone penuh.

    Menggunakan ``git ls-remote`` yang ringan (hanya membaca referensi remote).

    Returns:
        Short SHA-1 (40 chars) atau None jika gagal.
    """
    from config.settings import get_settings
    from src.tools.git_utils import inject_pat_into_url

    settings   = get_settings()
    github_pat = settings.github_pat
    gitlab_pat = settings.gitlab_pat

    # Inject PAT ke URL agar akses repo private bisa berhasil
    authed_url = inject_pat_into_url(repo_url, github_pat=github_pat, gitlab_pat=gitlab_pat)

    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "ls-remote", "--head", authed_url, "refs/heads/HEAD",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        lines = stdout.decode(errors="replace").strip().splitlines()

        # ls-remote tanpa branch spesifik bisa mengembalikan banyak ref;
        # kita ambil baris pertama yang merupakan HEAD default branch.
        if not lines:
            # Coba tanpa filter HEAD
            proc2 = await asyncio.create_subprocess_exec(
                "git", "ls-remote", "--symref", authed_url, "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout2, _ = await asyncio.wait_for(proc2.communicate(), timeout=timeout)
            lines = stdout2.decode(errors="replace").strip().splitlines()

        for line in lines:
            parts = line.split()
            if len(parts) >= 1 and len(parts[0]) == 40:
                return parts[0]  # full SHA-1

        return None
    except Exception as exc:
        logger.warning("RepoWatcher: ls-remote failed for %s: %s", repo_url, exc)
        return None


async def _review_repo(repo_url: str, new_hash: str, old_hash: Optional[str]) -> str:
    """Gunakan DeveloperInspectorAgent untuk mereview perubahan pada repo.

    Jika ini pertama kali dilihat (old_hash is None), berikan ringkasan
    singkat repo saja tanpa diff.
    """
    from src.agents.llm_client import LLMClient
    from src.agents.developer_inspector.agent import DeveloperInspectorAgent
    from src.memory.history import ConversationHistory

    llm     = LLMClient()
    history = ConversationHistory(max_messages=5)
    agent   = DeveloperInspectorAgent(llm=llm, history=history)

    if old_hash is None:
        # Pertama kali — lakukan inspeksi awal
        description = (
            f"Inspeksi awal repositori: {repo_url}\n"
            f"Commit HEAD terbaru: {new_hash[:8]}\n"
            "Berikan gambaran singkat struktur dan kondisi umum repositori."
        )
        action = "inspeksi awal"
    else:
        # Ada commit baru
        description = (
            f"Ada commit baru pada repositori: {repo_url}\n"
            f"Commit sebelumnya: {old_hash[:8]}\n"
            f"Commit terbaru   : {new_hash[:8]}\n"
            "Lakukan inspeksi dan identifikasi perubahan utama. "
            "Apakah ada potensi bug, breaking change, atau masalah kualitas kode?"
        )
        action = f"commit baru {old_hash[:8]} → {new_hash[:8]}"

    logger.info("RepoWatcher: running inspector for %s (%s)", repo_url, action)

    from src.memory.state import AgentTask
    task = AgentTask(
        session_id  = "proactive_repo_watcher",
        user_input  = description,
        current_mode = "all",
    )
    task.metadata["repo_url"] = repo_url

    try:
        task.mark_processing("developer_inspector")
        task = await agent.run(task)
        return task.result or "_(Inspector tidak menghasilkan laporan)_"
    except Exception as exc:
        logger.warning("RepoWatcher: inspector failed for %s: %s", repo_url, exc)
        return f"⚠️ Inspector gagal: {exc}"


async def _check_repo(repo_url: str, chat_id: str) -> None:
    """Periksa satu repo dan kirim laporan jika ada perubahan."""
    from src.proactive._bot_ref import get_bot

    bot = get_bot()
    if bot is None:
        logger.error("RepoWatcher: bot instance not set.")
        return

    head = await _get_remote_head(repo_url)
    if head is None:
        logger.warning("RepoWatcher: could not fetch HEAD for %s.", repo_url)
        return

    last = _last_seen.get(repo_url)

    if last == head:
        logger.debug("RepoWatcher: no new commits for %s (HEAD=%s).", repo_url, head[:8])
        return

    # Commit baru (atau pertama kali diperiksa)
    logger.info(
        "RepoWatcher: change detected for %s: %s → %s",
        repo_url, (last or "none")[:8], head[:8],
    )

    review = await _review_repo(repo_url, new_hash=head, old_hash=last)

    # Kirim laporan ke Telegram
    from telegram.constants import ParseMode
    from datetime import datetime, timezone

    now    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    action = "🔍 *Inspeksi Awal*" if last is None else "🆕 *Commit Baru Terdeteksi*"
    header = (
        f"{action}\n"
        f"📦 `{repo_url}`\n"
        f"🕐 {now}\n"
        + (f"🔢 {last[:8]} → {head[:8]}\n" if last else f"🔢 HEAD: `{head[:8]}`\n")
        + f"{'─' * 30}\n\n"
    )
    message = header + review

    MAX_LEN = 4000
    if len(message) > MAX_LEN:
        message = message[:MAX_LEN] + "\n\n_[laporan terpotong]_"

    try:
        await bot.send_message(
            chat_id    = chat_id,
            text       = message,
            parse_mode = ParseMode.MARKDOWN,
        )
        logger.info("RepoWatcher: report sent to chat_id=%s for %s", chat_id, repo_url)
    except Exception as exc:
        logger.error("RepoWatcher: failed to send report: %s", exc)

    # Update state setelah berhasil kirim
    _last_seen[repo_url] = head


async def _watch_all_repos(repos: list[str], chat_id: str) -> None:
    """Periksa semua repo secara paralel."""
    await asyncio.gather(*[_check_repo(repo, chat_id) for repo in repos])


# ── Public API ─────────────────────────────────────────────────────────────────

def start_repo_watcher_job(bot: "Bot") -> None:
    """
    Daftarkan repo watcher job ke APScheduler.

    Dipanggil sekali saat startup dari ``telegram_bot._send_startup_notification()``.
    Job hanya didaftarkan jika ``PROACTIVE_REPO_WATCHER_ENABLED=true``.

    Args:
        bot: Telegram Bot instance untuk mengirim laporan.
    """
    from config.settings import get_settings
    from src.agents.reminder_agent.scheduler import get_scheduler
    from src.proactive._bot_ref import set_bot

    settings = get_settings()

    if not settings.proactive_repo_watcher_enabled:
        logger.info("RepoWatcher: disabled (PROACTIVE_REPO_WATCHER_ENABLED not set).")
        return

    # Parse daftar repo
    raw_repos = settings.proactive_repo_watcher_repos.strip()
    repos     = [r.strip() for r in raw_repos.split(",") if r.strip()]
    if not repos:
        logger.warning(
            "RepoWatcher: no repos configured (set PROACTIVE_REPO_WATCHER_REPOS). Disabled."
        )
        return

    # Target chat_id
    chat_id = (
        settings.proactive_repo_watcher_chat_id.strip()
        or str(settings.admin_user_id)
    )
    if not chat_id or chat_id == "0":
        logger.warning(
            "RepoWatcher: no chat_id configured "
            "(set PROACTIVE_REPO_WATCHER_CHAT_ID or ADMIN_USER_ID). Disabled."
        )
        return

    interval_minutes = max(1, settings.proactive_repo_watcher_interval)

    set_bot(bot)

    sched = get_scheduler()

    # Daftarkan satu job per repo agar mudah diidentifikasi di log
    for repo_url in repos:
        job_id = _JOB_ID_PREFIX + repo_url.replace("https://", "").replace("/", "_")[:60]

        if sched.get_job(job_id):
            sched.remove_job(job_id)

        sched.add_job(
            _check_repo,
            trigger           = IntervalTrigger(minutes=interval_minutes),
            id                = job_id,
            args              = [repo_url, chat_id],
            misfire_grace_time = interval_minutes * 60,
        )
        logger.info(
            "RepoWatcher: job registered for %s (every %d min) → chat_id=%s",
            repo_url, interval_minutes, chat_id,
        )


def stop_repo_watcher_jobs() -> None:
    """Hapus semua repo watcher jobs dari scheduler (dipanggil saat shutdown)."""
    from src.agents.reminder_agent.scheduler import get_scheduler

    sched = get_scheduler()
    removed = 0
    for job in list(sched.get_jobs()):
        if job.id.startswith(_JOB_ID_PREFIX):
            sched.remove_job(job.id)
            removed += 1
    if removed:
        logger.info("RepoWatcher: %d job(s) removed.", removed)
