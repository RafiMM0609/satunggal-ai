#!/usr/bin/env bash
# =============================================================================
# helper_deploy.sh — Pull kode terbaru & restart service advance_ai
#
# Penggunaan:
#   chmod +x helper_deploy.sh
#   ./helper_deploy.sh              → pull + restart (foreground summary)
#   ./helper_deploy.sh --dry-run    → hanya tampilkan apa yang akan dilakukan
# =============================================================================

set -euo pipefail

# ── Direktori & file ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
DEPLOY_LOG="$LOG_DIR/deploy.log"
START_SH="$SCRIPT_DIR/start.sh"
DRY_RUN=false

mkdir -p "$LOG_DIR"

# ── Warna & logger ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

ts()      { date '+%Y-%m-%d %H:%M:%S'; }
log()     { echo "$(ts) $*" | tee -a "$DEPLOY_LOG"; }
info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; log "[INFO]  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; log "[OK]    $*"; }
warning() { echo -e "${YELLOW}[WARN]${RESET}  $*"; log "[WARN]  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; log "[ERROR] $*"; }
header()  { echo -e "\n${BOLD}${BLUE}=== $* ===${RESET}"; log "=== $* ==="; }

# ── Parse argumen ─────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) error "Argumen tidak dikenal: $arg"; exit 1 ;;
    esac
done

if $DRY_RUN; then
    warning "DRY-RUN aktif — tidak ada perubahan yang akan dilakukan."
fi

# ── Validasi ──────────────────────────────────────────────────────────────────
if [[ ! -f "$START_SH" ]]; then
    error "start.sh tidak ditemukan di $SCRIPT_DIR"
    exit 1
fi

if ! command -v git &>/dev/null; then
    error "git tidak ditemukan di PATH"
    exit 1
fi

# ── Catat info awal ───────────────────────────────────────────────────────────
DEPLOY_START=$(date +%s)
header "Deploy dimulai — $(ts)"
info "Direktori: $SCRIPT_DIR"
info "Log      : $DEPLOY_LOG"

# ── 1. Git pull ───────────────────────────────────────────────────────────────
header "1/4 — Git Pull"

cd "$SCRIPT_DIR"

BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
COMMIT_BEFORE=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
info "Branch saat ini : $BRANCH"
info "Commit sebelum  : $COMMIT_BEFORE"

if $DRY_RUN; then
    warning "[DRY-RUN] git pull dilewati."
else
    PULL_OUTPUT=$(LC_ALL=C git pull 2>&1) || {
        # Tangani kasus: file lokal yang tidak terlacak akan ditimpa oleh merge
        if echo "$PULL_OUTPUT" | grep -q "would be overwritten by merge"; then
            warning "File lokal tidak terlacak terdeteksi yang akan ditimpa. Menghapus dan mencoba lagi ..."
            while IFS= read -r conflict_file; do
                # Hapus leading whitespace (tab/spasi)
                conflict_file="${conflict_file#$'\t'}"
                conflict_file="${conflict_file## }"
                # Validasi: pastikan path tidak mengandung traversal dan tidak kosong
                if [[ -n "$conflict_file" ]] && [[ "$conflict_file" != *".."* ]] && [[ "$conflict_file" != /* ]]; then
                    warning "  Menghapus file konflik: $conflict_file"
                    rm -f "$SCRIPT_DIR/$conflict_file"
                    log "[WARN]  Menghapus file konflik: $conflict_file"
                fi
            done < <(echo "$PULL_OUTPUT" \
                | sed -n '/The following untracked working tree files would be overwritten/,/Please move or remove/p' \
                | grep -v "^error:" \
                | grep -v "The following untracked" \
                | grep -v "Please move or remove")

            # Coba git pull lagi setelah pembersihan
            PULL_OUTPUT=$(LC_ALL=C git pull 2>&1) || {
                error "git pull gagal setelah pembersihan file konflik:"
                echo "$PULL_OUTPUT" | tee -a "$DEPLOY_LOG"
                exit 1
            }
        else
            error "git pull gagal:"
            echo "$PULL_OUTPUT" | tee -a "$DEPLOY_LOG"
            exit 1
        fi
    }
    echo "$PULL_OUTPUT" | tee -a "$DEPLOY_LOG"

    COMMIT_AFTER=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    if [[ "$COMMIT_BEFORE" == "$COMMIT_AFTER" ]]; then
        info "Tidak ada perubahan kode baru (sudah up-to-date)."
    else
        success "Kode diperbarui dari $COMMIT_BEFORE → $COMMIT_AFTER"
    fi
fi

# ── 2. Update Python dependencies ────────────────────────────────────────────
header "2/4 — Update Python Dependencies"

VENV_DIR="$SCRIPT_DIR/.venv"
PIP="$VENV_DIR/bin/pip"

if $DRY_RUN; then
    warning "[DRY-RUN] pip install dilewati."
elif [[ ! -f "$PIP" ]]; then
    warning "Virtual environment belum ada di $VENV_DIR — akan dibuat saat restart."
else
    info "Mengupgrade pip ..."
    "$PIP" install --upgrade pip -q 2>&1 | tee -a "$DEPLOY_LOG"
    info "Menginstall dependencies dari requirements.txt ..."
    if ! "$PIP" install -r "$SCRIPT_DIR/requirements.txt" 2>&1 | tee -a "$DEPLOY_LOG"; then
        error "pip install gagal. Deploy dibatalkan."
        exit 1
    fi
    success "Python dependencies berhasil diupdate."
fi

# ── 3. Install Playwright browser binaries & system dependencies ──────────────
header "3/4 — Playwright Browser & System Dependencies"

PLAYWRIGHT_BIN="$VENV_DIR/bin/playwright"

if $DRY_RUN; then
    warning "[DRY-RUN] playwright install dilewati."
elif [[ ! -f "$PLAYWRIGHT_BIN" ]]; then
    warning "playwright tidak ditemukan di $PLAYWRIGHT_BIN. Lewati instalasi browser Playwright."
else
    info "Menginstall Playwright Chromium browser binary ..."
    if ! "$PLAYWRIGHT_BIN" install chromium 2>&1 | tee -a "$DEPLOY_LOG"; then
        warning "playwright install chromium gagal. Fitur web browsing mungkin tidak berfungsi."
    else
        success "Playwright Chromium berhasil diinstall."
    fi

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        info "Menginstall system dependencies untuk Playwright ..."
        if ! "$PLAYWRIGHT_BIN" install-deps chromium 2>&1 | tee -a "$DEPLOY_LOG"; then
            warning "playwright install-deps gagal. Coba jalankan manual: playwright install-deps"
        else
            success "Playwright system dependencies berhasil diinstall."
        fi
    fi
fi

# ── 4. Jadwalkan restart (fully detached) ─────────────────────────────────────
# PENTING: Tidak boleh memanggil start.sh --restart secara langsung dari sini,
# karena start.sh akan membunuh proses bot yang sedang menjalankan handler ini.
# Solusi: jadwalkan restart sebagai proses terpisah (setsid + disown + delay)
# agar bot sempat mengirim hasil deploy ke Telegram sebelum mati.
header "4/4 — Jadwalkan Restart Service"

RESTART_LOG="$LOG_DIR/restart_$(date +%Y%m%d_%H%M%S).log"
RESTART_DELAY=8  # detik — cukup untuk bot kirim pesan ke Telegram

if $DRY_RUN; then
    warning "[DRY-RUN] Penjadwalan restart dilewati."
else
    if ! command -v setsid &>/dev/null; then
        # Fallback tanpa setsid
        (sleep "$RESTART_DELAY" && bash "$START_SH" --restart >> "$RESTART_LOG" 2>&1) &
        disown $!
    else
        # setsid: proses masuk session baru, tidak ikut mati saat parent kill
        setsid bash -c "sleep $RESTART_DELAY && bash '$START_SH' --restart >> '$RESTART_LOG' 2>&1" &
        disown $!
    fi
    success "Restart dijadwalkan dalam ${RESTART_DELAY} detik."
    info "Log restart → $RESTART_LOG"
fi

# ── Ringkasan ─────────────────────────────────────────────────────────────────
DEPLOY_END=$(date +%s)
ELAPSED=$(( DEPLOY_END - DEPLOY_START ))
header "Deploy selesai dalam ${ELAPSED}s — $(ts)"
info "Bot akan restart dalam ~${RESTART_DELAY} detik."
