#!/usr/bin/env bash
# =============================================================================
# start.sh — Setup & launchscript untuk advance_ai
# Menangani instalasi dependency Python, sistem, dan Node.js
#
# Penggunaan:
#   chmod +x start.sh
#   ./start.sh               → setup + jalankan bot (foreground)
#   ./start.sh --install     → setup dependency saja (tanpa jalankan bot)
#   ./start.sh --background  → setup + jalankan bot di background (nohup)
#   ./start.sh --service     → install & aktifkan sebagai systemd user service
#   ./start.sh --stop        → hentikan bot yang berjalan di background
#   ./start.sh --restart     → restart bot di background
#   ./start.sh --status      → cek apakah bot sedang berjalan
#   ./start.sh --logs        → tail log bot (Ctrl+C untuk keluar)
# =============================================================================

set -euo pipefail

# ── Warna terminal ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warning() { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${BLUE}=== $* ===${RESET}"; }

# ── Deteksi OS & package manager ──────────────────────────────────────────────
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &>/dev/null; then echo "debian"
        elif command -v dnf   &>/dev/null; then echo "fedora"
        elif command -v pacman &>/dev/null; then echo "arch"
        else echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/bot.pid"
LOG_FILE="$SCRIPT_DIR/logs/bot.log"
SERVICE_NAME="advance_ai_bot"

# ── Parse argumen ─────────────────────────────────────────────────────────────
MODE="run"
for arg in "$@"; do
    case "$arg" in
        --install)           MODE="install"     ;;
        --background|-b)     MODE="background"  ;;
        --service)           MODE="service"     ;;
        --stop)              MODE="stop"        ;;
        --status)            MODE="status"      ;;
        --logs|-l)           MODE="logs"        ;;
        --restart)           MODE="restart"     ;;
    esac
done

# ── Fungsi manajemen background ───────────────────────────────────────────────
bot_pid()        { [[ -f "$PID_FILE" ]] && cat "$PID_FILE" || echo ""; }

bot_is_running() {
    local pid
    pid=$(bot_pid)
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

stop_bot() {
    if bot_is_running; then
        local pid
        pid=$(bot_pid)
        info "Menghentikan bot (PID $pid) ..."
        kill "$pid" 2>/dev/null && sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        success "Bot dihentikan."
    else
        warning "Bot tidak sedang berjalan."
        [[ -f "$PID_FILE" ]] && rm -f "$PID_FILE"
    fi
}

bot_status() {
    header "Status Bot"
    if bot_is_running; then
        local pid
        pid=$(bot_pid)
        success "Bot AKTIF  (PID: $pid)"
        echo -e "  Log  : $LOG_FILE"
        echo -e "  PID  : $PID_FILE"
    else
        warning "Bot TIDAK berjalan."
        [[ -f "$PID_FILE" ]] && { warning "PID file lama ditemukan, dihapus."; rm -f "$PID_FILE"; }
    fi
}

install_systemd_service() {
    local SERVICE_FILE="$HOME/.config/systemd/user/${SERVICE_NAME}.service"
    mkdir -p "$(dirname "$SERVICE_FILE")"
    mkdir -p "$(dirname "$LOG_FILE")"
    cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=advance_ai Bot
After=network.target

[Service]
Type=simple
WorkingDirectory=${SCRIPT_DIR}
ExecStart=${SCRIPT_DIR}/.venv/bin/python ${SCRIPT_DIR}/main.py
Restart=on-failure
RestartSec=10
StandardOutput=append:${LOG_FILE}
StandardError=append:${LOG_FILE}

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable "$SERVICE_NAME"
    systemctl --user restart "$SERVICE_NAME"
    success "Systemd user service '${SERVICE_NAME}' aktif."
    info "Perintah berguna:"
    echo "  systemctl --user status  ${SERVICE_NAME}"
    echo "  systemctl --user stop    ${SERVICE_NAME}"
    echo "  systemctl --user restart ${SERVICE_NAME}"
    echo "  journalctl --user -u ${SERVICE_NAME} -f"
    echo ""
    info "Agar bot tetap jalan saat SSH logout, aktifkan lingering (sekali saja):"
    echo "  loginctl enable-linger \$USER"
}

# ── Mode yang tidak butuh setup dependency ─────────────────────────────────────
case "$MODE" in
    stop)
        bot_status
        stop_bot
        exit 0
        ;;
    status)
        bot_status
        exit 0
        ;;
    logs)
        if [[ -f "$LOG_FILE" ]]; then
            info "Menampilkan log dari $LOG_FILE (Ctrl+C untuk keluar) ..."
            echo ""
            tail -f "$LOG_FILE"
        else
            warning "File log tidak ditemukan: $LOG_FILE"
            info "Jalankan bot dulu dengan: ./start.sh --background"
        fi
        exit 0
        ;;
    restart)
        stop_bot
        MODE="background"  # lanjut ke setup + background run
        ;;
esac

# ==============================================================================
# 1. Python virtual environment
# ==============================================================================
header "Python Virtual Environment"

VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
    info "Membuat virtual environment di .venv ..."
    python3 -m venv "$VENV_DIR"
    success "Virtual environment dibuat."
else
    success "Virtual environment sudah ada."
fi

# Aktifkan venv
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
PIP="$VENV_DIR/bin/pip"

info "Menginstall Python dependencies dari requirements.txt ..."
"$PIP" install --upgrade pip -q || { error "Gagal upgrade pip."; exit 1; }
if ! "$PIP" install -r requirements.txt; then
    error "Gagal menginstall satu atau lebih package dari requirements.txt."
    error "Cek output di atas untuk detail error."
    exit 1
fi
success "Python dependencies terinstall."

# ==============================================================================
# 2. Pandoc (sistem) — untuk konversi Markdown ke Word (.docx)
# ==============================================================================
header "Pandoc (Markdown → Word)"

if command -v pandoc &>/dev/null; then
    success "pandoc sudah terinstall: $(pandoc --version | head -1)"
else
    warning "pandoc belum terinstall. Mencoba menginstall ..."

    install_pandoc_from_github() {
        info "Mengunduh pandoc dari GitHub releases ..."
        local PANDOC_VERSION
        PANDOC_VERSION=$(curl -fsSL "https://api.github.com/repos/jgm/pandoc/releases/latest" \
            | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
        if [[ -z "$PANDOC_VERSION" ]]; then
            warning "Tidak bisa mendapatkan versi pandoc terbaru dari GitHub."
            return 1
        fi
        local DEB_URL="https://github.com/jgm/pandoc/releases/download/${PANDOC_VERSION}/pandoc-${PANDOC_VERSION}-1-amd64.deb"
        local TMP_DEB
        TMP_DEB=$(mktemp /tmp/pandoc-XXXXXX.deb)
        if curl -fsSL "$DEB_URL" -o "$TMP_DEB"; then
            sudo dpkg -i "$TMP_DEB" && rm -f "$TMP_DEB"
            success "pandoc ${PANDOC_VERSION} berhasil diinstall dari GitHub."
        else
            rm -f "$TMP_DEB"
            warning "Gagal mengunduh pandoc dari GitHub. Install manual: https://pandoc.org/installing.html"
            return 1
        fi
    }

    case "$OS" in
        debian)
            if ! sudo apt-get install -y pandoc 2>/dev/null; then
                warning "apt-get gagal. Mencoba unduh langsung dari GitHub ..."
                install_pandoc_from_github || true
            else
                success "pandoc berhasil diinstall."
            fi
            ;;
        fedora)
            sudo dnf install -y pandoc
            success "pandoc berhasil diinstall."
            ;;
        arch)
            sudo pacman -S --noconfirm pandoc
            success "pandoc berhasil diinstall."
            ;;
        macos)
            if command -v brew &>/dev/null; then
                brew install pandoc
                success "pandoc berhasil diinstall."
            else
                warning "Homebrew tidak ditemukan. Install pandoc manual: https://pandoc.org/installing.html"
            fi
            ;;
        *)
            warning "OS tidak dikenali. Install pandoc manual: https://pandoc.org/installing.html"
            ;;
    esac
fi

# ==============================================================================
# 3. Node.js & mermaid-cli — untuk render diagram Mermaid ke PNG
# ==============================================================================
header "mermaid-cli / mmdc (diagram renderer)"

install_node() {
    case "$OS" in
        debian)
            info "Menginstall Node.js via NodeSource ..."
            curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
            sudo apt-get install -y nodejs
            ;;
        fedora)
            sudo dnf install -y nodejs
            ;;
        arch)
            sudo pacman -S --noconfirm nodejs npm
            ;;
        macos)
            if command -v brew &>/dev/null; then
                brew install node
            else
                warning "Install Node.js manual: https://nodejs.org"
                return 1
            fi
            ;;
        *)
            warning "Install Node.js manual: https://nodejs.org"
            return 1
            ;;
    esac
}

if ! command -v node &>/dev/null; then
    warning "Node.js tidak ditemukan. Mencoba menginstall ..."
    install_node
fi

if command -v node &>/dev/null; then
    success "Node.js: $(node --version)"

    if command -v mmdc &>/dev/null; then
        success "mermaid-cli sudah terinstall: $(mmdc --version 2>/dev/null | head -1)"
    else
        info "Menginstall mermaid-cli secara global ..."
        npm install -g @mermaid-js/mermaid-cli --loglevel=error

        # Chromium headless diperlukan oleh mermaid-cli di Linux tanpa display
        if [[ "$OS" == "debian" ]]; then
            if ! command -v chromium-browser &>/dev/null && ! command -v chromium &>/dev/null; then
                info "Menginstall Chromium untuk headless rendering ..."
                sudo apt-get install -y chromium-browser 2>/dev/null \
                    || sudo apt-get install -y chromium 2>/dev/null \
                    || warning "Chromium tidak bisa diinstall otomatis. Install manual."
            fi
        fi

        if command -v mmdc &>/dev/null; then
            success "mermaid-cli (mmdc) berhasil diinstall."
        else
            warning "mmdc tidak ditemukan di PATH setelah instalasi. Coba: export PATH=\$PATH:\$(npm root -g)/.bin"
        fi
    fi
else
    warning "Node.js gagal diinstall. Diagram Mermaid akan menggunakan placeholder teks."
fi

# ==============================================================================
# 4. Verifikasi akhir
# ==============================================================================
header "Ringkasan Instalasi"

check() {
    local name="$1"; local cmd="$2"
    if command -v $cmd &>/dev/null; then
        success "$name : $(command -v $cmd)"
    else
        warning "$name : tidak ditemukan (fitur terkait akan terdegradasi)"
    fi
}

check "python"  "python3"
check "pip"     "pip"
check "pandoc"  "pandoc"
check "node"    "node"
check "mmdc"    "mmdc"

echo ""

# ==============================================================================
# 5. Jalankan bot
# ==============================================================================
case "$MODE" in
    install)
        success "Setup selesai. Pilih cara menjalankan bot:"
        echo ""
        echo "  ./start.sh               → foreground (lihat output langsung)"
        echo "  ./start.sh --background  → background, tetap aktif saat SSH logout*"
        echo "  ./start.sh --service     → systemd service (auto-restart, boot otomatis)"
        echo ""
        echo "  *) pastikan nohup aktif: nohup sudah dipakai secara otomatis."
        echo "     Untuk sesi persisten sejati gunakan --service + loginctl enable-linger"
        exit 0
        ;;

    background)
        header "Menjalankan Bot (Background)"
        if bot_is_running; then
            warning "Bot sudah berjalan (PID: $(bot_pid)). Gunakan --restart untuk restart."
            exit 1
        fi
        mkdir -p "$(dirname "$LOG_FILE")"
        info "Menjalankan bot di background ..."
        info "Log  → $LOG_FILE"
        nohup "$VENV_DIR/bin/python" -u main.py >> "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        sleep 2
        if bot_is_running; then
            success "Bot berjalan di background (PID: $(cat "$PID_FILE"))."
            echo ""
            info "Perintah berguna:"
            echo "  ./start.sh --status   → cek status"
            echo "  ./start.sh --logs     → lihat log real-time"
            echo "  ./start.sh --restart  → restart"
            echo "  ./start.sh --stop     → hentikan bot"
        else
            error "Bot gagal dijalankan. Cek log: $LOG_FILE"
            cat "$LOG_FILE" 2>/dev/null | tail -20 || true
            exit 1
        fi
        ;;

    service)
        header "Install Systemd User Service"
        install_systemd_service
        ;;

    run)
        header "Menjalankan Bot (Foreground)"
        info "Mengaktifkan venv dan menjalankan main.py ..."
        info "(Tekan Ctrl+C untuk menghentikan. Gunakan --background agar tetap aktif saat SSH logout.)"
        echo ""
        exec "$VENV_DIR/bin/python" -u main.py
        ;;
esac
