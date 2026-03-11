#!/usr/bin/env bash
# =============================================================================
# start.sh — Setup & launchscript untuk advance_ai
# Menangani instalasi dependency Python, sistem, dan Node.js
#
# Penggunaan:
#   chmod +x start.sh
#   ./start.sh            → setup + jalankan bot
#   ./start.sh --install  → setup dependency saja (tanpa jalankan bot)
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
INSTALL_ONLY=false
[[ "${1:-}" == "--install" ]] && INSTALL_ONLY=true

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
"$PIP" install --upgrade pip -q
"$PIP" install -r requirements.txt -q
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
# 5. Jalankan bot (jika tidak --install only)
# ==============================================================================
if [[ "$INSTALL_ONLY" == true ]]; then
    success "Setup selesai. Jalankan bot dengan: ./start.sh"
    exit 0
fi

header "Menjalankan Bot"
info "Mengaktifkan venv dan menjalankan main.py ..."
echo ""
exec "$VENV_DIR/bin/python" main.py
