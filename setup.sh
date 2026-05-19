#!/usr/bin/env bash
# UpscalyVid setup — verifies dependencies and optionally pre-downloads models
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

# ── colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓${NC} $*"; }
warn() { echo -e "${YELLOW}  ⚠${NC} $*"; }
fail() { echo -e "${RED}  ✗${NC} $*"; exit 1; }

echo ""
echo "=================================================="
echo "  UpscalyVid — setup check"
echo "=================================================="

# ── ffmpeg ────────────────────────────────────────────────────────────────────
echo ""
echo "[ ffmpeg ]"
if command -v ffmpeg &>/dev/null; then
    VER=$(ffmpeg -version 2>&1 | head -1)
    ok "$VER"
else
    fail "ffmpeg not found. Install with: sudo pacman -S ffmpeg   (or your distro's package manager)"
fi

# ── python ────────────────────────────────────────────────────────────────────
echo ""
echo "[ Python ]"
PYTHON=$(command -v python3 || command -v python || true)
if [[ -z "$PYTHON" ]]; then
    fail "python3 not found."
fi
PY_VER=$("$PYTHON" --version 2>&1)
ok "$PY_VER  →  $PYTHON"

# ── packages ──────────────────────────────────────────────────────────────────
echo ""
echo "[ Python packages ]"

check_pkg() {
    local name="$1"
    local import="$2"
    local hint="$3"
    if "$PYTHON" -c "import $import" 2>/dev/null; then
        VER=$("$PYTHON" -c "import $import; print(getattr($import, '__version__', 'ok'))" 2>/dev/null || echo "ok")
        ok "$name  $VER"
    else
        warn "$name not found — install with: $hint"
    fi
}

check_pkg "torch"  "torch"  "pip install torch --index-url https://download.pytorch.org/whl/cu128"
check_pkg "numpy"  "numpy"  "pip install numpy"
check_pkg "Pillow" "PIL"    "pip install Pillow"

# CUDA
echo ""
echo "[ CUDA ]"
"$PYTHON" - <<'EOF'
import sys
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            vram = p.total_memory / 1024**3
            print(f"  \033[0;32m  ✓\033[0m GPU {i}: {p.name}  ({vram:.1f} GB VRAM)")
    else:
        print("  \033[1;33m  ⚠\033[0m No CUDA GPU detected — will run on CPU (slow)")
except ImportError:
    print("  \033[1;33m  ⚠\033[0m torch not available, skipping CUDA check")
EOF

# ── model pre-download (optional) ─────────────────────────────────────────────
echo ""
echo "[ Models ]"
mkdir -p "$MODELS_DIR"

MODELS=(
    "realesrgan-x4plus|RealESRGAN_x4plus.pth|https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    "realesrgan-x4plus-anime|RealESRGAN_x4plus_anime_6B.pth|https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    "realesrgan-x2plus|RealESRGAN_x2plus.pth|https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    "realesrnet-x4plus|RealESRNet_x4plus.pth|https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name filename url <<< "$entry"
    path="$MODELS_DIR/$filename"
    if [[ -f "$path" ]]; then
        size=$(du -sh "$path" 2>/dev/null | cut -f1)
        ok "$name  ($size)"
    else
        warn "$name not cached — will auto-download on first use"
    fi
done

if [[ "${1:-}" == "--download-all" ]]; then
    echo ""
    echo "Pre-downloading all models …"
    for entry in "${MODELS[@]}"; do
        IFS='|' read -r name filename url <<< "$entry"
        path="$MODELS_DIR/$filename"
        if [[ ! -f "$path" ]]; then
            echo "  Downloading $name …"
            curl -L --progress-bar -o "$path" "$url"
            ok "Downloaded $filename"
        fi
    done
fi

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Setup check complete."
echo ""
echo "  Usage:"
echo "    python3 upscalyvid.py input.mp4 output.mp4"
echo "    python3 upscalyvid.py --help"
echo "    python3 upscalyvid.py --list-models"
echo ""
echo "  Pre-download all models:"
echo "    bash setup.sh --download-all"
echo "=================================================="
echo ""
