#!/bin/bash
# build-appimage.sh — builds UpscalyVid-x86_64.AppImage (Type 2 with auto-update)
# Usage: ./build-appimage.sh [VERSION]
#   VERSION defaults to output of `git describe --tags --always`

set -euo pipefail

VERSION="${1:-$(git describe --tags --always 2>/dev/null || echo "dev")}"
APPNAME="UpscalyVid"
APPDIR="${APPNAME}.AppDir"
OUTPUT="${APPNAME}-${VERSION}-x86_64.AppImage"

echo "==> Building ${OUTPUT}"

# ── Clean old AppDir ──────────────────────────────────────────────────────────
rm -rf "$APPDIR"
mkdir -p "$APPDIR/opt/${APPNAME}"
mkdir -p "$APPDIR/usr/share/icons/hicolor/256x256/apps"
mkdir -p "$APPDIR/usr/share/applications"

# ── Copy app files ────────────────────────────────────────────────────────────
cp upscalyvid.py gui.py requirements.txt "$APPDIR/opt/${APPNAME}/"
# setup.sh and launch.fish are not needed inside the AppImage
# models/ is excluded (downloaded at runtime into ~/.local/share/UpscalyVid)

# ── Icon ──────────────────────────────────────────────────────────────────────
cp upscalyvid.png "$APPDIR/upscalyvid.png"
cp upscalyvid.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/upscalyvid.png"
cp upscalyvid.svg "$APPDIR/upscalyvid.svg" 2>/dev/null || true

# ── Desktop file ─────────────────────────────────────────────────────────────
cat > "$APPDIR/upscalyvid.desktop" << 'EOF'
[Desktop Entry]
Type=Application
Name=UpscalyVid
Comment=AI Video Upscaler powered by Real-ESRGAN and HAT
Exec=upscalyvid
Icon=upscalyvid
Categories=Video;Graphics;AudioVideo;
Terminal=false
StartupNotify=true
StartupWMClass=upscalyvid
Keywords=upscale;video;ai;esrgan;hat;super resolution;
EOF
cp "$APPDIR/upscalyvid.desktop" "$APPDIR/usr/share/applications/"

# ── AppRun ────────────────────────────────────────────────────────────────────
cat > "$APPDIR/AppRun" << 'APPRUN'
#!/bin/bash
# AppRun — UpscalyVid AppImage launcher

SELF=$(realpath "$0")
HERE=$(dirname "$SELF")
APP_CODE="$HERE/opt/UpscalyVid"

# Data directory (writable, persists across runs)
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/UpscalyVid"
VENV_DIR="$DATA_DIR/venv"
MODELS_DIR="$DATA_DIR/models"
mkdir -p "$DATA_DIR" "$MODELS_DIR"

# ── Find best Python ──────────────────────────────────────────────────────────
find_python() {
    for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
        local bin
        bin=$(command -v "$candidate" 2>/dev/null) || continue
        # Prefer Python that already has torch with CUDA
        if "$bin" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo "$bin"
            return 0
        fi
    done
    # Fall back to any Python 3 that has torch (CPU mode)
    for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
        local bin
        bin=$(command -v "$candidate" 2>/dev/null) || continue
        if "$bin" -c "import torch" 2>/dev/null; then
            echo "$bin"
            return 0
        fi
    done
    # Last resort: any Python 3
    for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
        local bin
        bin=$(command -v "$candidate" 2>/dev/null) || continue
        if "$bin" -c "import sys; assert sys.version_info >= (3,10)" 2>/dev/null; then
            echo "$bin"
            return 0
        fi
    done
    echo ""
}

PYTHON=$(find_python)
if [ -z "$PYTHON" ]; then
    zenity --error --title="UpscalyVid" \
        --text="Python 3.10+ is required but was not found.\nInstall Python and try again." \
        2>/dev/null || \
    echo "ERROR: Python 3.10+ not found. Please install Python." >&2
    exit 1
fi

HAS_CUDA=$("$PYTHON" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")

# ── Create / update venv ──────────────────────────────────────────────────────
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[UpscalyVid] Setting up environment (first run)…"
    "$PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi

# Install / upgrade minimal deps (non-torch)
"$VENV_PIP" install -q --upgrade Pillow PyQt6 numpy 2>/dev/null || true

# If torch not available via system-site-packages, install CPU torch as fallback
if ! "$VENV_PYTHON" -c "import torch" 2>/dev/null; then
    echo "[UpscalyVid] Installing PyTorch (CPU mode — GPU requires system CUDA PyTorch)…"
    "$VENV_PIP" install -q torch --index-url https://download.pytorch.org/whl/cpu
fi

# Warn if GPU not available
if [ "$HAS_CUDA" = "False" ]; then
    echo "[UpscalyVid] WARNING: CUDA not available. Upscaling will run on CPU (very slow)." >&2
fi

# ── Run the app ───────────────────────────────────────────────────────────────
# Point models dir to writable location inside DATA_DIR
export UPSCALYVID_MODELS_DIR="$MODELS_DIR"
export UPSCALYVID_APP_DIR="$APP_CODE"

exec "$VENV_PYTHON" "$APP_CODE/gui.py" "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

# ── Embed update information and build AppImage ───────────────────────────────
# Format: gh-releases-zsync|USER|REPO|CHANNEL|FILENAME_PATTERN
UPDATE_INFO="gh-releases-zsync|Tamalero|UpscalyVid|latest|UpscalyVid-*-x86_64.AppImage.zsync"

echo "==> Running appimagetool…"
ARCH=x86_64 appimagetool \
    --updateinformation "$UPDATE_INFO" \
    "$APPDIR" \
    "$OUTPUT"

# ── Generate zsync file for AppImageUpdate ────────────────────────────────────
echo "==> Generating .zsync file…"
zsyncmake -C -u "$OUTPUT" -o "${OUTPUT}.zsync" "$OUTPUT"

# ── SHA256 checksum ───────────────────────────────────────────────────────────
sha256sum "$OUTPUT" > "${OUTPUT}.sha256"

echo ""
echo "==> Done!"
echo "    AppImage : $OUTPUT  ($(du -sh "$OUTPUT" | cut -f1))"
echo "    zsync    : ${OUTPUT}.zsync"
echo "    SHA256   : $(cat "${OUTPUT}.sha256" | cut -d' ' -f1)"
echo ""
echo "Upload both $OUTPUT and ${OUTPUT}.zsync to the GitHub release."
