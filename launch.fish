#!/usr/bin/env fish

set SCRIPT_DIR (realpath (dirname (status --filename)))
set VENV $SCRIPT_DIR/.venv
set REQS $SCRIPT_DIR/requirements.txt
set PYTHON $VENV/bin/python
set PIP $VENV/bin/pip

# ── create venv if missing ────────────────────────────────────────────────────
if not test -d $VENV
    echo "[launch] Creating virtual environment at .venv …"
    # --system-site-packages inherits the system CUDA-enabled torch
    python3 -m venv --system-site-packages $VENV
end

# ── install / sync dependencies ───────────────────────────────────────────────
set STAMP $VENV/.installed
if not test -f $STAMP; or test $REQS -nt $STAMP
    echo "[launch] Installing dependencies from requirements.txt …"
    $PIP install --quiet --upgrade pip
    $PIP install --quiet -r $REQS
    touch $STAMP
end

# ── verify PyQt6 ──────────────────────────────────────────────────────────────
if not $PYTHON -c "from PyQt6.QtWidgets import QApplication" 2>/dev/null
    echo "[launch] PyQt6 not found in venv — installing …"
    $PIP install --quiet PyQt6
end

# ── launch GUI ────────────────────────────────────────────────────────────────
echo "[launch] Starting UpscalyVid …"
exec $PYTHON $SCRIPT_DIR/gui.py $argv
