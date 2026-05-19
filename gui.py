#!/usr/bin/env python3
"""UpscalyVid GUI — PyQt6 frontend for upscalyvid.py"""

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QProcess, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QDesktopServices, QFont, QImage, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QMenuBar,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy,
    QSpinBox, QStatusBar, QToolButton, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).parent))
from upscalyvid import MODELS, MODELS_DIR, _VIDEO_EXTS
try:
    from upscalyvid import suggest_tile as _suggest_tile
    _SUGGEST_TILE_OK = True
except Exception:
    _SUGGEST_TILE_OK = False

PYTHON = sys.executable
CLI    = str(Path(__file__).parent / "upscalyvid.py")

# ---------------------------------------------------------------------------
# Persistent config — remembers last browsed directory across sessions
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path.home() / ".config" / "upscalyvid" / "config.json"


def _load_config() -> dict:
    try:
        return json.loads(_CONFIG_PATH.read_text())
    except Exception:
        return {}


def _save_config(cfg: dict) -> None:
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# SVG arrow icons — Qt doesn't render CSS border-triangle tricks
# ---------------------------------------------------------------------------

_ICONS_DIR = Path(__file__).parent / "icons"
_ICONS_DIR.mkdir(exist_ok=True)
_ARROW_DOWN = _ICONS_DIR / "arrow_down.svg"
_ARROW_UP   = _ICONS_DIR / "arrow_up.svg"
_ARROW_DOWN.write_text(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6">'
    '<path d="M0 0 L5 6 L10 0 Z" fill="#9399b2"/></svg>'
)
_ARROW_UP.write_text(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 6">'
    '<path d="M0 6 L5 0 L10 6 Z" fill="#9399b2"/></svg>'
)
_DN = str(_ARROW_DOWN).replace("\\", "/")
_UP = str(_ARROW_UP).replace("\\", "/")

# ---------------------------------------------------------------------------
# Palette constants
# ---------------------------------------------------------------------------

DARK_BG      = "#1e1e2e"
SURFACE      = "#252535"
SURFACE2     = "#2d2d42"
BORDER       = "#3d3d5c"
ACCENT       = "#7c6af7"
ACCENT_HOVER = "#9b8cf9"
ACCENT_PRESS = "#6355cc"
TEXT         = "#cdd6f4"
TEXT_DIM     = "#7f849c"
SUCCESS      = "#a6e3a1"
WARNING      = "#f9e2af"
ERROR_COLOR  = "#f38ba8"
LOG_BG       = "#13131f"
PREVIEW_BG   = "#0d0d1a"

STYLE = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT};
    font-family: "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
    color: {TEXT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: {TEXT_DIM};
    font-size: 11px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
QLabel {{ color: {TEXT_DIM}; font-size: 12px; }}
QLineEdit {{
    background-color: {SURFACE2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {TEXT};
    selection-background-color: {ACCENT};
}}
QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox {{
    background-color: {SURFACE2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {TEXT};
    min-width: 160px;
}}
QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{
    border: none;
    border-left: 1px solid {BORDER};
    width: 28px;
    border-radius: 0 6px 6px 0;
    background: {SURFACE2};
}}
QComboBox::down-arrow {{ image: url({_DN}); width: 10px; height: 6px; }}
QComboBox QAbstractItemView {{
    background-color: {SURFACE2};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    color: {TEXT};
    padding: 4px;
}}
QSpinBox {{
    background-color: {SURFACE2};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
    color: {TEXT};
}}
QSpinBox:focus {{ border-color: {ACCENT}; }}
QSpinBox::up-button {{
    background: {SURFACE}; border: none;
    border-left: 1px solid {BORDER}; border-bottom: 1px solid {BORDER};
    width: 20px; border-top-right-radius: 6px;
}}
QSpinBox::down-button {{
    background: {SURFACE}; border: none;
    border-left: 1px solid {BORDER};
    width: 20px; border-bottom-right-radius: 6px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background: {BORDER}; }}
QSpinBox::up-arrow   {{ image: url({_UP}); width: 8px; height: 5px; }}
QSpinBox::down-arrow {{ image: url({_DN}); width: 8px; height: 5px; }}
QCheckBox {{ spacing: 8px; color: {TEXT}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border-radius: 4px; border: 2px solid {BORDER}; background: {SURFACE2};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT}; border-color: {ACCENT}; image: none;
}}
QCheckBox::indicator:checked:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton {{
    background-color: {SURFACE2}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 7px 16px; color: {TEXT}; font-weight: 500;
}}
QPushButton:hover   {{ background-color: {BORDER}; }}
QPushButton:pressed {{ background-color: {SURFACE}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; background-color: {SURFACE}; }}
QPushButton#primary {{
    background-color: {ACCENT}; border-color: {ACCENT};
    color: white; font-weight: 600; padding: 8px 24px;
}}
QPushButton#primary:hover    {{ background-color: {ACCENT_HOVER}; }}
QPushButton#primary:pressed  {{ background-color: {ACCENT_PRESS}; }}
QPushButton#primary:disabled {{ background-color: {BORDER}; color: {TEXT_DIM}; }}
QPushButton#cancel {{
    background-color: transparent; border-color: {BORDER}; color: {ERROR_COLOR};
}}
QPushButton#cancel:hover {{ background-color: rgba(243,139,168,0.1); }}
QPushButton#lossless {{
    background-color: transparent; border-color: {BORDER};
    color: {WARNING}; padding: 5px 10px; font-size: 12px;
}}
QPushButton#lossless:hover {{ background-color: rgba(249,226,175,0.08); }}
QPushButton#reset {{
    background-color: transparent; border-color: {BORDER};
    color: {TEXT_DIM}; padding: 5px 10px; font-size: 12px;
}}
QPushButton#reset:hover {{ background-color: rgba(205,214,244,0.06); }}
QProgressBar {{
    background-color: {SURFACE2}; border: 1px solid {BORDER};
    border-radius: 4px; height: 8px; color: transparent;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 4px; }}
QPlainTextEdit {{
    background-color: {LOG_BG}; border: 1px solid {BORDER};
    border-radius: 6px; color: #b4befe;
    font-family: "JetBrains Mono", "Fira Code", "Cascadia Code", monospace;
    font-size: 11px; padding: 8px; selection-background-color: {ACCENT};
}}
QStatusBar {{
    background-color: {SURFACE}; border-top: 1px solid {BORDER};
    color: {TEXT_DIM}; font-size: 11px;
}}
QToolButton {{
    background-color: {SURFACE2}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 6px 10px; color: {TEXT};
}}
QToolButton:hover {{ background-color: {BORDER}; }}
QMenuBar {{
    background-color: {DARK_BG};
    color: {TEXT};
    border-bottom: 1px solid {BORDER};
    padding: 2px 4px;
}}
QMenuBar::item {{
    padding: 4px 14px;
    border-radius: 4px;
    color: {TEXT_DIM};
}}
QMenuBar::item:selected {{ background-color: {SURFACE2}; color: {TEXT}; }}
QMenuBar::item:pressed  {{ background-color: {ACCENT};   color: #ffffff; }}
QMenu {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
    color: {TEXT};
}}
QMenu::item {{
    padding: 7px 28px 7px 14px;
    border-radius: 4px;
    font-size: 12px;
}}
QMenu::item:selected {{ background-color: {ACCENT}; color: #ffffff; }}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}
"""

# CRF / CQ scale hint shown beneath the quality spinbox
_QUALITY_HINT = (
    "CRF for H.264/H.265  ·  CQ for NVENC\n"
    "0 = lossless  ·  18 = near-lossless  ·  28 = good  ·  51 = smallest file"
)

# Model description shown in the tooltip — includes content guidance
_MODEL_TIPS = {
    "realesrgan-x4plus":          "Best all-round quality. Good for live-action footage, landscapes, people.",
    "realesrgan-x4plus-anime":    "Optimised for flat-colour anime, cartoons, illustrations.",
    "realesrgan-x2plus":          "2x only. Use when you want HD→2K or 2K→4K without over-sharpening.",
    "realesrnet-x4plus":          "Softer output — good for faces or content that tends to over-sharpen.",
    "realesr-general-x4v3":       "Fast compact model. Great default for real-world video.",
    "realesr-general-wdn-x4v3":   "Same as above but with built-in denoising. Best for compressed, noisy, or VHS footage.",
    "4x-UltraSharp":              "Community favourite for fine organic textures: fur, feathers, hair, scales, fabric. "
                                  "Preserves high-frequency micro-detail better than the official models.",
    "realesr-general-wdn-x4v3-denoise":
                                  "Alias for wdn — listed here as a reminder for heavily degraded footage.",
}

# ---------------------------------------------------------------------------
# Download thread
# ---------------------------------------------------------------------------

class DownloadThread(QThread):
    log      = pyqtSignal(str)
    finished = pyqtSignal(bool)

    def run(self):
        import urllib.request
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ok = True
        seen: set[str] = set()
        for name, info in MODELS.items():
            dst = MODELS_DIR / info["filename"]
            if not info.get("url"):
                if info["filename"] not in seen:
                    if dst.exists():
                        self.log.emit(f"  local   {info['filename']}  (custom, no download needed)")
                    else:
                        self.log.emit(f"  missing {info['filename']}  (custom — place .pth in models/)")
                seen.add(info["filename"])
                continue
            if dst.exists():
                if info["filename"] not in seen:
                    self.log.emit(f"  cached  {info['filename']}")
                seen.add(info["filename"])
                continue
            if info["filename"] in seen:
                continue
            seen.add(info["filename"])
            self.log.emit(f"  Downloading {info['filename']} …")
            tmp = dst.with_suffix(".tmp")
            try:
                req = urllib.request.Request(
                    info["url"],
                    headers={"User-Agent": "UpscalyVid/1.0", "Accept": "*/*"},
                )
                with urllib.request.urlopen(req) as resp:
                    total = int(resp.headers.get("Content-Length", 0))
                    downloaded = 0
                    with open(tmp, "wb") as f:
                        while True:
                            chunk = resp.read(65536)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                pct = min(100, downloaded * 100 // total)
                                mb  = downloaded / 1024 / 1024
                                tmb = total / 1024 / 1024
                                self.log.emit(f"\r  {pct:3d}%  {mb:.1f}/{tmb:.1f} MB  ({name})")
                tmp.rename(dst)
                self.log.emit(f"  ✓ {info['filename']}")
            except Exception as e:
                if tmp.exists():
                    tmp.unlink()
                self.log.emit(f"  ✗ {name}: {e}")
                self.log.emit(f"    → Find it manually at https://openmodeldb.info")
                ok = False
        self.finished.emit(ok)


# ---------------------------------------------------------------------------
# File-picker row helper
# ---------------------------------------------------------------------------

def file_row(placeholder: str, save: bool = False,
             allow_dir: bool = False, config: dict | None = None):
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)

    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    btn = QToolButton()
    btn.setText("Browse…")
    btn.setFixedHeight(edit.sizeHint().height())

    def _start_dir() -> str:
        if config:
            last = config.get("last_dir", "")
            if last and Path(last).exists():
                return last
        return ""

    def browse():
        start = _start_dir()
        if save:
            path, _ = QFileDialog.getSaveFileName(
                None, "Save output file", start,
                "Video files (*.mp4 *.mkv *.mov *.webm);;All files (*)")
        else:
            path, _ = QFileDialog.getOpenFileName(
                None, "Select input video", start,
                "Video files (*.mp4 *.mkv *.mov *.avi *.webm *.flv *.wmv);;All files (*)")
        if path:
            edit.setText(path)
            if config is not None:
                config["last_dir"] = str(Path(path).parent)
                _save_config(config)

    btn.clicked.connect(browse)
    layout.addWidget(edit)
    layout.addWidget(btn)

    if allow_dir:
        dir_btn = QToolButton()
        dir_btn.setText("Folder…")
        dir_btn.setFixedHeight(edit.sizeHint().height())
        dir_btn.setToolTip("Select a folder to process all videos inside it (batch mode).")

        def browse_dir():
            start = _start_dir()
            path = QFileDialog.getExistingDirectory(None, "Select input folder", start)
            if path:
                edit.setText(path)
                if config is not None:
                    config["last_dir"] = path
                    _save_config(config)

        dir_btn.clicked.connect(browse_dir)
        layout.addWidget(dir_btn)

    return widget, edit


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UpscalyVid")
        self.setMinimumWidth(800)
        self.resize(900, 900)

        self._config:     dict                   = _load_config()
        self._process:    QProcess | None       = None
        self._dl_thread:  DownloadThread | None = None
        self._work_dir:   Path | None           = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(5000)
        self._preview_timer.timeout.connect(self._refresh_preview)

        # ── menu bar ─────────────────────────────────────────────────────────
        about_menu = self.menuBar().addMenu("About")

        act_github = about_menu.addAction("GitHub Project")
        act_github.setStatusTip("Open the UpscalyVid GitHub repository")
        act_github.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://github.com/Tamalero/UpscalyVid"))
        )

        about_menu.addSeparator()

        act_models = about_menu.addAction("Find Models  (OpenModelDB)")
        act_models.setStatusTip("Browse community AI upscaling models at openmodeldb.info")
        act_models.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://openmodeldb.info"))
        )

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 12)

        # ── header ───────────────────────────────────────────────────────────
        hdr = QLabel("UpscalyVid")
        hdr.setStyleSheet(f"font-size: 22px; font-weight: 700; color: {TEXT};")
        sub = QLabel("AI Video Upscaler powered by Real-ESRGAN")
        sub.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px;")
        root.addWidget(hdr)
        root.addWidget(sub)

        # ── files ────────────────────────────────────────────────────────────
        files_box    = QGroupBox("Files")
        files_layout = QVBoxLayout(files_box)
        files_layout.setSpacing(8)
        in_row,  self.input_edit  = file_row("Select input video or folder…",
                                              allow_dir=True, config=self._config)
        out_row, self.output_edit = file_row("Select output path…", save=True)
        for lbl_text, row in [("Input video", in_row), ("Output file", out_row)]:
            files_layout.addWidget(QLabel(lbl_text))
            files_layout.addWidget(row)
        self.input_edit.textChanged.connect(self._auto_output)
        root.addWidget(files_box)

        # ── model & codec ─────────────────────────────────────────────────────
        model_box  = QGroupBox("Model & Codec")
        model_grid = QHBoxLayout(model_box)
        model_grid.setSpacing(16)

        # Left — model + outscale
        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(QLabel("Upscaling model"))
        self.model_combo = QComboBox()
        for key, info in MODELS.items():
            self.model_combo.addItem(f"{key}  ({info['scale']}x)", userData=key)
            self.model_combo.setItemData(
                self.model_combo.count() - 1,
                _MODEL_TIPS.get(key, info["description"]),
                Qt.ItemDataRole.ToolTipRole,
            )
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        left.addWidget(self.model_combo)

        # Model description label (updates on selection change)
        self.model_desc = QLabel("")
        self.model_desc.setWordWrap(True)
        self.model_desc.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px; font-style: italic; padding-top: 2px;"
        )
        left.addWidget(self.model_desc)

        left.addSpacing(4)
        left.addWidget(QLabel("Output scale"))
        self.outscale_spin = QSpinBox()
        self.outscale_spin.setRange(1, 8)
        self.outscale_spin.setValue(4)
        self.outscale_spin.setToolTip(
            "Final scale relative to input.\n"
            "Can differ from model scale — e.g. outscale 2 with a 4x model → 2x output."
        )
        left.addWidget(self.outscale_spin)
        left.addStretch()
        model_grid.addLayout(left)

        # Right — codec + quality
        right = QVBoxLayout()
        right.setSpacing(6)
        right.addWidget(QLabel("Output codec"))
        self.codec_combo = QComboBox()
        for codec, label in [
            ("h264",       "H.264  (libx264 — software)"),
            ("h265",       "H.265  (libx265 — software)"),
            ("h264_nvenc", "H.264  (NVENC — GPU hardware)"),
            ("hevc_nvenc", "H.265  (NVENC — GPU hardware)"),
        ]:
            self.codec_combo.addItem(label, userData=codec)
        right.addWidget(self.codec_combo)

        right.addSpacing(4)

        # Quality label + spinbox + quick buttons
        qual_header = QHBoxLayout()
        qual_title  = QLabel("Output quality")
        qual_title.setStyleSheet(f"color: {TEXT_DIM};")
        qual_header.addWidget(qual_title)
        qual_header.addStretch()
        right.addLayout(qual_header)

        qual_controls = QHBoxLayout()
        qual_controls.setSpacing(6)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(0, 51)
        self.quality_spin.setValue(18)
        self.quality_spin.setFixedWidth(72)
        self.quality_spin.valueChanged.connect(self._on_quality_changed)

        self.btn_lossless = QPushButton("Lossless (0)")
        self.btn_lossless.setObjectName("lossless")
        self.btn_lossless.setToolTip(
            "Set quality to 0.\n"
            "For H.264/H.265 this is true lossless (very large files).\n"
            "For NVENC CQ=0 means best-quality VBR, not strictly lossless."
        )
        self.btn_lossless.clicked.connect(lambda: self.quality_spin.setValue(0))

        self.btn_reset_q = QPushButton("↺ Default (18)")
        self.btn_reset_q.setObjectName("reset")
        self.btn_reset_q.setToolTip("Reset to 18 — near-lossless quality, practical file size.")
        self.btn_reset_q.clicked.connect(lambda: self.quality_spin.setValue(18))

        qual_controls.addWidget(self.quality_spin)
        qual_controls.addWidget(self.btn_lossless)
        qual_controls.addWidget(self.btn_reset_q)
        qual_controls.addStretch()
        right.addLayout(qual_controls)

        # Quality hint line
        self.quality_hint = QLabel(_QUALITY_HINT)
        self.quality_hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
        self.quality_value_lbl = QLabel("near-lossless")
        self.quality_value_lbl.setStyleSheet(f"color: {SUCCESS}; font-size: 11px;")
        right.addWidget(self.quality_hint)
        right.addWidget(self.quality_value_lbl)
        right.addStretch()
        model_grid.addLayout(right)

        root.addWidget(model_box)
        self._on_model_changed()
        self._on_quality_changed(18)

        # ── options ───────────────────────────────────────────────────────────
        opts_box    = QGroupBox("Options")
        opts_layout = QVBoxLayout(opts_box)
        opts_layout.setSpacing(10)

        row1 = QHBoxLayout()
        self.chk_fp32    = QCheckBox("FP32 precision  (more VRAM, default is FP16)")
        self.chk_fp32.setToolTip("Full 32-bit precision — slower but marginally more accurate.")
        self.chk_cpu     = QCheckBox("Force CPU  (very slow)")
        self.chk_cpu.setToolTip("Disable GPU and run inference on CPU.")
        row1.addWidget(self.chk_fp32)
        row1.addSpacing(24)
        row1.addWidget(self.chk_cpu)
        row1.addStretch()

        row2 = QHBoxLayout()
        self.chk_keep    = QCheckBox("Keep temp frames")
        self.chk_keep.setToolTip("Retain extracted and upscaled frame dirs after processing.")
        self.chk_jpg     = QCheckBox("JPG frames  (faster I/O, minor quality loss)")
        self.chk_jpg.setToolTip("Use JPEG instead of PNG for intermediate frames.")
        self.chk_verbose = QCheckBox("Verbose output")
        self.chk_verbose.setToolTip(
            "Show detailed logs:\n"
            "  • ffmpeg stderr\n"
            "  • Per-frame timing and ETA\n"
            "  • VRAM allocated / peak\n"
            "  • Model parameter count and CUDA version"
        )
        row2.addWidget(self.chk_keep)
        row2.addSpacing(16)
        row2.addWidget(self.chk_jpg)
        row2.addSpacing(16)
        row2.addWidget(self.chk_verbose)
        row2.addStretch()

        tile_row = QHBoxLayout()
        tile_lbl = QLabel("Tile size  (0 = auto):")
        tile_lbl.setToolTip(
            "Split large frames into tiles to cap peak VRAM usage.\n"
            "0 = auto-tile based on free VRAM (recommended)."
        )
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(0, 2048)
        self.tile_spin.setSingleStep(64)
        self.tile_spin.setValue(0)
        self.tile_spin.setFixedWidth(90)
        self.tile_hint = QLabel("")
        self.tile_hint.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
        tile_row.addWidget(tile_lbl)
        tile_row.addWidget(self.tile_spin)
        tile_row.addSpacing(10)
        tile_row.addWidget(self.tile_hint)
        tile_row.addStretch()

        opts_layout.addLayout(row1)
        opts_layout.addLayout(row2)
        opts_layout.addLayout(tile_row)
        self.model_combo.currentIndexChanged.connect(self._update_tile_hint)
        self.chk_fp32.stateChanged.connect(self._update_tile_hint)
        root.addWidget(opts_box)
        self._update_tile_hint()

        # ── action bar ────────────────────────────────────────────────────────
        act_row = QHBoxLayout()
        act_row.setSpacing(8)
        self.btn_download = QPushButton("⬇  Pre-download models")
        self.btn_download.setToolTip("Download all model weights to ./models/ now.\nThey also auto-download on first use.")
        self.btn_download.clicked.connect(self._start_download)
        self.btn_list = QPushButton("≡  List models")
        self.btn_list.clicked.connect(self._list_models)
        self.btn_start  = QPushButton("▶  Start upscale")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self._start_upscale)
        self.btn_cancel = QPushButton("✕  Cancel")
        self.btn_cancel.setObjectName("cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        act_row.addWidget(self.btn_download)
        act_row.addWidget(self.btn_list)
        act_row.addStretch()
        act_row.addWidget(self.btn_cancel)
        act_row.addWidget(self.btn_start)
        root.addLayout(act_row)

        # ── progress + preview ────────────────────────────────────────────────
        prog_box    = QGroupBox("Progress")
        prog_outer  = QVBoxLayout(prog_box)
        prog_outer.setSpacing(6)

        # Progress bar row
        bar_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setTextVisible(False)
        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; min-width: 160px;")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        bar_row.addWidget(self.progress_bar)
        bar_row.addWidget(self.progress_label)
        prog_outer.addLayout(bar_row)

        # Log + preview side by side
        split = QHBoxLayout()
        split.setSpacing(10)

        # Log (left, expands)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        self.log.setMinimumHeight(180)
        mono = QFont("JetBrains Mono", 10)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.log.setFont(mono)
        split.addWidget(self.log, stretch=3)

        # Preview panel (right, fixed width)
        preview_col = QVBoxLayout()
        preview_col.setSpacing(4)

        preview_title = QLabel("Frame preview  (updates every 5 s)")
        preview_title.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        preview_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_col.addWidget(preview_title)

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(300, 170)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            f"background: {PREVIEW_BG}; border: 1px solid {BORDER}; border-radius: 6px; color: {TEXT_DIM};"
        )
        self.preview_label.setText("No preview yet")
        preview_col.addWidget(self.preview_label)

        self.preview_info = QLabel("")
        self.preview_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_info.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
        preview_col.addWidget(self.preview_info)
        preview_col.addStretch()

        split.addLayout(preview_col, stretch=0)
        prog_outer.addLayout(split)
        root.addWidget(prog_box)

        # ── status bar ────────────────────────────────────────────────────────
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _auto_output(self, text: str):
        if not text or self.output_edit.text():
            return
        p = Path(text.rstrip("/"))
        if p.is_dir():
            self.output_edit.setText(str(p / "upscaled"))
        else:
            self.output_edit.setText(str(p.parent / f"{p.stem}_upscaled.mp4"))

    def _on_model_changed(self):
        key = self.model_combo.currentData()
        if key:
            self.outscale_spin.setValue(MODELS[key]["scale"])
            self.model_desc.setText(_MODEL_TIPS.get(key, MODELS[key]["description"]))

    def _on_quality_changed(self, val: int):
        if val == 0:
            label, color = "lossless", WARNING
        elif val <= 18:
            label, color = "near-lossless", SUCCESS
        elif val <= 28:
            label, color = "good quality", SUCCESS
        elif val <= 35:
            label, color = "medium quality", WARNING
        else:
            label, color = "low quality / small file", ERROR_COLOR
        self.quality_value_lbl.setText(label)
        self.quality_value_lbl.setStyleSheet(f"color: {color}; font-size: 11px;")

    def _update_tile_hint(self, *_):
        if not _SUGGEST_TILE_OK:
            return
        try:
            import torch
            if not torch.cuda.is_available():
                return
            key = self.model_combo.currentData()
            use_half = not self.chk_fp32.isChecked()
            device = torch.device("cuda")
            suggested = _suggest_tile(key, use_half, device)
            if suggested == 0:
                self.tile_hint.setText("full frame fits in free VRAM")
                self.tile_hint.setStyleSheet(f"color: {SUCCESS}; font-size: 10px;")
            else:
                self.tile_hint.setText(f"suggested: {suggested}px (estimated from free VRAM)")
                self.tile_hint.setStyleSheet(f"color: {WARNING}; font-size: 10px;")
        except Exception:
            self.tile_hint.setText("")

    def _log(self, text: str):
        if text.startswith("\r"):
            cursor = self.log.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.movePosition(cursor.MoveOperation.StartOfLine, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(text.lstrip("\r").rstrip())
        else:
            self.log.appendPlainText(text.rstrip())
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    _LOCKABLE = property(lambda self: [
        self.input_edit, self.output_edit,
        self.model_combo, self.codec_combo,
        self.outscale_spin, self.quality_spin, self.tile_spin,
        self.chk_fp32, self.chk_cpu, self.chk_keep, self.chk_jpg, self.chk_verbose,
        self.btn_lossless, self.btn_reset_q,
    ])

    def _set_busy(self, busy: bool):
        self.btn_start.setEnabled(not busy)
        self.btn_download.setEnabled(not busy)
        self.btn_list.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        for w in [
            self.input_edit, self.output_edit,
            self.model_combo, self.codec_combo,
            self.outscale_spin, self.quality_spin, self.tile_spin,
            self.chk_fp32, self.chk_cpu, self.chk_keep, self.chk_jpg, self.chk_verbose,
            self.btn_lossless, self.btn_reset_q,
        ]:
            w.setEnabled(not busy)

    # ── preview ───────────────────────────────────────────────────────────────

    def _refresh_preview(self):
        if not self._work_dir:
            return
        upscaled_dir = self._work_dir / "upscaled"
        frames = sorted(upscaled_dir.glob("*.png")) + sorted(upscaled_dir.glob("*.jpg"))
        if not frames:
            return
        latest = frames[-1]
        try:
            pil_img = Image.open(latest).convert("RGB")
            orig_w, orig_h = pil_img.size
            pil_img.thumbnail(
                (self.preview_label.width(), self.preview_label.height()),
                Image.LANCZOS,
            )
            w, h = pil_img.size
            qimg = QImage(pil_img.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
        except Exception:
            return
        if pix.isNull():
            return
        self.preview_label.setPixmap(pix)
        self.preview_info.setText(f"{latest.name}  ·  {orig_w}×{orig_h}")

    def _reset_preview(self):
        self.preview_label.clear()
        self.preview_label.setText("No preview yet")
        self.preview_info.setText("")

    # ── model list ────────────────────────────────────────────────────────────

    def _list_models(self):
        self.log.clear()
        self._log("Available models:\n")
        for name, info in MODELS.items():
            cached = (MODELS_DIR / info["filename"]).exists()
            tag  = "✓ cached" if cached else "⬇ not downloaded"
            tags = "  |  tags: " + ", ".join(info.get("tags", [])) if info.get("tags") else ""
            self._log(f"  {name}  [{tag}]")
            self._log(f"    {info['scale']}x  ·  {info['description']}{tags}")
            self._log(f"    {_MODEL_TIPS.get(name, '')}")
            self._log("")

    # ── download ──────────────────────────────────────────────────────────────

    def _start_download(self):
        self._set_busy(True)
        self.log.clear()
        self._log("Pre-downloading all models …\n")
        self.progress_bar.setRange(0, 0)
        self.status.showMessage("Downloading models…")
        self._dl_thread = DownloadThread()
        self._dl_thread.log.connect(self._log)
        self._dl_thread.finished.connect(self._on_dl_finished)
        self._dl_thread.start()

    def _on_dl_finished(self, ok: bool):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100 if ok else 0)
        self._set_busy(False)
        if ok:
            self._log("\nAll models downloaded.")
            self.status.showMessage("Models ready.")
        else:
            self._log("\nSome downloads failed — check the log. Community models may need manual download.")
            self.status.showMessage("Download error — see log.")

    # ── upscaling ─────────────────────────────────────────────────────────────

    def _build_cmd(self) -> list[str]:
        inp = self.input_edit.text().strip()
        is_dir = Path(inp.rstrip("/")).is_dir()
        cmd = [
            PYTHON, CLI,
            inp.rstrip("/"),
            self.output_edit.text().strip(),
            "-m",         self.model_combo.currentData(),
            "--outscale", str(self.outscale_spin.value()),
            "--codec",    self.codec_combo.currentData(),
            "-q",         str(self.quality_spin.value()),
            "--tile",     str(self.tile_spin.value()),
        ]
        # --work-dir only for single-file mode — GUI preview needs to watch it
        if not is_dir:
            cmd += ["--work-dir", str(self._work_dir)]
        if self.chk_fp32.isChecked():    cmd.append("--fp32")
        if self.chk_cpu.isChecked():     cmd.append("--cpu")
        if self.chk_keep.isChecked():    cmd.append("--keep-frames")
        if self.chk_jpg.isChecked():     cmd += ["--frame-format", "jpg"]
        if self.chk_verbose.isChecked(): cmd.append("--verbose")
        return cmd

    def _validate(self) -> bool:
        inp = self.input_edit.text().strip()
        out = self.output_edit.text().strip()
        if not inp:
            QMessageBox.warning(self, "Missing input",
                                "Please select an input video file or folder.")
            return False
        p = Path(inp.rstrip("/"))
        if not p.exists():
            QMessageBox.warning(self, "Not found", f"Input does not exist:\n{inp}")
            return False
        if p.is_dir():
            videos = [f for f in p.iterdir() if f.suffix.lower() in _VIDEO_EXTS]
            if not videos:
                QMessageBox.warning(self, "No videos",
                                    f"No video files found in:\n{inp}\n\n"
                                    f"Supported: {', '.join(sorted(_VIDEO_EXTS))}")
                return False
        if not out:
            QMessageBox.warning(self, "Missing output", "Please specify an output path.")
            return False
        return True

    def _start_upscale(self):
        if not self._validate():
            return

        # Create a persistent work dir so the preview timer can watch it
        self._work_dir = Path(tempfile.mkdtemp(prefix="upscalyvid_"))
        (self._work_dir / "upscaled").mkdir()

        cmd = self._build_cmd()
        self.log.clear()
        self._log("$ " + " ".join(cmd))
        self._log("")

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting…")
        self._reset_preview()
        self._set_busy(True)
        self.status.showMessage("Upscaling…")

        self._preview_timer.start()

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._on_output)
        self._process.finished.connect(self._on_finished)
        self._process.start(cmd[0], cmd[1:])

    # Regexes for parsing CLI stdout
    _FRAME_RE   = re.compile(r"\[.+?\]\s+([\d.]+)%\s+(\d+)/(\d+)\s+frames")
    _EXTRACT_RE = re.compile(r"(\d+)\s+frames extracted")
    _STEP_RE    = re.compile(r"\[(\d)/3\]")

    def _on_output(self):
        raw = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in raw.splitlines(keepends=True):
            s = line.rstrip("\n")

            m = self._FRAME_RE.search(s)
            if m:
                pct, cur, total = float(m.group(1)), int(m.group(2)), int(m.group(3))
                self.progress_bar.setValue(int(pct))
                self.progress_label.setText(f"{pct:.1f}%  ({cur}/{total} frames)")
                self._log("\r" + s)
                continue

            if self._EXTRACT_RE.search(s):
                pass  # frame count captured if needed

            m = self._STEP_RE.search(s)
            if m:
                self.progress_bar.setValue((int(m.group(1)) - 1) * 30)

            self._log(s)

    def _on_finished(self, exit_code: int, _status):
        self._preview_timer.stop()
        self._refresh_preview()   # show the last frame
        self._set_busy(False)

        if exit_code == 0:
            self.progress_bar.setValue(100)
            self.progress_label.setText("Complete")
            self.status.showMessage(f"Done → {self.output_edit.text().strip()}")
            self._log("\n✓ Upscaling complete.")
        elif exit_code == 130:
            self.progress_label.setText("Cancelled")
            self.status.showMessage("Cancelled.")
            self._log("\n✗ Cancelled by user.")
        else:
            self.progress_label.setText("Error")
            self.status.showMessage(f"Error (exit {exit_code})")
            self._log(f"\n✗ Process exited with code {exit_code}.")

        self._process  = None
        self._work_dir = None

    def _cancel(self):
        if self._process:
            self._process.kill()
        if self._dl_thread and self._dl_thread.isRunning():
            self._dl_thread.terminate()
        self._preview_timer.stop()
        self.progress_label.setText("Cancelling…")
        self.status.showMessage("Cancelling…")

    def closeEvent(self, event):
        if self._process:
            self._process.kill()
        if self._dl_thread and self._dl_thread.isRunning():
            self._dl_thread.terminate()
        self._preview_timer.stop()
        if self._work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _ensure_desktop_file():
    """Write a .desktop file so the freedesktop portal can resolve this app's ID."""
    dst = Path.home() / ".local" / "share" / "applications" / "upscalyvid.desktop"
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=UpscalyVid\n"
        "Comment=AI Video Upscaler powered by Real-ESRGAN\n"
        f"Exec={sys.executable} {Path(__file__).resolve()}\n"
        "Icon=video-x-generic\n"
        "Terminal=false\n"
        "Categories=Video;Graphics;AudioVideo;\n"
        "StartupNotify=true\n"
        "StartupWMClass=upscalyvid\n"
    )


def main():
    _ensure_desktop_file()
    app = QApplication(sys.argv)
    app.setApplicationName("UpscalyVid")
    app.setDesktopFileName("upscalyvid")
    app.setStyleSheet(STYLE)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(DARK_BG))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base,            QColor(SURFACE2))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.Text,            QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button,          QColor(SURFACE2))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_DIM))
    app.setPalette(palette)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
