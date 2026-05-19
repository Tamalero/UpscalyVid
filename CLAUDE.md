# UpscalyVid — Claude Context

AI-powered video upscaler inspired by [Upscayl](https://github.com/upscayl/upscayl). Uses Real-ESRGAN and HAT models via a self-contained PyTorch implementation (no basicsr/opencv dependency). Includes a PyQt6 GUI and a CLI.

## GitHub repository

**URL**: https://github.com/Tamalero/UpscalyVid  
Releases include AppImage + .zsync + .sha256. CI workflow at `.github/workflows/release.yml` builds and publishes automatically on `v*` tag push.

## Project layout

```
UpscalyVid/
├── upscalyvid.py              # CLI + all AI logic (self-contained)
├── gui.py                     # PyQt6 GUI frontend
├── launch.fish                # Fish launcher — auto-creates venv and starts GUI
├── setup.sh                   # Bash env checker + optional model pre-download
├── requirements.txt           # Minimal: torch, numpy, Pillow
├── build-appimage.sh          # Builds AppImage Type 2 + zsync + sha256
├── upscalyvid.svg / .png      # App icon (SVG source + 256×256 PNG)
├── .github/workflows/         # GitHub Actions: release.yml builds AppImage on v* tag
├── models/                    # Downloaded .pth weights (auto-created, gitignore)
├── icons/                     # SVG arrow icons written at GUI startup
└── .venv/                     # Local venv (auto-created by launch.fish)
```

## How to run

```fish
./launch.fish                      # creates .venv if needed, installs deps, opens GUI
python3 upscalyvid.py --help       # CLI usage
python3 upscalyvid.py --list-models
bash setup.sh --download-all       # pre-download all model weights
```

## System environment (dev machine)

- **OS**: Linux (CachyOS / Arch-based), Fish shell
- **Python**: 3.14.4 (system)
- **PyTorch**: 2.11.0 with CUDA (system-installed)
- **GPU**: NVIDIA RTX 4090 — 23.5 GB VRAM
- **ffmpeg**: n8.1.1 (full build with NVENC, CUDA, Vulkan)
- **PyQt6**: 6.11.0 (system)

## Venv strategy

`launch.fish` creates `.venv` with `--system-site-packages` so the venv inherits the system CUDA-enabled PyTorch without re-downloading the multi-GB wheel. PyQt6, Pillow, and numpy come from the system too; the venv only layers on top.

## Architecture — upscalyvid.py

### AI pipeline

1. **ffmpeg** → extract all frames to `{work_dir}/frames/` (`frame_%08d.png` or `.jpg`)
2. **Real-ESRGAN / HAT** (PyTorch + CUDA, FP16 for RRDBNet/SRVGGNet, FP32 for HAT) → upscale each frame into `{work_dir}/upscaled/`
3. **ffmpeg** → re-encode upscaled frames + mux original audio back in

`work_dir` is a `tempfile.mkdtemp()` by default, but `--work-dir PATH` lets the caller (GUI) specify it so the preview panel can watch the upscaled dir.

### Neural network classes (self-contained, no basicsr)

| Class | Used by |
|---|---|
| `ResidualDenseBlock` | RRDBNet internals |
| `RRDB` | RRDBNet internals |
| `RRDBNet` | realesrgan-x4plus family |
| `SRVGGNetCompact` | realesr-general-x4v3 family |
| `CPBMLP` | HAT — continuous position bias MLP (one per WinAttnHead) |
| `WinAttnHead` | HAT — one head-group (3 heads) in WinAttn; owns rpe_biases + relative_position_index buffers |
| `SimpleGate` | HAT — gated FFN gate (DWConv + LayerNorm on half, then multiply) |
| `GatedFFN` | HAT — fc1 → SimpleGate → fc2 |
| `WinAttn` | HAT — shifted-window self-attention (even blocks) |
| `ChanAttn` | HAT — Restormer-style channel attention (odd blocks) |
| `WinAttnBlock` | HAT — even-indexed transformer block |
| `ChanAttnBlock` | HAT — odd-indexed transformer block |
| `ResidualGroup` | HAT — one residual layer (6 blocks + trailing Conv2d) |
| `HAT` | HAT top-level model (Hybrid Attention Transformer) |
| `RealESRGANer` | inference wrapper (tiling, auto-tiling, pre/post-pad, window-pad, strict, verbose) |

`RealESRGANer.enhance(img_rgb)` takes/returns **RGB numpy uint8**. Internally converts to BGR for the model (training convention) and back.

### Custom model discovery

`_load_custom_models()` runs at import time: scans `models/*.pth` for filenames not in the MODELS registry, registers each as `Custom-<stem>` with `arch="auto"` and a filename-based scale hint (looks for x2/x4/x8 in the stem).

`_safe_load_checkpoint(path)` tries `weights_only=True` first, falls back to `weights_only=False` for legacy pickle formats.

`_detect_arch_from_weights(weights)` inspects state dict keys to identify the architecture:
- `body.N.rdb*` keys → RRDBNet (counts blocks, detects scale from conv_up2/up3)
- `body.N.weight` flat (no rdb) → SRVGGNet
- `before_RG` + `layers.N.blocks.M.attn.attns.*` → HAT (delegates to `_detect_hat_params`)
- `model.*` keys → old ESRGAN format. For `arch="auto"` custom models: reported as unsupported (error with re-download note). For registered models (e.g. `4x-UltraSharp`): transparently remapped by `_remap_esrgan_keys()` before `load_state_dict` — these models load and run correctly without user intervention.
- other → unrecognised (error pointing to chaiNNer/ComfyUI)

`_detect_hat_params(weights)` extracts all HAT hyperparameters from key shapes: `embed_dim`, `num_heads`, `head_dim`, `window_size`, `shift_size`, `num_layers`, `num_blocks_per_layer`, `ffn_ratio`, `ch_sq`, `sp_sq`, `heads_per_attn`, `upsample_mid_ch`, `scale`.

### Model registry — `MODELS` dict

Registered models have:
```python
{
    "arch":        "rrdbnet" | "srvgg" | "hat" | "auto",
    "scale":       2 | 4 | 8,
    "num_block":   23 | 6,      # rrdbnet only
    "num_feat":    64,           # srvgg only
    "num_conv":    32,           # srvgg only
    "url":         "https://...",   # empty string for local-only custom models
    "filename":    "RealESRGAN_x4plus.pth",
    "description": "...",
    "tags":        ["general", "video", ...],
}
```

Custom models discovered by `_load_custom_models()` use `arch="auto"` and `url=""`. `build_upsampler()` probes these at load time via `_detect_arch_from_weights`. `download_model()` skips empty URLs (prints size from disk instead).

`build_upsampler()` reads `arch` and instantiates the right class. For `arch="auto"` it probes the checkpoint first. For `arch="hat"` it instantiates `HAT(...)` with `strict=False`, `half=False` (forced FP32), and passes `window_size` to `RealESRGANer`. HAT models are passed `compile_model=False` regardless of the CLI flag (cudagraphs replays wrong graph for differently-shaped tiles → rainbow artifacts). After constructing the upsampler it sets `upsampler._vram_per_pixel` based on arch and half-precision flag (RRDBNet FP16 = 3500, SRVGGNet FP16 = 1200, doubles for FP32; HAT = 0). `RealESRGANer.__init__` automatically remaps old ESRGAN `model.*` keys via `_remap_esrgan_keys()` before calling `load_state_dict`.

Weights are auto-downloaded to `./models/` on first use via `download_model()`. Downloads use `urllib.request` with `User-Agent: UpscalyVid/1.0` and `Accept: */*` headers (required by HuggingFace CDN — bare `urlretrieve` returns 401). Atomic `.tmp` rename on completion.

If a community model download fails, the error message points to `openmodeldb.info`.

### RealESRGANer

```python
RealESRGANer(model, model_path, scale, tile, tile_pad, pre_pad,
             half, device, verbose, strict=True, window_size=1)
```

- `strict`: passed to `model.load_state_dict()`. Set to `False` for HAT (stored `attn_mask` buffers are image-size-dependent and skipped; they are recomputed at forward time).
- `window_size`: when > 1, `_infer()` pads the input tensor to the next multiple of `window_size` before the model call and crops the output back. Required for HAT's window attention.
- `_vram_per_pixel`: internal attribute set by `build_upsampler()` after construction — empirical bytes-per-pixel peak estimate used by `_auto_tile` for RRDBNet/SRVGGNet. RRDBNet FP16 = 3500, FP32 = 7000; SRVGGNet FP16 = 1200, FP32 = 2400. HAT leaves it at 0 (uses the window-size formula instead).

**`_auto_tile(h, w)`**: called inside `enhance()` when `tile == 0`. Uses `torch.cuda.mem_get_info(device)` for actual free VRAM (not `total × 0.65` — that caused OOM when other processes held VRAM). Keeps 512 MB headroom. Three branches:
- `tile > 0`: return immediately (user-specified tile, no auto logic).
- `window_size > 1` (HAT): Formula: `3 × nW × ws⁴ × 4 B × 12`. The base `3 × nW × ws⁴ × 4 B` is the SDPA attn_mask size (one head-group, sequential so only one exists at a time). The **12× safety factor** accounts for Q/K/V projections, DWConv buffers, upsample intermediates, and other intermediate activations not captured by the attn_mask alone — empirically calibrated: a 720×944 input OOMs with 23 GB available, implying the base formula underestimates by ~11×. **Do not restore the old `2 ×` multiplier** — that was a different bug that over-estimated the attn_mask itself by 2×. **Do not reduce the 12× factor** — it causes OOM on full-frame HAT inference. Picks tile from ladder `ws × {48, 40, 32, 24, 16, 12, 8}`, first that fits in `avail_bytes / 2`. Minimum `ws × 8`. Prints chosen tile unconditionally.
- `_vram_per_pixel > 0` (RRDBNet/SRVGGNet): estimates `h_eff × w_eff × vpp`; tile = `√(avail / 2 / vpp)` rounded down to 64 px, min 128. On a 6 GB GPU with ~5 GB free, RRDBNet 1080p → tile ≈ 512.

**`_to_numpy(t)`**: calls `torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)` before `clamp(0,1)` to safely handle any NaN/inf that survive from fp16 operations.

**`enhance()` dark-pixel bicubic blend** (HAT only, `window_size > 1`): After the model output is produced and pre_pad is removed, `enhance()` corrects the HAT dark-area rainbow artifact with a two-stage blend. `SOLID_THR=100`, `DARK_THR=180`: for brightness ≤ 100 the output is replaced 100% with bicubic-upscaled input (no model contribution at all — model is definitely wrong here); for brightness 101–179 a linear taper reduces the bicubic fraction from 100% to 0%; for brightness ≥ 180 the model output is used as-is. A linear blend from 0 to some single threshold (e.g. 128) still leaves too much wrong model colour for pixels in the 60–100 range (e.g. brightness=60 → 53% model × wildly-wrong colour = still visibly rainbow). The hard cutoff at 100 eliminates the rainbow on dark fabric/shadow completely. The blend weight is upscaled to output resolution via `F.interpolate(mode='bilinear')` for smooth edges at fabric/skin boundaries. A PIL BICUBIC resize of the original input to 4× is computed only if any pixel needs blending (`bic_w.max() > 0`). Do not apply this to RRDBNet/SRVGGNet — they produce correct dark output without it.

`F.softmax` on CUDA silently upcasts fp16→fp32; `ChanAttn` casts back with `.to(q.dtype)` after softmax. `WinAttnHead` no longer calls `F.softmax` directly — it uses `F.scaled_dot_product_attention` (see below).

### HAT must run in FP32

HAT activations overflow fp16 after ~5 residual groups of accumulation. The overflow cascades into `LayerNorm` (inf − inf = NaN), producing all-NaN output that renders as all-black frames. `build_upsampler()` forces `half=False` for any model with `arch="hat"` regardless of the `--fp32` flag. RRDBNet and SRVGGNet are unaffected and still use FP16.

### WinAttnHead — SDPA

`WinAttnHead.forward()` uses `F.scaled_dot_product_attention` instead of the manual `q @ k.T * scale → softmax → @ v` sequence. The RPE bias `[1, heads, L, L]` and shift mask `[nW, L, L]` are summed into a single float `attn_mask` (`[nW, heads, L, L]`; expanded to `[nWB, heads, L, L]` only when batch > 1). SDPA with a float mask selects **Memory-Efficient Attention** on CUDA, which avoids materialising the `[nWB, heads, L, L]` score tensor in global memory. This was the source of the 2.10 GiB allocation that caused OOM on a 720×944 frame with ~1 GB free VRAM. On sm80+ (Ampere, Ada Lovelace) this backend is highly optimised; on sm100+ (Blackwell) future PyTorch versions will use Flash Attention 3 automatically via the same API call.

### Checkpoint loading

```python
weights = ckpt.get("params_ema") or ckpt.get("params") or ckpt
```
Handles Real-ESRGAN format (`params_ema`), older format (`params`), and bare state-dicts.

`_safe_load_checkpoint(path)` tries `weights_only=True`, then `weights_only=False`. If both fail, it peeks at the first 16 bytes and checks two patterns:
- `b"version https://"` → **Git LFS pointer** (from `git clone` without LFS) — delete and re-run.
- First 4 bytes `b"v1.0"` → **old Torch format from ~2021** (pre-Python ESRGAN era, e.g. models downloaded from old ESRGAN GUI tools) — delete and re-download from the correct HuggingFace URL.
Raises `RuntimeError` with an actionable hint rather than letting the raw pickle error propagate. Note: "invalid load key, 'v'" is the signature for BOTH cases (first byte 'v' = 0x76). The hint distinguishes them.

After a successful load, if the first weight key starts with `"model."`, `_remap_esrgan_keys(weights)` converts old ESRGAN key names to Real-ESRGAN names before `load_state_dict`. Key mapping: `model.0` → `conv_first`; `model.1.sub.N.RDB{k}.conv{j}.0` → `body.N.rdb{k}.conv{j}` (lowercase, no `.0`); `model.1.sub.{max_idx}` → `conv_body`; `model.3/6/8/10` → `conv_up1/conv_up2/conv_hr/conv_last`.

### Verbose mode (`-v` / `--verbose`)

When enabled:
- ffmpeg runs without `-loglevel error` (full stderr visible)
- Per-frame line: `frame N/total | X.XXs/frame | ETA Xs | VRAM used/peak GB`
- Model load prints parameter count, dtype, CUDA version, SM architecture + which SDPA backend applies (`Flash Attention (sm80+)` / `Memory-Efficient Attention (sm70+)` / `Math fallback`)
- `download_model()` prints the full URL
- `build_upsampler()` prints detected arch and scale; prints FP32 override note for HAT
- After the **first frame** is upscaled, two PNG files are saved to `work_dir/` for side-by-side quality inspection:
  - `source_frame.png` — the raw extracted frame at original resolution (RGB)
  - `sample_frame.png` — the upscaled result for that same frame
  Both paths are printed to stdout.

The `upscale_frames()` function also reports total frames and average fps on completion in all modes.

### `torch.compile` and Triton

`_triton_available()` helper checks `import triton` at runtime. Triton is **not installed** on the dev system. Based on the result:
- Triton present → `torch.compile(model, mode="reduce-overhead")` (inductor backend — fastest)
- Triton absent → `torch.compile(model, backend="cudagraphs")` (CUDA graph replay — no Triton, similar speedup for fixed-shape video)

**Critical:** `torch.compile()` itself always succeeds regardless of Triton. `TritonMissing` only surfaces at the first forward call. A try/except around `torch.compile()` does **not** protect against this — the Triton check must happen before calling compile.

`torch.set_float32_matmul_precision("high")` is set in `main()` when CUDA is available. Enables TF32 for FP32 matmul on sm80+ (free speedup for HAT inference) and suppresses the inductor warning about unset matmul precision.

### `vprint()` helper

```python
def vprint(msg: str, verbose: bool) -> None:
```
Used throughout to conditionally emit verbose-only lines without cluttering call sites with `if verbose` blocks.

## Models

### Full registry

| Key | Arch | Scale | Tags | Best for |
|---|---|---|---|---|
| `realesrgan-x4plus` | RRDBNet-23 | 4x | general, photo, video | Best all-round quality |
| `realesrgan-x4plus-anime` | RRDBNet-6 | 4x | anime, cartoon | Flat-colour anime / illustration |
| `realesrgan-x2plus` | RRDBNet-23 | 2x | general, 2x | HD→2K or 2K→4K |
| `realesrnet-x4plus` | RRDBNet-23 | 4x | general, restoration, soft | Softer output, less sharpening |
| `realesr-general-x4v3` | SRVGGNet | 4x | general, video, fast | Fast compact model, default for video |
| `realesr-general-wdn-x4v3` | SRVGGNet | 4x | general, video, fast, denoise | Noisy/compressed footage |
| `4x-UltraSharp` | RRDBNet-23 | 4x | fur, feathers, hair, texture, detail, community | Fine organic textures |
| `realesr-general-wdn-x4v3-denoise` | SRVGGNet | 4x | degraded, vhs, film, artefacts | Alias of wdn for degraded/VHS footage |

Custom `.pth` files placed in `models/` appear automatically as `Custom-<stem>` with `arch="auto"`. Example: `models/4x-UltraSharpV2.pth` → `Custom-4x-UltraSharpV2` (detected as HAT, 4x, embed_dim=180, forced FP32).

### Adding a new model

1. Add an entry to `MODELS` with the correct `arch` key and a `tags` list.
2. If it uses a new architecture, implement the class and add a branch in `build_upsampler()`. Also set `upsampler._vram_per_pixel` to an empirical bytes-per-pixel estimate so `_auto_tile` works on VRAM-limited GPUs.
3. Add a human-readable tip string to `_MODEL_TIPS` in `gui.py`.
4. The GUI combo box picks up new models automatically at startup.

**Supported architectures**: `rrdbnet`, `srvgg`, `hat`, `auto` (probed at load time)
**Compatible weight sources**: xinntao/Real-ESRGAN GitHub releases, HuggingFace `.pth` files using `params_ema` / `params` / bare state-dict. HAT models from Kim2091 and similar community authors.
**Community models**: hosted on HuggingFace. If download fails the error message directs to `openmodeldb.info`.

### HAT architecture details (4x-UltraSharpV2 family)

HAT = Hybrid Attention Transformer. Alternates window self-attention (even blocks) with Restormer-style channel attention (odd blocks) within each `ResidualGroup`.

For `4x-UltraSharpV2.pth` the detected hyperparameters are:
- `embed_dim=180`, `num_heads=6`, `head_dim=30`, `heads_per_attn=3`
- `window_size=16`, `shift_size=8`
- `num_layers=6`, `num_blocks_per_layer=6`
- `ffn_ratio=2`, `ch_sq=22`, `sp_sq=11`, `upsample_mid_ch=64`, `scale=4`

The checkpoint stores 18 `attn_mask_*` buffers (image-size-dependent, recomputed at runtime) — these are in `unexpected_keys` after `strict=False` loading and are expected.

VRAM usage (FP32, no tiling): ~4 GB for 1080p, ~16 GB for 4K before auto-tiling kicks in. Auto-tiling at tile=512 keeps peak well under 4 GB per tile.

## GUI — gui.py

Built with **PyQt6**. Dark Catppuccin-inspired theme.

### Key design notes

- **SVG arrows**: Qt does not render CSS border-triangle tricks. `gui.py` writes `icons/arrow_down.svg` and `icons/arrow_up.svg` at import time and references them via `image: url(path)` in the stylesheet. Always use this pattern for arrow icons — never the CSS border trick. The icons are written to `_ICONS_DIR = _DATA_DIR / "icons"` where `_DATA_DIR` is `UPSCALYVID_DATA_DIR` env var (set by AppRun to `~/.local/share/UpscalyVid`) or `Path(__file__).parent` for source installs.
- **Desktop file**: `_ensure_desktop_file()` is called at the top of `main()`, before `QApplication` is created. It writes `~/.local/share/applications/upscalyvid.desktop` once if missing. Required so `app.setDesktopFileName("upscalyvid")` can be resolved by the freedesktop portal — without it Qt logs a portal registration error on KDE/Wayland. When running from an AppImage the `Exec=` line uses the `$APPIMAGE` env var (points to the `.AppImage` file itself) rather than `sys.executable + __file__` (which would be the temporary squashfs mount path).
- **About menu**: `self.menuBar().addMenu("About")` added in `__init__` before the central widget. Three actions: "GitHub Project" → `https://github.com/Tamalero/UpscalyVid`, "Find Models (OpenModelDB)" → `https://openmodeldb.info`, and "Open Models Folder" → `QDesktopServices.openUrl(QUrl.fromLocalFile(str(MODELS_DIR)))` (also calls `MODELS_DIR.mkdir()` first to ensure it exists). All use `QDesktopServices.openUrl`. Styled via `QMenuBar` / `QMenu` blocks in `STYLE` to match the dark Catppuccin theme.
- **Subprocess model**: upscaling runs as a `QProcess` calling `upscalyvid.py` with `--work-dir` pointing to a temp dir created by the GUI before launch. stdout is parsed via regexes for progress bar updates.
- **Work dir ownership**: the GUI creates `_work_dir = Path(tempfile.mkdtemp(...))` before starting the subprocess and passes it via `--work-dir`. On `_on_finished` it sets `_work_dir = None`. The preview timer reads `_work_dir / "upscaled"`.
- **Preview panel**: a `QTimer` fires every 5 seconds during upscaling. `_refresh_preview` opens the latest frame with PIL (`Image.open().convert("RGB")`), calls `.thumbnail((300, 170))`, then converts via `QImage(bytes, w, h, stride, Format_RGB888)` → `QPixmap.fromImage()`. Never use `QPixmap(path)` directly for upscaled frames — Qt silently returns a valid but all-black pixmap for large PNGs (>~10 MB uncompressed).
- **Download thread**: model pre-downloads run in `DownloadThread(QThread)` — UI stays responsive. Uses `urllib.request` with `User-Agent: UpscalyVid/1.0` (required by HuggingFace). Uses a `seen: set[str]` to deduplicate shared filenames (e.g. wdn alias). Skips models with empty URL. On failure emits a message pointing to `openmodeldb.info`.
- **Auto output path**: selecting an input auto-fills the output to `{stem}_upscaled.mp4`. Fires only when output field is empty.
- **Outscale + model description sync**: `_on_model_changed()` updates both the outscale spinner and the italic description label below the combo.
- **Quality section**: spinbox (0–51) + `Lossless (0)` button + `↺ Default (18)` button + always-visible hint line + live status label. Lossless tooltip clarifies NVENC CQ=0 is not strictly lossless.
- **Verbose checkbox**: passes `-v` to the subprocess. More output in the log — useful for debugging frame timing, VRAM, codec info.

### `_MODEL_TIPS` dict (gui.py)

Plain-English content guidance for each model key, shown as:
1. Tooltip on combo items (`Qt.ItemDataRole.ToolTipRole`)
2. Italic description label below the combo (updates on selection)

### Stylesheet variables (top of gui.py)

```python
DARK_BG, SURFACE, SURFACE2, BORDER   # backgrounds
ACCENT, ACCENT_HOVER, ACCENT_PRESS   # purple highlight (#7c6af7)
TEXT, TEXT_DIM                        # foreground
SUCCESS, WARNING, ERROR_COLOR         # status colours
LOG_BG                                # monospace log area
PREVIEW_BG                            # dark background for preview panel
```

Named button object IDs used in QSS: `primary`, `cancel`, `lossless`, `reset`.

## CLI reference

```
upscalyvid.py [input] [output] [options]
upscalyvid.py /path/to/folder/           # batch — processes all videos, outputs to folder/upscaled/
upscalyvid.py /path/to/folder/ /out/dir/ # batch with custom output directory

  -m, --model       realesrgan-x4plus (default) | realesrgan-x4plus-anime |
                    realesrgan-x2plus | realesrnet-x4plus |
                    realesr-general-x4v3 | realesr-general-wdn-x4v3 |
                    4x-UltraSharp | realesr-general-wdn-x4v3-denoise |
                    Custom-<stem>  (any .pth in models/ not in the registry)
  --outscale        final scale factor (default: model native scale)
  --codec           h264 | h265 | h264_nvenc | hevc_nvenc  (default: h264)
  -q, --quality     CRF (h264/h265) or CQ (nvenc) — 0=lossless, 18=default
  --tile            tile size px, 0=auto from free VRAM (default: 0)
  --tile-pad        tile overlap padding (default: 10)
  --frame-format    png (default, lossless) | jpg
  --work-dir PATH   custom temp dir for frames — used by GUI for preview (single-file only)
  --fp32            use FP32 instead of FP16 (HAT always uses FP32 regardless)
  --cpu             force CPU (very slow)
  --keep-frames     retain temp frame dirs after completion
  -v, --verbose     detailed logs: ffmpeg output, per-frame timing, VRAM; also saves
                    source_frame.png and sample_frame.png next to source video after frame 1
  --no-compile      disable torch.compile (faster startup, slower per-frame)
  --list-models     print model table and exit (input/output not required)
```

`input` and `output` are `nargs="?"` so `--list-models` and `--help` work without them.  
When `input` is a directory, `output` defaults to `{input}/upscaled/`. The model is loaded once and reused for all files in batch mode.

## Known constraints

- `basicsr` and `realesrgan` pip packages are intentionally **not used** — they break on PyTorch 2.x. Everything is reimplemented inline.
- `opencv-python` is intentionally **not used** — PIL handles all image I/O and resizing.
- The venv uses `--system-site-packages` — do not install a CPU-only torch into the venv, it would shadow the system CUDA torch.
- `setup.sh` is Bash. `launch.fish` is the primary launcher for the Fish shell environment.
- The `UPSCALED_DIR=...` line printed to stdout by the CLI is informational only; the GUI does not parse it — it already knows the path because it created `_work_dir` before launching the process.
- HuggingFace CDN requires a `User-Agent` header — bare `urllib.request.urlretrieve` returns HTTP 401. Always use `urllib.request.Request` with headers.
- HAT models load with `strict=False` because `attn_mask` buffers in the checkpoint are image-size-dependent and excluded from the model definition (recomputed at runtime).
- **HAT must run in FP32.** fp16 accumulates overflow across 6 residual groups → `LayerNorm` receives inf → NaN cascade → all-black frames. `build_upsampler` hard-forces `half=False` for HAT; do not attempt to add fp16 support without fixing the numerical stability first.
- `F.softmax` on CUDA can upcast fp16 input to fp32 — always cast back to the input dtype after softmax in fp16 attention layers. This applies to `ChanAttn`; `WinAttnHead` uses SDPA and no longer calls `F.softmax` directly.
- **`WinAttnHead` uses `F.scaled_dot_product_attention`.** The combined float `attn_mask` (RPE bias + shift mask) causes PyTorch to select the Memory-Efficient Attention backend, not Flash Attention. Do not switch back to manual `q @ k.T`; the memory savings are critical for large frames.
- **`_auto_tile` uses actual free VRAM** via `torch.cuda.mem_get_info()`. Do not revert to `total_memory × 0.65` — that caused OOM when other processes held VRAM. The 512 MB headroom is intentional.
- **HAT auto-tile formula is `3 × nW × ws⁴ × 4 B × 12`**. The 12× safety factor accounts for intermediate activations beyond the attn_mask (Q/K/V, DWConv, upsample). Do not remove the 12× — full-frame HAT inference OOMs on a 23 GB GPU for 720×944 input, calibrated to this. Do not restore the old `2 ×` multiplier inside the attn_mask term — that caused tile=128 instead of tile=256, producing rainbow boundary artifacts.
- **`_vram_per_pixel` must be set in `build_upsampler`** for any new non-HAT arch, otherwise `_auto_tile` will never tile for that arch (returns 0 from the `elif` branch).
- **`torch.compile` needs Triton for `mode="reduce-overhead"`.** Triton is not installed on this system. `_triton_available()` detects this and falls back to `backend="cudagraphs"`. The `TritonMissing` error surfaces at the first forward call, not at `torch.compile()` — a try/except around the compile call does not protect against it.
- **HAT must NOT be compiled.** `torch.compile(backend="cudagraphs")` records a kernel graph from the first tile's H/W. HAT tiles differ in H/W per tile (window-size padding depends on remainder), so the replayed graph produces wrong shift masks → rainbow color artifacts in all frames. Fix: `compile_model=compile_model and not is_hat` in `build_upsampler`. Do not re-enable compile for HAT without first making `_shift_mask` produce consistent shapes across tiles.
- **HAT dark-pixel artifact — bicubic blend fix.** HAT was trained on BGR images in [0,1] range WITHOUT mean normalization. For dark inputs, the model's transformer layers diverge: biases dominate the near-zero signal, LayerNorm normalises to std=1, each residual group compounds by ~2.5× → ±900K at layer 6 for pure black. The final LayerNorm rescues the range but wrong per-channel biases produce bright rainbow output (orange, teal, purple) for dark areas — visibly on black fabric like a swimsuit. **A simple linear blend from 0 to a threshold is not enough**: at brightness=60 a linear blend from 0–128 still mixes in 47% of the wildly-wrong model colour, which remains visibly rainbow. Fix uses a **two-stage blend** in `enhance()` when `window_size > 1`: `SOLID_THR=100` (hard cutoff — brightness ≤ 100 → 100% bicubic, zero model contribution), `DARK_THR=180` (brightness ≥ 180 → 100% model). Linear taper between 100 and 180. Blend weight is upscaled to 4× via `F.interpolate(mode='bilinear')` for smooth transitions at edges. **Do NOT add mean normalization to `HAT.forward()`** — the model was trained without it; any mean subtraction maps midtone inputs near zero and triggers the same divergence.
- **Old ESRGAN format** (`model.*` keys) is transparently handled by `_remap_esrgan_keys()` for registered models. Do not add explicit error handling for `model.*` keys in registered rrdbnet models — the remapper fixes it silently.
- **Old Torch format files** (first 4 bytes `b"v1.0"`) produce `invalid load key, 'v'` and cannot be loaded by modern PyTorch at all — `weights_only=False` also fails. `_safe_load_checkpoint` detects this and tells the user to delete and re-download. These are models from ~2021 downloaded from non-HuggingFace ESRGAN GUI tools.
- **Git LFS pointer files** produce `invalid load key, 'v'` when loaded with `torch.load`. If a cached `.pth` fails with this error, delete it and re-run. `_safe_load_checkpoint` detects the `b"version https://"` magic and prints an actionable message.
- **Never use `QPixmap(path)` for upscaled frames.** Qt silently returns a valid but all-black pixmap for large PNGs. Always go through PIL thumbnail → `QImage` → `QPixmap.fromImage()`.
- **`setDesktopFileName` requires a matching `.desktop` file** registered in `~/.local/share/applications/` — without it the freedesktop portal logs a registration error. `_ensure_desktop_file()` writes it once at startup.
- HAT without tiling on 4K exceeds 24 GB VRAM — `_auto_tile()` handles this automatically, but keep this in mind when reasoning about VRAM.
- **Model is unloaded from VRAM after upscaling.** In `main()`, after `upscale_frames()` returns, `upsampler.model = None` + `del upsampler` + `torch.cuda.empty_cache()` frees GPU memory before the ffmpeg re-encode step. Do not reference `upsampler` after this point.
- **`.safetensors` format is intentionally not supported.** Community HAT models are distributed as `.pth` files; `.safetensors` contains identical weights in a different container. Adding safetensors loading would require a new dependency (`safetensors` pip package) with no benefit to output quality or model behavior. If a user has a `.safetensors` file, they should convert it to `.pth` with `safetensors.torch.load_file()` + `torch.save()`.
- **Verbose mode saves two comparison frames next to the source video.** `upscale_frames()` saves `source_frame.png` (original resolution) and `sample_frame.png` (upscaled first frame) to `sample_dir` (= `input_path.parent` when verbose, `None` otherwise). The `sample_dir` parameter was added to `upscale_frames()` for this purpose; when `None` the frames are not saved.
- **Log file created on every run.** `main()` opens `_TeeLogger` (installs itself as `sys.stdout`) immediately after argument validation, writing `upscalyvid_YYYYMMDD_HHMMSS.log` in the source video's parent directory (or the batch input directory). The tee is closed in the outer `finally` block, so cancellation and crashes are captured too.
- **Batch mode.** When `input` is a directory, `main()` discovers all `_VIDEO_EXTS` files, builds a `file_pairs` list, downloads the model once, and loops over each file. Each file gets its own temp work dir (inner `try/finally` cleans it). `--work-dir` is ignored in batch mode.
- **`suggest_tile()` function.** Mirrors `_auto_tile()` logic for a reference 1080p frame using current free VRAM. Called in `main()` for the `[Device]` section tile hint, and imported by `gui.py` for the tile hint label. Returns 0 if full frame fits.
- **Stale temp cleanup.** `_cleanup_stale_temp()` called at the top of `main()` removes `upscalyvid_*` dirs in `tempfile.gettempdir()` older than 2 hours.
- **MODELS_DIR env override.** `MODELS_DIR` checks `os.environ.get("UPSCALYVID_MODELS_DIR", ...)` before falling back to `Path(__file__).parent / "models"`. The AppRun script sets this to `~/.local/share/UpscalyVid/models` so model downloads work from the read-only AppImage squashfs mount. Normal (non-AppImage) runs are unaffected.
- **AppImage.** Built with `build-appimage.sh` using `appimagetool`. Type 2 (squashfs/ELF). Update info embedded: `gh-releases-zsync|Tamalero|UpscalyVid|latest|UpscalyVid-*-x86_64.AppImage.zsync`. AppRun finds system Python with CUDA torch, creates venv in `~/.local/share/UpscalyVid/venv`, installs missing deps. GitHub Actions (`release.yml`) auto-builds on `v*` tag push.
- **GUI config persistence.** `~/.config/upscalyvid/config.json` stores `last_dir`. Loaded at startup in `MainWindow.__init__` (`self._config`). Written via `_save_config()` whenever a file or folder is selected. Both Browse and Folder… buttons use `config.get("last_dir")` as the QFileDialog start directory.
- **GUI tile hint label.** `self.tile_hint` QLabel sits next to `tile_spin`. Updated by `_update_tile_hint()` which calls `suggest_tile()` — connected to `model_combo.currentIndexChanged` and `chk_fp32.stateChanged`. `_update_tile_hint()` is called after `opts_box` is built (tile_hint must exist before the method runs).
- **GUI batch input.** `file_row(allow_dir=True)` adds a "Folder…" QToolButton that opens `QFileDialog.getExistingDirectory`. `_auto_output()` detects `Path(text).is_dir()` and suggests `{dir}/upscaled/`. `_validate()` checks for video files if input is a dir. `_build_cmd()` omits `--work-dir` when input is a directory (preview does not work for batch).
- **GUI closeEvent cleanup.** `closeEvent` calls `shutil.rmtree(self._work_dir, ignore_errors=True)` before `event.accept()` to remove any leftover temp dirs when the window is closed mid-upscale or after completion.
- **AppImage read-only filesystem — icons dir.** `gui.py` writes `arrow_down.svg` and `arrow_up.svg` at import time. The target directory `_ICONS_DIR` is derived from `_DATA_DIR = Path(os.environ.get("UPSCALYVID_DATA_DIR", Path(__file__).parent))`. AppRun exports `UPSCALYVID_DATA_DIR=$DATA_DIR` (`~/.local/share/UpscalyVid`), redirecting writes out of the squashfs. For source installs the env var is unset, so the icons go to `__file__.parent/icons/` as before. Do not write any other per-run files relative to `__file__` — always check for `UPSCALYVID_DATA_DIR` first.
- **AppImage read-only filesystem — desktop file Exec=.** `_ensure_desktop_file()` now uses `os.environ.get("APPIMAGE")` for the `Exec=` line when set (the AppImage runtime always sets `$APPIMAGE` to the path of the `.AppImage` file). Without this, `Exec=` would point to the temporary squashfs mount path (`/tmp/.mount_xxx/...`) which disappears after unmount. Source installs are unaffected (`$APPIMAGE` is unset).
- **About menu.** Added via `self.menuBar().addMenu("About")` in `MainWindow.__init__`, before the central widget setup. Contains three actions: "GitHub Project" → `https://github.com/Tamalero/UpscalyVid`, "Find Models (OpenModelDB)" → `https://openmodeldb.info`, and "Open Models Folder" → opens `MODELS_DIR` in the system file manager via `QDesktopServices.openUrl(QUrl.fromLocalFile(str(MODELS_DIR)))` (also calls `MODELS_DIR.mkdir(parents=True, exist_ok=True)` to ensure the folder exists before opening). All three use `QDesktopServices.openUrl`. The status bar shows the full models path on hover for the "Open Models Folder" action (`act_open_models_dir.setStatusTip(str(MODELS_DIR))`). The `STYLE` string includes `QMenuBar` and `QMenu` rules matching the dark Catppuccin theme. Imports added: `QUrl` from `QtCore`, `QDesktopServices` from `QtGui`, `QMenu`/`QMenuBar` from `QtWidgets`.
