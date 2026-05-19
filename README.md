# UpscalyVid

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Platform](https://img.shields.io/badge/platform-Linux%20x86__64-lightgrey) ![License](https://img.shields.io/badge/license-MIT-green)

AI-powered video upscaler using Real-ESRGAN and HAT (Hybrid Attention Transformer) models. Self-contained PyTorch implementation — no `basicsr` or `opencv` dependency. Includes a PyQt6 GUI and a CLI.

---

## Requirements

- Linux x86_64
- Python 3.10+
- NVIDIA GPU + CUDA + PyTorch 2.x (system-installed) — CPU fallback is available but very slow
- ffmpeg (frame extraction and encoding)
- PyQt6 6.x (for the GUI)

---

## Installation

### Option 1 — AppImage (easiest)

Download `UpscalyVid-v1.0.1-x86_64.AppImage` from the [latest release](https://github.com/Tamalero/UpscalyVid/releases/latest):

```bash
chmod +x UpscalyVid-v1.0.1-x86_64.AppImage
./UpscalyVid-v1.0.1-x86_64.AppImage
```

First launch automatically sets up a Python venv. GPU acceleration requires a system CUDA + PyTorch installation.

**Auto-update:**

```bash
AppImageUpdate UpscalyVid-*.AppImage
```

### Option 2 — From source (Linux with system PyTorch + CUDA)

```bash
git clone https://github.com/Tamalero/UpscalyVid.git
cd UpscalyVid

# Fish shell (recommended — handles venv automatically)
./launch.fish

# Or manually with any shell
python3 -m venv --system-site-packages .venv
.venv/bin/pip install Pillow PyQt6 numpy
.venv/bin/python gui.py
```

The venv is created with `--system-site-packages` so it inherits the system CUDA-enabled PyTorch without re-downloading the multi-GB wheel.

### Option 3 — Pre-download all model weights

```bash
bash setup.sh --download-all
```

Models are otherwise downloaded automatically on first use.

---

## Quick start

### GUI

```bash
./launch.fish
# or: .venv/bin/python gui.py
```

### CLI

```bash
python3 upscalyvid.py input.mp4 output.mp4
python3 upscalyvid.py input.mp4 output.mp4 -m 4x-UltraSharp -v
python3 upscalyvid.py /path/to/folder/        # batch — outputs to folder/upscaled/
python3 upscalyvid.py --list-models
```

---

## CLI reference

```
python3 upscalyvid.py [input] [output] [options]
python3 upscalyvid.py /path/to/folder/          # batch — outputs to folder/upscaled/

Options:
  -m, --model MODEL     Model name (default: realesrgan-x4plus)
  --outscale FLOAT      Final output scale (default: model native)
  --codec CODEC         h264 | h265 | h264_nvenc | hevc_nvenc (default: h264)
  -q, --quality INT     CRF/CQ 0–51 (0=lossless, 18=default)
  --tile INT            Tile size px, 0=auto from free VRAM (default: 0)
  --tile-pad INT        Tile overlap padding (default: 10)
  --frame-format FMT    png (default, lossless) | jpg
  --fp32                Use FP32 instead of FP16
  --cpu                 Force CPU (very slow)
  --keep-frames         Retain temp frame dirs
  -v, --verbose         Per-frame timing, VRAM, saves source_frame.png + sample_frame.png next to source video
  --list-models         Print model table and exit
  --no-compile          Disable torch.compile
```

---

## GUI overview

| Control | Description |
|---|---|
| Input | File picker + "Folder..." button for batch processing |
| Output | Auto-filled as `{stem}_upscaled.mp4` (or `{folder}/upscaled/` for batch) |
| Model | Combo with tooltip and description for each model |
| Output scale | 1–8x (defaults to model native scale) |
| Codec | H.264/H.265 software or NVENC hardware |
| Quality | 0–51 CRF/CQ — 0 = lossless, 18 = default near-lossless |
| Tile size | 0 = auto (tile size suggestion shown from free VRAM) |
| FP32 / CPU / Keep frames / JPG frames / Verbose | Misc options |
| Preview | Live frame preview updated every 5 seconds during upscaling |
| Pre-download | Downloads all registered model weights up front |
| Log | Full output log panel |

---

## Supported models

Models are downloaded automatically to `models/` on first use.

| Key | Arch | Scale | Best for |
|---|---|---|---|
| `realesrgan-x4plus` | RRDBNet | 4x | General photos and video (default) |
| `realesrgan-x4plus-anime` | RRDBNet | 4x | Anime, cartoons, flat-colour art |
| `realesrgan-x2plus` | RRDBNet | 2x | HD→2K or 2K→4K |
| `realesrnet-x4plus` | RRDBNet | 4x | Softer output, less sharpening |
| `realesr-general-x4v3` | SRVGGNet | 4x | Fast, good for video |
| `realesr-general-wdn-x4v3` | SRVGGNet | 4x | Noisy or compressed footage |
| `realesr-general-wdn-x4v3-denoise` | SRVGGNet | 4x | Degraded/VHS/film |
| `4x-UltraSharp` | RRDBNet | 4x | Fur, feathers, fine textures |

### Custom models

Drop any `.pth` file into the `models/` directory. It is automatically registered as `Custom-<stem>` and the architecture (RRDBNet, SRVGGNet, or HAT) is detected from the weights. HAT models like `4x-UltraSharpV2.pth` work automatically — no configuration needed.

---

## Logging

Every run writes a log file (`upscalyvid_YYYYMMDD_HHMMSS.log`) to the source video's directory, capturing all output including cancellations.

**Verbose mode** (`-v` / `--verbose`) additionally saves two comparison images next to the source video after the first frame is processed:

- `source_frame.png` — original resolution raw frame
- `sample_frame.png` — upscaled result for that same frame

---

## Building the AppImage

```bash
./build-appimage.sh v1.0.1
# Produces: UpscalyVid-v1.0.1-x86_64.AppImage + .zsync + .sha256
```

GitHub Actions automatically builds the AppImage and creates a release whenever a `v*` tag is pushed.

---

## Technical notes

- **HAT models always use FP32.** FP16 accumulates overflow across residual groups, causing NaN cascades and all-black frames. This is forced automatically — the `--fp32` flag has no effect on HAT.
- **Auto-tile uses actual free VRAM** via `torch.cuda.mem_get_info()`, not a fixed fraction of total VRAM. This avoids OOM when other processes are using the GPU.
- **torch.compile** uses the `cudagraphs` backend — Triton is not required.
- **Memory-efficient attention (SDPA)** is used in the HAT window attention layers, avoiding materialisation of the full attention score tensor and preventing OOM on large frames.

---

## License

MIT
