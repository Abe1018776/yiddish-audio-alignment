# How We Built the Yiddish Audio Alignment Feature

A step-by-step walkthrough of how this feature was built from scratch — from research to a live production endpoint.

---

## The Problem

We needed a way to take Yiddish audio recordings and their corresponding text transcripts, and produce **word-level timestamps** — knowing exactly when each word starts and ends in the audio. This is critical for building synchronized audio-text experiences (karaoke-style playback, transcript navigation, quality verification).

## Step 1: Research — How ivrit-ai Does It

We started by analyzing the alignment code from [ivrit-ai/ivrit.ai](https://github.com/ivrit-ai/ivrit.ai/tree/master/alignment), which handles Hebrew audio alignment. Their approach uses:

- **stable-ts** (stable_whisper) — a library that improves Whisper's timestamp accuracy
- **faster-whisper** — a CTranslate2-based Whisper implementation (4x faster than OpenAI's)
- A **confusion-aware alignment algorithm** that detects when the model gets confused and retries

### The Confusion-Aware Algorithm

The key insight from ivrit-ai is that Whisper alignment sometimes produces garbage — the model gets "confused" on certain audio segments. Their algorithm:

1. Aligns the full audio to the text
2. Uses a sliding window to monitor word confidence scores
3. When the ratio of low-confidence to high-confidence words exceeds a threshold (0.8), it marks a "confusion zone"
4. Re-attempts alignment from a high-confidence point before the confusion
5. If retry still fails, skips that section and moves on
6. Stitches all the good segments together

This produces much more reliable alignments than naive Whisper alignment.

### Key Files We Referenced

- `ivrit_align.py` — Main alignment loop with confusion detection
- `ivrit_breakable_aligner.py` — Custom aligner that can terminate early when quality drops
- `ivrit_seekable_audio_loader.py` — Efficient audio seeking using ffmpeg
- `ivrit_utils.py` — Confidence ratio calculation, confusion zone detection

## Step 2: Adapt for Yiddish

We adapted the ivrit-ai code with two key changes:

1. **Model**: Changed from Hebrew Whisper to `ivrit-ai/yi-whisper-large-v3-turbo-ct2` — a fine-tuned Yiddish Whisper model
2. **Language**: Set language to `"yi"` (Yiddish) instead of `"he"` (Hebrew). This is critical — the model's language detection is unreliable, so the language must be explicitly specified

### Files Created

```
alignment/
├── __init__.py                  # Package init
├── align.py                     # Main alignment function (confusion-aware)
├── breakable_aligner.py         # Aligner that stops on quality drops
├── seekable_audio_loader.py     # Fast audio seeking via ffmpeg -ss
└── utils.py                     # Confidence ratios, confusion detection
```

### Core Function

```python
from alignment.align import align_transcript_to_audio

result = align_transcript_to_audio(
    audio_file="recording.mp3",
    transcript="דער ייִדישער טעקסט...",
    model=model,           # yi-whisper model
    language="yi",         # Must be explicit
)
# result.segments[0].words[0].start → 0.42
# result.segments[0].words[0].end   → 0.87
# result.segments[0].words[0].word  → "דער"
```

## Step 3: Flask Blueprint (Local Endpoint)

We created a Flask Blueprint (`blueprints/alignment_bp.py`) that exposes the alignment as an HTTP API:

- `POST /api/alignment/align` — Upload audio + text, get word timestamps
- `GET /api/alignment/health` — Health check

This worked locally but was extremely slow without a GPU (Whisper on CPU takes minutes per audio file).

## Step 4: RunPod Serverless (GPU in the Cloud)

To solve the speed problem, we deployed the model to RunPod serverless:

### Docker Image

Built a Docker image (`runpod/Dockerfile`) based on PyTorch with CUDA:

```dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
# Install dependencies
RUN pip3 install runpod stable-ts faster-whisper torch ...
# BAKE the model into the image (no runtime download!)
RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/yi-whisper-large-v3-turbo-ct2")'
```

**Key decision**: The ~1.5GB Whisper model is downloaded and cached at Docker build time. This means:
- No model download delay on cold starts
- Every worker starts with the model ready
- Consistent, reproducible deployments

### RunPod Handler

The handler (`runpod/handler.py`) supports two modes:

1. **Transcribe** — Audio in, text + timestamps out
2. **Align** — Audio + text in, word-level timestamps out

Audio can be provided as a URL or base64-encoded data.

### RunPod Client

The Python client (`runpod_client.py`) handles:
- Submitting jobs via `runsync` (blocking, up to ~30s)
- Polling for long-running jobs
- Health checks

### Deployment

```
Image: ghcr.io/abe1018776/yiddish-alignment:latest
Endpoint ID: 7u8848gpu4vjvf
GPU: AMPERE_16 (16GB VRAM)
Workers: 0 min, 1 max (scales to zero when idle)
Idle timeout: 10 seconds
```

### Performance

| Metric | Value |
|--------|-------|
| Cold start (image pull + model load) | ~2.5 minutes |
| Warm alignment (short audio) | 0.5-1 second |
| Warm transcription (short audio) | 7-8 seconds |
| Model load (first request on warm worker) | ~6 seconds |

## Step 5: Updated Flask Blueprint

Updated `blueprints/alignment_bp.py` to support dual backends:

- **RunPod mode** (when `RUNPOD_ENDPOINT_ID` + `RUNPOD_API_KEY` are set) — proxies to GPU
- **Local mode** (fallback) — runs model directly (slow without GPU)

Also added a `/api/alignment/transcribe` endpoint (RunPod only).

## Step 6: Cloudflare Pages UI

Built a web interface deployed to `align.kohnai.ai`:

### Architecture

```
Browser (align.kohnai.ai)
    ↓ POST /api/align (audio as base64 + text)
Cloudflare Pages Function (site/functions/api/align.js)
    ↓ POST runsync (proxied, API key stays server-side)
RunPod Serverless (GPU worker)
    ↓ stable-ts alignment with yi-whisper
    ↓ Returns word-level timestamps
Cloudflare Pages Function
    ↓ Returns JSON to browser
Browser renders interactive results
```

### UI Features

- Drag-and-drop audio upload
- RTL Yiddish text input
- Mode toggle: Align (audio + text) or Transcribe (audio only)
- **Karaoke-style playback**: Click any word to start playing from there, words highlight as they're spoken
- Confidence color coding (green = high, yellow = mid, red = low)
- Segments table view
- Raw JSON view with download

### Deployment Stack

| Component | Service | URL |
|-----------|---------|-----|
| Static HTML/JS | Cloudflare Pages | align.kohnai.ai |
| API Proxy | Cloudflare Pages Functions | align.kohnai.ai/api/align |
| GPU Inference | RunPod Serverless | api.runpod.ai/v2/7u8848gpu4vjvf |
| Docker Image | GitHub Container Registry | ghcr.io/abe1018776/yiddish-alignment:latest |
| Model | HuggingFace | ivrit-ai/yi-whisper-large-v3-turbo-ct2 |

## Step 7: Testing

- **Unit tests** (12 passing): Confidence ratios, confusion detection, result formatting
- **Endpoint validation**: Health check, missing file/text error handling
- **E2E through Cloudflare**: base64 audio → Pages Function → RunPod → word timestamps
- **Live test with Yiddish text**: Aligned "דאָס איז אַ טעסט" to audio with per-word timestamps

---

## File Map

```
clean-yiddish-transcripts/
├── alignment/                    # Core alignment library
│   ├── __init__.py
│   ├── align.py                  # Confusion-aware alignment algorithm
│   ├── breakable_aligner.py      # Quality-monitoring aligner
│   ├── seekable_audio_loader.py  # Fast audio seeking
│   └── utils.py                  # Helpers (confidence, confusion zones)
├── blueprints/
│   └── alignment_bp.py           # Flask endpoints (RunPod + local fallback)
├── runpod/
│   ├── Dockerfile                # GPU image with baked-in model
│   ├── handler.py                # RunPod serverless handler
│   └── README.md                 # Deployment instructions
├── runpod_client.py              # Python client for RunPod endpoint
├── site/
│   ├── public/
│   │   └── index.html            # Web UI (align.kohnai.ai)
│   └── functions/
│       └── api/
│           └── align.js          # Cloudflare Pages Function (API proxy)
├── tests/
│   └── test_alignment.py         # Unit tests (12 tests)
├── reference/                    # Read-only ivrit-ai source for reference
└── docs/
    ├── ALIGNMENT_BUILD_GUIDE.md  # This file
    └── ALIGNMENT_API.md          # Direct API access docs
```
