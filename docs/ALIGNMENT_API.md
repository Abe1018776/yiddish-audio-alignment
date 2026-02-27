# Yiddish Audio Alignment API

Direct API access for aligning Yiddish audio to text or transcribing audio — no UI needed.

---

## Endpoints

There are three ways to access the alignment service:

| Method | URL | Auth Required |
|--------|-----|---------------|
| **Cloudflare Proxy** (recommended) | `https://align.kohnai.ai/api/align` | No |
| **RunPod Direct** | `https://api.runpod.ai/v2/7u8848gpu4vjvf/runsync` | Yes (API key) |
| **Flask Local** | `http://localhost:5050/api/alignment/align` | No |

---

## 1. Cloudflare Proxy (Recommended)

No API key needed — the proxy handles authentication server-side.

### Align Text to Audio

Send audio + text, get word-level timestamps.

```bash
curl -X POST https://align.kohnai.ai/api/align \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "align",
    "audio_base64": "<base64-encoded-audio>",
    "audio_format": ".mp3",
    "text": "דער טעקסט פון דער רעדע",
    "language": "yi"
  }'
```

### Transcribe Audio

Send audio only, get transcription with word timestamps.

```bash
curl -X POST https://align.kohnai.ai/api/align \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "transcribe",
    "audio_base64": "<base64-encoded-audio>",
    "audio_format": ".mp3",
    "language": "yi"
  }'
```

### Using Audio URL Instead of Base64

If your audio is already hosted publicly:

```bash
curl -X POST https://align.kohnai.ai/api/align \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "align",
    "audio_url": "https://example.com/audio.mp3",
    "text": "דער טעקסט פון דער רעדע",
    "language": "yi"
  }'
```

### Quick Test with Base64

```bash
# Encode an audio file to base64
AUDIO_B64=$(base64 -w0 my_audio.mp3)

# Align
curl -X POST https://align.kohnai.ai/api/align \
  -H "Content-Type: application/json" \
  -d "{
    \"mode\": \"align\",
    \"audio_base64\": \"$AUDIO_B64\",
    \"audio_format\": \".mp3\",
    \"text\": \"דאָס איז אַ טעסט\",
    \"language\": \"yi\"
  }"
```

---

## 2. RunPod Direct

For higher control or if you want to bypass the Cloudflare proxy.

**API Key**: Required in `Authorization` header.

### Align

```bash
curl -X POST "https://api.runpod.ai/v2/7u8848gpu4vjvf/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "align",
      "audio_url": "https://example.com/audio.mp3",
      "text": "דער טעקסט פון דער רעדע",
      "language": "yi"
    }
  }'
```

### Transcribe

```bash
curl -X POST "https://api.runpod.ai/v2/7u8848gpu4vjvf/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "transcribe",
      "audio_url": "https://example.com/audio.mp3",
      "language": "yi",
      "word_timestamps": true
    }
  }'
```

### Async (Long Audio)

For audio longer than ~30 seconds, use the async endpoint:

```bash
# Submit job
curl -X POST "https://api.runpod.ai/v2/7u8848gpu4vjvf/run" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "mode": "align",
      "audio_url": "https://example.com/long_audio.mp3",
      "text": "...",
      "language": "yi"
    }
  }'
# Returns: {"id": "job-abc123", "status": "IN_QUEUE"}

# Poll for result
curl "https://api.runpod.ai/v2/7u8848gpu4vjvf/status/job-abc123" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
# Returns: {"status": "COMPLETED", "output": {...}}
```

### Health Check

```bash
curl "https://api.runpod.ai/v2/7u8848gpu4vjvf/health" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY"
```

---

## 3. Python Client

```python
from runpod_client import RunPodAlignmentClient

client = RunPodAlignmentClient(
    api_key="YOUR_RUNPOD_API_KEY",
    endpoint_id="7u8848gpu4vjvf"
)

# Align
result = client.align(
    audio_url="https://example.com/audio.mp3",
    text="דער טעקסט פון דער רעדע",
    language="yi"
)

# Transcribe
result = client.transcribe(
    audio_url="https://example.com/audio.mp3",
    language="yi"
)

# Health
health = client.health()
```

Or using environment variables:

```bash
export RUNPOD_API_KEY=your_key_here
export RUNPOD_ENDPOINT_ID=7u8848gpu4vjvf
```

```python
client = RunPodAlignmentClient()  # reads from env
```

---

## Request Schema

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `mode` | string | No | `"transcribe"` | `"align"` or `"transcribe"` |
| `audio_url` | string | One of audio_url/audio_base64 | — | Public URL to audio file |
| `audio_base64` | string | One of audio_url/audio_base64 | — | Base64-encoded audio data |
| `audio_format` | string | No | `".mp3"` | File extension (`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`) |
| `text` | string | Yes for align mode | — | Yiddish text to align to audio |
| `language` | string | No | `"yi"` | Language code (always use `"yi"` for Yiddish) |
| `word_timestamps` | bool | No | `true` | Include word-level timestamps (transcribe mode) |

---

## Response Schema

```json
{
  "full_text": "דאָס איז אַ טעסט",
  "segments": [
    {
      "start": 0.0,
      "end": 2.98,
      "text": "דאָס איז אַ טעסט",
      "words": [
        {
          "word": "דאָס",
          "start": 0.0,
          "end": 0.18,
          "confidence": 0.85
        },
        {
          "word": "איז",
          "start": 0.18,
          "end": 0.56,
          "confidence": 0.92
        },
        {
          "word": "אַ",
          "start": 0.56,
          "end": 1.62,
          "confidence": 0.78
        },
        {
          "word": "טעסט",
          "start": 1.62,
          "end": 2.98,
          "confidence": 0.88
        }
      ]
    }
  ],
  "timestamps": [
    {
      "start": 0.0,
      "end": 0.18,
      "text": "דאָס",
      "confidence": 0.85,
      "type": "word"
    },
    {
      "start": 0.18,
      "end": 0.56,
      "text": "איז",
      "confidence": 0.92,
      "type": "word"
    }
  ],
  "model": "ivrit-ai/yi-whisper-large-v3-turbo-ct2",
  "language": "yi",
  "mode": "align"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `full_text` | string | Complete transcribed/aligned text |
| `segments` | array | Audio segments with nested word timestamps |
| `segments[].start` | float | Segment start time in seconds |
| `segments[].end` | float | Segment end time in seconds |
| `segments[].text` | string | Segment text |
| `segments[].words` | array | Word-level timestamps within the segment |
| `segments[].words[].word` | string | The word |
| `segments[].words[].start` | float | Word start time in seconds |
| `segments[].words[].end` | float | Word end time in seconds |
| `segments[].words[].confidence` | float\|null | Model confidence (0.0-1.0) |
| `timestamps` | array | Flat list of all word timestamps (convenience format) |
| `timestamps[].type` | string | `"word"` or `"segment"` (segment if no word-level data) |
| `model` | string | Model used for inference |
| `language` | string | Language code used |
| `mode` | string | `"align"` or `"transcribe"` |

---

## Error Responses

```json
{"error": "Provide 'audio_base64' or 'audio_url'"}          // 400
{"error": "'text' is required for align mode"}               // 400
{"error": "RunPod job failed"}                               // 502
{"error": "Server misconfigured: missing RunPod credentials"} // 500
```

---

## Notes

- **Language must be `"yi"`**: The model's automatic language detection is unreliable. Always pass `language: "yi"` explicitly.
- **Cold starts**: If the RunPod worker has been idle for >10 seconds, it scales to zero. First request after idle takes ~2.5 minutes (image pull + model load). Subsequent requests are sub-second.
- **Audio size**: Base64 encoding increases size by ~33%. For large files, prefer `audio_url` with a publicly accessible link.
- **Supported formats**: MP3, WAV, M4A, OGG, FLAC (anything ffmpeg can decode).
- **Max audio length**: No hard limit, but very long audio (>1hr) may hit the 5-minute RunPod timeout. Use the async `/run` endpoint for long audio.
- **Confidence scores**: Range 0.0-1.0. Above 0.5 is generally good. Low confidence may indicate misalignment or unclear audio.
