# Yiddish Audio Alignment API

Align Yiddish audio to transcript text and get word-level timestamps.

**Endpoint:** `POST https://align.kohnai.ai/api/align`

---

## Request

**Content-Type:** `application/json`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `mode` | String | Yes | — | Must be `"align"` |
| `audio_url` | String | One of* | — | Public URL to audio file |
| `audio_base64` | String | One of* | — | Base64-encoded audio data |
| `audio_format` | String | No | `".wav"` | File extension: `".mp3"`, `".wav"`, `".m4a"`, `".ogg"`, `".flac"` |
| `text` | String | Yes | — | Transcript text to align against |
| `language` | String | No | `"yi"` | Language code |

*Provide either `audio_url` or `audio_base64`, not both.

### Example — Audio URL

```bash
curl -X POST https://align.kohnai.ai/api/align \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "align",
    "audio_url": "https://example.com/recording.mp3",
    "text": "דער רבי האט געזאגט",
    "language": "yi"
  }'
```

### Example — Base64 Audio

```bash
curl -X POST https://align.kohnai.ai/api/align \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "align",
    "audio_base64": "<base64-encoded-audio>",
    "audio_format": ".mp3",
    "text": "דער רבי האט געזאגט",
    "language": "yi"
  }'
```

---

## Response (200)

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 2.98,
      "text": "דער רבי האט געזאגט",
      "words": [
        { "word": "דער", "start": 0.0, "end": 0.18, "confidence": 0.92 },
        { "word": "רבי", "start": 0.20, "end": 0.55, "confidence": 0.88 },
        { "word": "האט", "start": 0.58, "end": 0.80, "confidence": 0.95 },
        { "word": "געזאגט", "start": 0.82, "end": 1.20, "confidence": 0.91 }
      ]
    }
  ],
  "timestamps": [
    { "start": 0.0, "end": 0.18, "text": "דער", "confidence": 0.92, "type": "word" },
    { "start": 0.20, "end": 0.55, "text": "רבי", "confidence": 0.88, "type": "word" },
    { "start": 0.58, "end": 0.80, "text": "האט", "confidence": 0.95, "type": "word" },
    { "start": 0.82, "end": 1.20, "text": "געזאגט", "confidence": 0.91, "type": "word" }
  ],
  "full_text": "דער רבי האט געזאגט"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `segments` | Array | Phrase-level groups with nested word timestamps |
| `segments[].start` | Float | Segment start time (seconds) |
| `segments[].end` | Float | Segment end time (seconds) |
| `segments[].text` | String | Segment text |
| `segments[].words` | Array | Word-level timestamps within the segment |
| `timestamps` | Array | Flat list of every word timestamp |
| `timestamps[].start` | Float | Word start time (seconds) |
| `timestamps[].end` | Float | Word end time (seconds) |
| `timestamps[].text` | String | The word |
| `timestamps[].confidence` | Float | Confidence score (0–1) |
| `full_text` | String | Complete aligned text |

---

## Errors

| Status | Cause |
|--------|-------|
| 400 | Missing `text`, missing audio, or invalid JSON |
| 500 | Server misconfigured (missing RunPod env vars) |
| 502 | RunPod backend failure |

```json
{ "error": "Description of what went wrong" }
```
