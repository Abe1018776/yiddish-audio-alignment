# RunPod Serverless — Yiddish Alignment & Transcription

GPU-accelerated serverless endpoint for Yiddish audio alignment and transcription.
Model `ivrit-ai/yi-whisper-large-v3-turbo-ct2` is baked into the Docker image.

## Build & Push

```bash
docker build -t your-registry/yiddish-alignment:latest .
docker push your-registry/yiddish-alignment:latest
```

## Deploy to RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Create new endpoint → Custom container
3. Image: `your-registry/yiddish-alignment:latest`
4. GPU: 16GB+ (e.g. A4000, L4, A100)
5. Min workers: 0, Max workers: 3
6. Container disk: 20GB+

## API Usage

### Transcribe
```json
{
  "input": {
    "mode": "transcribe",
    "audio_url": "https://example.com/audio.mp3",
    "language": "yi",
    "word_timestamps": true
  }
}
```

### Align text to audio
```json
{
  "input": {
    "mode": "align",
    "audio_url": "https://example.com/audio.mp3",
    "text": "דער טעקסט פון דער רעדע...",
    "language": "yi"
  }
}
```

### Call via RunPod API
```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"mode": "transcribe", "audio_url": "...", "language": "yi"}}'
```
