"""RunPod serverless handler for Yiddish audio alignment + transcription.

Supports two modes:
  - "transcribe": Transcribe audio with word-level timestamps
  - "align": Align existing text to audio with word-level timestamps

Model ivrit-ai/yi-whisper-large-v3-turbo-ct2 is baked into the Docker image.
"""
import dataclasses
import json
import logging
import os
import tempfile
import time
import urllib.request
from pathlib import Path

import runpod
import stable_whisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default model when the request doesn't specify one. Reads MODEL_NAME from
# the endpoint's env vars so the active model can be swapped without rebuilding
# the image (set via RunPod GraphQL when an admin "Promotes" a TrainingRun).
DEFAULT_MODEL_NAME = os.environ.get(
    "MODEL_NAME", "ivrit-ai/yi-whisper-large-v3-turbo-ct2"
)

# Single-slot model cache — loaded once per name, swapped when a request asks
# for a different one. Mirrors ivrit-ai's transcribe_core() pattern (see
# reference/ivrit_infer_runpod.py): we don't need a multi-model LRU because
# any given worker rarely sees more than one model in use at a time, and the
# memory footprint of two large CT2 models is too large to keep both resident.
_current_model_name: str | None = None
_current_model = None


def get_model(model_name: str):
    """Return a loaded model for `model_name`, swapping if a different one is requested."""
    global _current_model, _current_model_name
    if _current_model_name == model_name and _current_model is not None:
        return _current_model

    logger.info(f"Loading model: {model_name}")
    start = time.time()
    _current_model = stable_whisper.load_faster_whisper(
        model_name, device="cuda", compute_type="int8"
    )
    _current_model_name = model_name
    logger.info(f"Model loaded in {time.time() - start:.1f}s")
    return _current_model


def download_audio(url: str) -> str:
    """Download audio from URL to temp file."""
    suffix = Path(url.split("?")[0]).suffix or ".mp3"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    req = urllib.request.Request(url, headers={"User-Agent": "YiddishAligner/1.0"})
    with urllib.request.urlopen(req) as response, open(tmp.name, "wb") as out:
        out.write(response.read())
    return tmp.name


def handler(job):
    """RunPod serverless handler.

    Input schema:
    {
        "mode": "transcribe" | "align",
        "audio_url": "https://...",          # URL to audio file
        "audio_base64": "...",               # OR base64-encoded audio
        "text": "transcript text...",         # Required for "align" mode
        "language": "yi",                    # Default: "yi" (Yiddish)
        "word_timestamps": true,             # Default: true
        "model_name": "user/yi-v3-ct2"       # Optional: HF repo to load.
                                             # Falls back to MODEL_NAME env var.
    }

    Output:
    {
        "full_text": "...",
        "segments": [...],
        "timestamps": [
            {"start": 0.0, "end": 0.5, "text": "word", "confidence": 0.92, "type": "word"}
        ],
        "duration_seconds": 123.4,
        "model": "ivrit-ai/yi-whisper-large-v3-turbo-ct2",
        "language": "yi",
        "mode": "transcribe|align"
    }
    """
    input_data = job["input"]
    mode = input_data.get("mode", "transcribe")
    language = input_data.get("language", "yi")
    word_timestamps = input_data.get("word_timestamps", True)
    # Per-request model override (Yiddish Cleaner sends this when an admin
    # picks a published TrainingRun in the transcribe UI). Falls back to the
    # endpoint's MODEL_NAME env var when the caller doesn't specify one.
    model_name = input_data.get("model_name") or DEFAULT_MODEL_NAME

    # Get audio
    audio_path = None
    try:
        if "audio_url" in input_data:
            audio_path = download_audio(input_data["audio_url"])
        elif "audio_base64" in input_data:
            import base64
            suffix = input_data.get("audio_format", ".mp3")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(base64.b64decode(input_data["audio_base64"]))
            tmp.close()
            audio_path = tmp.name
        else:
            return {"error": "Provide 'audio_url' or 'audio_base64'"}

        model = get_model(model_name)

        if mode == "align":
            text = input_data.get("text")
            if not text:
                return {"error": "'text' is required for align mode"}

            logger.info(f"Aligning {len(text)} chars to audio...")
            result = model.align(
                audio_path,
                text,
                language=language,
            )
        else:
            # Transcribe mode
            logger.info(f"Transcribing audio...")
            result = model.transcribe(
                audio_path,
                language=language,
                word_timestamps=word_timestamps,
                vad=True,
                regroup=True,
            )

        return format_result(result, mode, language, model_name)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        if audio_path and os.path.exists(audio_path):
            os.unlink(audio_path)


def format_result(result, mode, language, model_name):
    """Convert stable_whisper result to standardized output format."""
    segments = []
    timestamps = []

    for seg in result.segments:
        seg_data = {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": [],
        }

        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                word_data = {
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "confidence": round(w.probability, 4) if hasattr(w, "probability") else None,
                }
                seg_data["words"].append(word_data)
                timestamps.append({
                    "start": word_data["start"],
                    "end": word_data["end"],
                    "text": word_data["word"],
                    "confidence": word_data["confidence"],
                    "type": "word",
                })
        else:
            timestamps.append({
                "start": seg_data["start"],
                "end": seg_data["end"],
                "text": seg_data["text"],
                "confidence": None,
                "type": "segment",
            })

        segments.append(seg_data)

    return {
        "full_text": result.text.strip() if hasattr(result, "text") else "",
        "segments": segments,
        "timestamps": timestamps,
        "model": model_name,
        "language": language,
        "mode": mode,
    }


runpod.serverless.start({"handler": handler})
