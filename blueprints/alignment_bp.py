"""Flask Blueprint for Yiddish audio-text alignment endpoint.

Provides word-level timestamp alignment using the ivrit-ai confusion-aware
algorithm with the yi-whisper-large-v3 Yiddish model.
"""
import json
import logging
import os
import tempfile
from pathlib import Path

from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

alignment_bp = Blueprint("alignment", __name__, url_prefix="/api/alignment")

# Lazy-loaded model cache
_cached_model = None


def _get_model():
    """Lazy-load the alignment model."""
    global _cached_model
    if _cached_model is None:
        from alignment.utils import get_breakable_align_model

        model_name = os.getenv("ALIGNMENT_MODEL", "ivrit-ai/yi-whisper-large-v3-turbo-ct2")
        device = os.getenv("ALIGNMENT_DEVICE", "auto")
        compute_type = os.getenv("ALIGNMENT_COMPUTE_TYPE", "int8")

        logger.info(f"Loading alignment model: {model_name} on {device}")
        _cached_model = get_breakable_align_model(model_name, device, compute_type)

    return _cached_model


def _format_result(result):
    """Convert WhisperResult to JSON-serializable dict with word timestamps."""
    segments = []
    for seg in result.segments:
        words = []
        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                words.append({
                    "word": w.word.strip() if hasattr(w, "word") else str(w.text).strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "confidence": round(w.probability, 4) if hasattr(w, "probability") else None,
                })
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": words,
        })

    # Flatten to shared timestamp format
    timestamps = []
    for seg in segments:
        if seg["words"]:
            for w in seg["words"]:
                timestamps.append({
                    "start": w["start"],
                    "end": w["end"],
                    "text": w["word"],
                    "confidence": w["confidence"],
                    "type": "word",
                })
        else:
            timestamps.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "confidence": None,
                "type": "segment",
            })

    return {
        "segments": segments,
        "timestamps": timestamps,
        "full_text": result.text.strip() if hasattr(result, "text") else "",
    }


@alignment_bp.route("/align", methods=["POST"])
def align_audio():
    """Align audio to text with word-level timestamps.

    Accepts multipart form data:
        - audio: (required) Audio file (MP3, WAV, M4A, OGG, FLAC)
        - text: (required) Transcript text to align
        - language: (optional) Language code, default "yi"
        - entry_id: (optional) ID for logging

    Returns JSON with segments, word timestamps, and confidence scores.
    """
    if "audio" not in request.files:
        return jsonify({"error": "Missing 'audio' file"}), 400

    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    language = request.form.get("language", "yi")
    entry_id = request.form.get("entry_id", None)

    audio_file = request.files["audio"]
    suffix = Path(audio_file.filename).suffix or ".wav"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp)
            tmp_path = Path(tmp.name)

        from alignment.align import align_transcript_to_audio

        model = _get_model()
        result = align_transcript_to_audio(
            audio_file=tmp_path,
            transcript=text,
            model=model,
            language=language,
            entry_id=entry_id,
        )

        return jsonify(_format_result(result))

    except Exception as e:
        logger.exception("Alignment failed")
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@alignment_bp.route("/health", methods=["GET"])
def alignment_health():
    """Health check for alignment service."""
    try:
        import stable_whisper  # noqa: F401
        import faster_whisper  # noqa: F401
        deps_ok = True
    except ImportError:
        deps_ok = False

    return jsonify({
        "status": "ok" if deps_ok else "missing_dependencies",
        "model": os.getenv("ALIGNMENT_MODEL", "ivrit-ai/yi-whisper-large-v3-turbo-ct2"),
        "dependencies_installed": deps_ok,
    })
