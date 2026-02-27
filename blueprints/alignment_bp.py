"""Flask Blueprint for Yiddish audio-text alignment endpoint.

Provides word-level timestamp alignment using the ivrit-ai confusion-aware
algorithm with the yi-whisper-large-v3 Yiddish model.

Supports two backends:
  - RunPod serverless (default when RUNPOD_ENDPOINT_ID is set)
  - Local model (fallback, requires GPU for reasonable speed)
"""
import logging
import os
import tempfile
from pathlib import Path

from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

alignment_bp = Blueprint("alignment", __name__, url_prefix="/api/alignment")


def _use_runpod():
    """Check if RunPod backend is configured."""
    return bool(os.getenv("RUNPOD_ENDPOINT_ID")) and bool(os.getenv("RUNPOD_API_KEY"))


def _format_runpod_result(result):
    """Pass through RunPod result which already matches our format."""
    return {
        "segments": result.get("segments", []),
        "timestamps": result.get("timestamps", []),
        "full_text": result.get("full_text", ""),
    }


def _format_local_result(result):
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

    When RunPod is configured (RUNPOD_ENDPOINT_ID + RUNPOD_API_KEY),
    the audio is uploaded and processed on GPU via RunPod serverless.
    Otherwise falls back to local model (slow without GPU).
    """
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    if "audio" not in request.files and "audio_url" not in request.form:
        return jsonify({"error": "Missing 'audio' file or 'audio_url' parameter"}), 400

    language = request.form.get("language", "yi")
    entry_id = request.form.get("entry_id", None)

    try:
        if _use_runpod():
            return _align_via_runpod(text, language)
        else:
            return _align_locally(text, language, entry_id)
    except Exception as e:
        logger.exception("Alignment failed")
        return jsonify({"error": str(e)}), 500


def _align_via_runpod(text, language):
    """Align using RunPod serverless endpoint."""
    from runpod_client import RunPodAlignmentClient

    client = RunPodAlignmentClient()

    # If audio_url is provided directly, use it
    audio_url = request.form.get("audio_url")
    if audio_url:
        result = client.align(audio_url=audio_url, text=text, language=language)
        return jsonify(_format_runpod_result(result))

    # Otherwise, the audio file needs to be accessible via URL.
    # For now, save locally and provide a publicly accessible URL.
    # In production, upload to S3/R2/GCS and pass the URL.
    audio_file = request.files["audio"]
    suffix = Path(audio_file.filename).suffix or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp)
        tmp_path = Path(tmp.name)

    try:
        # Check if a file upload URL helper is configured
        upload_fn = os.getenv("AUDIO_UPLOAD_FUNCTION")
        if upload_fn:
            # Import and call the configured upload function
            module_name, func_name = upload_fn.rsplit(".", 1)
            import importlib
            mod = importlib.import_module(module_name)
            upload = getattr(mod, func_name)
            audio_url = upload(tmp_path)
        else:
            # Fallback: use the request's host as base URL for local serving
            # This only works if the server can serve the temp file
            return jsonify({
                "error": "RunPod requires audio_url (public URL). "
                         "Either pass audio_url in form data, or set "
                         "AUDIO_UPLOAD_FUNCTION env var to a module.function "
                         "that uploads a file and returns a URL."
            }), 400

        result = client.align(audio_url=audio_url, text=text, language=language)
        return jsonify(_format_runpod_result(result))

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _align_locally(text, language, entry_id):
    """Align using local model (slow without GPU)."""
    if "audio" not in request.files:
        return jsonify({"error": "Missing 'audio' file for local mode"}), 400

    audio_file = request.files["audio"]
    suffix = Path(audio_file.filename).suffix or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp)
        tmp_path = Path(tmp.name)

    try:
        from alignment.align import align_transcript_to_audio
        from alignment.utils import get_breakable_align_model

        model_name = os.getenv("ALIGNMENT_MODEL", "ivrit-ai/yi-whisper-large-v3-turbo-ct2")
        device = os.getenv("ALIGNMENT_DEVICE", "auto")
        compute_type = os.getenv("ALIGNMENT_COMPUTE_TYPE", "int8")

        logger.info(f"Loading local alignment model: {model_name} on {device}")
        model = get_breakable_align_model(model_name, device, compute_type)

        result = align_transcript_to_audio(
            audio_file=tmp_path,
            transcript=text,
            model=model,
            language=language,
            entry_id=entry_id,
        )
        return jsonify(_format_local_result(result))

    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@alignment_bp.route("/transcribe", methods=["POST"])
def transcribe_audio():
    """Transcribe audio with word-level timestamps (RunPod only).

    Accepts:
        - audio_url: (required) Public URL to audio file
        - language: (optional) Language code, default "yi"
    """
    if not _use_runpod():
        return jsonify({"error": "Transcription requires RunPod backend. Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY."}), 501

    audio_url = request.form.get("audio_url") or request.json.get("audio_url") if request.is_json else request.form.get("audio_url")
    if not audio_url:
        return jsonify({"error": "Missing 'audio_url' parameter"}), 400

    language = (request.form.get("language") or (request.json.get("language") if request.is_json else None)) or "yi"

    try:
        from runpod_client import RunPodAlignmentClient
        client = RunPodAlignmentClient()
        result = client.transcribe(audio_url=audio_url, language=language)
        return jsonify(_format_runpod_result(result))
    except Exception as e:
        logger.exception("Transcription failed")
        return jsonify({"error": str(e)}), 500


@alignment_bp.route("/health", methods=["GET"])
def alignment_health():
    """Health check for alignment service."""
    backend = "runpod" if _use_runpod() else "local"
    runpod_ok = None

    if backend == "runpod":
        try:
            from runpod_client import RunPodAlignmentClient
            client = RunPodAlignmentClient()
            health = client.health()
            runpod_ok = True
        except Exception as e:
            runpod_ok = False
            logger.warning(f"RunPod health check failed: {e}")

    try:
        import stable_whisper  # noqa: F401
        import faster_whisper  # noqa: F401
        local_deps_ok = True
    except ImportError:
        local_deps_ok = False

    return jsonify({
        "status": "ok",
        "backend": backend,
        "model": os.getenv("ALIGNMENT_MODEL", "ivrit-ai/yi-whisper-large-v3-turbo-ct2"),
        "runpod_healthy": runpod_ok,
        "local_dependencies_installed": local_deps_ok,
    })
