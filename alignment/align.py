"""Confusion-aware audio-text alignment for Yiddish.

Adapted from ivrit-ai/ivrit.ai alignment algorithm. Aligns transcript to audio
by breaking alignment into pieces when confusion zones are detected.

Key difference from ivrit-ai: default language="yi" and model uses
ivrit-ai/yi-whisper-large-v3-turbo-ct2 for Yiddish.
"""
from pathlib import Path
from typing import Union

import stable_whisper
from faster_whisper import WhisperModel
from stable_whisper.whisper_compatibility import SAMPLE_RATE
from tqdm import tqdm

from alignment.seekable_audio_loader import SeekableAudioLoader
from alignment.utils import (
    create_transcript_from_segments,
    find_probable_segment_before_time,
    get_breakable_align_model,
    get_confusion_zone,
    get_text_from_segments,
)

# Default Yiddish model
DEFAULT_MODEL = "ivrit-ai/yi-whisper-large-v3-turbo-ct2"


def align_transcript_to_audio(
    audio_file: Path,
    transcript: Union[Path, str, stable_whisper.result.WhisperResult],
    model: Union[str, WhisperModel] = DEFAULT_MODEL,
    device: str = "auto",
    align_model_compute_type: str = "int8",
    language: str = "yi",
    pre_confusion_zone_backward_skip_search_duration_window: int = 30,
    max_pre_confusion_zone_tries_before_skip: int = 2,
    unaligned_start_text_match_search_radius: int = 270,
    zero_duration_segments_failure_ratio: float = 0.2,
    max_confusion_zone_skip_duration: int = 120,
    min_confusion_zone_skip_duration: int = 15,
    entry_id: str = None,
) -> stable_whisper.WhisperResult:
    """Align a Yiddish transcript to audio using confusion-aware algorithm.

    Args:
        audio_file: Path to the audio file.
        transcript: Either a Path to a VTT file, a plain text string,
            or a WhisperResult object.
        model: Model name or loaded WhisperModel instance.
        device: "auto", "cpu", "cuda", or "cuda:N".
        language: Language code — "yi" for Yiddish.
        entry_id: Optional ID for logging in parallel processing.

    Returns:
        WhisperResult with word-level timestamps and confidence scores.
    """
    # Handle transcript input types
    if isinstance(transcript, str) and not Path(transcript).exists():
        # Plain text — create a simple WhisperResult with the text
        unaligned = stable_whisper.WhisperResult({
            "segments": [{"start": 0, "end": 0, "text": transcript}]
        })
    elif isinstance(transcript, (str, Path)):
        transcript_path = Path(transcript)
        if transcript_path.suffix == ".vtt":
            from stable_whisper.result import WhisperResult
            unaligned = WhisperResult(str(transcript_path))
        else:
            # Plain text file
            text = transcript_path.read_text(encoding="utf-8")
            unaligned = stable_whisper.WhisperResult({
                "segments": [{"start": 0, "end": 0, "text": text}]
            })
    else:
        unaligned = transcript

    # Load model if needed
    if isinstance(model, str):
        model = get_breakable_align_model(model, device, align_model_compute_type)

    audio_metadata = stable_whisper.audio.utils.get_metadata(str(audio_file))
    audio_duration = audio_metadata["duration"] or 0

    # Alignment loop state
    slice_start = 0
    top_matched_unaligned_timestamp = 0
    done = False
    aligned_pieces = []
    min_confusion_zone_start = 0
    max_confusion_zone_end = 0
    current_pre_confusion_zone_tries = 0
    to_align_next = unaligned.text

    progress_bar = tqdm(
        total=audio_duration, unit="sec", desc=f"Aligning {entry_id or 'Entry'}"
    )

    while not done:
        audio = SeekableAudioLoader(
            str(audio_file),
            sr=SAMPLE_RATE,
            stream=True,
            load_sections=[[slice_start, None]],
            test_first_chunk=False,
            buffer_size=300 * SAMPLE_RATE,
        )

        aligned = model.align(
            audio,
            to_align_next,
            language=language,
            failure_threshold=zero_duration_segments_failure_ratio,
        )

        if not aligned.segments:
            break

        any_good_alignments = aligned.segments[0].start != aligned.segments[-1].end

        if not any_good_alignments:
            confusion_zone_start = slice_start
            confusion_zone_end = confusion_zone_start + min_confusion_zone_skip_duration
        else:
            confusion_zone_start, confusion_zone_end = get_confusion_zone(aligned)

        # No confusion zone — alignment is complete
        if confusion_zone_start is None:
            aligned_pieces.extend(aligned.segments)
            break

        # Track confusion zone boundaries
        if confusion_zone_start > max_confusion_zone_end:
            min_confusion_zone_start = confusion_zone_start
            max_confusion_zone_end = confusion_zone_end
            current_pre_confusion_zone_tries = 0
        else:
            min_confusion_zone_start = min(min_confusion_zone_start, confusion_zone_start)
            max_confusion_zone_end = max(max_confusion_zone_end, confusion_zone_end)

        probable_segment_before = find_probable_segment_before_time(
            aligned,
            confusion_zone_start,
            pre_confusion_zone_backward_skip_search_duration_window,
        )

        if probable_segment_before is not None:
            aligned_pieces.extend(aligned.segments[: probable_segment_before.id + 1])
            slice_start = probable_segment_before.end
            progress_bar.update(slice_start - progress_bar.n)
            to_align_next = get_text_from_segments(
                aligned.segments[probable_segment_before.id + 1:]
            )
            if current_pre_confusion_zone_tries < max_pre_confusion_zone_tries_before_skip:
                current_pre_confusion_zone_tries += 1
                continue

        # Skip the confusion zone
        min_confusion_zone_start = slice_start
        max_confusion_zone_end = max(
            max_confusion_zone_end,
            min_confusion_zone_start + min_confusion_zone_skip_duration,
        )
        max_confusion_zone_end = min(
            min_confusion_zone_start + max_confusion_zone_skip_duration,
            max_confusion_zone_end,
        )

        progress_bar.write(
            f"Skipping confusion zone: {min_confusion_zone_start} - {max_confusion_zone_end}"
        )
        current_pre_confusion_zone_tries = 0

        # Text matching to find skip boundaries
        text_prefix = to_align_next[: unaligned_start_text_match_search_radius // 3]
        search_tries_left = 3
        search_radius = unaligned_start_text_match_search_radius
        found_at_text_idx = -1

        while search_tries_left > 0:
            search_tries_left -= 1
            search_around = (
                min_confusion_zone_start
                if probable_segment_before is None
                else probable_segment_before.end
            )
            search_start = max(top_matched_unaligned_timestamp, search_around - search_radius)
            search_end = search_around + search_radius

            segments_around = unaligned.get_content_by_time(
                (search_start, search_end), segment_level=True
            )
            text_around = get_text_from_segments(segments_around)
            found_at_text_idx = text_around.find(text_prefix)

            if found_at_text_idx == -1:
                search_radius *= 1.5
            else:
                break

        if found_at_text_idx == -1:
            found_at_text_idx = 0

        top_aligned_timestamp = aligned_pieces[-1].end if aligned_pieces else 0

        segments_after = unaligned.get_content_by_time(
            (max_confusion_zone_end, unaligned.segments[-1].end), segment_level=True
        )
        if segments_after and segments_after[0].start < max_confusion_zone_end:
            segments_after = segments_after[1:]

        # Process skipped segments
        if segments_around:
            curr_idx = 0
            idx_within = found_at_text_idx
            text_len = len(segments_around[curr_idx].text)
            while text_len <= found_at_text_idx and curr_idx + 1 < len(segments_around):
                idx_within = found_at_text_idx - text_len
                curr_idx += 1
                text_len += len(segments_around[curr_idx].text)

            initial_seg = segments_around[curr_idx]
            initial_seg_id = initial_seg.id

            if idx_within > 0:
                initial_seg = stable_whisper.result.Segment(
                    start=min(max(top_aligned_timestamp, slice_start), initial_seg.end),
                    end=max(top_aligned_timestamp, initial_seg.end),
                    text=initial_seg.text[idx_within:],
                )

            if not segments_after:
                done = True
                first_continue = None
                skip_segs = unaligned.segments[initial_seg_id + 1:]
            else:
                first_continue = segments_after[0]
                skip_segs = unaligned.segments[initial_seg_id + 1: first_continue.id]

            skip_segs = [initial_seg] + skip_segs
            skipped_text = get_text_from_segments(skip_segs)
            skip_start = max(top_aligned_timestamp, skip_segs[0].start)
            skip_end = first_continue.start if first_continue else None

            audio = SeekableAudioLoader(
                str(audio_file),
                sr=SAMPLE_RATE,
                stream=True,
                load_sections=[[skip_start, skip_end]],
                test_first_chunk=False,
            )
            aligned_skip = model.align(
                audio, skipped_text, language=language,
                failure_threshold=zero_duration_segments_failure_ratio,
            )

            for segment in aligned_skip:
                for word in segment.words:
                    word.start = max(word.start, top_aligned_timestamp)
                    word.end = max(word.end, word.start)
                    if skip_end:
                        word.end = min(skip_end, word.end)
                    word.start = min(word.end, word.start)

            aligned_pieces.extend(aligned_skip.segments)
            top_matched_unaligned_timestamp = skip_segs[-1].end
        else:
            if not segments_after:
                done = True
                first_continue = None
            else:
                first_continue = segments_after[0]

        if not done and first_continue:
            to_align_next = get_text_from_segments(segments_after)
            slice_start = first_continue.start
            progress_bar.update(slice_start - progress_bar.n)

        min_confusion_zone_start = 0
        max_confusion_zone_end = 0

    final_aligned = create_transcript_from_segments(aligned_pieces)
    progress_bar.close()
    return final_aligned
