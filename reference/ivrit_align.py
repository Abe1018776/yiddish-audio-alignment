"""Reference: ivrit-ai/ivrit.ai/alignment/align.py - DO NOT MODIFY, reference only."""
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
from utils.vtt import vtt_to_whisper_result


def align_transcript_to_audio(
    audio_file: Path,
    transcript: Union[Path, stable_whisper.result.WhisperResult],
    model: Union[str, WhisperModel] = "ivrit-ai/whisper-large-v3-turbo-ct2",
    device: str = "auto",
    align_model_compute_type: str = "int8",
    language: str = "he",
    pre_confusion_zone_backward_skip_search_duration_window: int = 30,
    max_pre_confusion_zone_tries_before_skip: int = 2,
    unaligned_start_text_match_search_radius: int = 270,
    zero_duration_segments_failure_ratio: float = 0.2,
    max_confusion_zone_skip_duration: int = 120,
    min_confusion_zone_skip_duration: int = 15,
    entry_id: str = None,
) -> stable_whisper.WhisperResult:
    if isinstance(transcript, Path):
        unaligned = vtt_to_whisper_result(str(transcript))
    else:
        unaligned = transcript

    if isinstance(model, str):
        model = get_breakable_align_model(model, device, align_model_compute_type)

    audio_metadata = stable_whisper.audio.utils.get_metadata(str(audio_file))
    audio_duration = audio_metadata["duration"] or 0

    slice_start = 0
    top_matched_unaligned_timestamp = 0
    done = False
    aligned_pieces = []
    min_confusion_zone_start = 0
    max_confusion_zone_end = 0
    current_pre_confusion_zone_tries = 0
    to_align_next = unaligned.text

    progress_bar = tqdm(total=audio_duration, unit="sec", desc=f"Aligning {entry_id or 'Entry'}")
    while not done:
        audio = SeekableAudioLoader(
            str(audio_file), sr=SAMPLE_RATE, stream=True,
            load_sections=[[slice_start, None]], test_first_chunk=False,
            buffer_size=300 * SAMPLE_RATE,
        )
        aligned = model.align(audio, to_align_next, language=language, failure_threshold=zero_duration_segments_failure_ratio)

        any_good_alignemnts = aligned.segments[0].start != aligned.segments[-1].end
        if not any_good_alignemnts:
            confusion_zone_start = slice_start
            confusion_zone_end = confusion_zone_start + min_confusion_zone_skip_duration
        else:
            confusion_zone_start, confusion_zone_end = get_confusion_zone(aligned)

        if confusion_zone_start is None:
            aligned_pieces.extend(aligned.segments)
            break

        if confusion_zone_start > max_confusion_zone_end:
            min_confusion_zone_start = confusion_zone_start
            max_confusion_zone_end = confusion_zone_end
            current_pre_confusion_zone_tries = 0
        else:
            min_confusion_zone_start = min(min_confusion_zone_start, confusion_zone_start)
            max_confusion_zone_end = max(max_confusion_zone_end, confusion_zone_end)

        probable_segment_before_confusion_zone = find_probable_segment_before_time(
            aligned, confusion_zone_start, pre_confusion_zone_backward_skip_search_duration_window,
        )

        if probable_segment_before_confusion_zone is not None:
            segments_already_aligned = aligned.segments[: probable_segment_before_confusion_zone.id + 1]
            aligned_pieces.extend(segments_already_aligned)
            slice_start = probable_segment_before_confusion_zone.end
            progress_bar.update(slice_start - progress_bar.n)
            to_align_next = get_text_from_segments(aligned.segments[probable_segment_before_confusion_zone.id + 1 :])
            if current_pre_confusion_zone_tries < max_pre_confusion_zone_tries_before_skip:
                current_pre_confusion_zone_tries += 1
                continue

        min_confusion_zone_start = slice_start
        max_confusion_zone_end = max(max_confusion_zone_end, min_confusion_zone_start + min_confusion_zone_skip_duration)
        max_confusion_zone_end = min(min_confusion_zone_start + max_confusion_zone_skip_duration, max_confusion_zone_end)
        current_pre_confusion_zone_tries = 0

        text_at_start_of_confusion_zone = to_align_next[: unaligned_start_text_match_search_radius // 3]
        search_tries_left = 3
        search_radius_to_try = unaligned_start_text_match_search_radius
        while search_tries_left > 0:
            search_tries_left -= 1
            search_around_time = min_confusion_zone_start if probable_segment_before_confusion_zone is None else probable_segment_before_confusion_zone.end
            search_in_unaligned_window_time_start = max(top_matched_unaligned_timestamp, search_around_time - search_radius_to_try)
            search_in_unaligned_window_time_end = search_around_time + search_radius_to_try
            segments_around_confusion_zone = unaligned.get_content_by_time(
                (search_in_unaligned_window_time_start, search_in_unaligned_window_time_end), segment_level=True,
            )
            text_around_confusion_zone = get_text_from_segments(segments_around_confusion_zone)
            found_at_text_idx = text_around_confusion_zone.find(text_at_start_of_confusion_zone)
            if found_at_text_idx == -1:
                search_radius_to_try *= 1.5
            else:
                break

        if found_at_text_idx == -1:
            found_at_text_idx = 0

        top_aligned_timestamp = aligned_pieces[-1].end if aligned_pieces else 0
        segments_after_assumed_confusion_zone = unaligned.get_content_by_time(
            (max_confusion_zone_end, unaligned.segments[-1].end), segment_level=True
        )
        if segments_after_assumed_confusion_zone and segments_after_assumed_confusion_zone[0].start < max_confusion_zone_end:
            segments_after_assumed_confusion_zone = segments_after_assumed_confusion_zone[1:]

        curr_matched_segment_idx = 0
        index_within_segment = found_at_text_idx
        if segments_around_confusion_zone:
            text_len_so_far = len(segments_around_confusion_zone[curr_matched_segment_idx].text)
            while text_len_so_far <= found_at_text_idx:
                index_within_segment = found_at_text_idx - text_len_so_far
                curr_matched_segment_idx += 1
                text_len_so_far += len(segments_around_confusion_zone[curr_matched_segment_idx].text)

            initial_unaligned_segment_in_confusion_zone = segments_around_confusion_zone[curr_matched_segment_idx]
            initial_unaligned_segment_id_in_confusion_zone = initial_unaligned_segment_in_confusion_zone.id
            if index_within_segment > 0:
                initial_unaligned_segment_in_confusion_zone = stable_whisper.result.Segment(
                    start=min(max(top_aligned_timestamp, slice_start), initial_unaligned_segment_in_confusion_zone.end),
                    end=max(top_aligned_timestamp, initial_unaligned_segment_in_confusion_zone.end),
                    text=initial_unaligned_segment_in_confusion_zone.text[index_within_segment:],
                )

        if not segments_after_assumed_confusion_zone:
            done = True
            first_segment_to_continue_aligning = None
        else:
            first_segment_to_continue_aligning = segments_after_assumed_confusion_zone[0]

        if segments_around_confusion_zone:
            if done:
                confusing_segments_to_skip = unaligned.segments[initial_unaligned_segment_id_in_confusion_zone + 1:]
            else:
                confusing_segments_to_skip = unaligned.segments[initial_unaligned_segment_id_in_confusion_zone + 1: first_segment_to_continue_aligning.id]
            confusing_segments_to_skip = [initial_unaligned_segment_in_confusion_zone] + confusing_segments_to_skip

            skipped_text_to_align = get_text_from_segments(confusing_segments_to_skip)
            align_skipped_start_from = max(top_aligned_timestamp, confusing_segments_to_skip[0].start)
            align_skipped_end_at = first_segment_to_continue_aligning.start if first_segment_to_continue_aligning else None
            audio = SeekableAudioLoader(
                str(audio_file), sr=SAMPLE_RATE, stream=True,
                load_sections=[[align_skipped_start_from, align_skipped_end_at]], test_first_chunk=False,
            )
            aligned_skipped = model.align(audio, skipped_text_to_align, language=language, failure_threshold=zero_duration_segments_failure_ratio)
            for segment in aligned_skipped:
                for word in segment.words:
                    word.start = max(word.start, top_aligned_timestamp)
                    word.end = max(word.end, word.start)
                    if align_skipped_end_at:
                        word.end = min(align_skipped_end_at, word.end)
                    word.start = min(word.end, word.start)
            aligned_pieces.extend(aligned_skipped.segments)
            top_matched_unaligned_timestamp = confusing_segments_to_skip[-1].end

        if not done:
            to_align_next = get_text_from_segments(segments_after_assumed_confusion_zone)
            slice_start = first_segment_to_continue_aligning.start
            progress_bar.update(slice_start - progress_bar.n)

        min_confusion_zone_start = 0
        max_confusion_zone_end = 0

    final_aligned = create_transcript_from_segments(aligned_pieces)
    progress_bar.close()
    return final_aligned
