"""Alignment utility functions: confusion zone detection, segment helpers.

Adapted from ivrit-ai for Yiddish alignment.
"""
from collections import deque
from types import MethodType

import numpy as np
import stable_whisper


def calculate_bad_good_prob_ratio(words, good_seg_prob_threshold=0.4):
    """Calculate ratio of low-confidence to high-confidence word durations."""
    bad_durations = [
        max(0.1, w.end - w.start) for w in words if w.probability < good_seg_prob_threshold
    ]
    good_durations = [
        w.end - w.start for w in words if w.probability >= good_seg_prob_threshold
    ]
    total_bad = sum(bad_durations)
    total_good = sum(good_durations)
    ratio = float("inf") if total_good == 0 else (total_bad / total_good)
    return ratio, total_bad, total_good


def get_confusion_zone(
    aligned,
    detection_window_duration=120,
    hop_length=30,
    good_seg_prob_threshold=0.4,
    bad_to_good_probs_detection_threshold=0.8,
):
    """Detect regions where alignment quality drops using sliding window."""
    all_words = aligned.all_words()
    if not all_words:
        return None, None

    start_time = all_words[0].start
    end_time = all_words[-1].end
    window_start = start_time
    window_end = window_start + detection_window_duration
    window_words = deque()
    word_index = 0

    while window_end < end_time:
        while word_index < len(all_words) and all_words[word_index].start < window_end:
            word = all_words[word_index]
            if word.end > window_start:
                window_words.append(word)
            word_index += 1

        while window_words and window_words[0].end <= window_start:
            window_words.popleft()

        ratio, _, _ = calculate_bad_good_prob_ratio(window_words, good_seg_prob_threshold)

        if ratio > bad_to_good_probs_detection_threshold:
            return window_start, window_end

        window_start += hop_length
        window_end = window_start + detection_window_duration

    return None, None


def create_transcript_from_segments(slice_segments):
    """Create a WhisperResult from a list of Segment objects."""
    slice_segments_as_dict = [s.to_dict() for s in slice_segments]
    return stable_whisper.WhisperResult({"segments": slice_segments_as_dict})


def get_text_from_segments(segments):
    """Concatenate text from a list of segments."""
    return "".join([s.text for s in segments])


def find_probable_segment_before_time(
    aligned,
    find_before_time,
    go_back_duration,
    minimal_seg_prob_to_consider_retry_start_segment=0.8,
    max_seek_start_seg_backward_hops=6,
):
    """Find a high-confidence segment before a given time for re-alignment."""
    segment_to_start_after = None
    search_for_segment_start_time = find_before_time - go_back_duration

    while not segment_to_start_after and max_seek_start_seg_backward_hops > 0:
        pre_confusion_zone_segments = aligned.get_content_by_time(
            (search_for_segment_start_time, search_for_segment_start_time + go_back_duration),
            segment_level=True,
        )
        if pre_confusion_zone_segments:
            for pre_seg in pre_confusion_zone_segments:
                seg_word_probs = [w.probability for w in pre_seg.words]
                avg_seg_prob = np.mean(seg_word_probs) if seg_word_probs else 0
                if avg_seg_prob >= minimal_seg_prob_to_consider_retry_start_segment:
                    segment_to_start_after = pre_seg
                    break

        if not segment_to_start_after:
            max_seek_start_seg_backward_hops -= 1
            search_for_segment_start_time -= go_back_duration
            if search_for_segment_start_time < 0:
                break

    return segment_to_start_after


def get_breakable_align_model(model_name, device, compute_type):
    """Load a faster-whisper model with breakable alignment capability."""
    from alignment.breakable_aligner import breakable_align

    device_index = 0
    if len(device.split(":")) == 2:
        device, device_index = device.split(":")
        device_index = int(device_index)

    model = stable_whisper.load_faster_whisper(
        model_name, device=device, device_index=device_index, compute_type=compute_type,
    )
    model.align = MethodType(breakable_align, model)
    return model
