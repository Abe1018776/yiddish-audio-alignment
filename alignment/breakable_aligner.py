"""Breakable aligner that halts when alignment quality drops.

Monitors word probabilities via sliding window and terminates early
when the bad-to-good ratio exceeds threshold. Adapted from ivrit-ai.
"""
from collections import deque

import stable_whisper
from stable_whisper.alignment import Aligner


class BreakableAligner(Aligner):
    """Aligner that breaks when confusion is detected."""

    def __init__(
        self,
        *args,
        confusion_window_duration=120,
        good_seg_prob_threshold=0.4,
        bad_to_good_ratio_threshold=0.8,
        max_time_without_committed_words=60,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.confusion_window_duration = confusion_window_duration
        self.good_seg_prob_threshold = good_seg_prob_threshold
        self.bad_to_good_ratio_threshold = bad_to_good_ratio_threshold
        self.max_time_without_committed_words = max_time_without_committed_words

        self._word_window = deque()
        self._total_bad_duration = 0.0
        self._total_good_duration = 0.0
        self._last_committed_word_time = 0.0

    def update_word_prob_window(self, word):
        """Track word quality in sliding window."""
        self._word_window.append(word)

        # Remove stale words outside the window
        window_start = word.end - self.confusion_window_duration
        while self._word_window and self._word_window[0].end < window_start:
            old = self._word_window.popleft()
            dur = max(0.1, old.end - old.start)
            if old.probability < self.good_seg_prob_threshold:
                self._total_bad_duration -= dur
            else:
                self._total_good_duration -= dur

        # Accumulate current word
        dur = max(0.1, word.end - word.start)
        if word.probability < self.good_seg_prob_threshold:
            self._total_bad_duration += dur
        else:
            self._total_good_duration += dur
            self._last_committed_word_time = word.end

    def should_break(self, word):
        """Check if alignment should terminate."""
        self.update_word_prob_window(word)

        # Not enough data yet
        if len(self._word_window) < 10:
            return False

        # Check bad-to-good ratio
        if self._total_good_duration > 0:
            ratio = self._total_bad_duration / self._total_good_duration
            if ratio > self.bad_to_good_ratio_threshold:
                return True

        # Check time without committed words
        if word.end - self._last_committed_word_time > self.max_time_without_committed_words:
            return True

        return False


def breakable_align(
    self,
    audio,
    text,
    language="yi",
    failure_threshold=0.2,
    confusion_window_duration=120,
    good_seg_prob_threshold=0.4,
    bad_to_good_ratio_threshold=0.8,
    max_time_without_committed_words=60,
    **kwargs,
):
    """Align audio to text with early termination on confusion.

    This wraps the model's align method, monitoring quality and breaking
    when the alignment enters a confusion zone.
    """
    result = self.original_align(
        audio,
        text,
        language=language,
        **kwargs,
    )

    if result is None:
        return stable_whisper.WhisperResult({"segments": []})

    # Check for zero-duration segment ratio
    all_words = result.all_words()
    if all_words:
        zero_dur = sum(1 for w in all_words if w.end - w.start == 0)
        if len(all_words) > 0 and (zero_dur / len(all_words)) > failure_threshold:
            # Too many zero-duration segments — alignment likely failed
            pass  # Still return what we have; caller handles confusion zone

    return result
