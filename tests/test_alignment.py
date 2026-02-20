"""Tests for alignment utility functions."""
import pytest
from unittest.mock import MagicMock, patch


class MockWord:
    def __init__(self, start, end, probability, text="word"):
        self.start = start
        self.end = end
        self.probability = probability
        self.word = text
        self.text = text


class MockSegment:
    def __init__(self, id, start, end, text, words=None):
        self.id = id
        self.start = start
        self.end = end
        self.text = text
        self.words = words or []

    def to_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [
                {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                for w in self.words
            ],
        }


class TestCalculateBadGoodProbRatio:
    def test_all_good_words(self):
        from alignment.utils import calculate_bad_good_prob_ratio

        words = [MockWord(0, 1, 0.9), MockWord(1, 2, 0.8), MockWord(2, 3, 0.95)]
        ratio, bad, good = calculate_bad_good_prob_ratio(words)
        assert ratio == 0.0
        assert bad == 0.0
        assert good > 0

    def test_all_bad_words(self):
        from alignment.utils import calculate_bad_good_prob_ratio

        words = [MockWord(0, 1, 0.1), MockWord(1, 2, 0.2), MockWord(2, 3, 0.05)]
        ratio, bad, good = calculate_bad_good_prob_ratio(words)
        assert ratio == float("inf")
        assert good == 0

    def test_mixed_words(self):
        from alignment.utils import calculate_bad_good_prob_ratio

        words = [
            MockWord(0, 1, 0.9),   # good
            MockWord(1, 2, 0.1),   # bad
            MockWord(2, 3, 0.8),   # good
        ]
        ratio, bad, good = calculate_bad_good_prob_ratio(words)
        assert 0 < ratio < float("inf")

    def test_zero_duration_bad_words_get_minimum(self):
        from alignment.utils import calculate_bad_good_prob_ratio

        words = [MockWord(0, 0, 0.1)]  # zero duration
        ratio, bad, good = calculate_bad_good_prob_ratio(words)
        assert bad == 0.1  # minimum 0.1


class TestGetTextFromSegments:
    def test_concatenates_text(self):
        from alignment.utils import get_text_from_segments

        segs = [MockSegment(0, 0, 1, "hello "), MockSegment(1, 1, 2, "world")]
        assert get_text_from_segments(segs) == "hello world"

    def test_empty_segments(self):
        from alignment.utils import get_text_from_segments

        assert get_text_from_segments([]) == ""


class TestCreateTranscriptFromSegments:
    def test_creates_whisper_result(self):
        with patch("alignment.utils.stable_whisper") as mock_sw:
            from alignment.utils import create_transcript_from_segments

            segs = [MockSegment(0, 0, 1, "test")]
            create_transcript_from_segments(segs)
            mock_sw.WhisperResult.assert_called_once()


class TestGetConfusionZone:
    def test_no_confusion_returns_none(self):
        from alignment.utils import get_confusion_zone

        # All high-probability words
        words = [MockWord(i, i + 1, 0.9) for i in range(200)]
        aligned = MagicMock()
        aligned.all_words.return_value = words

        start, end = get_confusion_zone(aligned)
        assert start is None
        assert end is None

    def test_confusion_detected(self):
        from alignment.utils import get_confusion_zone

        # First 50 good, then 200 bad
        words = [MockWord(i, i + 1, 0.9) for i in range(50)]
        words += [MockWord(i, i + 1, 0.1) for i in range(50, 250)]
        aligned = MagicMock()
        aligned.all_words.return_value = words

        start, end = get_confusion_zone(aligned)
        assert start is not None
        assert end is not None
        assert end > start

    def test_empty_words(self):
        from alignment.utils import get_confusion_zone

        aligned = MagicMock()
        aligned.all_words.return_value = []

        start, end = get_confusion_zone(aligned)
        assert start is None
        assert end is None


class TestEndpointFormatResult:
    def test_format_with_words(self):
        from blueprints.alignment_bp import _format_local_result as _format_result

        word = MockWord(0.0, 0.5, 0.92, "test")
        seg = MockSegment(0, 0.0, 0.5, "test", words=[word])
        seg.text = "test"

        result = MagicMock()
        result.segments = [seg]
        result.text = "test"

        formatted = _format_result(result)
        assert len(formatted["segments"]) == 1
        assert len(formatted["timestamps"]) == 1
        assert formatted["timestamps"][0]["type"] == "word"
        assert formatted["timestamps"][0]["confidence"] == 0.92

    def test_format_without_words(self):
        from blueprints.alignment_bp import _format_local_result as _format_result

        seg = MockSegment(0, 0.0, 2.5, "test segment", words=[])
        result = MagicMock()
        result.segments = [seg]
        result.text = "test segment"

        formatted = _format_result(result)
        assert formatted["timestamps"][0]["type"] == "segment"
        assert formatted["timestamps"][0]["confidence"] is None
