"""Reference: ivrit-ai/ivrit.ai/alignment/breakable_aligner.py - DO NOT MODIFY, reference only.
BreakableAligner extends stable_whisper's Aligner to halt when quality metrics drop.
Uses sliding window of word probabilities to detect confusion zones.
Key params: confusion_window_duration=120s, good_seg_prob_threshold=0.4,
bad_to_good_ratio_threshold=0.8, max_time_without_committed_words=60s.
"""
