"""Optimized audio loader with ffmpeg seek for mid-file restarts.

Adapted from ivrit-ai. When load_sections is specified with a start offset,
instructs ffmpeg to seek directly instead of chunk-by-chunk seeking.
"""
import subprocess
import warnings

from stable_whisper.audio import AudioLoader, load_source


class SeekableAudioLoader(AudioLoader):
    """AudioLoader that uses ffmpeg -ss for fast seeking in breakable alignment."""

    def _audio_loading_process(self):
        if not isinstance(self.source, str) or not self._stream:
            return

        only_ffmpeg = False
        source = load_source(
            self.source, verbose=self.verbose, only_ffmpeg=only_ffmpeg, return_dict=True
        )

        if isinstance(source, dict):
            info = source
            source = info.pop("popen")
        else:
            info = None

        if info and info["duration"]:
            self._duration_estimation = info["duration"]
            if not self._stream and info["is_live"]:
                warnings.warn(
                    "The audio appears to be a continuous stream "
                    "but setting was set to `stream=False`."
                )

        if isinstance(source, subprocess.Popen):
            self._extra_process, stdin = source, source.stdout
        else:
            stdin = None

        try:
            seek_start_cmd_parts = []
            if self.load_sections:
                start_at = self.load_sections[0][0]
                if start_at:
                    self._prev_seek = int(round(start_at * self._sr))
                    self._accum_samples = self._prev_seek
                    seek_start_cmd_parts = ["-ss", str(start_at)]

            cmd = [
                "ffmpeg", "-loglevel", "panic", "-nostdin", "-threads", "0",
                *seek_start_cmd_parts,
                "-i", self.source if stdin is None else "pipe:",
                "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
                "-ar", str(self._sr), "-",
            ]
            out = subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.PIPE)

        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to load audio: {e}") from e

        return out
