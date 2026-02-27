"""Client for calling the Yiddish alignment RunPod serverless endpoint."""
import json
import os
import time
import logging
import requests

logger = logging.getLogger(__name__)

RUNPOD_API_BASE = "https://api.runpod.ai/v2"


class RunPodAlignmentClient:
    """Client for Yiddish alignment/transcription via RunPod serverless."""

    def __init__(self, api_key: str = None, endpoint_id: str = None):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = endpoint_id or os.getenv("RUNPOD_ENDPOINT_ID")
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY required")
        if not self.endpoint_id:
            raise ValueError("RUNPOD_ENDPOINT_ID required")

        self.base_url = f"{RUNPOD_API_BASE}/{self.endpoint_id}"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def align(self, audio_url: str, text: str, language: str = "yi") -> dict:
        """Align text to audio. Returns word-level timestamps."""
        return self._run({
            "mode": "align",
            "audio_url": audio_url,
            "text": text,
            "language": language,
        })

    def transcribe(self, audio_url: str, language: str = "yi", word_timestamps: bool = True) -> dict:
        """Transcribe audio. Returns text with word-level timestamps."""
        return self._run({
            "mode": "transcribe",
            "audio_url": audio_url,
            "language": language,
            "word_timestamps": word_timestamps,
        })

    def _run(self, input_data: dict, timeout: int = 300) -> dict:
        """Submit job and poll until complete."""
        # Try runsync first (blocks up to ~30s)
        resp = requests.post(
            f"{self.base_url}/runsync",
            headers=self.headers,
            json={"input": input_data},
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()

        status = result.get("status")
        if status == "COMPLETED":
            return result.get("output", {})

        # If still running, poll
        job_id = result.get("id")
        if not job_id:
            raise RuntimeError(f"Unexpected response: {result}")

        return self._poll(job_id, timeout)

    def _poll(self, job_id: str, timeout: int = 300) -> dict:
        """Poll for job completion."""
        start = time.time()
        while time.time() - start < timeout:
            resp = requests.get(
                f"{self.base_url}/status/{job_id}",
                headers=self.headers,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()

            status = result.get("status")
            if status == "COMPLETED":
                return result.get("output", {})
            elif status == "FAILED":
                raise RuntimeError(f"Job failed: {result.get('error', 'unknown')}")
            elif status in ("IN_QUEUE", "IN_PROGRESS"):
                time.sleep(2)
            else:
                raise RuntimeError(f"Unknown status: {status}")

        raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

    def health(self) -> dict:
        """Check endpoint health."""
        resp = requests.get(
            f"{self.base_url}/health",
            headers=self.headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
