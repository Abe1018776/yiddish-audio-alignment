# Multi-Agent Build Plan: Clean Yiddish Transcripts Feature Expansion

Created: 2026-02-19
Author: architect-agent (Opus 4.6)

---

## Codebase Overview

The existing application is a **Flask web app** (`app.py`, port 5050) for cleaning Yiddish transcripts. It uses a **plugin-based architecture** with:

- **ProcessorRegistry** / **WriterRegistry** (`registry.py`) -- decorator-based plugin registration
- **BaseProcessor** (`processors/base.py`) -- abstract base for text cleaning plugins
- **TranscriptCleaner** (`cleaner.py`) -- orchestrates cleaning profiles (chains of processors)
- **DocumentProcessor** (`document_processor.py`) -- handles .docx/.doc parsing, text extraction, output generation
- **LLM Processor** (`llm_processor.py`) -- standalone module for LLM-based cleaning (OpenAI, Anthropic, Google, Groq, OpenRouter, Ollama)
- **Diff Utils** (`diff_utils.py`) -- line-by-line and word-by-word diff generation
- **Document Model** (`document_model.py`) -- dataclass-based document representation (Paragraph, TextRun, RunStyle)
- **Writers** (`writers/`) -- output format plugins (docx, txt)
- **Templates** (`templates/index.html`) -- single-page frontend (Tailwind CSS, ~2700 lines)

**Key patterns to follow:**
1. New features should be **standalone Python modules** at the project root (like `llm_processor.py`)
2. Routes are added directly to `app.py` but agents should create **Flask Blueprints** in separate files to avoid merge conflicts
3. The registry pattern (`@ProcessorRegistry.register`) is available for new processors
4. The app uses `python-dotenv` for configuration via `.env`
5. Frontend is a single HTML file with inline JS -- new UI should be added as separate template pages or via API-only endpoints

**External data source (kol-yid sites):**
- Audio samples at: `https://kol-yid.on-forge.com/audio-samples/{id}` and `https://kolyid.dynamiq.dev/audio-samples/{id}`
- Transcriptions at: `https://kol-yid.on-forge.com/transcriptions/{id}`
- These sites require authentication (redirect to `/login` for unauthenticated requests)
- The URL structure implies a REST-like data model with numeric IDs linking audio samples to transcriptions

---

## Shared Contracts (All Agents Must Agree On)

### 1. Timestamp Format (Standard Across All Agents)

```json
{
  "timestamps": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "word or segment text",
      "confidence": 0.95,
      "type": "word"
    }
  ]
}
```

- `start` / `end`: float, seconds from audio start
- `text`: the transcribed text for this segment
- `confidence`: float 0.0-1.0 (optional, may be null)
- `type`: `"word"` | `"segment"` | `"sentence"`

### 2. Audio-Text Mapping Format

```json
{
  "version": "1.0",
  "created": "2026-02-19T00:00:00Z",
  "mappings": [
    {
      "id": "mapping_001",
      "audio_url": "https://kol-yid.on-forge.com/audio-samples/42",
      "audio_id": 42,
      "transcript_url": "https://kol-yid.on-forge.com/transcriptions/35",
      "transcript_id": 35,
      "title": "5720 Shevat 10 Maamar",
      "year": "5720",
      "month": "Shevat",
      "type": "maamar|sicha",
      "duration_minutes": null,
      "is_eval": false
    }
  ],
  "eval_set": [
    {
      "audio_id": 28,
      "transcript_id": 21,
      "title": "5711-tamuz-12-sicha",
      "reason": "reserved for evaluation"
    }
  ],
  "statistics": {
    "total_mappings": 50,
    "total_hours": 50.0,
    "year_distribution": {"5710": 3, "5711": 4},
    "month_distribution": {"Tishrei": 5, "Shevat": 4}
  }
}
```

### 3. Transcription Result Format (YiddishLabs Output)

```json
{
  "audio_id": 42,
  "source_url": "https://kol-yid.on-forge.com/audio-samples/42",
  "transcription": {
    "full_text": "the complete transcribed text...",
    "timestamps": [
      {"start": 0.0, "end": 2.5, "text": "word", "confidence": 0.9}
    ],
    "language": "yi",
    "duration_seconds": 3600.0
  },
  "metadata": {
    "model": "yiddishlabs",
    "transcribed_at": "2026-02-19T00:00:00Z",
    "rapid": false
  }
}
```

### 4. Cleaned Transcript Diff Format (For Diff Viewer)

```json
{
  "original_text": "full original text...",
  "cleaned_text": "full cleaned text...",
  "diff_rows": [
    {
      "row_index": 0,
      "type": "unchanged|modified|removed|added",
      "original_line": "original line text",
      "cleaned_line": "cleaned line text",
      "original_line_number": 1,
      "cleaned_line_number": 1,
      "timestamp_start": 0.0,
      "timestamp_end": 5.2
    }
  ]
}
```

### 5. Stable-TS Alignment Output Format

```json
{
  "audio_file": "path_or_url",
  "aligned_segments": [
    {
      "start": 0.0,
      "end": 2.5,
      "text": "aligned text segment",
      "words": [
        {"word": "text", "start": 0.0, "end": 0.5, "confidence": 0.92}
      ]
    }
  ],
  "model": "openai/whisper-large-v3",
  "language": "yi"
}
```

---

## Execution Order & Dependencies

```
Agent 1 (Audio-Text Mapper)  ──────────────────────────────┐
                                                            │
Agent 2 (YiddishLabs Transcription) ─── needs Agent 1's ───┤
                                        mapping output      │
                                                            │
Agent 3 (LLM Cleaning Flow)  ──── independent ─────────────┤
                                                            │
Agent 5 (Stable-TS Alignment) ─── independent ─────────────┤
                                                            │
Agent 4 (Diff Viewer + Audio) ─── needs Agent 2 & 3 ───────┘
                                   (timestamps + cleaned text)
```

**Parallel Group 1 (can run simultaneously):**
- Agent 1: Audio-Text Mapper
- Agent 3: LLM Cleaning Flow
- Agent 5: Stable-TS Alignment

**Sequential Group 2 (after Group 1):**
- Agent 2: YiddishLabs Transcription (needs Agent 1 mapping output format, but can stub it)

**Sequential Group 3 (after Group 2):**
- Agent 4: Diff Viewer with Audio (needs Agent 2 timestamps + Agent 3 cleaning output)

**Practical recommendation:** All 5 agents CAN run in parallel if they code against the shared contracts above. Agent 4 should use mock timestamp data during development.

---

## Agent 1: Audio-Text Mapper

**Branch:** `feature/audio-text-mapping`

### Purpose
Identify and document 50 hours of matching audio + transcript pairs from the kol-yid platform, with even distribution across years and months. Exclude 5 specific eval-set items.

### Files to Create

| File | Purpose |
|------|---------|
| `audio_text_mapper.py` | Core mapping logic -- scraping/API client for kol-yid sites |
| `mapping_config.py` | Configuration: base URLs, eval set exclusions, distribution targets |
| `mappings/README.md` | Documentation for the mapping data |
| `mappings/audio_text_mappings.json` | Output: the 50-hour mapping document |
| `mappings/eval_set.json` | The 5 excluded eval items with metadata |
| `tests/test_mapper.py` | Unit tests for mapping logic |

### Detailed Implementation Instructions

**1. `mapping_config.py`**
```python
"""Configuration for audio-text mapping."""

# Base URLs for the kol-yid platform
BASE_URLS = [
    "https://kol-yid.on-forge.com",
    "https://kolyid.dynamiq.dev",
]

# Eval set -- these audio-transcript pairs are EXCLUDED from training data
EVAL_SET = [
    {"audio_id": 28, "transcript_id": 21, "title": "5711-tamuz-12-sicha"},
    {"audio_id": 30, "transcript_id": 23, "title": "5715-Tamuz 13d Sicha"},
    {"audio_id": 80, "transcript_id": None, "title": "5741-Nissan 11 Mamar"},
    {"audio_id": 1,  "transcript_id": None, "title": "5711 10 Shevat Maamar"},
    {"audio_id": 10, "transcript_id": None, "title": "5742 19 Kislev Sicha"},
]

EVAL_AUDIO_IDS = {item["audio_id"] for item in EVAL_SET}

# Target: 50 hours of audio distributed across all available years and months
TARGET_HOURS = 50
TARGET_DISTRIBUTION = "even"  # Aim for even spread across years/months

# Hebrew month order for sorting
HEBREW_MONTHS = [
    "Tishrei", "Cheshvan", "Kislev", "Teves", "Shevat", "Adar",
    "Adar I", "Adar II", "Nissan", "Iyar", "Sivan",
    "Tamuz", "Av", "Elul"
]
```

**2. `audio_text_mapper.py`**

This module should:
- Connect to the kol-yid platform (handle authentication if needed -- check for API keys/tokens in `.env`)
- Crawl or query the audio-samples and transcriptions endpoints
- Match audio samples to their corresponding transcriptions
- Filter out the 5 eval-set items
- Select ~50 hours of content with even year/month distribution
- Output the mapping as JSON following the shared contract format

Key functions:
```python
class AudioTextMapper:
    def __init__(self, base_url: str, auth_token: str = None):
        """Initialize with platform credentials."""
    
    def fetch_audio_samples(self, page: int = 1, per_page: int = 100) -> List[dict]:
        """Fetch audio sample listings from the platform."""
    
    def fetch_transcriptions(self, page: int = 1, per_page: int = 100) -> List[dict]:
        """Fetch transcription listings from the platform."""
    
    def match_audio_to_transcripts(self, audio_samples: List, transcriptions: List) -> List[dict]:
        """Match audio samples to their transcriptions by title/metadata."""
    
    def filter_eval_set(self, mappings: List[dict]) -> List[dict]:
        """Remove eval-set items from the mapping."""
    
    def select_distributed_subset(self, mappings: List[dict], target_hours: float = 50) -> List[dict]:
        """Select a subset with even year/month distribution targeting N hours."""
    
    def generate_mapping_document(self, output_path: str = "mappings/audio_text_mappings.json"):
        """Run the full pipeline and output the mapping document."""
    
    def generate_statistics(self, mappings: List[dict]) -> dict:
        """Compute distribution statistics for the selected mappings."""
```

**3. Integration approach:**
- Add env vars to `.env.example`: `KOL_YID_BASE_URL`, `KOL_YID_AUTH_TOKEN`
- The mapper runs as a **standalone script** (`python audio_text_mapper.py`) and also exposes functions importable by other modules
- Output lives in `mappings/` directory (gitignored binary audio, committed JSON metadata)
- Optionally add a Flask route `/api/mappings` to serve the mapping data

**Important notes for the agent:**
- The kol-yid sites require authentication (they redirect to `/login` for unauthenticated requests). The agent MUST handle authentication. Check if there's a session/cookie/token-based auth or an API endpoint.
- If the site has a REST API, use it. If it's a rendered web app, the agent may need to use `requests.Session` with login credentials or document that manual export is needed.
- The agent should provide a **fallback mode** where mappings can be populated manually from a CSV or spreadsheet if automated scraping is not feasible.
- Parse year from titles (e.g., "5711-tamuz-12-sicha" -> year=5711, month=Tamuz)

### Dependencies
```
requests>=2.31.0
```

### Testing Approach
- Unit test the title parser (extract year/month from title strings)
- Unit test the distribution algorithm
- Unit test eval-set filtering
- Integration test with mock API responses
- Validate output JSON against the shared contract schema

---

## Agent 2: YiddishLabs Transcription Integration

**Branch:** `feature/yiddishlabs-transcription`

### Purpose
Build a module that sends audio files to the YiddishLabs API for transcription and retrieves timestamped results.

### Files to Create

| File | Purpose |
|------|---------|
| `yiddishlabs_client.py` | API client for YiddishLabs transcription service |
| `transcription_manager.py` | Batch transcription orchestration, progress tracking |
| `blueprints/transcription_bp.py` | Flask Blueprint with transcription API routes |
| `tests/test_yiddishlabs.py` | Unit tests for the client |

### Detailed Implementation Instructions

**1. `yiddishlabs_client.py`**

```python
"""Client for the YiddishLabs transcription API."""

import requests
import os
from typing import Optional, BinaryIO
from pathlib import Path


class YiddishLabsClient:
    """Client for YiddishLabs transcription API."""
    
    DEFAULT_BASE_URL = "https://api.yiddishlabs.com"  # Confirm actual URL
    SUPPORTED_FORMATS = {"mp3", "wav", "m4a", "ogg", "flac"}
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("YIDDISHLABS_API_KEY")
        self.base_url = base_url or os.getenv("YIDDISHLABS_BASE_URL", self.DEFAULT_BASE_URL)
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
    
    def transcribe(
        self,
        audio_file: BinaryIO,
        name: Optional[str] = None,
        context: Optional[str] = None,
        webhook_url: Optional[str] = None,
        timestamps: bool = True,        # MUST be True per requirements
        language: str = "yi",            # Default to Yiddish
        rapid: bool = False
    ) -> dict:
        """
        Submit audio file for transcription.
        
        Args:
            audio_file: Binary file object (MP3, WAV, M4A, OGG, FLAC)
            name: Optional name for the transcription
            context: Optional context to help transcription
            webhook_url: Optional webhook for async notification
            timestamps: Request word-level timestamps (ALWAYS True)
            language: Language code (yi, en, he, lk, auto)
            rapid: Use rapid transcription mode
            
        Returns:
            dict with transcription result including timestamps
        """
        files = {"file": audio_file}
        data = {}
        params = {}
        
        if name:
            data["name"] = name
        if context:
            data["context"] = context
        if webhook_url:
            data["webhook_url"] = webhook_url
        
        # timestamps is a query parameter
        params["timestamps"] = "true" if timestamps else "false"
        
        if language:
            data["language"] = language
        if rapid:
            data["rapid"] = "true"
        
        response = self.session.post(
            f"{self.base_url}/transcribe",
            files=files,
            data=data,
            params=params,
            timeout=600  # 10 min timeout for long audio
        )
        response.raise_for_status()
        return response.json()
    
    def transcribe_file(self, file_path: str, **kwargs) -> dict:
        """Transcribe from a file path."""
        path = Path(file_path)
        if path.suffix.lstrip('.').lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}. Supported: {self.SUPPORTED_FORMATS}")
        
        with open(file_path, "rb") as f:
            kwargs.setdefault("name", path.stem)
            return self.transcribe(f, **kwargs)
    
    def get_transcription_status(self, transcription_id: str) -> dict:
        """Check status of an async transcription (if webhook mode)."""
        response = self.session.get(f"{self.base_url}/transcriptions/{transcription_id}")
        response.raise_for_status()
        return response.json()
```

**2. `transcription_manager.py`**

```python
"""Batch transcription manager for processing multiple audio files."""

import json
import os
import time
import logging
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class TranscriptionManager:
    """Manages batch transcription jobs using YiddishLabs."""
    
    def __init__(self, client, output_dir: str = "transcriptions"):
        self.client = client
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def transcribe_from_mapping(
        self,
        mapping_file: str,
        audio_dir: str = None,
        skip_existing: bool = True
    ) -> List[dict]:
        """
        Transcribe all audio files from Agent 1's mapping document.
        
        Reads mappings/audio_text_mappings.json and processes each audio file.
        """
        ...
    
    def transcribe_single(
        self,
        audio_path: str,
        audio_id: int = None,
        save_result: bool = True
    ) -> dict:
        """Transcribe a single audio file and optionally save result."""
        ...
    
    def save_transcription(self, result: dict, audio_id: int):
        """Save transcription result to JSON file in output_dir."""
        output_path = os.path.join(self.output_dir, f"transcription_{audio_id}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def normalize_timestamps(self, raw_result: dict) -> List[dict]:
        """Convert YiddishLabs timestamp format to shared contract format."""
        ...
```

**3. `blueprints/transcription_bp.py`**

```python
"""Flask Blueprint for YiddishLabs transcription routes."""

from flask import Blueprint, request, jsonify
import os

transcription_bp = Blueprint('transcription', __name__, url_prefix='/api/transcription')

@transcription_bp.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Transcribe an uploaded audio file via YiddishLabs."""
    ...

@transcription_bp.route('/transcribe-url', methods=['POST'])
def transcribe_from_url():
    """Transcribe audio from a URL."""
    ...

@transcription_bp.route('/status/<transcription_id>', methods=['GET'])
def get_status(transcription_id):
    """Check transcription status."""
    ...

@transcription_bp.route('/results', methods=['GET'])
def list_results():
    """List all transcription results."""
    ...
```

**4. Integration with existing app:**
- The Blueprint is registered in `app.py` via: `from blueprints.transcription_bp import transcription_bp; app.register_blueprint(transcription_bp)`
- However, to avoid modifying `app.py` on this branch, the agent should create an `__init__.py` or `register.py` that documents how to register the Blueprint
- Add env vars: `YIDDISHLABS_API_KEY`, `YIDDISHLABS_BASE_URL`

### Dependencies
```
requests>=2.31.0
```
(Already in the project via Flask's dependencies, but good to be explicit)

### Testing Approach
- Unit test client with mocked HTTP responses
- Test timestamp normalization with sample YiddishLabs response data
- Test error handling (400, 401, 429, 500, timeout)
- Test file format validation
- Integration test: end-to-end with a small test audio file (if API key available)

---

## Agent 3: LLM Cleaning Flow

**Branch:** `feature/llm-cleaning-flow`

### Purpose
Build a module where users provide a prompt and an LLM removes commentary/non-speech content from manual transcripts. This extends the existing `llm_processor.py` as a modular pipeline.

### Files to Create

| File | Purpose |
|------|---------|
| `llm_cleaning_flow.py` | Core cleaning flow -- prompt management, multi-pass cleaning, result comparison |
| `prompt_templates.py` | Library of reusable prompt templates for different cleaning scenarios |
| `blueprints/llm_cleaning_bp.py` | Flask Blueprint for LLM cleaning routes |
| `tests/test_llm_cleaning.py` | Unit tests |

### Detailed Implementation Instructions

**1. `prompt_templates.py`**

```python
"""Library of prompt templates for LLM-based transcript cleaning."""

TEMPLATES = {
    "default": {
        "name": "Default Cleaning",
        "description": "Standard removal of editorial content",
        "template": "... (reuse/extend from llm_processor.py DEFAULT_PROMPT) ..."
    },
    "aggressive": {
        "name": "Aggressive Cleaning", 
        "description": "Remove all non-speech content including citations",
        "template": "..."
    },
    "minimal": {
        "name": "Minimal Cleaning",
        "description": "Only remove obvious editorial notes and timestamps",
        "template": "..."
    },
    "custom": {
        "name": "Custom",
        "description": "User-provided prompt",
        "template": "{user_prompt}\n\n---\n\n{document_text}"
    }
}

def get_template(name: str) -> dict:
    """Get a prompt template by name."""
    return TEMPLATES.get(name, TEMPLATES["default"])

def list_templates() -> list:
    """List all available templates with metadata."""
    return [{"id": k, **{kk: vv for kk, vv in v.items() if kk != "template"}} 
            for k, v in TEMPLATES.items()]
```

**2. `llm_cleaning_flow.py`**

This module builds on top of the existing `llm_processor.py` (imports and uses `process_with_llm`):

```python
"""LLM-based cleaning flow for Yiddish transcripts.

Extends the existing llm_processor module with:
- Multi-pass cleaning
- Prompt template library
- Side-by-side comparison with original
- Integration with diff_utils for change tracking
"""

from llm_processor import process_with_llm, get_available_providers
from diff_utils import generate_line_diff, get_diff_summary
from prompt_templates import get_template, list_templates
from typing import Optional, List


class LLMCleaningFlow:
    """Orchestrates LLM-based transcript cleaning."""
    
    def __init__(self, provider: str, api_key: str, model: str = None):
        self.provider = provider
        self.api_key = api_key
        self.model = model
    
    def clean(
        self,
        text: str,
        prompt_template: str = None,
        template_name: str = "default",
        custom_instructions: str = None,
        passes: int = 1
    ) -> dict:
        """
        Clean a transcript using LLM.
        
        Args:
            text: The transcript text to clean
            prompt_template: Full prompt template (overrides template_name)
            template_name: Named template from prompt_templates
            custom_instructions: Additional user instructions to prepend
            passes: Number of cleaning passes (1 is usually sufficient)
            
        Returns:
            dict with cleaned_text, diff, statistics, etc.
        """
        ...
    
    def clean_with_comparison(
        self,
        original_text: str,
        manual_transcript: str,
        prompt: str
    ) -> dict:
        """
        Clean a manual transcript and compare against original.
        Returns diff data for the diff viewer.
        """
        ...
    
    def multi_pass_clean(self, text: str, prompt: str, passes: int = 2) -> dict:
        """Run multiple cleaning passes, tracking changes at each step."""
        ...
```

**3. `blueprints/llm_cleaning_bp.py`**

```python
"""Flask Blueprint for LLM cleaning flow."""

from flask import Blueprint, request, jsonify

llm_cleaning_bp = Blueprint('llm_cleaning', __name__, url_prefix='/api/llm-clean')

@llm_cleaning_bp.route('/templates', methods=['GET'])
def get_templates():
    """List available prompt templates."""
    ...

@llm_cleaning_bp.route('/clean', methods=['POST'])
def clean_transcript():
    """Clean a transcript using LLM with selected template/prompt."""
    # Accepts: text, provider, api_key, model, template_name, custom_prompt, passes
    ...

@llm_cleaning_bp.route('/clean-file', methods=['POST'])
def clean_file():
    """Clean an uploaded document file using LLM."""
    ...

@llm_cleaning_bp.route('/compare', methods=['POST'])
def compare_results():
    """Compare original vs cleaned transcript, return diff."""
    ...
```

**4. Integration:**
- IMPORTS from existing `llm_processor.py` and `diff_utils.py` -- does NOT modify them
- Uses the same provider/model patterns already in the app
- Blueprint registration documented in a README or `register.py`
- The existing `/process-llm` route in `app.py` remains untouched

### Dependencies
No new dependencies (uses existing `openai`, `anthropic`, `google-generativeai`)

### Testing Approach
- Unit test prompt template selection and rendering
- Unit test multi-pass cleaning logic with mocked LLM responses
- Test diff generation between original and cleaned text
- Test custom prompt injection
- Test error handling for LLM failures

---

## Agent 4: Diff Viewer with Audio Playback

**Branch:** `feature/diff-viewer-audio`

### Purpose
Build a diff viewer that shows cleanup results row-by-row between original and cleaned transcripts, with audio playback synchronized to timestamps from Agent 2's transcription.

### Files to Create

| File | Purpose |
|------|---------|
| `diff_viewer.py` | Backend: merge diff data with timestamps, produce enriched diff rows |
| `blueprints/diff_viewer_bp.py` | Flask Blueprint for diff viewer API |
| `templates/diff_viewer.html` | Standalone page for the diff viewer with audio player |
| `static/diff_viewer.js` | Frontend JS for diff interaction and audio sync |
| `static/diff_viewer.css` | Styles for the diff viewer |
| `tests/test_diff_viewer.py` | Unit tests |

### Detailed Implementation Instructions

**1. `diff_viewer.py`**

```python
"""Diff viewer with timestamp-aligned audio playback.

Merges diff data (from diff_utils.py) with timestamp data (from YiddishLabs
transcription) to create an enriched diff view where each row can be clicked
to play the corresponding audio segment.
"""

from diff_utils import generate_line_diff, get_diff_summary
from typing import List, Dict, Optional
import json


class DiffViewerData:
    """Prepares diff data enriched with audio timestamps."""
    
    def __init__(
        self,
        original_text: str,
        cleaned_text: str,
        timestamps: List[dict] = None,
        audio_url: str = None
    ):
        self.original_text = original_text
        self.cleaned_text = cleaned_text
        self.timestamps = timestamps or []
        self.audio_url = audio_url
    
    def generate_enriched_diff(self) -> dict:
        """
        Generate diff data with timestamp information per row.
        
        Each diff row gets a timestamp_start and timestamp_end based on
        aligning the original text positions to the timestamp data.
        """
        diff_data = generate_line_diff(self.original_text, self.cleaned_text)
        
        if self.timestamps:
            diff_data['changes'] = self._align_timestamps_to_rows(
                diff_data['changes']
            )
        
        diff_data['audio_url'] = self.audio_url
        diff_data['has_timestamps'] = bool(self.timestamps)
        
        return diff_data
    
    def _align_timestamps_to_rows(self, rows: List[dict]) -> List[dict]:
        """
        Align timestamp data to diff rows.
        
        Strategy: Map character positions in the original text to timestamps.
        Each diff row covers a range of characters -> find corresponding time range.
        """
        # Build a character-position-to-time mapping from timestamps
        char_to_time = self._build_char_time_map()
        
        for row in rows:
            original_line = row.get('original', '')
            if original_line and char_to_time:
                # Find the time range for this line
                row['timestamp_start'] = self._find_start_time(original_line, char_to_time)
                row['timestamp_end'] = self._find_end_time(original_line, char_to_time)
            else:
                row['timestamp_start'] = None
                row['timestamp_end'] = None
        
        return rows
    
    def _build_char_time_map(self) -> dict:
        """Build mapping from character positions to timestamps."""
        ...
    
    def _find_start_time(self, line: str, char_map: dict) -> Optional[float]:
        ...
    
    def _find_end_time(self, line: str, char_map: dict) -> Optional[float]:
        ...
```

**2. `templates/diff_viewer.html`**

A standalone HTML page (NOT modifying `templates/index.html`) that provides:
- Side-by-side or unified diff view of original vs cleaned text
- Each row is clickable
- An HTML5 `<audio>` element at the top/bottom
- Clicking a row seeks the audio to the corresponding timestamp
- Color-coded rows: green=unchanged, red=removed, yellow=modified, blue=added
- RTL text support (Yiddish is right-to-left)
- Row highlighting when audio is playing

Key frontend logic:
```javascript
// When a diff row is clicked:
function playFromRow(row) {
    const audio = document.getElementById('audio-player');
    const startTime = parseFloat(row.dataset.timestampStart);
    if (!isNaN(startTime)) {
        audio.currentTime = startTime;
        audio.play();
        highlightRow(row);
    }
}

// Highlight current row based on audio time:
audio.addEventListener('timeupdate', () => {
    const currentTime = audio.currentTime;
    document.querySelectorAll('.diff-row').forEach(row => {
        const start = parseFloat(row.dataset.timestampStart);
        const end = parseFloat(row.dataset.timestampEnd);
        if (currentTime >= start && currentTime < end) {
            row.classList.add('active-row');
        } else {
            row.classList.remove('active-row');
        }
    });
});
```

**3. `blueprints/diff_viewer_bp.py`**

```python
"""Flask Blueprint for diff viewer with audio playback."""

from flask import Blueprint, request, jsonify, render_template

diff_viewer_bp = Blueprint('diff_viewer', __name__, url_prefix='/api/diff-viewer')

@diff_viewer_bp.route('/view', methods=['GET'])
def view_diff():
    """Render the diff viewer page."""
    return render_template('diff_viewer.html')

@diff_viewer_bp.route('/generate', methods=['POST'])
def generate_diff_with_audio():
    """Generate enriched diff data with audio timestamps."""
    # Accepts: original_text, cleaned_text, timestamps (JSON), audio_url
    ...

@diff_viewer_bp.route('/proxy-audio', methods=['GET'])
def proxy_audio():
    """Proxy audio from kol-yid platform (handles auth)."""
    # This may be needed if audio URLs require authentication
    ...
```

**4. Integration:**
- Uses existing `diff_utils.py` (imports `generate_line_diff`, `get_diff_summary`)
- Consumes timestamp data from Agent 2's transcription output (shared contract format)
- Serves its own template page at `/api/diff-viewer/view`
- Audio can come from: a URL (kol-yid platform), an uploaded file, or the browser's local file

### Dependencies
No new Python dependencies. Frontend uses vanilla JS + HTML5 Audio API.

### Testing Approach
- Unit test timestamp-to-row alignment algorithm
- Unit test with various diff scenarios (all unchanged, all removed, mixed)
- Test with mock timestamp data
- Test RTL text handling
- Manual browser testing for audio sync

---

## Agent 5: Stable-TS Alignment

**Branch:** `feature/stable-ts-alignment`

### Purpose
Build a flow for the stable-ts algorithm with a Yiddish Whisper model, producing word-level timestamps and optional confidence scoring.

### Files to Create

| File | Purpose |
|------|---------|
| `stable_ts_aligner.py` | Core alignment module using stable-ts with Whisper |
| `alignment_config.py` | Configuration: model paths, parameters, defaults |
| `blueprints/alignment_bp.py` | Flask Blueprint for alignment API routes |
| `tests/test_stable_ts.py` | Unit tests |

### Detailed Implementation Instructions

**1. `alignment_config.py`**

```python
"""Configuration for stable-ts alignment."""

import os

# Whisper model for Yiddish -- use a fine-tuned model if available
# Common options: "openai/whisper-large-v3", a HuggingFace Yiddish model, 
# or a local path to a fine-tuned model
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", None)  # Local model path

# stable-ts parameters
DEFAULT_ALIGNMENT_CONFIG = {
    "language": "yi",              # Yiddish
    "vad": True,                   # Voice Activity Detection
    "demucs": False,               # Audio source separation (slow but cleaner)
    "word_timestamps": True,       # Word-level timestamps
    "refine_whisper_precision": True,  # Refine timestamp precision
}

# Confidence scoring
ENABLE_CONFIDENCE = os.getenv("STABLE_TS_CONFIDENCE", "true").lower() == "true"
```

**2. `stable_ts_aligner.py`**

```python
"""Stable-TS alignment for Yiddish audio with Whisper model.

Uses stable-ts to produce highly accurate word-level timestamps
from audio files using a Yiddish-capable Whisper model.
"""

import os
import json
import logging
from typing import Optional, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class StableTSAligner:
    """Aligner using stable-ts for word-level timestamp generation."""
    
    def __init__(
        self,
        model_name: str = None,
        model_path: str = None,
        device: str = None
    ):
        """
        Initialize the aligner.
        
        Args:
            model_name: Whisper model name (e.g., "large-v3")
            model_path: Path to a local fine-tuned model
            device: "cuda" or "cpu" (auto-detected if None)
        """
        from alignment_config import WHISPER_MODEL, WHISPER_MODEL_PATH
        
        self.model_name = model_name or WHISPER_MODEL
        self.model_path = model_path or WHISPER_MODEL_PATH
        self.device = device or self._detect_device()
        self._model = None
    
    def _detect_device(self) -> str:
        """Detect available compute device."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    
    def _load_model(self):
        """Lazy-load the Whisper model via stable-ts."""
        if self._model is None:
            import stable_whisper
            
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading local model from {self.model_path}")
                self._model = stable_whisper.load_model(self.model_path, device=self.device)
            else:
                logger.info(f"Loading Whisper model: {self.model_name}")
                self._model = stable_whisper.load_model(self.model_name, device=self.device)
        
        return self._model
    
    def align(
        self,
        audio_path: str,
        text: str = None,
        language: str = "yi",
        word_timestamps: bool = True,
        include_confidence: bool = True
    ) -> dict:
        """
        Align audio with optional reference text.
        
        If text is provided, performs forced alignment.
        If text is None, performs transcription + alignment.
        
        Args:
            audio_path: Path to audio file
            text: Optional reference text for forced alignment
            language: Language code
            word_timestamps: Include word-level timestamps
            include_confidence: Include confidence scores per word
            
        Returns:
            dict following the shared Stable-TS alignment output format
        """
        model = self._load_model()
        
        if text:
            # Forced alignment mode
            result = model.align(audio_path, text, language=language)
        else:
            # Transcribe + align mode
            result = model.transcribe(
                audio_path,
                language=language,
                word_timestamps=word_timestamps,
                vad=True,
                regroup=True
            )
        
        return self._format_result(result, audio_path, language, include_confidence)
    
    def _format_result(self, result, audio_path: str, language: str, include_confidence: bool) -> dict:
        """Convert stable-ts result to shared contract format."""
        segments = []
        
        for segment in result.segments:
            seg_data = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": []
            }
            
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    word_data = {
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end,
                    }
                    if include_confidence and hasattr(word, 'probability'):
                        word_data["confidence"] = round(word.probability, 4)
                    else:
                        word_data["confidence"] = None
                    seg_data["words"].append(word_data)
            
            segments.append(seg_data)
        
        return {
            "audio_file": audio_path,
            "aligned_segments": segments,
            "model": self.model_name,
            "language": language,
            "full_text": result.text.strip() if hasattr(result, 'text') else "",
            "timestamps": self._flatten_to_shared_format(segments)
        }
    
    def _flatten_to_shared_format(self, segments: List[dict]) -> List[dict]:
        """Flatten segments to the shared timestamp format for cross-agent compatibility."""
        timestamps = []
        for seg in segments:
            if seg.get("words"):
                for w in seg["words"]:
                    timestamps.append({
                        "start": w["start"],
                        "end": w["end"],
                        "text": w["word"],
                        "confidence": w.get("confidence"),
                        "type": "word"
                    })
            else:
                timestamps.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "confidence": None,
                    "type": "segment"
                })
        return timestamps
    
    def align_batch(
        self,
        audio_text_pairs: List[Dict[str, str]],
        output_dir: str = "alignments"
    ) -> List[dict]:
        """
        Align multiple audio-text pairs.
        
        Args:
            audio_text_pairs: List of {"audio_path": ..., "text": ..., "id": ...}
            output_dir: Directory to save alignment results
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for pair in audio_text_pairs:
            try:
                result = self.align(
                    audio_path=pair["audio_path"],
                    text=pair.get("text"),
                    language=pair.get("language", "yi")
                )
                
                # Save result
                output_path = os.path.join(output_dir, f"alignment_{pair.get('id', 'unknown')}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                result["status"] = "success"
                results.append(result)
            except Exception as e:
                logger.error(f"Alignment failed for {pair.get('audio_path')}: {e}")
                results.append({
                    "audio_file": pair.get("audio_path"),
                    "status": "error",
                    "error": str(e)
                })
        
        return results
```

**3. `blueprints/alignment_bp.py`**

```python
"""Flask Blueprint for stable-ts alignment routes."""

from flask import Blueprint, request, jsonify
import os

alignment_bp = Blueprint('alignment', __name__, url_prefix='/api/alignment')

@alignment_bp.route('/align', methods=['POST'])
def align_audio():
    """Align an uploaded audio file, optionally with reference text."""
    # Accepts: audio file (multipart), text (optional), language, word_timestamps
    ...

@alignment_bp.route('/align-url', methods=['POST'])
def align_from_url():
    """Align audio from a URL with optional reference text."""
    ...

@alignment_bp.route('/batch', methods=['POST'])
def batch_align():
    """Batch align multiple audio-text pairs."""
    ...

@alignment_bp.route('/models', methods=['GET'])
def list_models():
    """List available Whisper models."""
    ...
```

**4. Integration:**
- Standalone module that can be used via CLI (`python stable_ts_aligner.py`) or Blueprint
- Outputs timestamps in the same shared contract format as Agent 2
- The `_flatten_to_shared_format` method ensures timestamps are compatible with Agent 4's diff viewer
- Model loading is lazy (only loads on first use) to avoid slow Flask startup

### Dependencies
```
stable-ts>=2.16.0
openai-whisper>=20231117
torch>=2.0.0
```

**Note:** These are heavy dependencies (several GB for Whisper models). The agent should:
- Add them to a separate `requirements-alignment.txt` (not the main `requirements.txt`)
- Document GPU requirements and fallback to CPU
- Consider making this an optional module with graceful import failure

### Testing Approach
- Unit test result formatting with mock stable-ts output objects
- Unit test device detection
- Unit test the flatten-to-shared-format conversion
- Test graceful failure when stable-ts/torch not installed
- Integration test with a small audio sample (if GPU available)
- Test forced alignment vs transcription-only mode

---

## Blueprint Registration Guide

Each agent creates a Flask Blueprint in `blueprints/`. After all agents finish, a final integration step registers them in `app.py`. Each agent should document registration in their Blueprint file:

```python
# To register this Blueprint, add to app.py:
#
#   from blueprints.transcription_bp import transcription_bp
#   app.register_blueprint(transcription_bp)
```

The `blueprints/` directory needs an `__init__.py`:

```python
"""Flask Blueprints for feature modules.

Register Blueprints in app.py:

    from blueprints.transcription_bp import transcription_bp
    from blueprints.llm_cleaning_bp import llm_cleaning_bp
    from blueprints.diff_viewer_bp import diff_viewer_bp
    from blueprints.alignment_bp import alignment_bp

    app.register_blueprint(transcription_bp)
    app.register_blueprint(llm_cleaning_bp)
    app.register_blueprint(diff_viewer_bp)
    app.register_blueprint(alignment_bp)
"""
```

Each agent should create this `blueprints/__init__.py` if it does not exist, but only add their own Blueprint's import -- the integration step merges them.

---

## Directory Structure After All Agents Complete

```
clean-yiddish-transcripts/
├── app.py                          # Existing (modified only during integration)
├── audio_text_mapper.py            # Agent 1: Mapping logic
├── mapping_config.py               # Agent 1: Mapping configuration
├── yiddishlabs_client.py           # Agent 2: API client
├── transcription_manager.py        # Agent 2: Batch transcription
├── llm_cleaning_flow.py            # Agent 3: LLM cleaning orchestration
├── prompt_templates.py             # Agent 3: Prompt template library
├── diff_viewer.py                  # Agent 4: Diff + timestamp alignment
├── stable_ts_aligner.py            # Agent 5: Stable-TS alignment
├── alignment_config.py             # Agent 5: Alignment configuration
├── blueprints/
│   ├── __init__.py
│   ├── transcription_bp.py         # Agent 2
│   ├── llm_cleaning_bp.py          # Agent 3
│   ├── diff_viewer_bp.py           # Agent 4
│   └── alignment_bp.py             # Agent 5
├── mappings/
│   ├── README.md                   # Agent 1
│   ├── audio_text_mappings.json    # Agent 1: Output
│   └── eval_set.json               # Agent 1: Eval exclusions
├── transcriptions/                  # Agent 2: Output dir
├── alignments/                      # Agent 5: Output dir
├── static/
│   ├── diff_viewer.js              # Agent 4
│   └── diff_viewer.css             # Agent 4
├── templates/
│   ├── index.html                  # Existing (untouched)
│   └── diff_viewer.html            # Agent 4
├── tests/
│   ├── test_mapper.py              # Agent 1
│   ├── test_yiddishlabs.py         # Agent 2
│   ├── test_llm_cleaning.py        # Agent 3
│   ├── test_diff_viewer.py         # Agent 4
│   └── test_stable_ts.py           # Agent 5
├── requirements.txt                 # Existing
├── requirements-alignment.txt       # Agent 5: Heavy ML deps
├── cleaner.py                       # Existing (untouched)
├── llm_processor.py                 # Existing (untouched, Agent 3 imports from it)
├── diff_utils.py                    # Existing (untouched, Agent 4 imports from it)
└── ... (other existing files)
```

---

## Environment Variables to Add (`.env.example`)

```bash
# Agent 1: Audio-Text Mapper
KOL_YID_BASE_URL=https://kol-yid.on-forge.com
KOL_YID_AUTH_TOKEN=

# Agent 2: YiddishLabs Transcription
YIDDISHLABS_API_KEY=
YIDDISHLABS_BASE_URL=https://api.yiddishlabs.com

# Agent 5: Stable-TS Alignment
WHISPER_MODEL=large-v3
WHISPER_MODEL_PATH=
STABLE_TS_CONFIDENCE=true
```

---

## Risks & Mitigations

| Risk | Impact | Agent | Mitigation |
|------|--------|-------|------------|
| kol-yid sites require auth, no public API | High | Agent 1 | Build fallback manual-entry mode; document auth requirements |
| YiddishLabs API format unknown | Medium | Agent 2 | Build flexible response parser; use adapter pattern |
| Merge conflicts in `app.py` | Medium | All | Use Blueprints, register only during final integration |
| Stable-TS + Whisper are heavy deps | Medium | Agent 5 | Separate requirements file; lazy loading; document GPU needs |
| Timestamp alignment accuracy | Medium | Agent 4 | Build tolerance/fuzzy matching; handle missing timestamps gracefully |
| Yiddish RTL text rendering in diff | Low | Agent 4 | Use `dir="rtl"` on HTML elements; test with actual Yiddish text |
| LLM rate limits during batch cleaning | Low | Agent 3 | Add retry logic with exponential backoff |

---

## Success Criteria

1. **Agent 1:** Produces a valid `audio_text_mappings.json` with ~50 hours of content, eval set excluded, distribution statistics show coverage across years and months
2. **Agent 2:** Can transcribe a single audio file via YiddishLabs API and return timestamped results in shared contract format
3. **Agent 3:** User can select a prompt template, submit transcript text, and receive cleaned output with diff comparison
4. **Agent 4:** Diff viewer renders correctly with RTL Yiddish text, clicking a row seeks audio to correct timestamp
5. **Agent 5:** Can process an audio file through stable-ts and produce word-level timestamps with confidence scores

---

## Resolved Questions

- [x] **YiddishLabs API base URL**: `https://app.yiddishlabs.com` — API docs at `/developer/api-docs` (JS-rendered SPA)
- [x] **kol-yid platform**: Same codebase as this repo, deployed to that domain. Use the same logic/patterns.
- [x] **Yiddish Whisper model**: `ivrit-ai/yi-whisper-large-v3` on HuggingFace — fine-tuned from openai/whisper-large-v3, Apache-2.0, 2B params. MUST set language="yi" explicitly (language detection degraded). Translation task NOT supported.
- [x] **Diff viewer**: Integrate into existing `index.html` tabs unless too complex, then separate page.
- [ ] What is the expected audio format on the kol-yid platform? (MP3? WAV? Affects download/transcription pipeline)
