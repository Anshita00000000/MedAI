"""
tests/test_demo.py

Tests for the Streamlit demo helper functions (src/demo/app.py).

Streamlit's widget calls cannot run outside a browser session, so we test
only the pure-Python utility functions that contain business logic:

- ``_parse_transcript()``  — DOCTOR/PATIENT turn parsing
- ``SAMPLE_CASES``         — all five sample transcripts load and parse
- ``_api_headers()``       — header construction with/without key
- ``_run_pipeline()``      — API call wrapper (mocked with responses)
- Session-state defaults   — ``_STATE_DEFAULTS`` has expected keys
- ``_load_file_bytes()``   — file reading helper
- ``_check_health()``      — health probe (mocked)

Run with::

    pytest tests/test_demo.py -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Repo root on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Patch streamlit before importing app ─────────────────────────────────────
# The demo module calls st.set_page_config() and widget functions at import
# time.  We replace the entire streamlit module with a MagicMock so the
# module can be imported without a running Streamlit session.

import types

def _make_ctx() -> MagicMock:
    """Return a context-manager MagicMock (used for st.columns, st.expander, etc.)."""
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    return m


_st_mock = MagicMock()
# set_page_config must not raise
_st_mock.set_page_config = MagicMock(return_value=None)
# session_state behaves like a real dict
_st_mock.session_state = {}
# buttons always return False so no if-body widget code is executed
_st_mock.button = MagicMock(return_value=False)
# text_input / text_area return empty strings
_st_mock.text_input = MagicMock(return_value="")
_st_mock.text_area = MagicMock(return_value="")
# selectbox returns a valid first key (avoids KeyError on SAMPLE_CASES lookup)
_st_mock.selectbox = MagicMock(return_value="Case 1: Hypertension Follow-up")
# checkbox / radio return sensible defaults
_st_mock.checkbox = MagicMock(return_value=True)
_st_mock.radio = MagicMock(return_value="Template Fallback")
# progress / empty / spinner are no-ops
_st_mock.progress = MagicMock(return_value=_make_ctx())
_st_mock.empty = MagicMock(return_value=_make_ctx())
_st_mock.spinner = MagicMock(return_value=_make_ctx())
_st_mock.expander = MagicMock(return_value=_make_ctx())
_st_mock.container = MagicMock(return_value=_make_ctx())
# columns: side_effect returns a tuple of n context managers
def _columns_side_effect(*args, **kwargs):
    n = args[0] if args else 2
    if not isinstance(n, int):
        n = 2
    return tuple(_make_ctx() for _ in range(n))
_st_mock.columns.side_effect = _columns_side_effect
# tabs returns three context managers
_st_mock.tabs.return_value = (_make_ctx(), _make_ctx(), _make_ctx())
# sidebar is a context manager
_st_mock.sidebar = _make_ctx()

sys.modules["streamlit"] = _st_mock
sys.modules["streamlit.components"] = MagicMock()
sys.modules["streamlit.components.v1"] = MagicMock()

# Now safe to import the demo utilities
from medai.src.demo.app import (  # noqa: E402
    SAMPLE_CASES,
    _STATE_DEFAULTS,
    _parse_transcript,
    _api_headers,
    _run_pipeline,
    _load_file_bytes,
    _check_health,
)


# ---------------------------------------------------------------------------
# _parse_transcript
# ---------------------------------------------------------------------------

class TestParseTranscript(unittest.TestCase):
    """Tests for the transcript parser."""

    def test_simple_two_turn(self):
        """Parses a minimal DOCTOR/PATIENT exchange."""
        text = "DOCTOR: Hello.\nPATIENT: Hi doctor."
        turns = _parse_transcript(text)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["speaker"], "DOCTOR")
        self.assertEqual(turns[0]["text"], "Hello.")
        self.assertEqual(turns[1]["speaker"], "PATIENT")
        self.assertEqual(turns[1]["text"], "Hi doctor.")

    def test_case_insensitive_labels(self):
        """Labels are matched case-insensitively."""
        text = "Doctor: How are you?\npatient: Fine."
        turns = _parse_transcript(text)
        self.assertEqual(turns[0]["speaker"], "DOCTOR")
        self.assertEqual(turns[1]["speaker"], "PATIENT")

    def test_continuation_lines_appended(self):
        """Lines without a speaker label are appended to the previous turn."""
        text = "DOCTOR: Hello.\ncontinued here.\nPATIENT: Yes."
        turns = _parse_transcript(text)
        self.assertEqual(len(turns), 2)
        self.assertIn("continued here", turns[0]["text"])

    def test_blank_lines_ignored(self):
        """Blank lines do not create empty turns."""
        text = "DOCTOR: Hello.\n\n\nPATIENT: Hi."
        turns = _parse_transcript(text)
        self.assertEqual(len(turns), 2)

    def test_empty_string_returns_empty(self):
        """Empty input returns an empty list."""
        self.assertEqual(_parse_transcript(""), [])

    def test_no_speaker_labels_returns_empty(self):
        """Text with no recognised labels returns an empty list."""
        self.assertEqual(_parse_transcript("Hello, how are you?"), [])

    def test_multi_turn_conversation(self):
        """Multi-turn conversation is parsed in order."""
        lines = [
            "DOCTOR: Question one?",
            "PATIENT: Answer one.",
            "DOCTOR: Question two?",
            "PATIENT: Answer two.",
        ]
        turns = _parse_transcript("\n".join(lines))
        self.assertEqual(len(turns), 4)
        speakers = [t["speaker"] for t in turns]
        self.assertEqual(speakers, ["DOCTOR", "PATIENT", "DOCTOR", "PATIENT"])

    def test_turn_text_stripped(self):
        """Leading/trailing whitespace in turn text is stripped."""
        text = "DOCTOR:   spaces around   \nPATIENT: ok."
        turns = _parse_transcript(text)
        self.assertEqual(turns[0]["text"], "spaces around")


# ---------------------------------------------------------------------------
# SAMPLE_CASES
# ---------------------------------------------------------------------------

class TestSampleCases(unittest.TestCase):
    """Tests that all five sample transcripts load and parse correctly."""

    def test_five_cases_defined(self):
        """Exactly five sample cases are defined."""
        self.assertEqual(len(SAMPLE_CASES), 5)

    def test_all_cases_have_non_empty_text(self):
        """Every sample case has non-empty transcript text."""
        for name, text in SAMPLE_CASES.items():
            self.assertGreater(len(text.strip()), 0, f"Empty transcript for: {name}")

    def test_all_cases_parse_to_turns(self):
        """Every sample case parses to at least 2 turns."""
        for name, text in SAMPLE_CASES.items():
            turns = _parse_transcript(text)
            self.assertGreaterEqual(len(turns), 2, f"Too few turns for: {name}")

    def test_all_cases_have_doctor_and_patient(self):
        """Every sample has both DOCTOR and PATIENT turns."""
        for name, text in SAMPLE_CASES.items():
            turns = _parse_transcript(text)
            speakers = {t["speaker"] for t in turns}
            self.assertIn("DOCTOR", speakers, f"No DOCTOR turns in: {name}")
            self.assertIn("PATIENT", speakers, f"No PATIENT turns in: {name}")

    def test_case1_is_hypertension(self):
        """Case 1 contains hypertension-related content."""
        text = SAMPLE_CASES["Case 1: Hypertension Follow-up"]
        self.assertIn("blood pressure", text.lower())

    def test_case2_is_migraine(self):
        """Case 2 contains migraine-related content."""
        text = SAMPLE_CASES["Case 2: Migraine Presentation"]
        self.assertIn("headache", text.lower())

    def test_case3_is_respiratory(self):
        """Case 3 contains respiratory-related content."""
        text = SAMPLE_CASES["Case 3: Respiratory Complaint"]
        self.assertIn("cough", text.lower())

    def test_case4_is_orthopaedic(self):
        """Case 4 contains knee/orthopaedic content."""
        text = SAMPLE_CASES["Case 4: Knee Pain (Orthopaedic)"]
        self.assertIn("knee", text.lower())

    def test_case5_is_urinary(self):
        """Case 5 contains urinary symptom content."""
        text = SAMPLE_CASES["Case 5: Urinary Symptoms"]
        self.assertIn("urin", text.lower())


# ---------------------------------------------------------------------------
# _STATE_DEFAULTS  (session state initialisation)
# ---------------------------------------------------------------------------

class TestStateDefaults(unittest.TestCase):
    """Tests that session state initialises with the expected keys."""

    _REQUIRED_KEYS = {
        "pipeline_result",
        "transcript_text",
        "patient_name",
        "patient_dob",
        "patient_mrn",
        "generate_pdf",
        "generate_fhir",
        "last_api_url",
        "fhir_bundle",
        "reference_note",
    }

    def test_all_required_keys_present(self):
        """_STATE_DEFAULTS contains all expected session-state keys."""
        self.assertTrue(
            self._REQUIRED_KEYS.issubset(_STATE_DEFAULTS.keys()),
            f"Missing keys: {self._REQUIRED_KEYS - _STATE_DEFAULTS.keys()}",
        )

    def test_pipeline_result_default_is_none(self):
        """pipeline_result defaults to None (no result before first run)."""
        self.assertIsNone(_STATE_DEFAULTS["pipeline_result"])

    def test_transcript_text_default_is_empty(self):
        """transcript_text defaults to empty string."""
        self.assertEqual(_STATE_DEFAULTS["transcript_text"], "")

    def test_generate_pdf_default_true(self):
        """generate_pdf defaults to True."""
        self.assertTrue(_STATE_DEFAULTS["generate_pdf"])

    def test_generate_fhir_default_true(self):
        """generate_fhir defaults to True."""
        self.assertTrue(_STATE_DEFAULTS["generate_fhir"])

    def test_last_api_url_default(self):
        """last_api_url defaults to localhost:8000."""
        self.assertIn("8000", _STATE_DEFAULTS["last_api_url"])


# ---------------------------------------------------------------------------
# _api_headers
# ---------------------------------------------------------------------------

class TestApiHeaders(unittest.TestCase):
    """Tests for _api_headers()."""

    def test_content_type_always_present(self):
        """Content-Type header is always included."""
        h = _api_headers("")
        self.assertEqual(h["Content-Type"], "application/json")

    def test_no_api_key_header_when_empty(self):
        """X-API-Key header absent when key is empty."""
        self.assertNotIn("X-API-Key", _api_headers(""))

    def test_no_api_key_header_when_whitespace(self):
        """X-API-Key header absent when key is only whitespace."""
        self.assertNotIn("X-API-Key", _api_headers("   "))

    def test_api_key_header_added_when_set(self):
        """X-API-Key header added when key is non-empty."""
        h = _api_headers("secret-key")
        self.assertEqual(h["X-API-Key"], "secret-key")

    def test_api_key_stripped(self):
        """Leading/trailing whitespace stripped from API key."""
        h = _api_headers("  trimmed  ")
        self.assertEqual(h["X-API-Key"], "trimmed")


# ---------------------------------------------------------------------------
# _run_pipeline  (mocked requests)
# ---------------------------------------------------------------------------

class TestRunPipeline(unittest.TestCase):
    """Tests for _run_pipeline() with mocked HTTP calls."""

    _TURNS = [
        {"speaker": "DOCTOR", "text": "How are you?"},
        {"speaker": "PATIENT", "text": "Fine thanks."},
    ]

    def _mock_response(self, status: int = 200, json_data: dict | None = None):
        """Build a mock requests.Response."""
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.json.return_value = json_data or {"status": "success"}
        if status >= 400:
            from requests.exceptions import HTTPError
            mock_resp.raise_for_status.side_effect = HTTPError(
                response=mock_resp
            )
        else:
            mock_resp.raise_for_status.return_value = None
        return mock_resp

    @patch("medai.src.demo.app.requests.post")
    def test_success_returns_result_dict(self, mock_post):
        """Successful API call returns (result_dict, '') ."""
        mock_post.return_value = self._mock_response(
            200, {"conversation_id": "c1", "status": "success"}
        )
        result, err = _run_pipeline("http://localhost:8000", "", self._TURNS, None, True, True, False)
        self.assertIsNotNone(result)
        self.assertEqual(err, "")
        self.assertEqual(result["status"], "success")

    @patch("medai.src.demo.app.requests.post")
    def test_connection_error_returns_message(self, mock_post):
        """ConnectionError returns (None, user-friendly error message)."""
        import requests as _requests
        mock_post.side_effect = _requests.exceptions.ConnectionError()
        result, err = _run_pipeline("http://bad-host", "", self._TURNS, None, True, True, False)
        self.assertIsNone(result)
        self.assertIn("Cannot reach", err)

    @patch("medai.src.demo.app.requests.post")
    def test_timeout_returns_message(self, mock_post):
        """Timeout returns (None, timeout error message)."""
        import requests as _requests
        mock_post.side_effect = _requests.exceptions.Timeout()
        result, err = _run_pipeline("http://localhost:8000", "", self._TURNS, None, True, True, False)
        self.assertIsNone(result)
        self.assertIn("timed out", err)

    @patch("medai.src.demo.app.requests.post")
    def test_http_500_returns_error_message(self, mock_post):
        """HTTP 500 from API returns (None, error message with status code)."""
        mock_post.return_value = self._mock_response(500, {"detail": "Internal error"})
        result, err = _run_pipeline("http://localhost:8000", "", self._TURNS, None, True, True, False)
        self.assertIsNone(result)
        self.assertIn("500", err)

    @patch("medai.src.demo.app.requests.post")
    def test_http_401_returns_error_message(self, mock_post):
        """HTTP 401 from API returns (None, error message with status code)."""
        mock_post.return_value = self._mock_response(401, {"detail": "Unauthorized"})
        result, err = _run_pipeline("http://localhost:8000", "", self._TURNS, None, True, True, False)
        self.assertIsNone(result)
        self.assertIn("401", err)

    @patch("medai.src.demo.app.requests.post")
    def test_api_key_passed_in_header(self, mock_post):
        """API key is included in the request headers."""
        mock_post.return_value = self._mock_response(200, {"status": "success"})
        _run_pipeline("http://localhost:8000", "my-key", self._TURNS, None, True, True, False)
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"]["X-API-Key"], "my-key")


# ---------------------------------------------------------------------------
# _check_health  (mocked requests)
# ---------------------------------------------------------------------------

class TestCheckHealth(unittest.TestCase):
    """Tests for _check_health()."""

    @patch("medai.src.demo.app.requests.get")
    def test_returns_dict_on_success(self, mock_get):
        """Returns parsed JSON dict on HTTP 200."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_resp
        result = _check_health("http://localhost:8000", "")
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "healthy")

    @patch("medai.src.demo.app.requests.get")
    def test_returns_none_on_connection_error(self, mock_get):
        """Returns None when the API server is unreachable."""
        import requests as _requests
        mock_get.side_effect = _requests.exceptions.ConnectionError()
        self.assertIsNone(_check_health("http://bad-host", ""))

    @patch("medai.src.demo.app.requests.get")
    def test_returns_none_on_timeout(self, mock_get):
        """Returns None when the request times out."""
        import requests as _requests
        mock_get.side_effect = _requests.exceptions.Timeout()
        self.assertIsNone(_check_health("http://localhost:8000", ""))


# ---------------------------------------------------------------------------
# _load_file_bytes
# ---------------------------------------------------------------------------

class TestLoadFileBytes(unittest.TestCase):
    """Tests for _load_file_bytes()."""

    def test_reads_existing_file(self):
        """Returns bytes for an existing file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello medai")
            tmp_path = f.name
        result = _load_file_bytes(tmp_path)
        self.assertEqual(result, b"hello medai")
        Path(tmp_path).unlink(missing_ok=True)

    def test_returns_none_for_missing_file(self):
        """Returns None for a non-existent path."""
        result = _load_file_bytes("/tmp/definitely_does_not_exist_xyz.pdf")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
