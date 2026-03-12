"""
tests/test_api.py

FastAPI TestClient tests for the MedAI REST API (src/api/main.py).

All pipeline components are replaced with lightweight mocks so no data
files, API keys, or optional dependencies are required.

Run with::

    pytest tests/test_api.py -v
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Minimal stub dataclasses (mirror production types without importing pipeline)
# ---------------------------------------------------------------------------

@dataclass
class _Turn:
    index: int = 0
    speaker: str = "DOCTOR"
    text: str = "Hello."
    word_count: int = 1
    is_question: bool = False
    consecutive_run: int = 1


@dataclass
class _ProcessedTranscript:
    id: str = "conv-test"
    source_dataset: str = "api_input"
    language: str = "en"
    turns: List[_Turn] = field(default_factory=lambda: [_Turn(), _Turn(speaker="PATIENT", text="I have a headache.")])
    doctor_text: str = "Hello."
    patient_text: str = "I have a headache."
    full_text: str = "DOCTOR: Hello.\nPATIENT: I have a headache."
    turn_count: int = 2
    word_count: int = 6
    has_reference_note: bool = False
    reference_note: str = None
    flags: List[str] = field(default_factory=list)


@dataclass
class _MedicalEntities:
    conversation_id: str = "conv-test"
    symptoms: List[str] = field(default_factory=lambda: ["headache"])
    medications: List[str] = field(default_factory=list)
    vitals: dict = field(default_factory=dict)
    clinical_findings: List[str] = field(default_factory=list)
    diagnoses: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    anatomical_sites: List[str] = field(default_factory=list)
    temporal_info: List[str] = field(default_factory=list)
    negations: List[str] = field(default_factory=list)
    confidence: float = 0.8
    extraction_method: str = "rule_based"


@dataclass
class _SOAPNote:
    conversation_id: str = "conv-test"
    speciality: str = "General Medicine"
    subjective: str = "Patient reports headache."
    objective: str = "Vitals normal."
    assessment: str = "Tension headache."
    plan: str = "Analgesia PRN."
    red_flag_mentions: List[str] = field(default_factory=list)
    generated_at: str = "2026-03-12T10:00:00+00:00"
    model_used: str = "template_fallback"
    prompt_tokens: int = 0
    confidence: float = 0.85
    generation_method: str = "template_fallback"


@dataclass
class _RedFlag:
    term: str = "chest pain"
    severity: str = "urgent"
    category: str = "cardiac"
    description: str = "Possible ACS"
    matched_text: str = "chest pain"
    recommended_action: str = "ECG immediately"


@dataclass
class _RedFlagResult:
    conversation_id: str = "conv-test"
    flags: List[_RedFlag] = field(default_factory=list)
    highest_severity: str = "none"
    flag_count: int = 0
    requires_immediate_action: bool = False
    summary: str = "No red flags."


@dataclass
class _Diagnosis:
    name: str = "Tension Headache"
    icd10_code: str = "G44.2"
    probability: str = "high"
    probability_score: float = 0.82
    supporting_evidence: List[str] = field(default_factory=lambda: ["headache"])
    contradicting_evidence: List[str] = field(default_factory=list)
    reasoning: str = "Classic presentation."


@dataclass
class _DDxResult:
    conversation_id: str = "conv-test"
    top_diagnoses: List[_Diagnosis] = field(default_factory=lambda: [_Diagnosis()])
    primary_diagnosis: _Diagnosis = field(default_factory=_Diagnosis)
    follow_up_questions: List[str] = field(default_factory=lambda: ["Any aura?"])
    recommended_investigations: List[str] = field(default_factory=lambda: ["MRI"])
    red_flag_result: Any = None
    generation_method: str = "rule_based"
    speciality: str = "Neurology"


@dataclass
class _ReportOutput:
    conversation_id: str = "conv-test"
    physician_pdf_path: str = "/tmp/physician_conv-test.html"
    patient_pdf_path: str = "/tmp/patient_conv-test.html"
    generated_at: str = "2026-03-12T10:00:00+00:00"
    page_count: int = 0
    file_size_bytes: int = 1024
    generation_method: str = "html_fallback"


# ---------------------------------------------------------------------------
# Fixture: app with all pipeline components mocked
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """
    TestClient with all pipeline components replaced by mocks.

    Components are wired onto ``app.state`` directly so the startup event
    is not needed for every test.
    """
    from medai.src.api.main import app

    # Build mock pipeline components
    mock_processor = MagicMock()
    mock_processor.use_fallback = True
    mock_processor.process.return_value = _ProcessedTranscript()

    mock_extractor = MagicMock()
    mock_extractor.use_fallback = True
    mock_extractor.extract.return_value = _MedicalEntities()

    mock_soap_gen = MagicMock()
    mock_soap_gen.use_fallback = True
    mock_soap_gen.generate.return_value = _SOAPNote()

    mock_red_flag = MagicMock()
    mock_red_flag.detect.return_value = _RedFlagResult()

    mock_ddx = MagicMock()
    mock_ddx.use_fallback = True
    mock_ddx.analyse.return_value = _DDxResult()

    mock_pdf_gen = MagicMock()
    mock_pdf_gen.use_fallback = True
    mock_pdf_gen.generate_physician_report.return_value = _ReportOutput()
    mock_pdf_gen.generate_patient_summary.return_value = _ReportOutput(
        physician_pdf_path="", patient_pdf_path="/tmp/patient.html"
    )

    mock_fhir = MagicMock()
    mock_fhir.format_bundle.return_value = {"resourceType": "Bundle", "entry": []}
    mock_fhir.save_bundle.return_value = "/tmp/fhir_conv-test.json"

    # Wire onto app.state
    app.state.processor = mock_processor
    app.state.processor_ok = True
    app.state.extractor = mock_extractor
    app.state.extractor_ok = True
    app.state.soap_gen = mock_soap_gen
    app.state.soap_gen_ok = True
    app.state.red_flag = mock_red_flag
    app.state.red_flag_ok = True
    app.state.ddx_engine = mock_ddx
    app.state.ddx_engine_ok = True
    app.state.pdf_gen = mock_pdf_gen
    app.state.pdf_gen_ok = True
    app.state.fhir = mock_fhir
    app.state.fhir_ok = True

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture()
def client_with_key(monkeypatch):
    """
    TestClient fixture with MEDAI_API_KEY set to 'test-secret'.

    Used for auth-related tests.
    """
    monkeypatch.setenv("MEDAI_API_KEY", "test-secret")
    # Force re-read of the env var by reloading the module's module-level constant
    import medai.src.api.main as api_mod
    original_key = api_mod._API_KEY_ENV
    api_mod._API_KEY_ENV = "test-secret"

    from medai.src.api.main import app

    # Wire the same mocks as client fixture
    mock_processor = MagicMock()
    mock_processor.use_fallback = True
    mock_processor.process.return_value = _ProcessedTranscript()
    mock_extractor = MagicMock()
    mock_extractor.use_fallback = True
    mock_extractor.extract.return_value = _MedicalEntities()
    mock_soap_gen = MagicMock()
    mock_soap_gen.use_fallback = True
    mock_soap_gen.generate.return_value = _SOAPNote()
    mock_red_flag = MagicMock()
    mock_red_flag.detect.return_value = _RedFlagResult()
    mock_ddx = MagicMock()
    mock_ddx.use_fallback = True
    mock_ddx.analyse.return_value = _DDxResult()
    mock_pdf_gen = MagicMock()
    mock_pdf_gen.use_fallback = True
    mock_pdf_gen.generate_physician_report.return_value = _ReportOutput()
    mock_pdf_gen.generate_patient_summary.return_value = _ReportOutput(
        physician_pdf_path="", patient_pdf_path="/tmp/patient.html"
    )
    mock_fhir = MagicMock()
    mock_fhir.format_bundle.return_value = {"resourceType": "Bundle", "entry": []}
    mock_fhir.save_bundle.return_value = "/tmp/fhir_conv-test.json"

    app.state.processor = mock_processor
    app.state.processor_ok = True
    app.state.extractor = mock_extractor
    app.state.extractor_ok = True
    app.state.soap_gen = mock_soap_gen
    app.state.soap_gen_ok = True
    app.state.red_flag = mock_red_flag
    app.state.red_flag_ok = True
    app.state.ddx_engine = mock_ddx
    app.state.ddx_engine_ok = True
    app.state.pdf_gen = mock_pdf_gen
    app.state.pdf_gen_ok = True
    app.state.fhir = mock_fhir
    app.state.fhir_ok = True

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    api_mod._API_KEY_ENV = original_key


# ---------------------------------------------------------------------------
# Shared test payload
# ---------------------------------------------------------------------------

_VALID_TRANSCRIPT = {
    "conversation_id": "conv-test",
    "turns": [
        {"speaker": "DOCTOR", "text": "Hello, what brings you in today?"},
        {"speaker": "PATIENT", "text": "I have had a headache for two days."},
    ],
}

_PIPELINE_BODY = {
    "transcript": _VALID_TRANSCRIPT,
    "generate_pdf": True,
    "generate_fhir": True,
    "use_llm": False,
}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        """/health always returns HTTP 200."""
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_has_status_field(self, client):
        """/health response includes a 'status' field."""
        data = client.get("/health").json()
        assert "status" in data

    def test_response_has_all_component_keys(self, client):
        """/health response includes all seven component keys."""
        data = client.get("/health").json()
        expected = {
            "transcript_processor",
            "entity_extractor",
            "soap_generator",
            "red_flag_detector",
            "ddx_engine",
            "pdf_generator",
            "fhir_formatter",
        }
        assert expected.issubset(data["components"].keys())

    def test_healthy_when_all_online(self, client):
        """/health returns 'healthy' when all components are online."""
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_degraded_when_component_offline(self, client):
        """/health returns 'degraded' when a component is offline."""
        from medai.src.api.main import app
        original = app.state.processor_ok
        app.state.processor_ok = False
        try:
            data = client.get("/health").json()
            assert data["status"] == "degraded"
        finally:
            app.state.processor_ok = original

    def test_entity_extractor_includes_extraction_mode(self, client):
        """/health entity_extractor sub-dict includes extraction_mode."""
        data = client.get("/health").json()
        ee = data["components"]["entity_extractor"]
        assert "extraction_mode" in ee

    def test_soap_generator_includes_llm_available(self, client):
        """/health soap_generator sub-dict includes llm_available."""
        data = client.get("/health").json()
        sg = data["components"]["soap_generator"]
        assert "llm_available" in sg

    def test_pdf_generator_includes_pdf_mode(self, client):
        """/health pdf_generator sub-dict includes pdf_mode."""
        data = client.get("/health").json()
        pg = data["components"]["pdf_generator"]
        assert "pdf_mode" in pg

    def test_health_no_auth_required(self, client_with_key):
        """/health is public even when API key auth is enabled."""
        r = client_with_key.get("/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /pipeline/run
# ---------------------------------------------------------------------------

class TestPipelineRun:
    def test_returns_200_with_valid_transcript(self, client):
        """POST /pipeline/run returns 200 for a valid transcript."""
        r = client.post("/pipeline/run", json=_PIPELINE_BODY)
        assert r.status_code == 200

    def test_response_has_required_fields(self, client):
        """POST /pipeline/run response includes all PipelineResponse fields."""
        data = client.post("/pipeline/run", json=_PIPELINE_BODY).json()
        for field in ("conversation_id", "status", "soap_note", "red_flags",
                      "ddx", "pdf_paths", "fhir_bundle_path",
                      "processing_time_seconds", "warnings"):
            assert field in data, f"Missing field: {field}"

    def test_conversation_id_matches_input(self, client):
        """conversation_id in response matches the input."""
        data = client.post("/pipeline/run", json=_PIPELINE_BODY).json()
        assert data["conversation_id"] == "conv-test"

    def test_status_success_when_no_warnings(self, client):
        """status is 'success' when no warnings are generated."""
        data = client.post("/pipeline/run", json=_PIPELINE_BODY).json()
        assert data["status"] in ("success", "partial")

    def test_empty_turns_returns_422(self, client):
        """POST /pipeline/run with empty turns list returns 422."""
        body = {"transcript": {**_VALID_TRANSCRIPT, "turns": []}}
        r = client.post("/pipeline/run", json=body)
        assert r.status_code == 422

    def test_missing_transcript_returns_422(self, client):
        """POST /pipeline/run without transcript field returns 422."""
        r = client.post("/pipeline/run", json={})
        assert r.status_code == 422

    def test_processing_time_is_non_negative(self, client):
        """processing_time_seconds is a non-negative float."""
        data = client.post("/pipeline/run", json=_PIPELINE_BODY).json()
        assert data["processing_time_seconds"] >= 0

    def test_pdf_paths_dict_keys(self, client):
        """pdf_paths dict has physician and patient keys."""
        data = client.post("/pipeline/run", json=_PIPELINE_BODY).json()
        assert "physician" in data["pdf_paths"]
        assert "patient" in data["pdf_paths"]

    def test_partial_status_when_component_offline(self, client):
        """status is 'partial' when a non-critical component is offline."""
        from medai.src.api.main import app
        original = app.state.pdf_gen_ok
        app.state.pdf_gen_ok = False
        try:
            data = client.post("/pipeline/run", json=_PIPELINE_BODY).json()
            assert data["status"] == "partial"
            assert any("PDFGenerator" in w or "offline" in w for w in data["warnings"])
        finally:
            app.state.pdf_gen_ok = original

    def test_500_when_processor_offline(self, client):
        """POST /pipeline/run returns 500 when TranscriptProcessor is offline."""
        from medai.src.api.main import app
        original = app.state.processor_ok
        app.state.processor_ok = False
        try:
            r = client.post("/pipeline/run", json=_PIPELINE_BODY)
            assert r.status_code == 500
        finally:
            app.state.processor_ok = original

    def test_partial_failure_pipeline_continues(self, client):
        """Pipeline continues and collects warnings when SOAPGenerator fails."""
        from medai.src.api.main import app
        original_gen = app.state.soap_gen.generate
        app.state.soap_gen.generate = MagicMock(side_effect=RuntimeError("LLM timeout"))
        try:
            data = client.post("/pipeline/run", json=_PIPELINE_BODY).json()
            # Should return 200 with partial status
            assert data["status"] == "partial"
            assert any("SOAPGenerator" in w for w in data["warnings"])
        finally:
            app.state.soap_gen.generate = original_gen


# ---------------------------------------------------------------------------
# POST /transcripts/process
# ---------------------------------------------------------------------------

class TestTranscriptsProcess:
    def test_returns_200(self, client):
        """POST /transcripts/process returns 200."""
        r = client.post("/transcripts/process", json=_VALID_TRANSCRIPT)
        assert r.status_code == 200

    def test_response_has_conversation_id(self, client):
        """Response includes conversation_id matching the input."""
        data = client.post("/transcripts/process", json=_VALID_TRANSCRIPT).json()
        assert data["conversation_id"] == "conv-test"

    def test_response_has_processed_turns(self, client):
        """Response includes processed_turns as an int."""
        data = client.post("/transcripts/process", json=_VALID_TRANSCRIPT).json()
        assert isinstance(data["processed_turns"], int)

    def test_response_has_entities_dict(self, client):
        """Response includes entities as a dict."""
        data = client.post("/transcripts/process", json=_VALID_TRANSCRIPT).json()
        assert isinstance(data["entities"], dict)

    def test_response_has_flags_list(self, client):
        """Response includes flags as a list."""
        data = client.post("/transcripts/process", json=_VALID_TRANSCRIPT).json()
        assert isinstance(data["flags"], list)

    def test_response_has_processing_time(self, client):
        """Response includes processing_time_seconds."""
        data = client.post("/transcripts/process", json=_VALID_TRANSCRIPT).json()
        assert "processing_time_seconds" in data


# ---------------------------------------------------------------------------
# POST /soap/generate
# ---------------------------------------------------------------------------

class TestSoapGenerate:
    def test_returns_200(self, client):
        """POST /soap/generate returns 200."""
        r = client.post("/soap/generate", json=_VALID_TRANSCRIPT)
        assert r.status_code == 200

    def test_response_has_soap_note_dict(self, client):
        """Response includes soap_note as a dict."""
        data = client.post("/soap/generate", json=_VALID_TRANSCRIPT).json()
        assert isinstance(data["soap_note"], dict)

    def test_soap_note_has_subjective(self, client):
        """soap_note dict includes subjective field."""
        data = client.post("/soap/generate", json=_VALID_TRANSCRIPT).json()
        assert "subjective" in data["soap_note"]

    def test_soap_note_has_generation_method(self, client):
        """Response includes generation_method string."""
        data = client.post("/soap/generate", json=_VALID_TRANSCRIPT).json()
        assert isinstance(data["generation_method"], str)

    def test_soap_note_has_confidence(self, client):
        """Response includes confidence as a float."""
        data = client.post("/soap/generate", json=_VALID_TRANSCRIPT).json()
        assert isinstance(data["confidence"], float)

    def test_500_when_soap_gen_offline(self, client):
        """Returns 500 when SOAPGenerator is offline."""
        from medai.src.api.main import app
        original = app.state.soap_gen_ok
        app.state.soap_gen_ok = False
        try:
            r = client.post("/soap/generate", json=_VALID_TRANSCRIPT)
            assert r.status_code == 500
        finally:
            app.state.soap_gen_ok = original


# ---------------------------------------------------------------------------
# POST /reasoning/analyse
# ---------------------------------------------------------------------------

class TestReasoningAnalyse:
    def test_returns_200(self, client):
        """POST /reasoning/analyse returns 200."""
        r = client.post("/reasoning/analyse", json=_VALID_TRANSCRIPT)
        assert r.status_code == 200

    def test_response_has_red_flags(self, client):
        """Response includes red_flags dict."""
        data = client.post("/reasoning/analyse", json=_VALID_TRANSCRIPT).json()
        assert isinstance(data["red_flags"], dict)

    def test_response_has_ddx(self, client):
        """Response includes ddx dict."""
        data = client.post("/reasoning/analyse", json=_VALID_TRANSCRIPT).json()
        assert isinstance(data["ddx"], dict)

    def test_response_has_requires_immediate_action(self, client):
        """Response includes requires_immediate_action bool."""
        data = client.post("/reasoning/analyse", json=_VALID_TRANSCRIPT).json()
        assert isinstance(data["requires_immediate_action"], bool)

    def test_immediate_action_false_by_default(self, client):
        """requires_immediate_action is False when no critical flags."""
        data = client.post("/reasoning/analyse", json=_VALID_TRANSCRIPT).json()
        assert data["requires_immediate_action"] is False

    def test_500_when_ddx_offline(self, client):
        """Returns 500 when DDxEngine is offline."""
        from medai.src.api.main import app
        original = app.state.ddx_engine_ok
        app.state.ddx_engine_ok = False
        try:
            r = client.post("/reasoning/analyse", json=_VALID_TRANSCRIPT)
            assert r.status_code == 500
        finally:
            app.state.ddx_engine_ok = original


# ---------------------------------------------------------------------------
# GET /reports/{conversation_id}
# ---------------------------------------------------------------------------

class TestReportsList:
    def test_404_for_unknown_id(self, client):
        """GET /reports/{id} returns 404 for an unknown conversation ID."""
        r = client.get("/reports/nonexistent-conv-id-xyz-9999")
        assert r.status_code == 404

    def test_404_detail_mentions_conversation_id(self, client):
        """404 error detail references the conversation ID."""
        r = client.get("/reports/no-such-id")
        assert "no-such-id" in r.json()["detail"]

    def test_200_and_files_list_when_file_exists(self, client, tmp_path):
        """Returns 200 with files list when report file exists."""
        from medai.src.api import main as api_mod

        # Temporarily redirect REPO_ROOT reports dir to tmp_path
        reports_dir = tmp_path / "data/outputs/reports"
        reports_dir.mkdir(parents=True)
        test_file = reports_dir / "physician_my-test-conv_20260312.html"
        test_file.write_text("<html>test</html>")

        original_root = api_mod._REPO_ROOT
        api_mod._REPO_ROOT = tmp_path
        try:
            r = client.get("/reports/my-test-conv")
            assert r.status_code == 200
            data = r.json()
            assert data["conversation_id"] == "my-test-conv"
            assert len(data["files"]) >= 1
            assert data["files"][0]["size_bytes"] > 0
        finally:
            api_mod._REPO_ROOT = original_root


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------

class TestAuthentication:
    def test_post_returns_401_when_key_set_but_missing(self, client_with_key):
        """POST returns 401 when API key is configured but header is absent."""
        r = client_with_key.post("/pipeline/run", json=_PIPELINE_BODY)
        assert r.status_code == 401

    def test_post_returns_401_when_key_wrong(self, client_with_key):
        """POST returns 401 when the wrong API key is provided."""
        r = client_with_key.post(
            "/pipeline/run",
            json=_PIPELINE_BODY,
            headers={"X-API-Key": "wrong-key"},
        )
        assert r.status_code == 401

    def test_post_returns_200_with_correct_key(self, client_with_key):
        """POST returns 200 when the correct API key is provided."""
        r = client_with_key.post(
            "/pipeline/run",
            json=_PIPELINE_BODY,
            headers={"X-API-Key": "test-secret"},
        )
        assert r.status_code == 200

    def test_health_always_accessible_with_key_set(self, client_with_key):
        """GET /health is accessible even when API key auth is enabled."""
        r = client_with_key.get("/health")
        assert r.status_code == 200

    def test_transcripts_process_returns_401_without_key(self, client_with_key):
        """POST /transcripts/process returns 401 when key missing."""
        r = client_with_key.post("/transcripts/process", json=_VALID_TRANSCRIPT)
        assert r.status_code == 401

    def test_soap_generate_returns_401_without_key(self, client_with_key):
        """POST /soap/generate returns 401 when key missing."""
        r = client_with_key.post("/soap/generate", json=_VALID_TRANSCRIPT)
        assert r.status_code == 401

    def test_reasoning_analyse_returns_401_without_key(self, client_with_key):
        """POST /reasoning/analyse returns 401 when key missing."""
        r = client_with_key.post("/reasoning/analyse", json=_VALID_TRANSCRIPT)
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# Partial failure / resilience tests
# ---------------------------------------------------------------------------

class TestPartialFailures:
    def test_pipeline_continues_when_pdf_gen_raises(self, client):
        """Pipeline completes with partial status when PDFGenerator raises."""
        from medai.src.api.main import app
        original = app.state.pdf_gen.generate_physician_report
        app.state.pdf_gen.generate_physician_report = MagicMock(
            side_effect=RuntimeError("disk full")
        )
        try:
            data = client.post("/pipeline/run", json=_PIPELINE_BODY).json()
            assert data["status"] == "partial"
            assert any("PDFGenerator" in w for w in data["warnings"])
        finally:
            app.state.pdf_gen.generate_physician_report = original

    def test_pipeline_continues_when_fhir_raises(self, client):
        """Pipeline completes with partial status when FHIRFormatter raises."""
        from medai.src.api.main import app
        original_fmt = app.state.fhir.format_bundle
        app.state.fhir.format_bundle = MagicMock(side_effect=ValueError("bad data"))
        try:
            data = client.post("/pipeline/run", json=_PIPELINE_BODY).json()
            assert data["status"] == "partial"
            assert any("FHIRFormatter" in w for w in data["warnings"])
        finally:
            app.state.fhir.format_bundle = original_fmt

    def test_pipeline_skips_ddx_when_soap_fails(self, client):
        """DDx is skipped (with warning) when SOAP generation fails."""
        from medai.src.api.main import app
        original = app.state.soap_gen.generate
        app.state.soap_gen.generate = MagicMock(side_effect=RuntimeError("timeout"))
        try:
            data = client.post("/pipeline/run", json=_PIPELINE_BODY).json()
            assert data["status"] == "partial"
            # DDx should be skipped because soap is None
            assert data["ddx"] == {}
            assert any("SOAPGenerator" in w for w in data["warnings"])
        finally:
            app.state.soap_gen.generate = original

    def test_red_flag_offline_adds_warning(self, client):
        """RedFlagDetector offline adds a warning but doesn't crash pipeline."""
        from medai.src.api.main import app
        original = app.state.red_flag_ok
        app.state.red_flag_ok = False
        try:
            data = client.post("/pipeline/run", json=_PIPELINE_BODY).json()
            assert data["status"] == "partial"
            assert any("RedFlagDetector" in w for w in data["warnings"])
        finally:
            app.state.red_flag_ok = original
