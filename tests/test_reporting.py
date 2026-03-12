"""
tests/test_reporting.py

Tests for Step 6: PDFGenerator and FHIRFormatter.

All tests use lightweight stub dataclasses so no data files, API keys,
or optional dependencies (WeasyPrint) are required.
"""

from __future__ import annotations

import json
import sys
import os
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

# Ensure the repo root is on the path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Minimal stub dataclasses (mirror production types without importing pipeline)
# ---------------------------------------------------------------------------

@dataclass
class _Diagnosis:
    name: str = "Tension Headache"
    icd10_code: str = "G44.2"
    probability: str = "high"
    probability_score: float = 0.82
    supporting_evidence: List[str] = field(default_factory=lambda: ["headache", "stress"])
    contradicting_evidence: List[str] = field(default_factory=list)
    reasoning: str = "Bilateral pressure headache with stress trigger."


@dataclass
class _DDxResult:
    conversation_id: str = "conv-001"
    top_diagnoses: List[_Diagnosis] = field(default_factory=lambda: [_Diagnosis()])
    primary_diagnosis: _Diagnosis = field(default_factory=_Diagnosis)
    follow_up_questions: List[str] = field(
        default_factory=lambda: ["Any aura before headache?", "Family history?"]
    )
    recommended_investigations: List[str] = field(
        default_factory=lambda: ["MRI brain", "Blood pressure monitoring"]
    )
    red_flag_result: object = None
    generation_method: str = "rule_based"
    speciality: str = "Neurology"


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
    conversation_id: str = "conv-001"
    flags: List[_RedFlag] = field(default_factory=list)
    highest_severity: str = "none"
    flag_count: int = 0
    requires_immediate_action: bool = False
    summary: str = "No red flags detected."


@dataclass
class _SOAPNote:
    conversation_id: str = "conv-001"
    speciality: str = "General Medicine"
    subjective: str = "Patient reports headache for 2 days."
    objective: str = "BP 120/80, HR 72. Neurological exam normal."
    assessment: str = "Tension-type headache, likely stress-related."
    plan: str = "Analgesics PRN. Follow up if not resolved in 1 week."
    red_flag_mentions: List[str] = field(default_factory=list)
    generated_at: str = "2026-03-12T10:00:00+00:00"
    model_used: str = "template_fallback"
    prompt_tokens: int = 0
    confidence: float = 0.85
    generation_method: str = "template_fallback"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_soap(**kwargs) -> _SOAPNote:
    return _SOAPNote(**kwargs)


def _make_ddx(**kwargs) -> _DDxResult:
    return _DDxResult(**kwargs)


def _make_rfr(with_flags: bool = False, **kwargs) -> _RedFlagResult:
    if with_flags:
        return _RedFlagResult(
            flags=[_RedFlag()],
            highest_severity="urgent",
            flag_count=1,
            requires_immediate_action=False,
            **kwargs,
        )
    return _RedFlagResult(**kwargs)


# ---------------------------------------------------------------------------
# PDFGenerator tests
# ---------------------------------------------------------------------------

class TestPDFGeneratorFallback(unittest.TestCase):
    """Tests that run without WeasyPrint (always use HTML fallback)."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        # Patch WeasyPrint to be unavailable
        with patch.dict("sys.modules", {"weasyprint": None}):
            # Re-import with patched modules
            import importlib
            import medai.src.reporting.pdf_generator as _mod
            importlib.reload(_mod)
            self._mod = _mod

        from medai.src.reporting.pdf_generator import PDFGenerator
        self.gen = PDFGenerator(output_dir=self.tmp_dir)
        # Force fallback regardless of installed state
        self.gen.use_fallback = True

    def test_generate_physician_report_returns_report_output(self):
        """generate_physician_report returns a ReportOutput with correct fields."""
        soap = _make_soap()
        ddx = _make_ddx()
        rfr = _make_rfr()

        result = self.gen.generate_physician_report(soap, ddx, rfr)

        from medai.src.reporting.pdf_generator import ReportOutput
        self.assertIsInstance(result, ReportOutput)
        self.assertEqual(result.conversation_id, "conv-001")
        self.assertIn("physician_", result.physician_pdf_path)
        self.assertNotEqual(result.generated_at, "")
        self.assertEqual(result.generation_method, "html_fallback")

    def test_generate_physician_report_creates_file(self):
        """generate_physician_report writes a file to disk."""
        soap = _make_soap()
        ddx = _make_ddx()
        rfr = _make_rfr()

        result = self.gen.generate_physician_report(soap, ddx, rfr)
        self.assertTrue(Path(result.physician_pdf_path).exists())
        self.assertGreater(result.file_size_bytes, 0)

    def test_html_fallback_saves_html_extension(self):
        """HTML fallback produces a .html file, not .pdf."""
        soap = _make_soap()
        ddx = _make_ddx()
        rfr = _make_rfr()

        result = self.gen.generate_physician_report(soap, ddx, rfr)
        self.assertTrue(result.physician_pdf_path.endswith(".html"))

    def test_generate_patient_summary_returns_report_output(self):
        """generate_patient_summary returns a ReportOutput with correct fields."""
        soap = _make_soap()
        rfr = _make_rfr()

        result = self.gen.generate_patient_summary(soap, rfr)

        from medai.src.reporting.pdf_generator import ReportOutput
        self.assertIsInstance(result, ReportOutput)
        self.assertEqual(result.conversation_id, "conv-001")
        self.assertIn("patient_", result.patient_pdf_path)
        self.assertEqual(result.generation_method, "html_fallback")


class TestPDFGeneratorHtmlBuilders(unittest.TestCase):
    """Tests for the internal HTML-building methods."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        from medai.src.reporting.pdf_generator import PDFGenerator
        self.gen = PDFGenerator(output_dir=self.tmp_dir)
        self.gen.use_fallback = True

    def test_build_physician_html_contains_all_soap_sections(self):
        """_build_physician_html includes all four SOAP section labels."""
        soap = _make_soap()
        ddx = _make_ddx()
        rfr = _make_rfr()

        html = self.gen._build_physician_html(soap, ddx, rfr, {})

        for section in ["Subjective", "Objective", "Assessment", "Plan"]:
            self.assertIn(section, html, f"Missing SOAP section: {section}")

    def test_build_physician_html_includes_soap_content(self):
        """_build_physician_html renders the actual SOAP text."""
        soap = _make_soap(subjective="Terrible migraine for 3 days.")
        ddx = _make_ddx()
        rfr = _make_rfr()

        html = self.gen._build_physician_html(soap, ddx, rfr, {})
        self.assertIn("Terrible migraine for 3 days.", html)

    def test_build_physician_html_includes_ddx_content(self):
        """_build_physician_html renders diagnosis name and ICD-10 code."""
        soap = _make_soap()
        ddx = _make_ddx()
        rfr = _make_rfr()

        html = self.gen._build_physician_html(soap, ddx, rfr, {})
        self.assertIn("Tension Headache", html)
        self.assertIn("G44.2", html)

    def test_build_physician_html_red_flags_absent_when_no_flags(self):
        """Red flag section heading is not rendered when there are no flags."""
        soap = _make_soap()
        ddx = _make_ddx()
        rfr = _make_rfr(with_flags=False)

        html = self.gen._build_physician_html(soap, ddx, rfr, {})
        # The section heading "Red Flags / Alerts" is only injected when flags exist
        self.assertNotIn("Red Flags / Alerts", html)

    def test_build_physician_html_red_flags_present_when_flags_exist(self):
        """Red flag section heading is rendered when flags are present."""
        soap = _make_soap()
        ddx = _make_ddx()
        rfr = _make_rfr(with_flags=True)

        html = self.gen._build_physician_html(soap, ddx, rfr, {})
        self.assertIn("Red Flags / Alerts", html)

    def test_build_physician_html_patient_info(self):
        """Patient info is rendered in the physician report."""
        soap = _make_soap()
        ddx = _make_ddx()
        rfr = _make_rfr()
        patient_info = {"name": "Jane Doe", "dob": "1985-06-15", "mrn": "MRN-999"}

        html = self.gen._build_physician_html(soap, ddx, rfr, patient_info)
        self.assertIn("Jane Doe", html)
        self.assertIn("MRN-999", html)

    def test_build_patient_html_uses_plain_language_headings(self):
        """_build_patient_html uses plain-English headings (no LOINC codes)."""
        soap = _make_soap()
        rfr = _make_rfr()

        html = self.gen._build_patient_html(soap, rfr, {})

        # Plain language headings should be present
        self.assertIn("What brought you in today", html)
        self.assertIn("What we found", html)
        self.assertIn("Our assessment", html)
        self.assertIn("Your care plan", html)
        self.assertIn("When to seek emergency care", html)

    def test_build_patient_html_no_loinc_codes(self):
        """_build_patient_html does not contain raw LOINC codes."""
        soap = _make_soap()
        rfr = _make_rfr()

        html = self.gen._build_patient_html(soap, rfr, {})

        # LOINC codes like 10164-2 should not appear in patient reports
        for loinc in ["10164-2", "10210-3", "51848-0", "18776-5"]:
            self.assertNotIn(loinc, html, f"LOINC code {loinc} found in patient HTML")

    def test_build_patient_html_includes_emergency_advice(self):
        """Patient summary always includes the emergency safety-net text."""
        soap = _make_soap()
        rfr = _make_rfr()

        html = self.gen._build_patient_html(soap, rfr, {})
        self.assertIn("emergency", html.lower())

    def test_build_patient_html_shows_red_flags_in_plain_language(self):
        """Patient summary includes urgent concerns in plain text when flags exist."""
        soap = _make_soap()
        rfr = _make_rfr(with_flags=True)

        html = self.gen._build_patient_html(soap, rfr, {})
        self.assertIn("Urgent concerns", html)

    def test_build_patient_html_no_urgent_section_when_no_flags(self):
        """Patient summary omits urgent concerns section when no flags."""
        soap = _make_soap()
        rfr = _make_rfr(with_flags=False)

        html = self.gen._build_patient_html(soap, rfr, {})
        self.assertNotIn("Urgent concerns noted", html)


class TestPDFGeneratorBatch(unittest.TestCase):
    """Tests for generate_batch()."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        from medai.src.reporting.pdf_generator import PDFGenerator
        self.gen = PDFGenerator(output_dir=self.tmp_dir)
        self.gen.use_fallback = True

    def test_generate_batch_returns_list_of_report_outputs(self):
        """generate_batch returns one ReportOutput per matched record."""
        soaps = [_make_soap(conversation_id=f"c{i}") for i in range(3)]
        ddxs = [_make_ddx(conversation_id=f"c{i}") for i in range(3)]
        rfrs = [_make_rfr(conversation_id=f"c{i}") for i in range(3)]

        results = self.gen.generate_batch(soaps, ddxs, rfrs, show_progress=False)
        self.assertEqual(len(results), 3)

    def test_generate_batch_isolates_failures(self):
        """generate_batch skips records missing DDx and continues processing."""
        soaps = [_make_soap(conversation_id="c0"), _make_soap(conversation_id="c1")]
        ddxs = [_make_ddx(conversation_id="c0")]   # c1 has no DDx
        rfrs = [_make_rfr(conversation_id="c0"), _make_rfr(conversation_id="c1")]

        results = self.gen.generate_batch(soaps, ddxs, rfrs, show_progress=False)
        # Only c0 should succeed; c1 is missing ddx
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].conversation_id, "c0")

    def test_generate_batch_sets_both_paths(self):
        """generate_batch sets physician and patient paths on each ReportOutput."""
        soaps = [_make_soap()]
        ddxs = [_make_ddx()]
        rfrs = [_make_rfr()]

        results = self.gen.generate_batch(soaps, ddxs, rfrs, show_progress=False)
        self.assertEqual(len(results), 1)
        self.assertNotEqual(results[0].physician_pdf_path, "")
        self.assertNotEqual(results[0].patient_pdf_path, "")

    def test_generate_batch_exception_isolation(self):
        """generate_batch isolates per-record exceptions without crashing."""
        soaps = [_make_soap(conversation_id="good"), _make_soap(conversation_id="bad")]
        ddxs = [_make_ddx(conversation_id="good"), _make_ddx(conversation_id="bad")]
        rfrs = [_make_rfr(conversation_id="good"), _make_rfr(conversation_id="bad")]

        # Make the second soap raise when accessing .subjective
        bad_soap = MagicMock(spec=_SOAPNote)
        bad_soap.conversation_id = "bad"
        bad_soap.subjective = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        soaps[1] = bad_soap

        results = self.gen.generate_batch(soaps, ddxs, rfrs, show_progress=False)
        # "good" should still succeed even though "bad" raised
        conv_ids = [r.conversation_id for r in results]
        self.assertIn("good", conv_ids)


# ---------------------------------------------------------------------------
# FHIRFormatter tests
# ---------------------------------------------------------------------------

class TestFHIRFormatterBundle(unittest.TestCase):
    """Tests for FHIRFormatter.format_bundle()."""

    def setUp(self):
        from medai.src.reporting.fhir_formatter import FHIRFormatter
        self.fmt = FHIRFormatter()

    def test_format_bundle_returns_dict(self):
        """format_bundle returns a dict."""
        bundle = self.fmt.format_bundle(_make_soap(), _make_ddx(), _make_rfr())
        self.assertIsInstance(bundle, dict)

    def test_format_bundle_resource_type_is_bundle(self):
        """format_bundle sets resourceType to Bundle."""
        bundle = self.fmt.format_bundle(_make_soap(), _make_ddx(), _make_rfr())
        self.assertEqual(bundle["resourceType"], "Bundle")

    def test_format_bundle_type_is_document(self):
        """format_bundle sets type to document."""
        bundle = self.fmt.format_bundle(_make_soap(), _make_ddx(), _make_rfr())
        self.assertEqual(bundle["type"], "document")

    def test_format_bundle_has_timestamp(self):
        """format_bundle includes a timestamp field."""
        bundle = self.fmt.format_bundle(_make_soap(), _make_ddx(), _make_rfr())
        self.assertIn("timestamp", bundle)
        self.assertIsInstance(bundle["timestamp"], str)

    def test_format_bundle_entry_is_non_empty_list(self):
        """format_bundle entry is a non-empty list."""
        bundle = self.fmt.format_bundle(_make_soap(), _make_ddx(), _make_rfr())
        self.assertIsInstance(bundle["entry"], list)
        self.assertGreater(len(bundle["entry"]), 0)

    def test_format_bundle_contains_patient(self):
        """Bundle entry list includes a Patient resource."""
        bundle = self.fmt.format_bundle(_make_soap(), _make_ddx(), _make_rfr())
        rts = [e["resource"]["resourceType"] for e in bundle["entry"]]
        self.assertIn("Patient", rts)

    def test_format_bundle_contains_composition(self):
        """Bundle entry list includes a Composition resource."""
        bundle = self.fmt.format_bundle(_make_soap(), _make_ddx(), _make_rfr())
        rts = [e["resource"]["resourceType"] for e in bundle["entry"]]
        self.assertIn("Composition", rts)

    def test_format_bundle_contains_conditions(self):
        """Bundle entry list includes Condition resources."""
        bundle = self.fmt.format_bundle(_make_soap(), _make_ddx(), _make_rfr())
        rts = [e["resource"]["resourceType"] for e in bundle["entry"]]
        self.assertIn("Condition", rts)


class TestFHIRFormatterPatient(unittest.TestCase):
    """Tests for _build_patient_resource()."""

    def setUp(self):
        from medai.src.reporting.fhir_formatter import FHIRFormatter
        self.fmt = FHIRFormatter()

    def test_build_patient_resource_type(self):
        """_build_patient_resource returns a Patient resource."""
        res = self.fmt._build_patient_resource({})
        self.assertEqual(res["resourceType"], "Patient")

    def test_build_patient_resource_deidentified_defaults(self):
        """Without patient_info, defaults to de-identified placeholder."""
        res = self.fmt._build_patient_resource({})
        self.assertEqual(res["name"][0]["text"], "De-identified Patient")
        self.assertEqual(res["gender"], "unknown")

    def test_build_patient_resource_uses_provided_info(self):
        """Uses provided name, dob, gender when given."""
        info = {"name": "Alice Smith", "dob": "1990-01-15", "gender": "female"}
        res = self.fmt._build_patient_resource(info)
        self.assertEqual(res["name"][0]["text"], "Alice Smith")
        self.assertEqual(res["birthDate"], "1990-01-15")
        self.assertEqual(res["gender"], "female")


class TestFHIRFormatterComposition(unittest.TestCase):
    """Tests for _build_composition_resource()."""

    def setUp(self):
        from medai.src.reporting.fhir_formatter import FHIRFormatter
        self.fmt = FHIRFormatter()

    def test_composition_resource_type(self):
        """_build_composition_resource returns a Composition resource."""
        res = self.fmt._build_composition_resource(_make_soap())
        self.assertEqual(res["resourceType"], "Composition")

    def test_composition_has_four_sections(self):
        """Composition has exactly four SOAP sections."""
        res = self.fmt._build_composition_resource(_make_soap())
        self.assertEqual(len(res["section"]), 4)

    def test_composition_section_loinc_codes(self):
        """Each SOAP section carries the correct LOINC code."""
        res = self.fmt._build_composition_resource(_make_soap())
        loinc_codes = {
            sec["code"]["coding"][0]["code"] for sec in res["section"]
        }
        for expected in ["10164-2", "10210-3", "51848-0", "18776-5"]:
            self.assertIn(expected, loinc_codes)

    def test_composition_type_loinc(self):
        """Composition type carries LOINC 34133-9."""
        res = self.fmt._build_composition_resource(_make_soap())
        self.assertEqual(res["type"]["coding"][0]["code"], "34133-9")


class TestFHIRFormatterConditions(unittest.TestCase):
    """Tests for _build_condition_resources()."""

    def setUp(self):
        from medai.src.reporting.fhir_formatter import FHIRFormatter
        self.fmt = FHIRFormatter()

    def test_one_condition_per_diagnosis(self):
        """Creates exactly one Condition resource per top diagnosis."""
        diags = [_Diagnosis(name=f"Dx{i}", icd10_code=f"X{i:02d}") for i in range(3)]
        ddx = _make_ddx(top_diagnoses=diags)
        conditions = self.fmt._build_condition_resources(ddx)
        self.assertEqual(len(conditions), 3)

    def test_condition_resource_type(self):
        """Each resource has resourceType Condition."""
        conditions = self.fmt._build_condition_resources(_make_ddx())
        for cond in conditions:
            self.assertEqual(cond["resourceType"], "Condition")

    def test_condition_icd10_code(self):
        """Condition carries the correct ICD-10 code from Diagnosis."""
        ddx = _make_ddx(top_diagnoses=[_Diagnosis(icd10_code="G44.2")])
        conditions = self.fmt._build_condition_resources(ddx)
        code = conditions[0]["code"]["coding"][0]["code"]
        self.assertEqual(code, "G44.2")

    def test_condition_icd10_system(self):
        """Condition uses the ICD-10 coding system URI."""
        conditions = self.fmt._build_condition_resources(_make_ddx())
        system = conditions[0]["code"]["coding"][0]["system"]
        self.assertEqual(system, "http://hl7.org/fhir/sid/icd-10")

    def test_condition_empty_when_no_diagnoses(self):
        """Returns empty list when top_diagnoses is empty."""
        ddx = _make_ddx(top_diagnoses=[])
        conditions = self.fmt._build_condition_resources(ddx)
        self.assertEqual(conditions, [])


class TestFHIRFormatterObservations(unittest.TestCase):
    """Tests for _build_observation_resources()."""

    def setUp(self):
        from medai.src.reporting.fhir_formatter import FHIRFormatter
        self.fmt = FHIRFormatter()

    def test_empty_vitals_returns_empty_list(self):
        """Returns empty list when vitals dict is empty."""
        self.assertEqual(self.fmt._build_observation_resources({}), [])

    def test_bp_maps_to_loinc_55284_4(self):
        """BP vital maps to LOINC 55284-4."""
        obs = self.fmt._build_observation_resources({"BP": "120/80 mmHg"})
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["code"]["coding"][0]["code"], "55284-4")

    def test_hr_maps_to_loinc_8867_4(self):
        """HR vital maps to LOINC 8867-4."""
        obs = self.fmt._build_observation_resources({"HR": "72 bpm"})
        self.assertEqual(obs[0]["code"]["coding"][0]["code"], "8867-4")

    def test_temp_maps_to_loinc_8310_5(self):
        """temp vital maps to LOINC 8310-5."""
        obs = self.fmt._build_observation_resources({"temp": "37.2°C"})
        self.assertEqual(obs[0]["code"]["coding"][0]["code"], "8310-5")

    def test_rr_maps_to_loinc_9279_1(self):
        """RR vital maps to LOINC 9279-1."""
        obs = self.fmt._build_observation_resources({"RR": "16"})
        self.assertEqual(obs[0]["code"]["coding"][0]["code"], "9279-1")

    def test_spo2_maps_to_loinc_2708_6(self):
        """SpO2 vital maps to LOINC 2708-6."""
        obs = self.fmt._build_observation_resources({"SpO2": "98%"})
        self.assertEqual(obs[0]["code"]["coding"][0]["code"], "2708-6")

    def test_multiple_vitals_creates_multiple_observations(self):
        """Multiple vitals create one Observation each."""
        vitals = {"BP": "120/80", "HR": "72", "temp": "37.0"}
        obs = self.fmt._build_observation_resources(vitals)
        self.assertEqual(len(obs), 3)

    def test_observation_resource_type(self):
        """Each resource has resourceType Observation."""
        obs = self.fmt._build_observation_resources({"HR": "80"})
        self.assertEqual(obs[0]["resourceType"], "Observation")

    def test_observation_status_final(self):
        """Observations have status 'final'."""
        obs = self.fmt._build_observation_resources({"HR": "80"})
        self.assertEqual(obs[0]["status"], "final")

    def test_observation_value_string(self):
        """valueString carries the original vital value."""
        obs = self.fmt._build_observation_resources({"HR": "80 bpm"})
        self.assertEqual(obs[0]["valueString"], "80 bpm")

    def test_unknown_vital_key_ignored(self):
        """Unrecognised vital keys are silently ignored."""
        obs = self.fmt._build_observation_resources({"UnknownKey": "42"})
        self.assertEqual(obs, [])


class TestFHIRFormatterValidation(unittest.TestCase):
    """Tests for validate_bundle()."""

    def setUp(self):
        from medai.src.reporting.fhir_formatter import FHIRFormatter
        self.fmt = FHIRFormatter()

    def _valid_bundle(self) -> dict:
        """Return a minimal structurally-valid bundle."""
        return self.fmt.format_bundle(_make_soap(), _make_ddx(), _make_rfr())

    def test_valid_bundle_passes(self):
        """A well-formed bundle passes validation."""
        report = self.fmt.validate_bundle(self._valid_bundle())
        self.assertTrue(report["valid"])
        self.assertEqual(report["errors"], [])

    def test_wrong_resource_type_fails(self):
        """Wrong resourceType is flagged as an error."""
        bundle = self._valid_bundle()
        bundle["resourceType"] = "Observation"
        report = self.fmt.validate_bundle(bundle)
        self.assertFalse(report["valid"])
        self.assertTrue(any("resourceType" in e for e in report["errors"]))

    def test_missing_type_fails(self):
        """Missing 'type' field is flagged as an error."""
        bundle = self._valid_bundle()
        del bundle["type"]
        report = self.fmt.validate_bundle(bundle)
        self.assertFalse(report["valid"])
        self.assertTrue(any("type" in e for e in report["errors"]))

    def test_empty_entry_list_fails(self):
        """Empty entry list is flagged as an error."""
        bundle = self._valid_bundle()
        bundle["entry"] = []
        report = self.fmt.validate_bundle(bundle)
        self.assertFalse(report["valid"])
        self.assertTrue(any("entry" in e.lower() for e in report["errors"]))

    def test_missing_composition_fails(self):
        """Bundle without Composition resource fails validation."""
        bundle = self._valid_bundle()
        bundle["entry"] = [
            e for e in bundle["entry"]
            if e["resource"].get("resourceType") != "Composition"
        ]
        report = self.fmt.validate_bundle(bundle)
        self.assertFalse(report["valid"])
        self.assertTrue(any("Composition" in e for e in report["errors"]))

    def test_missing_patient_fails(self):
        """Bundle without Patient resource fails validation."""
        bundle = self._valid_bundle()
        bundle["entry"] = [
            e for e in bundle["entry"]
            if e["resource"].get("resourceType") != "Patient"
        ]
        report = self.fmt.validate_bundle(bundle)
        self.assertFalse(report["valid"])
        self.assertTrue(any("Patient" in e for e in report["errors"]))

    def test_entry_missing_resource_key(self):
        """Entry without 'resource' key is flagged."""
        bundle = self._valid_bundle()
        bundle["entry"].append({"bad_key": "no resource"})
        report = self.fmt.validate_bundle(bundle)
        # Should report error about missing 'resource' key
        self.assertTrue(any("resource" in e.lower() for e in report["errors"]))


class TestFHIRFormatterSaveBundle(unittest.TestCase):
    """Tests for save_bundle()."""

    def setUp(self):
        from medai.src.reporting.fhir_formatter import FHIRFormatter
        self.fmt = FHIRFormatter()
        self.tmp_dir = tempfile.mkdtemp()

    def test_save_bundle_creates_file(self):
        """save_bundle creates a file at the specified path."""
        bundle = self.fmt.format_bundle(_make_soap(), _make_ddx(), _make_rfr())
        out_path = os.path.join(self.tmp_dir, "test_bundle.json")
        result = self.fmt.save_bundle(bundle, output_path=out_path)
        self.assertTrue(Path(result).exists())

    def test_save_bundle_returns_path(self):
        """save_bundle returns the path used."""
        bundle = self.fmt.format_bundle(_make_soap(), _make_ddx(), _make_rfr())
        out_path = os.path.join(self.tmp_dir, "test_bundle.json")
        result = self.fmt.save_bundle(bundle, output_path=out_path)
        self.assertEqual(result, out_path)

    def test_save_bundle_valid_json(self):
        """save_bundle writes valid JSON that round-trips correctly."""
        bundle = self.fmt.format_bundle(_make_soap(), _make_ddx(), _make_rfr())
        out_path = os.path.join(self.tmp_dir, "test_bundle.json")
        self.fmt.save_bundle(bundle, output_path=out_path)
        with open(out_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["resourceType"], "Bundle")

    def test_save_bundle_default_path_under_fhir_dir(self):
        """save_bundle uses data/outputs/fhir/ as default directory."""
        bundle = self.fmt.format_bundle(_make_soap(), _make_ddx(), _make_rfr())
        result = self.fmt.save_bundle(bundle, conversation_id="smoke-test-conv")
        self.assertIn("fhir", result)
        self.assertIn("smoke-test-conv", result)
        self.assertTrue(result.endswith(".json"))
        # Clean up
        if Path(result).exists():
            Path(result).unlink()


if __name__ == "__main__":
    unittest.main()
