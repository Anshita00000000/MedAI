"""
tests/test_soap_generator.py

Pytest suite for medai.src.clinical.soap_generator.

Run with:
    pytest tests/test_soap_generator.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medai.src.clinical.soap_generator import (
    SOAPGenerator,
    SOAPNote,
    _RED_FLAG_TERMS,
    _tokenise,
)
from medai.src.clinical.entity_extractor import MedicalEntities
from medai.src.voice.transcript_processor import ProcessedTranscript, TranscriptProcessor, Turn


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------

def _make_entities(
    id_: str = "t_001",
    symptoms: list[str] | None = None,
    medications: list[str] | None = None,
    vitals: dict | None = None,
    negations: list[str] | None = None,
    speciality_hints: list[str] | None = None,
    medical_history: list[str] | None = None,
    confidence: float = 0.5,
) -> MedicalEntities:
    return MedicalEntities(
        conversation_id=id_,
        symptoms=symptoms or [],
        clinical_findings=[],
        medications=medications or [],
        medical_history=medical_history or [],
        vitals=vitals or {},
        allergies=[],
        speciality_hints=speciality_hints or [],
        negations=negations or [],
        extraction_method="rule_based",
        confidence=confidence,
    )


def _make_transcript(
    id_: str = "t_001",
    patient_text: str = "I have had chest pain and shortness of breath for two days.",
    doctor_text: str = "Tell me more about the pain.",
    reference_note: str | None = None,
) -> ProcessedTranscript:
    raw = {
        "id": id_,
        "source_dataset": "test",
        "language": "en",
        "turns": [
            {"speaker": "DOCTOR", "text": doctor_text},
            {"speaker": "PATIENT", "text": patient_text},
            {"speaker": "DOCTOR", "text": "Any other symptoms?"},
            {"speaker": "PATIENT", "text": "Some nausea as well."},
            {"speaker": "DOCTOR", "text": "How long has this been going on?"},
            {"speaker": "PATIENT", "text": "About two days now."},
        ],
        "reference_note": reference_note,
        "is_synthetic": True,
        "turn_count": 6,
    }
    return TranscriptProcessor().process(raw)


@pytest.fixture
def generator() -> SOAPGenerator:
    """Always uses template fallback — no API key in CI."""
    return SOAPGenerator(api_key=None)


@pytest.fixture
def basic_transcript() -> ProcessedTranscript:
    return _make_transcript(
        patient_text="I have severe chest pain and shortness of breath.",
        reference_note="SUBJECTIVE: Patient reports chest pain.",
    )


@pytest.fixture
def basic_entities() -> MedicalEntities:
    return _make_entities(
        symptoms=["chest pain", "shortness of breath"],
        medications=["aspirin"],
        vitals={"BP": "130/85 mmHg", "HR": "88 bpm"},
        negations=["fever"],
        speciality_hints=["cardiology"],
        confidence=0.6,
    )


# ---------------------------------------------------------------------------
# generate() — return type and field correctness
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_returns_soap_note(self, generator, basic_transcript, basic_entities):
        result = generator.generate(basic_transcript, basic_entities)
        assert isinstance(result, SOAPNote)

    def test_conversation_id_matches_transcript(self, generator, basic_transcript, basic_entities):
        result = generator.generate(basic_transcript, basic_entities)
        assert result.conversation_id == basic_transcript.id

    def test_speciality_inferred(self, generator, basic_transcript, basic_entities):
        result = generator.generate(basic_transcript, basic_entities)
        # cardiology hint → "Cardiology"
        assert result.speciality == "Cardiology"

    def test_speciality_defaults_to_general_medicine(self, generator, basic_transcript):
        entities = _make_entities(speciality_hints=[])
        result = generator.generate(basic_transcript, entities)
        assert result.speciality == "General Medicine"

    def test_generated_at_is_set(self, generator, basic_transcript, basic_entities):
        result = generator.generate(basic_transcript, basic_entities)
        assert result.generated_at != ""
        # Should be ISO format
        assert "T" in result.generated_at

    def test_confidence_inherited(self, generator, basic_transcript):
        entities = _make_entities(confidence=0.72)
        result = generator.generate(basic_transcript, entities)
        assert result.confidence == 0.72

    def test_generation_method_template_fallback(self, generator, basic_transcript, basic_entities):
        result = generator.generate(basic_transcript, basic_entities)
        assert result.generation_method == "template_fallback"

    def test_subjective_is_non_empty_string(self, generator, basic_transcript, basic_entities):
        result = generator.generate(basic_transcript, basic_entities)
        assert isinstance(result.subjective, str)
        assert len(result.subjective) > 0

    def test_objective_is_non_empty_string(self, generator, basic_transcript, basic_entities):
        result = generator.generate(basic_transcript, basic_entities)
        assert isinstance(result.objective, str)
        assert len(result.objective) > 0

    def test_assessment_is_non_empty_string(self, generator, basic_transcript, basic_entities):
        result = generator.generate(basic_transcript, basic_entities)
        assert isinstance(result.assessment, str)
        assert len(result.assessment) > 0

    def test_plan_is_non_empty_string(self, generator, basic_transcript, basic_entities):
        result = generator.generate(basic_transcript, basic_entities)
        assert isinstance(result.plan, str)
        assert len(result.plan) > 0

    def test_red_flag_mentions_is_list(self, generator, basic_transcript, basic_entities):
        result = generator.generate(basic_transcript, basic_entities)
        assert isinstance(result.red_flag_mentions, list)

    def test_never_raises(self, generator):
        """generate() must not raise even with degenerate inputs."""
        t = _make_transcript(patient_text="", doctor_text="")
        e = _make_entities()
        # Should not raise
        result = generator.generate(t, e)
        assert isinstance(result, SOAPNote)


# ---------------------------------------------------------------------------
# _parse_soap_response()
# ---------------------------------------------------------------------------

class TestParseSoapResponse:
    @pytest.fixture(autouse=True)
    def _setup(self, generator):
        self.gen = generator

    def test_parses_all_four_sections(self):
        response = (
            "SUBJECTIVE: Patient reports chest pain.\n"
            "OBJECTIVE: BP 130/80 mmHg, HR 88 bpm.\n"
            "ASSESSMENT: Possible angina.\n"
            "PLAN: ECG, troponin, refer to cardiology."
        )
        soap = self.gen._parse_soap_response(response, "t_001")
        assert "chest pain" in soap.subjective
        assert "130/80" in soap.objective
        assert "angina" in soap.assessment.lower()
        assert "ECG" in soap.plan

    def test_parses_sections_case_insensitively(self):
        response = (
            "subjective: Headache for 3 days.\n"
            "objective: Temp 38.5°C.\n"
            "assessment: Viral illness.\n"
            "plan: Rest and paracetamol."
        )
        soap = self.gen._parse_soap_response(response, "t_002")
        assert "headache" in soap.subjective.lower()
        assert "viral" in soap.assessment.lower()

    def test_missing_section_filled_with_not_documented(self):
        # Only SUBJECTIVE and PLAN present
        response = (
            "SUBJECTIVE: Patient has fever.\n"
            "PLAN: Prescribe antipyretics."
        )
        soap = self.gen._parse_soap_response(response, "t_003")
        assert soap.objective == "Not documented"
        assert soap.assessment == "Not documented"

    def test_all_sections_missing_returns_not_documented(self):
        soap = self.gen._parse_soap_response("No sections here at all.", "t_004")
        assert soap.subjective == "Not documented"
        assert soap.objective == "Not documented"
        assert soap.assessment == "Not documented"
        assert soap.plan == "Not documented"

    def test_conversation_id_set_correctly(self):
        response = "SUBJECTIVE: x\nOBJECTIVE: y\nASSESSMENT: z\nPLAN: w"
        soap = self.gen._parse_soap_response(response, "conv_42")
        assert soap.conversation_id == "conv_42"

    def test_strips_whitespace_from_sections(self):
        response = (
            "SUBJECTIVE:   \n  Patient is well.   \n"
            "OBJECTIVE:  \n  No abnormalities.  \n"
            "ASSESSMENT:  \n  Healthy.  \n"
            "PLAN:  \n  Continue current management.  \n"
        )
        soap = self.gen._parse_soap_response(response, "t_005")
        assert not soap.subjective.startswith(" ")
        assert not soap.plan.endswith(" ")

    def test_sections_with_bullet_points_preserved(self):
        response = (
            "SUBJECTIVE: Patient reports:\n- chest pain\n- dyspnoea\n"
            "OBJECTIVE: Vitals stable.\n"
            "ASSESSMENT: Possible ACS.\n"
            "PLAN:\n- ECG\n- Troponin\n- Aspirin 300mg"
        )
        soap = self.gen._parse_soap_response(response, "t_006")
        assert "chest pain" in soap.subjective
        assert "ECG" in soap.plan

    def test_extra_text_before_subjective_ignored(self):
        response = (
            "Here is the SOAP note:\n\n"
            "SUBJECTIVE: Cough for one week.\n"
            "OBJECTIVE: Temp 37.2°C.\n"
            "ASSESSMENT: URTI.\n"
            "PLAN: Symptomatic treatment."
        )
        soap = self.gen._parse_soap_response(response, "t_007")
        assert "cough" in soap.subjective.lower()


# ---------------------------------------------------------------------------
# _generate_template_fallback()
# ---------------------------------------------------------------------------

class TestTemplateFallback:
    @pytest.fixture(autouse=True)
    def _setup(self, generator):
        self.gen = generator

    def test_returns_soap_note_type(self, basic_transcript, basic_entities):
        result = self.gen._generate_template_fallback(basic_transcript, basic_entities)
        assert isinstance(result, SOAPNote)

    def test_generation_method_is_template_fallback(self, basic_transcript, basic_entities):
        result = self.gen._generate_template_fallback(basic_transcript, basic_entities)
        assert result.generation_method == "template_fallback"

    def test_model_used_is_template_fallback(self, basic_transcript, basic_entities):
        result = self.gen._generate_template_fallback(basic_transcript, basic_entities)
        assert result.model_used == "template_fallback"

    def test_symptoms_in_subjective(self, basic_transcript, basic_entities):
        result = self.gen._generate_template_fallback(basic_transcript, basic_entities)
        assert "chest pain" in result.subjective or "shortness of breath" in result.subjective

    def test_vitals_in_objective(self, basic_transcript, basic_entities):
        result = self.gen._generate_template_fallback(basic_transcript, basic_entities)
        assert "130/85" in result.objective or "BP" in result.objective

    def test_medications_in_objective(self, basic_transcript, basic_entities):
        result = self.gen._generate_template_fallback(basic_transcript, basic_entities)
        assert "aspirin" in result.objective.lower()

    def test_negations_in_subjective(self, basic_transcript, basic_entities):
        result = self.gen._generate_template_fallback(basic_transcript, basic_entities)
        # "fever" is negated
        assert "fever" in result.subjective.lower()

    def test_speciality_hint_in_assessment(self, basic_transcript, basic_entities):
        result = self.gen._generate_template_fallback(basic_transcript, basic_entities)
        assert "cardiology" in result.assessment.lower()

    def test_empty_entities_does_not_crash(self):
        t = _make_transcript()
        e = _make_entities()
        gen = SOAPGenerator(api_key=None)
        result = gen._generate_template_fallback(t, e)
        assert isinstance(result, SOAPNote)

    def test_long_patient_text_truncated_in_subjective(self):
        long_text = " ".join(["symptom"] * 300)
        t = _make_transcript(patient_text=long_text)
        e = _make_entities()
        gen = SOAPGenerator(api_key=None)
        result = gen._generate_template_fallback(t, e)
        # Should contain truncation marker
        assert "[...]" in result.subjective

    def test_plan_contains_physician_phrase(self, basic_transcript, basic_entities):
        result = self.gen._generate_template_fallback(basic_transcript, basic_entities)
        assert "physician" in result.plan.lower()

    def test_medical_history_in_assessment(self):
        t = _make_transcript()
        e = _make_entities(medical_history=["hypertension", "type 2 diabetes"])
        gen = SOAPGenerator(api_key=None)
        result = gen._generate_template_fallback(t, e)
        assert "hypertension" in result.assessment.lower()


# ---------------------------------------------------------------------------
# evaluate_against_reference()
# ---------------------------------------------------------------------------

class TestEvaluateAgainstReference:
    @pytest.fixture(autouse=True)
    def _setup(self, generator):
        self.gen = generator

    def _make_soap(self, text: str) -> SOAPNote:
        return SOAPNote(
            conversation_id="eval_001",
            speciality="General Medicine",
            subjective=text,
            objective=text,
            assessment=text,
            plan=text,
        )

    def test_returns_dict_with_required_keys(self):
        soap = self._make_soap("patient has fever")
        result = self.gen.evaluate_against_reference(soap, "patient has fever")
        assert "jaccard" in result
        assert "recall" in result
        assert "conversation_id" in result

    def test_perfect_overlap_gives_jaccard_one(self):
        text = "patient has fever and cough"
        soap = self._make_soap(text)
        result = self.gen.evaluate_against_reference(soap, text)
        assert result["jaccard"] == 1.0

    def test_no_overlap_gives_jaccard_zero(self):
        soap = self._make_soap("fever cough pain")
        result = self.gen.evaluate_against_reference(soap, "banana orange mango")
        assert result["jaccard"] == 0.0

    def test_partial_overlap_jaccard_in_range(self):
        soap = self._make_soap("patient has fever and cough")
        result = self.gen.evaluate_against_reference(soap, "patient has chest pain and fever")
        assert 0.0 < result["jaccard"] < 1.0

    def test_recall_one_when_reference_subset_of_generated(self):
        soap = self._make_soap("patient has fever cough and pain and nausea")
        # Reference is a strict subset of the generated text words
        result = self.gen.evaluate_against_reference(soap, "patient has fever")
        assert result["recall"] == 1.0

    def test_recall_zero_when_no_overlap(self):
        soap = self._make_soap("fever cough pain")
        result = self.gen.evaluate_against_reference(soap, "banana orange")
        assert result["recall"] == 0.0

    def test_empty_reference_returns_zeros(self):
        soap = self._make_soap("some text here")
        result = self.gen.evaluate_against_reference(soap, "")
        assert result["jaccard"] == 0.0
        assert result["recall"] == 0.0

    def test_conversation_id_in_result(self):
        soap = self._make_soap("text")
        result = self.gen.evaluate_against_reference(soap, "text")
        assert result["conversation_id"] == "eval_001"

    def test_case_insensitive_matching(self):
        soap = self._make_soap("Patient Has FEVER")
        result = self.gen.evaluate_against_reference(soap, "patient has fever")
        assert result["jaccard"] == 1.0

    def test_jaccard_is_symmetric(self):
        soap_a = self._make_soap("fever pain cough")
        soap_b = self._make_soap("pain nausea cough")
        r_a = self.gen.evaluate_against_reference(soap_a, "pain nausea cough")
        r_b = self.gen.evaluate_against_reference(soap_b, "fever pain cough")
        # Jaccard is symmetric
        assert r_a["jaccard"] == r_b["jaccard"]


# ---------------------------------------------------------------------------
# Red flag detection
# ---------------------------------------------------------------------------

class TestRedFlagDetection:
    @pytest.fixture(autouse=True)
    def _setup(self, generator):
        self.gen = generator

    def _soap_with_text(self, text: str) -> SOAPNote:
        return SOAPNote(
            conversation_id="rf_001",
            speciality="General Medicine",
            subjective=text,
            objective="",
            assessment="",
            plan="",
        )

    def test_detects_chest_pain(self):
        soap = self._soap_with_text("Patient reports chest pain since yesterday.")
        flags = SOAPGenerator._detect_red_flags(soap)
        assert "chest pain" in flags

    def test_detects_shortness_of_breath(self):
        soap = self._soap_with_text("Presenting with shortness of breath.")
        flags = SOAPGenerator._detect_red_flags(soap)
        assert "shortness of breath" in flags

    def test_detects_seizure(self):
        soap = self._soap_with_text("History of seizure disorder.")
        flags = SOAPGenerator._detect_red_flags(soap)
        assert "seizure" in flags

    def test_detects_stroke(self):
        soap = self._soap_with_text("Assessment: possible stroke.")
        flags = SOAPGenerator._detect_red_flags(soap)
        assert "stroke" in flags

    def test_detects_suicidal(self):
        soap = self._soap_with_text("Patient expressed suicidal ideation.")
        flags = SOAPGenerator._detect_red_flags(soap)
        assert "suicidal" in flags

    def test_detects_sepsis(self):
        soap = self._soap_with_text("Plan: rule out sepsis.")
        flags = SOAPGenerator._detect_red_flags(soap)
        assert "sepsis" in flags

    def test_no_flags_in_clean_note(self):
        soap = self._soap_with_text(
            "Patient reports mild cold symptoms. No concerning features."
        )
        flags = SOAPGenerator._detect_red_flags(soap)
        assert flags == []

    def test_detection_case_insensitive(self):
        soap = self._soap_with_text("CHEST PAIN reported on exertion.")
        flags = SOAPGenerator._detect_red_flags(soap)
        assert "chest pain" in flags

    def test_flags_scanned_across_all_sections(self):
        soap = SOAPNote(
            conversation_id="rf_002",
            speciality="General Medicine",
            subjective="mild headache",
            objective="BP elevated",
            assessment="Possible stroke event",
            plan="CT head urgent",
        )
        flags = SOAPGenerator._detect_red_flags(soap)
        assert "stroke" in flags

    def test_generate_populates_red_flag_mentions(self):
        t = _make_transcript(
            patient_text="I have severe chest pain and shortness of breath."
        )
        e = _make_entities(symptoms=["chest pain", "shortness of breath"])
        gen = SOAPGenerator(api_key=None)
        soap = gen.generate(t, e)
        # Template fallback includes symptoms in subjective
        assert isinstance(soap.red_flag_mentions, list)

    def test_partial_word_not_matched(self):
        # "strokelike" should not match "stroke" with word boundary
        soap = self._soap_with_text("Symptoms are strokelike in nature.")
        flags = SOAPGenerator._detect_red_flags(soap)
        assert "stroke" not in flags


# ---------------------------------------------------------------------------
# generate_batch() — ID matching and failure isolation
# ---------------------------------------------------------------------------

class TestGenerateBatch:
    def _make_pair(self, id_: str):
        t = _make_transcript(id_=id_)
        e = _make_entities(id_=id_)
        return t, e

    def test_returns_list(self, generator):
        t, e = self._make_pair("b_001")
        results = generator.generate_batch([t], [e], show_progress=False)
        assert isinstance(results, list)

    def test_all_matched_records_succeed(self, generator):
        pairs = [self._make_pair(f"b_{i:03d}") for i in range(3)]
        ts, es = zip(*pairs)
        results = generator.generate_batch(list(ts), list(es), show_progress=False)
        assert len(results) == 3

    def test_results_are_soap_notes(self, generator):
        t, e = self._make_pair("b_001")
        results = generator.generate_batch([t], [e], show_progress=False)
        assert all(isinstance(r, SOAPNote) for r in results)

    def test_mismatched_ids_skipped(self, generator):
        t = _make_transcript(id_="transcript_A")
        e = _make_entities(id_="entities_B")   # different id
        results = generator.generate_batch([t], [e], show_progress=False)
        # Transcript has no matching entities → skipped
        assert results == []

    def test_partial_mismatch_only_matched_returned(self, generator):
        t1 = _make_transcript(id_="match_1")
        t2 = _make_transcript(id_="no_match")
        e1 = _make_entities(id_="match_1")
        # e for t2 is absent
        results = generator.generate_batch([t1, t2], [e1], show_progress=False)
        assert len(results) == 1
        assert results[0].conversation_id == "match_1"

    def test_empty_inputs_return_empty(self, generator):
        assert generator.generate_batch([], [], show_progress=False) == []

    def test_ids_in_results_match_transcripts(self, generator):
        pairs = [self._make_pair(f"id_{i}") for i in range(3)]
        ts, es = zip(*pairs)
        results = generator.generate_batch(list(ts), list(es), show_progress=False)
        result_ids = {r.conversation_id for r in results}
        assert result_ids == {f"id_{i}" for i in range(3)}

    def test_show_progress_does_not_crash(self, generator, capsys):
        t, e = self._make_pair("prog_001")
        generator.generate_batch([t], [e], show_progress=True)
        captured = capsys.readouterr()
        assert "succeeded" in captured.out


# ---------------------------------------------------------------------------
# Fallback mode (google-generativeai not installed)
# ---------------------------------------------------------------------------

class TestFallbackMode:
    def test_initialises_without_genai(self):
        with patch("medai.src.clinical.soap_generator._GENAI_AVAILABLE", False):
            gen = SOAPGenerator(api_key="fake-key")
        assert gen.use_fallback is True
        assert gen._client is None

    def test_no_api_key_sets_fallback(self):
        with patch.dict("os.environ", {}, clear=True):
            gen = SOAPGenerator(api_key=None)
        assert gen.use_fallback is True

    def test_fallback_generate_returns_soap_note(self):
        with patch("medai.src.clinical.soap_generator._GENAI_AVAILABLE", False):
            gen = SOAPGenerator(api_key="fake-key")
        t = _make_transcript()
        e = _make_entities()
        result = gen.generate(t, e)
        assert isinstance(result, SOAPNote)

    def test_fallback_method_is_template_fallback(self):
        with patch("medai.src.clinical.soap_generator._GENAI_AVAILABLE", False):
            gen = SOAPGenerator(api_key="fake-key")
        t = _make_transcript()
        e = _make_entities()
        result = gen.generate(t, e)
        assert result.generation_method == "template_fallback"

    def test_llm_exception_falls_back_to_template(self):
        """If _call_gemini raises, generate() must fall back silently."""
        gen = SOAPGenerator.__new__(SOAPGenerator)
        gen.model_name = "gemini-1.5-flash"
        gen.use_fallback = False
        gen._prompt_template = "dummy {symptoms}{clinical_findings}{medications}{medical_history}{vitals}{negations}{speciality_hints}{transcript}"

        # Mock client that always raises
        mock_client = MagicMock()
        mock_client.generate_content.side_effect = RuntimeError("API down")
        gen._client = mock_client

        t = _make_transcript()
        e = _make_entities()
        result = gen.generate(t, e)
        # Should have fallen back without raising
        assert isinstance(result, SOAPNote)
        assert result.generation_method == "template_fallback"


# ---------------------------------------------------------------------------
# _build_prompt()
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_symptoms(self, generator, basic_transcript, basic_entities):
        prompt = generator._build_prompt(basic_transcript, basic_entities)
        assert "chest pain" in prompt

    def test_contains_vitals(self, generator, basic_transcript, basic_entities):
        prompt = generator._build_prompt(basic_transcript, basic_entities)
        assert "130/85" in prompt or "BP" in prompt

    def test_transcript_truncated_at_max_words(self, generator):
        long_text = " ".join([f"PATIENT: word{i}" for i in range(3000)])
        t = _make_transcript()
        t = ProcessedTranscript(
            id=t.id, source_dataset=t.source_dataset, language=t.language,
            turns=t.turns, doctor_text=t.doctor_text, patient_text=t.patient_text,
            full_text=long_text, turn_count=t.turn_count, word_count=3000,
            has_reference_note=False, reference_note=None,
        )
        e = _make_entities()
        prompt = generator._build_prompt(t, e)
        assert "truncated" in prompt

    def test_no_api_key_does_not_affect_build_prompt(self, basic_transcript, basic_entities):
        gen = SOAPGenerator(api_key=None)
        prompt = gen._build_prompt(basic_transcript, basic_entities)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ---------------------------------------------------------------------------
# _infer_speciality()
# ---------------------------------------------------------------------------

class TestInferSpeciality:
    def test_returns_general_medicine_when_no_hints(self):
        e = _make_entities(speciality_hints=[])
        assert SOAPGenerator._infer_speciality(e) == "General Medicine"

    def test_returns_capitalised_hint(self):
        e = _make_entities(speciality_hints=["cardiology"])
        assert SOAPGenerator._infer_speciality(e) == "Cardiology"

    def test_returns_most_common_hint(self):
        e = _make_entities(speciality_hints=["cardiology", "neurology", "cardiology"])
        assert SOAPGenerator._infer_speciality(e) == "Cardiology"

    def test_single_hint_returned(self):
        e = _make_entities(speciality_hints=["oncology"])
        assert SOAPGenerator._infer_speciality(e) == "Oncology"


# ---------------------------------------------------------------------------
# _tokenise() helper
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_lowercases_text(self):
        assert "fever" in _tokenise("FEVER")

    def test_strips_punctuation(self):
        result = _tokenise("pain, fever.")
        assert "pain" in result
        assert "fever" in result
        assert "," not in result

    def test_returns_set(self):
        assert isinstance(_tokenise("hello world"), set)

    def test_empty_string_returns_empty_set(self):
        assert _tokenise("") == set()

    def test_deduplicates_words(self):
        result = _tokenise("fever fever fever")
        assert len(result) == 1
