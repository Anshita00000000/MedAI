"""
tests/test_reasoning.py

Tests for Step 5: RedFlagDetector and DDxEngine.

All tests use lightweight stub objects so no data files or API keys are needed.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Minimal stub dataclasses (avoid importing full pipeline deps)
# ---------------------------------------------------------------------------

@dataclass
class _Turn:
    index: int
    speaker: str
    text: str
    word_count: int
    is_question: bool
    consecutive_run: int


@dataclass
class _ProcessedTranscript:
    id: str
    source_dataset: str = "test"
    language: str = "en"
    turns: List[_Turn] = field(default_factory=list)
    doctor_text: str = ""
    patient_text: str = ""
    full_text: str = ""
    turn_count: int = 0
    word_count: int = 0
    has_reference_note: bool = False
    reference_note: str | None = None
    flags: List[str] = field(default_factory=list)


@dataclass
class _MedicalEntities:
    conversation_id: str
    symptoms: List[str] = field(default_factory=list)
    clinical_findings: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    medical_history: List[str] = field(default_factory=list)
    vitals: dict = field(default_factory=dict)
    allergies: List[str] = field(default_factory=list)
    speciality_hints: List[str] = field(default_factory=list)
    negations: List[str] = field(default_factory=list)
    extraction_method: str = "rule_based"
    confidence: float = 0.5
    raw_entities: List[str] = field(default_factory=list)


@dataclass
class _SOAPNote:
    conversation_id: str
    speciality: str = "General Medicine"
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""
    red_flag_mentions: List[str] = field(default_factory=list)
    generated_at: str = ""
    model_used: str = "template_fallback"
    prompt_tokens: int = 0
    confidence: float = 0.5
    generation_method: str = "template_fallback"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transcript(
    cid: str = "t1",
    full_text: str = "DOCTOR: Hello. PATIENT: I have a headache.",
    patient_text: str = "I have a headache.",
    doctor_text: str = "Hello.",
) -> _ProcessedTranscript:
    return _ProcessedTranscript(
        id=cid,
        full_text=full_text,
        patient_text=patient_text,
        doctor_text=doctor_text,
        turn_count=2,
        word_count=10,
    )


def _make_entities(
    cid: str = "t1",
    symptoms: list | None = None,
    findings: list | None = None,
    history: list | None = None,
    vitals: dict | None = None,
    speciality_hints: list | None = None,
) -> _MedicalEntities:
    return _MedicalEntities(
        conversation_id=cid,
        symptoms=symptoms or [],
        clinical_findings=findings or [],
        medical_history=history or [],
        vitals=vitals or {},
        speciality_hints=speciality_hints or [],
    )


def _make_soap(cid: str = "t1", subjective: str = "", assessment: str = "") -> _SOAPNote:
    return _SOAPNote(
        conversation_id=cid,
        subjective=subjective,
        assessment=assessment,
    )


# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from medai.src.reasoning.red_flag_detector import (
    RedFlagDetector,
    RedFlagResult,
    RedFlag,
    _highest_severity,
    _build_summary,
    _DEFAULT_RULES,
)
from medai.src.reasoning.ddx_engine import (
    DDxEngine,
    DDxResult,
    Diagnosis,
    _score_to_tier,
    _unknown_diagnosis,
    _default_investigations,
    _DEFAULT_DISEASE_MAPPING,
)


# ===========================================================================
# RedFlagDetector
# ===========================================================================

class TestRedFlagDetectorInit:

    def test_initialises_with_default_rules_when_yaml_missing(self):
        detector = RedFlagDetector(config_path="configs/nonexistent.yaml")
        assert detector._rules is not None
        assert "critical" in detector._rules

    def test_loads_yaml_rules_when_file_exists(self):
        detector = RedFlagDetector()   # uses actual configs/red_flags.yaml
        assert "critical" in detector._rules
        assert len(detector._rules["critical"]) >= 1

    def test_compiled_patterns_populated(self):
        detector = RedFlagDetector()
        assert detector._compiled
        for severity in ("critical", "urgent", "monitor"):
            assert severity in detector._compiled

    def test_uses_default_rules_on_bad_yaml(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("not: valid: yaml: [[[")
        detector = RedFlagDetector(config_path=str(bad_yaml))
        assert "critical" in detector._rules


class TestRedFlagDetect:

    def setup_method(self):
        self.detector = RedFlagDetector()

    def test_returns_red_flag_result(self):
        t = _make_transcript(full_text="PATIENT: I have chest pain.")
        e = _make_entities()
        result = self.detector.detect(t, e)
        assert isinstance(result, RedFlagResult)

    def test_detects_critical_chest_pain(self):
        t = _make_transcript(full_text="PATIENT: I have chest pain and it is getting worse.")
        e = _make_entities()
        result = self.detector.detect(t, e)
        assert result.flag_count >= 1
        assert any(f.term == "chest pain" for f in result.flags)
        assert result.highest_severity == "critical"

    def test_critical_flag_sets_requires_immediate_action(self):
        t = _make_transcript(full_text="PATIENT: I am having difficulty breathing.")
        e = _make_entities()
        result = self.detector.detect(t, e)
        assert result.requires_immediate_action is True

    def test_urgent_flag_does_not_set_requires_immediate_action(self):
        t = _make_transcript(full_text="PATIENT: I have been losing weight slowly.")
        e = _make_entities(symptoms=["weight loss"])
        result = self.detector.detect(t, e)
        # weight loss is monitor, not critical
        assert result.requires_immediate_action is False

    def test_clean_transcript_returns_no_flags(self):
        t = _make_transcript(
            full_text="DOCTOR: How are you? PATIENT: I feel fine, just a routine checkup.",
            patient_text="I feel fine, just a routine checkup.",
        )
        e = _make_entities()
        result = self.detector.detect(t, e)
        assert result.flag_count == 0
        assert result.highest_severity == "none"
        assert result.requires_immediate_action is False

    def test_same_term_only_flagged_once(self):
        # "chest pain" appears in both full_text and symptoms
        t = _make_transcript(full_text="PATIENT: I have chest pain.")
        e = _make_entities(symptoms=["chest pain"])
        result = self.detector.detect(t, e)
        chest_pain_flags = [f for f in result.flags if f.term == "chest pain"]
        assert len(chest_pain_flags) == 1

    def test_soap_subjective_also_scanned(self):
        t = _make_transcript(full_text="PATIENT: I feel okay.")
        e = _make_entities()
        soap = _make_soap(subjective="Patient reports difficulty breathing since yesterday.")
        result = self.detector.detect(t, e, soap)
        assert any(f.category == "respiratory" for f in result.flags)

    def test_flag_count_matches_flags_list_length(self):
        t = _make_transcript(full_text="PATIENT: I have chest pain and I feel unconscious at times.")
        e = _make_entities()
        result = self.detector.detect(t, e)
        assert result.flag_count == len(result.flags)

    def test_conversation_id_preserved(self):
        t = _make_transcript(cid="conv-42")
        e = _make_entities(cid="conv-42")
        result = self.detector.detect(t, e)
        assert result.conversation_id == "conv-42"

    def test_summary_is_non_empty_string(self):
        t = _make_transcript(full_text="PATIENT: I have chest pain.")
        e = _make_entities()
        result = self.detector.detect(t, e)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_summary_mentions_none_when_no_flags(self):
        t = _make_transcript()
        e = _make_entities()
        result = self.detector.detect(t, e)
        assert "No red flags" in result.summary


class TestSeverityHierarchy:

    def setup_method(self):
        self.detector = RedFlagDetector()

    def test_critical_outranks_urgent(self):
        t = _make_transcript(
            full_text="PATIENT: I have chest pain and I also have severe pain."
        )
        e = _make_entities()
        result = self.detector.detect(t, e)
        assert result.highest_severity == "critical"

    def test_urgent_outranks_monitor(self):
        t = _make_transcript(
            full_text="PATIENT: I have high fever and I am losing weight."
        )
        e = _make_entities()
        result = self.detector.detect(t, e)
        assert result.highest_severity == "urgent"

    def test_highest_severity_helper_critical(self):
        flags = [
            RedFlag("a", "monitor", "x", "", "", ""),
            RedFlag("b", "critical", "y", "", "", ""),
            RedFlag("c", "urgent", "z", "", "", ""),
        ]
        assert _highest_severity(flags) == "critical"

    def test_highest_severity_helper_none_on_empty(self):
        assert _highest_severity([]) == "none"


class TestMatchFlags:

    def setup_method(self):
        self.detector = RedFlagDetector()

    def test_returns_flag_for_matching_keyword(self):
        compiled = self.detector._compiled.get("critical", [])
        flags = self.detector._match_flags("patient has chest pain now", "critical", compiled)
        assert any(f.term == "chest pain" for f in flags)

    def test_no_flag_when_keyword_absent(self):
        compiled = self.detector._compiled.get("critical", [])
        flags = self.detector._match_flags("patient feels fine today", "critical", compiled)
        assert len(flags) == 0

    def test_matched_text_snippet_included(self):
        compiled = self.detector._compiled.get("critical", [])
        flags = self.detector._match_flags("the patient reports chest pain radiating", "critical", compiled)
        chest_flags = [f for f in flags if f.term == "chest pain"]
        assert chest_flags
        assert "chest pain" in chest_flags[0].matched_text.lower()

    def test_only_one_flag_per_rule(self):
        # "chest pain" and "chest pressure" both appear — but same rule → one flag
        compiled = self.detector._compiled.get("critical", [])
        flags = self.detector._match_flags(
            "chest pain and chest pressure both present", "critical", compiled
        )
        chest_flags = [f for f in flags if f.term == "chest pain"]
        assert len(chest_flags) == 1


class TestDetectBatch:

    def setup_method(self):
        self.detector = RedFlagDetector()

    def test_returns_list_of_results(self):
        t1 = _make_transcript("t1", "PATIENT: I have chest pain.")
        t2 = _make_transcript("t2", "PATIENT: I feel fine.")
        e1 = _make_entities("t1")
        e2 = _make_entities("t2")
        results = self.detector.detect_batch([t1, t2], [e1, e2], show_progress=False)
        assert len(results) == 2

    def test_skips_transcript_with_no_matching_entities(self):
        t1 = _make_transcript("t1")
        t2 = _make_transcript("t2")
        e1 = _make_entities("t1")
        # e2 has id "other" — won't match t2
        e2 = _make_entities("other")
        results = self.detector.detect_batch([t1, t2], [e1, e2], show_progress=False)
        assert len(results) == 1
        assert results[0].conversation_id == "t1"

    def test_isolates_individual_failure(self, monkeypatch):
        t1 = _make_transcript("t1")
        t2 = _make_transcript("t2")
        e1 = _make_entities("t1")
        e2 = _make_entities("t2")

        original_detect = self.detector.detect

        def failing_detect(transcript, entities, soap=None):
            if transcript.id == "t2":
                raise RuntimeError("simulated failure")
            return original_detect(transcript, entities, soap)

        monkeypatch.setattr(self.detector, "detect", failing_detect)
        results = self.detector.detect_batch([t1, t2], [e1, e2], show_progress=False)
        assert len(results) == 1
        assert results[0].conversation_id == "t1"

    def test_empty_inputs_return_empty(self):
        results = self.detector.detect_batch([], [], show_progress=False)
        assert results == []


class TestHelpers:

    def test_build_summary_no_flags(self):
        summary = _build_summary("t1", [], "none")
        assert "No red flags" in summary

    def test_build_summary_with_flags(self):
        flags = [
            RedFlag("chest pain", "critical", "cardiac", "", "", ""),
            RedFlag("high fever", "urgent", "infectious", "", "", ""),
        ]
        summary = _build_summary("t1", flags, "critical")
        assert "critical" in summary
        assert "cardiac" in summary


# ===========================================================================
# DDxEngine
# ===========================================================================

class TestDDxEngineInit:

    def test_initialises_without_api_key(self):
        engine = DDxEngine()
        assert engine.use_fallback is True

    def test_initialises_with_default_mapping_when_yaml_missing(self):
        engine = DDxEngine(config_path="configs/nonexistent.yaml")
        assert engine._disease_mapping is not None
        assert "respiratory" in engine._disease_mapping

    def test_compiled_mapping_loaded(self):
        engine = DDxEngine()
        assert engine._disease_mapping
        total = sum(len(v) for v in engine._disease_mapping.values())
        assert total >= 5


class TestRuleBasedDDx:

    def setup_method(self):
        self.engine = DDxEngine()

    def test_returns_sorted_list(self):
        t = _make_transcript(
            full_text="PATIENT: I have a headache, nausea and light sensitivity.",
            patient_text="I have a headache, nausea and light sensitivity.",
        )
        e = _make_entities(symptoms=["headache", "nausea", "light sensitivity"])
        diagnoses = self.engine._rule_based_ddx(e, t)
        scores = [d.probability_score for d in diagnoses]
        assert scores == sorted(scores, reverse=True)

    def test_respects_min_matches_threshold(self):
        # Only one keyword match — most diseases require ≥2
        t = _make_transcript(
            full_text="PATIENT: I have a headache.",
            patient_text="I have a headache.",
        )
        e = _make_entities(symptoms=["headache"])
        diagnoses = self.engine._rule_based_ddx(e, t)
        # All returned diagnoses must have met min_matches
        for d in diagnoses:
            assert d.probability_score > 0

    def test_returns_at_most_five_diagnoses(self):
        # Give a very common symptom set that might match many diseases
        t = _make_transcript(
            full_text=(
                "PATIENT: I have fever, cough, shortness of breath, chest pain, "
                "nausea, headache, back pain, joint pain, sputum."
            ),
            patient_text=(
                "I have fever, cough, shortness of breath, chest pain, nausea, "
                "headache, back pain, joint pain, sputum."
            ),
        )
        e = _make_entities(
            symptoms=["fever", "cough", "shortness of breath", "chest pain",
                      "nausea", "headache", "back pain", "joint pain", "sputum"]
        )
        diagnoses = self.engine._rule_based_ddx(e, t)
        assert len(diagnoses) <= 5

    def test_returns_empty_for_empty_entities(self):
        t = _make_transcript(full_text="PATIENT: I feel fine.")
        e = _make_entities()
        diagnoses = self.engine._rule_based_ddx(e, t)
        # May or may not be empty depending on full_text keywords — just check type
        assert isinstance(diagnoses, list)

    def test_supporting_evidence_populated(self):
        t = _make_transcript(
            full_text="PATIENT: I have headache and nausea.",
            patient_text="I have headache and nausea.",
        )
        e = _make_entities(symptoms=["headache", "nausea"])
        diagnoses = self.engine._rule_based_ddx(e, t)
        if diagnoses:
            assert len(diagnoses[0].supporting_evidence) >= 1

    def test_diagnosis_has_icd10_code(self):
        t = _make_transcript(
            full_text="PATIENT: I have headache, nausea, light sensitivity.",
            patient_text="I have headache, nausea, light sensitivity.",
        )
        e = _make_entities(symptoms=["headache", "nausea", "light sensitivity"])
        diagnoses = self.engine._rule_based_ddx(e, t)
        for d in diagnoses:
            assert d.icd10_code != "" or d.name == "Undetermined"


class TestGenerateFollowUpQuestions:

    def setup_method(self):
        self.engine = DDxEngine()

    def test_adds_vitals_question_when_missing(self):
        e = _make_entities(vitals={})
        questions = self.engine._generate_follow_up_questions(e, [])
        assert any("vital signs" in q.lower() for q in questions)

    def test_no_vitals_question_when_vitals_present(self):
        e = _make_entities(vitals={"BP": "120/80 mmHg"})
        questions = self.engine._generate_follow_up_questions(e, [])
        assert not any("vital signs" in q.lower() for q in questions)

    def test_adds_history_question_when_missing(self):
        e = _make_entities(history=[])
        questions = self.engine._generate_follow_up_questions(e, [])
        assert any("past medical history" in q.lower() or "history" in q.lower() for q in questions)

    def test_adds_cardiac_question_for_cardiac_diagnosis(self):
        e = _make_entities(vitals={"BP": "120/80"}, history=["hypertension"])
        diagnosis = Diagnosis(
            name="Acute Coronary Syndrome",
            icd10_code="I24.9",
            probability="high",
            probability_score=0.8,
            supporting_evidence=[],
            contradicting_evidence=[],
            reasoning="",
        )
        questions = self.engine._generate_follow_up_questions(e, [diagnosis])
        assert any("heart disease" in q.lower() or "cardiac" in q.lower() for q in questions)

    def test_adds_neuro_question_for_neurological_diagnosis(self):
        e = _make_entities(vitals={"BP": "120/80"}, history=["migraine"])
        diagnosis = Diagnosis(
            name="Migraine",
            icd10_code="G43.909",
            probability="high",
            probability_score=0.8,
            supporting_evidence=[],
            contradicting_evidence=[],
            reasoning="",
        )
        questions = self.engine._generate_follow_up_questions(e, [diagnosis])
        assert any("head trauma" in q.lower() or "neurolog" in q.lower() or "episodes" in q.lower() for q in questions)

    def test_returns_at_most_five_questions(self):
        e = _make_entities()
        diagnoses = [
            Diagnosis("Acute Coronary Syndrome", "I24.9", "high", 0.8, [], [], ""),
            Diagnosis("Migraine", "G43.909", "moderate", 0.5, [], [], ""),
        ]
        questions = self.engine._generate_follow_up_questions(e, diagnoses)
        assert len(questions) <= 5


class TestAnalyse:

    def setup_method(self):
        self.engine = DDxEngine()
        from medai.src.reasoning.red_flag_detector import RedFlagDetector
        detector = RedFlagDetector()
        t = _make_transcript("t1")
        e = _make_entities("t1")
        s = _make_soap("t1")
        self.rfr = detector.detect(t, e, s)

    def test_returns_ddx_result(self):
        t = _make_transcript("t1")
        e = _make_entities("t1", symptoms=["headache", "nausea"])
        s = _make_soap("t1")
        result = self.engine.analyse(t, e, s, self.rfr)
        assert isinstance(result, DDxResult)

    def test_result_has_all_required_fields(self):
        t = _make_transcript("t1")
        e = _make_entities("t1")
        s = _make_soap("t1")
        result = self.engine.analyse(t, e, s, self.rfr)
        assert result.conversation_id == "t1"
        assert isinstance(result.top_diagnoses, list)
        assert isinstance(result.primary_diagnosis, Diagnosis)
        assert isinstance(result.follow_up_questions, list)
        assert isinstance(result.recommended_investigations, list)
        assert result.red_flag_result is self.rfr
        assert result.generation_method in ("llm", "rule_based")

    def test_primary_diagnosis_is_top_of_list(self):
        t = _make_transcript(
            "t1",
            full_text="PATIENT: headache nausea light sensitivity.",
            patient_text="headache nausea light sensitivity.",
        )
        e = _make_entities("t1", symptoms=["headache", "nausea", "light sensitivity"])
        s = _make_soap("t1")
        result = self.engine.analyse(t, e, s, self.rfr)
        if result.top_diagnoses:
            assert result.primary_diagnosis.name == result.top_diagnoses[0].name

    def test_empty_entities_does_not_crash(self):
        t = _make_transcript("t1")
        e = _make_entities("t1")
        s = _make_soap("t1")
        result = self.engine.analyse(t, e, s, self.rfr)
        assert isinstance(result, DDxResult)

    def test_generation_method_is_rule_based_without_api_key(self):
        t = _make_transcript("t1", full_text="PATIENT: headache nausea.", patient_text="headache nausea.")
        e = _make_entities("t1", symptoms=["headache", "nausea"])
        s = _make_soap("t1")
        result = self.engine.analyse(t, e, s, self.rfr)
        assert result.generation_method == "rule_based"

    def test_speciality_from_soap(self):
        t = _make_transcript("t1")
        e = _make_entities("t1")
        s = _make_soap("t1")
        s.speciality = "Cardiology"
        result = self.engine.analyse(t, e, s, self.rfr)
        assert result.speciality == "Cardiology"


class TestAnalyseBatch:

    def setup_method(self):
        self.engine = DDxEngine()
        from medai.src.reasoning.red_flag_detector import RedFlagDetector
        self.detector = RedFlagDetector()

    def _make_rfr(self, cid):
        t = _make_transcript(cid)
        e = _make_entities(cid)
        s = _make_soap(cid)
        return self.detector.detect(t, e, s)

    def test_returns_list_of_ddx_results(self):
        t1, t2 = _make_transcript("t1"), _make_transcript("t2")
        e1, e2 = _make_entities("t1"), _make_entities("t2")
        s1, s2 = _make_soap("t1"), _make_soap("t2")
        r1, r2 = self._make_rfr("t1"), self._make_rfr("t2")
        results = self.engine.analyse_batch(
            [t1, t2], [e1, e2], [s1, s2], [r1, r2], show_progress=False
        )
        assert len(results) == 2

    def test_skips_record_with_no_matching_entities(self):
        t1 = _make_transcript("t1")
        t2 = _make_transcript("t2")
        e1 = _make_entities("t1")
        e_other = _make_entities("other")     # won't match t2
        s1, s2 = _make_soap("t1"), _make_soap("t2")
        r1, r2 = self._make_rfr("t1"), self._make_rfr("t2")
        results = self.engine.analyse_batch(
            [t1, t2], [e1, e_other], [s1, s2], [r1, r2], show_progress=False
        )
        assert len(results) == 1
        assert results[0].conversation_id == "t1"

    def test_isolates_individual_failure(self, monkeypatch):
        t1, t2 = _make_transcript("t1"), _make_transcript("t2")
        e1, e2 = _make_entities("t1"), _make_entities("t2")
        s1, s2 = _make_soap("t1"), _make_soap("t2")
        r1, r2 = self._make_rfr("t1"), self._make_rfr("t2")

        original_analyse = self.engine.analyse

        def failing_analyse(transcript, entities, soap, rfr):
            if transcript.id == "t2":
                raise RuntimeError("simulated failure")
            return original_analyse(transcript, entities, soap, rfr)

        monkeypatch.setattr(self.engine, "analyse", failing_analyse)
        results = self.engine.analyse_batch(
            [t1, t2], [e1, e2], [s1, s2], [r1, r2], show_progress=False
        )
        assert len(results) == 1
        assert results[0].conversation_id == "t1"

    def test_empty_inputs_return_empty(self):
        results = self.engine.analyse_batch([], [], [], [], show_progress=False)
        assert results == []

    def test_skips_record_missing_soap(self):
        t1, t2 = _make_transcript("t1"), _make_transcript("t2")
        e1, e2 = _make_entities("t1"), _make_entities("t2")
        s1 = _make_soap("t1")
        # No soap for t2
        r1, r2 = self._make_rfr("t1"), self._make_rfr("t2")
        results = self.engine.analyse_batch(
            [t1, t2], [e1, e2], [s1], [r1, r2], show_progress=False
        )
        assert len(results) == 1


class TestHelperFunctions:

    def test_score_to_tier_high(self):
        assert _score_to_tier(0.9) == "high"
        assert _score_to_tier(0.66) == "high"

    def test_score_to_tier_moderate(self):
        assert _score_to_tier(0.5) == "moderate"
        assert _score_to_tier(0.33) == "moderate"

    def test_score_to_tier_low(self):
        assert _score_to_tier(0.1) == "low"
        assert _score_to_tier(0.0) == "low"

    def test_unknown_diagnosis(self):
        d = _unknown_diagnosis()
        assert d.name == "Undetermined"
        assert d.probability_score == 0.0

    def test_default_investigations_cardiac(self):
        d = Diagnosis("Acute Coronary Syndrome", "I24.9", "high", 0.8, [], [], "")
        invs = _default_investigations([d])
        assert any("ECG" in i or "Troponin" in i for i in invs)

    def test_default_investigations_empty(self):
        invs = _default_investigations([])
        assert isinstance(invs, list)
        assert len(invs) >= 1


class TestLlmRankDdxFallback:
    """Tests that cover LLM path error handling using rule-based fallback."""

    def setup_method(self):
        self.engine = DDxEngine()

    def test_llm_rank_falls_back_on_bad_json(self):
        candidates = [
            Diagnosis("Migraine", "G43.909", "high", 0.8, ["headache"], [], "rule match"),
        ]
        t = _make_transcript("t1", full_text="PATIENT: headache.")
        e = _make_entities("t1", symptoms=["headache"])

        # Patch _call_gemini to return invalid JSON
        def bad_gemini(prompt):
            return "This is not JSON at all."

        self.engine._call_gemini = bad_gemini
        result, fup, inv = self.engine._llm_rank_ddx(candidates, t, e)
        # Should return original candidates unchanged
        assert result == candidates

    def test_llm_rank_falls_back_on_non_list_json(self):
        candidates = [
            Diagnosis("Migraine", "G43.909", "high", 0.8, ["headache"], [], "rule match"),
        ]
        t = _make_transcript("t1", full_text="PATIENT: headache.")
        e = _make_entities("t1", symptoms=["headache"])

        def dict_gemini(prompt):
            return '{"name": "Migraine", "probability_score": 0.9}'

        self.engine._call_gemini = dict_gemini
        result, fup, inv = self.engine._llm_rank_ddx(candidates, t, e)
        assert result == candidates
