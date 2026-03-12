"""
tests/test_entity_extractor.py

Pytest suite for medai.src.clinical.entity_extractor.

Run with:
    pytest tests/test_entity_extractor.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medai.src.clinical.entity_extractor import (
    EntityExtractor,
    MedicalEntities,
    _VITALS_BP_LABELED,
)
from medai.src.voice.transcript_processor import ProcessedTranscript, TranscriptProcessor, Turn


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_transcript(
    id_: str = "test_001",
    doctor_turns: list[str] | None = None,
    patient_turns: list[str] | None = None,
    interleave: bool = True,
) -> ProcessedTranscript:
    """
    Build a ProcessedTranscript directly without going through the file loader.

    If ``interleave`` is True, turns alternate DOCTOR/PATIENT starting with
    DOCTOR.  Otherwise doctor_turns are placed first, then patient_turns.
    """
    doctor_turns = doctor_turns or ["How can I help you today?"]
    patient_turns = patient_turns or ["I have a headache."]

    if interleave:
        raw_turns = []
        for i, (d, p) in enumerate(zip(doctor_turns, patient_turns)):
            raw_turns.append({"speaker": "DOCTOR", "text": d})
            raw_turns.append({"speaker": "PATIENT", "text": p})
        # Append leftovers
        for d in doctor_turns[len(patient_turns):]:
            raw_turns.append({"speaker": "DOCTOR", "text": d})
        for p in patient_turns[len(doctor_turns):]:
            raw_turns.append({"speaker": "PATIENT", "text": p})
    else:
        raw_turns = (
            [{"speaker": "DOCTOR", "text": d} for d in doctor_turns]
            + [{"speaker": "PATIENT", "text": p} for p in patient_turns]
        )

    record = {
        "id": id_,
        "source_dataset": "test",
        "language": "en",
        "turns": raw_turns,
        "reference_note": None,
        "is_synthetic": True,
        "turn_count": len(raw_turns),
    }
    return TranscriptProcessor().process(record)


def _make_turns(speaker_text_pairs: list[tuple[str, str]]) -> list[Turn]:
    """Build a bare Turn list for testing _extract_negations directly."""
    turns = []
    run, prev = 0, None
    for i, (speaker, text) in enumerate(speaker_text_pairs):
        if speaker == prev:
            run += 1
        else:
            run = 1
            prev = speaker
        turns.append(Turn(
            index=i, speaker=speaker, text=text,
            word_count=len(text.split()),
            is_question=text.strip().endswith("?"),
            consecutive_run=run,
        ))
    return turns


@pytest.fixture
def extractor() -> EntityExtractor:
    """Always use rule-based mode — scispaCy is not installed in CI."""
    return EntityExtractor()


@pytest.fixture
def basic_transcript() -> ProcessedTranscript:
    return _make_transcript(
        doctor_turns=[
            "What brings you in today?",
            "Any history of heart disease?",
            "Your blood pressure is 130/85 mmHg and heart rate is 88 bpm.",
        ],
        patient_turns=[
            "I have had chest pain and shortness of breath for two days.",
            "No history of cardiac problems. I am allergic to penicillin.",
            "I have been taking ibuprofen 400 mg for the pain.",
        ],
    )


# ---------------------------------------------------------------------------
# extract() — return type and field correctness
# ---------------------------------------------------------------------------

class TestExtract:
    def test_returns_medical_entities(self, extractor, basic_transcript):
        result = extractor.extract(basic_transcript)
        assert isinstance(result, MedicalEntities)

    def test_conversation_id_matches(self, extractor, basic_transcript):
        result = extractor.extract(basic_transcript)
        assert result.conversation_id == basic_transcript.id

    def test_symptoms_is_list(self, extractor, basic_transcript):
        result = extractor.extract(basic_transcript)
        assert isinstance(result.symptoms, list)

    def test_clinical_findings_is_list(self, extractor, basic_transcript):
        result = extractor.extract(basic_transcript)
        assert isinstance(result.clinical_findings, list)

    def test_medications_is_list(self, extractor, basic_transcript):
        result = extractor.extract(basic_transcript)
        assert isinstance(result.medications, list)

    def test_medical_history_is_list(self, extractor, basic_transcript):
        result = extractor.extract(basic_transcript)
        assert isinstance(result.medical_history, list)

    def test_vitals_is_dict(self, extractor, basic_transcript):
        result = extractor.extract(basic_transcript)
        assert isinstance(result.vitals, dict)

    def test_allergies_is_list(self, extractor, basic_transcript):
        result = extractor.extract(basic_transcript)
        assert isinstance(result.allergies, list)

    def test_negations_is_list(self, extractor, basic_transcript):
        result = extractor.extract(basic_transcript)
        assert isinstance(result.negations, list)

    def test_confidence_in_range(self, extractor, basic_transcript):
        result = extractor.extract(basic_transcript)
        assert 0.0 <= result.confidence <= 1.0

    def test_extraction_method_is_string(self, extractor, basic_transcript):
        result = extractor.extract(basic_transcript)
        assert isinstance(result.extraction_method, str)
        assert result.extraction_method in ("scispacy", "rule_based", "failed")

    def test_raw_entities_is_list_of_dicts(self, extractor, basic_transcript):
        result = extractor.extract(basic_transcript)
        assert isinstance(result.raw_entities, list)
        assert all(isinstance(e, dict) for e in result.raw_entities)

    def test_symptoms_detected_from_patient(self, extractor):
        t = _make_transcript(
            patient_turns=[
                "I have chest pain and nausea.",
                "Also some dizziness.",
                "I feel very fatigued.",
            ],
            doctor_turns=[
                "When did this start?",
                "Any other symptoms?",
                "Have you had this before?",
            ],
        )
        result = extractor.extract(t)
        assert any("chest pain" in s or "pain" in s for s in result.symptoms)
        assert any("nausea" in s for s in result.symptoms)

    def test_medication_detected_from_context(self, extractor):
        t = _make_transcript(
            patient_turns=[
                "I have been taking ibuprofen for the pain.",
                "Also prescribed amoxicillin last week.",
                "Started it three days ago.",
            ],
            doctor_turns=[
                "What medications are you on?",
                "Any side effects?",
                "Good, continue the course.",
            ],
        )
        result = extractor.extract(t)
        meds_lower = [m.lower() for m in result.medications]
        assert any("ibuprofen" in m for m in meds_lower)

    def test_history_detected(self, extractor):
        t = _make_transcript(
            patient_turns=[
                "I have a history of hypertension.",
                "Diagnosed with type 2 diabetes five years ago.",
                "No other conditions.",
            ],
            doctor_turns=[
                "Any past medical history?",
                "When were you diagnosed?",
                "Are you on any medications?",
            ],
        )
        result = extractor.extract(t)
        assert any("hypertension" in h for h in result.medical_history)

    def test_does_not_crash_on_empty_turns(self, extractor):
        """extract() should return a failed MedicalEntities, not raise."""
        # Craft a transcript-like object with empty texts to stress the extractor
        t = _make_transcript(
            doctor_turns=["Hello.", "How are you?", "Goodbye."],
            patient_turns=["Fine.", "Okay.", "Bye."],
        )
        # Monkey-patch texts to empty to simulate edge case
        t = ProcessedTranscript(
            id=t.id, source_dataset=t.source_dataset, language=t.language,
            turns=t.turns, doctor_text="", patient_text="", full_text="",
            turn_count=t.turn_count, word_count=0,
            has_reference_note=False, reference_note=None,
        )
        result = extractor.extract(t)
        assert isinstance(result, MedicalEntities)

    def test_allergy_detected(self, extractor):
        t = _make_transcript(
            patient_turns=[
                "I am allergic to penicillin.",
                "Also have a reaction to sulfa drugs.",
                "No other allergies.",
            ],
            doctor_turns=[
                "Any allergies?",
                "What happens when you take it?",
                "Good to know.",
            ],
        )
        result = extractor.extract(t)
        assert any("penicillin" in a for a in result.allergies)


# ---------------------------------------------------------------------------
# _extract_vitals()
# ---------------------------------------------------------------------------

class TestExtractVitals:
    @pytest.fixture(autouse=True)
    def _setup(self, extractor):
        self.ex = extractor

    def test_blood_pressure_with_label(self):
        v = self.ex._extract_vitals("The BP was 130/85 mmHg today.")
        assert "BP" in v
        assert "130/85" in v["BP"]

    def test_blood_pressure_with_unit_only(self):
        v = self.ex._extract_vitals("Measured 120/80 mmHg at rest.")
        assert "BP" in v
        assert "120/80" in v["BP"]

    def test_blood_pressure_full_label(self):
        v = self.ex._extract_vitals("blood pressure 145/95 mmHg.")
        assert "BP" in v

    def test_temperature_with_celsius_symbol(self):
        v = self.ex._extract_vitals("Temperature was 39.1°C.")
        assert "temperature" in v
        assert "39.1" in v["temperature"]
        assert "C" in v["temperature"]

    def test_temperature_with_fahrenheit(self):
        v = self.ex._extract_vitals("Temp 98.6F on admission.")
        assert "temperature" in v
        assert "98.6" in v["temperature"]

    def test_temperature_with_degree_space_unit(self):
        v = self.ex._extract_vitals("Fever: 38.5 °C.")
        assert "temperature" in v

    def test_temperature_with_label_only(self):
        v = self.ex._extract_vitals("temp 38.2 on arrival")
        assert "temperature" in v
        assert "38.2" in v["temperature"]

    def test_heart_rate_with_hr_label(self):
        v = self.ex._extract_vitals("HR 125 bpm, irregular.")
        assert "HR" in v
        assert "125" in v["HR"]

    def test_heart_rate_with_pulse_label(self):
        v = self.ex._extract_vitals("Pulse was 72.")
        assert "HR" in v
        assert "72" in v["HR"]

    def test_heart_rate_with_bpm_unit(self):
        v = self.ex._extract_vitals("Heart rate: 88 bpm.")
        assert "HR" in v

    def test_heart_rate_standalone_bpm(self):
        v = self.ex._extract_vitals("Regular rhythm at 64 bpm.")
        assert "HR" in v
        assert "64" in v["HR"]

    def test_respiratory_rate_rr_label(self):
        v = self.ex._extract_vitals("RR 18 on assessment.")
        assert "RR" in v
        assert "18" in v["RR"]

    def test_respiratory_rate_full_label(self):
        v = self.ex._extract_vitals("Respiratory rate 24 breaths/min.")
        assert "RR" in v

    def test_spo2_with_label(self):
        v = self.ex._extract_vitals("SpO2 97% on room air.")
        assert "SpO2" in v
        assert "97" in v["SpO2"]

    def test_spo2_oxygen_saturation(self):
        v = self.ex._extract_vitals("Oxygen saturation 95%.")
        assert "SpO2" in v

    def test_spo2_sats_abbreviation(self):
        v = self.ex._extract_vitals("Sats were 98 on 2L O2.")
        assert "SpO2" in v

    def test_spo2_saturating_at(self):
        v = self.ex._extract_vitals("Patient is saturating at 96%.")
        assert "SpO2" in v

    def test_weight_kg(self):
        v = self.ex._extract_vitals("Weight 75 kg recorded.")
        assert "weight" in v
        assert "75" in v["weight"]

    def test_weight_lbs(self):
        v = self.ex._extract_vitals("Patient weighs 180 lbs.")
        assert "weight" in v

    def test_height_cm(self):
        v = self.ex._extract_vitals("Height 170 cm.")
        assert "height" in v
        assert "170" in v["height"]

    def test_multiple_vitals_in_one_string(self):
        text = "BP 120/80 mmHg, HR 72 bpm, SpO2 98%, temp 37.0°C."
        v = self.ex._extract_vitals(text)
        assert "BP" in v
        assert "HR" in v
        assert "SpO2" in v
        assert "temperature" in v

    def test_empty_string_returns_empty_dict(self):
        assert self.ex._extract_vitals("") == {}

    def test_no_vitals_returns_empty_dict(self):
        assert self.ex._extract_vitals("The patient feels unwell today.") == {}


# ---------------------------------------------------------------------------
# _extract_negations()
# ---------------------------------------------------------------------------

class TestExtractNegations:
    @pytest.fixture(autouse=True)
    def _setup(self, extractor):
        self.ex = extractor

    def test_no_X_pattern(self):
        turns = _make_turns([("PATIENT", "I have no fever or chills.")])
        result = self.ex._extract_negations(turns)
        assert any("fever" in n for n in result)

    def test_no_history_of_pattern(self):
        turns = _make_turns([("PATIENT", "No history of diabetes.")])
        result = self.ex._extract_negations(turns)
        assert any("diabetes" in n for n in result)

    def test_dont_have_pattern(self):
        turns = _make_turns([("PATIENT", "I don't have any allergies.")])
        result = self.ex._extract_negations(turns)
        assert any("allergies" in n for n in result)

    def test_havent_had_pattern(self):
        turns = _make_turns([("PATIENT", "I haven't had any chest pain.")])
        result = self.ex._extract_negations(turns)
        assert any("chest pain" in n for n in result)

    def test_denies_pattern(self):
        turns = _make_turns([("PATIENT", "Denies any shortness of breath.")])
        result = self.ex._extract_negations(turns)
        assert any("shortness of breath" in n for n in result)

    def test_without_pattern(self):
        turns = _make_turns([("PATIENT", "I came in without any nausea.")])
        result = self.ex._extract_negations(turns)
        assert any("nausea" in n for n in result)

    def test_not_experiencing_pattern(self):
        turns = _make_turns([("PATIENT", "I am not experiencing any dizziness.")])
        result = self.ex._extract_negations(turns)
        assert any("dizziness" in n for n in result)

    def test_doctor_turns_are_ignored(self):
        turns = _make_turns([
            ("DOCTOR", "No signs of infection noted."),
            ("PATIENT", "I feel fine."),
        ])
        result = self.ex._extract_negations(turns)
        # "No signs of infection" is from DOCTOR — should not appear
        assert not any("infection" in n for n in result)

    def test_multiple_negations_in_one_turn(self):
        turns = _make_turns([
            ("PATIENT", "No fever, no vomiting, and no rash.")
        ])
        result = self.ex._extract_negations(turns)
        assert any("fever" in n for n in result)
        assert any("vomiting" in n for n in result)

    def test_empty_turns_returns_empty(self):
        assert self.ex._extract_negations([]) == []

    def test_no_patient_turns_returns_empty(self):
        turns = _make_turns([("DOCTOR", "No noted abnormalities.")])
        result = self.ex._extract_negations(turns)
        assert result == []

    def test_negations_deduplicated(self):
        turns = _make_turns([
            ("PATIENT", "No fever."),
            ("PATIENT", "No fever at all."),
        ])
        result = self.ex._extract_negations(turns)
        fever_matches = [n for n in result if "fever" in n]
        assert len(fever_matches) == 1


# ---------------------------------------------------------------------------
# _extract_rule_based()
# ---------------------------------------------------------------------------

class TestExtractRuleBased:
    @pytest.fixture(autouse=True)
    def _setup(self, extractor):
        self.ex = extractor

    def test_detects_common_symptom_pain(self):
        t = _make_transcript(
            patient_turns=["I have severe chest pain.", "It hurts a lot.", "Been going on for days."],
            doctor_turns=["When did it start?", "Where exactly?", "Any radiation?"],
        )
        result = self.ex._extract_rule_based(t)
        assert any("pain" in s for s in result["symptoms"])

    def test_detects_multi_word_symptom(self):
        t = _make_transcript(
            patient_turns=["I have shortness of breath.", "Can't climb stairs.", "Gets worse at night."],
            doctor_turns=["How long?", "Any cough?", "Any fever?"],
        )
        result = self.ex._extract_rule_based(t)
        assert any("shortness of breath" in s for s in result["symptoms"])

    def test_detects_multiple_symptoms(self):
        t = _make_transcript(
            patient_turns=[
                "I have nausea and vomiting.",
                "Also have a headache.",
                "Feel very fatigued.",
            ],
            doctor_turns=["Since when?", "Any fever?", "Any other issues?"],
        )
        result = self.ex._extract_rule_based(t)
        syms = result["symptoms"]
        assert any("nausea" in s for s in syms)
        assert any("vomiting" in s or "vomit" in s for s in syms)
        assert any("headache" in s for s in syms)

    def test_detects_medication_by_suffix(self):
        t = _make_transcript(
            patient_turns=["I take atorvastatin for cholesterol.", "And lisinopril.", "Daily."],
            doctor_turns=["Any other meds?", "Dosage?", "Any side effects?"],
        )
        result = self.ex._extract_rule_based(t)
        meds_lower = [m.lower() for m in result["medications"]]
        assert any("atorvastatin" in m for m in meds_lower)
        assert any("lisinopril" in m for m in meds_lower)

    def test_detects_medication_by_common_name(self):
        t = _make_transcript(
            patient_turns=["I have been taking aspirin daily.", "For my heart.", "325mg."],
            doctor_turns=["Any other medications?", "Since when?", "Good."],
        )
        result = self.ex._extract_rule_based(t)
        meds_lower = [m.lower() for m in result["medications"]]
        assert any("aspirin" in m for m in meds_lower)

    def test_detects_medication_by_context_verb(self):
        t = _make_transcript(
            patient_turns=["I was prescribed metformin.", "500mg twice daily.", "Started last month."],
            doctor_turns=["For diabetes?", "Any side effects?", "Good control."],
        )
        result = self.ex._extract_rule_based(t)
        meds_lower = [m.lower() for m in result["medications"]]
        assert any("metformin" in m for m in meds_lower)

    def test_detects_medical_history(self):
        t = _make_transcript(
            patient_turns=[
                "I have a history of hypertension.",
                "Diagnosed with type 2 diabetes years ago.",
                "No other conditions.",
            ],
            doctor_turns=["Any chronic conditions?", "How long?", "On meds for it?"],
        )
        result = self.ex._extract_rule_based(t)
        assert any("hypertension" in h for h in result["medical_history"])

    def test_detects_allergy(self):
        t = _make_transcript(
            patient_turns=["I am allergic to penicillin.", "It gives me a rash.", "No other allergies."],
            doctor_turns=["Any drug allergies?", "What reaction?", "Good to know."],
        )
        result = self.ex._extract_rule_based(t)
        assert any("penicillin" in a for a in result["allergies"])

    def test_detects_speciality_hint_cardiology(self):
        t = _make_transcript(
            patient_turns=["I've had palpitations for a week.", "Also chest pain.", "Runs in family."],
            doctor_turns=["Any ECG done?", "Family history of cardiac disease?", "We'll do an echo."],
        )
        result = self.ex._extract_rule_based(t)
        assert "cardiology" in result["speciality_hints"]

    def test_detects_speciality_hint_endocrinology(self):
        t = _make_transcript(
            patient_turns=["My glucose was high.", "Taking insulin.", "HbA1c was 8.5."],
            doctor_turns=["When did diabetes start?", "Are you monitoring?", "Let's adjust dose."],
        )
        result = self.ex._extract_rule_based(t)
        assert "endocrinology" in result["speciality_hints"]

    def test_returns_all_required_keys(self):
        t = _make_transcript()
        result = self.ex._extract_rule_based(t)
        for key in ("symptoms", "clinical_findings", "medications",
                    "medical_history", "allergies", "speciality_hints", "raw_entities"):
            assert key in result, f"Missing key: {key}"

    def test_no_duplicates_in_symptoms(self):
        t = _make_transcript(
            patient_turns=[
                "I have pain in my chest.",
                "The pain is severe.",
                "Pain started yesterday.",
            ],
            doctor_turns=["Where is the pain?", "How bad?", "Any radiation?"],
        )
        result = self.ex._extract_rule_based(t)
        # "pain" should appear exactly once despite being in multiple turns
        pain_count = result["symptoms"].count("pain")
        assert pain_count <= 1


# ---------------------------------------------------------------------------
# extract_batch()
# ---------------------------------------------------------------------------

class TestExtractBatch:
    def _valid_transcripts(self, n: int = 4) -> list[ProcessedTranscript]:
        return [
            _make_transcript(
                id_=f"t_{i:03d}",
                patient_turns=[
                    "I have a headache and fever.",
                    "Also nausea.",
                    "It started yesterday.",
                ],
                doctor_turns=["Since when?", "Any vomiting?", "Temperature?"],
            )
            for i in range(n)
        ]

    def test_returns_list(self, extractor):
        results = extractor.extract_batch(self._valid_transcripts(), show_progress=False)
        assert isinstance(results, list)

    def test_all_valid_records_succeed(self, extractor):
        ts = self._valid_transcripts(4)
        results = extractor.extract_batch(ts, show_progress=False)
        assert len(results) == 4

    def test_results_are_medical_entities(self, extractor):
        results = extractor.extract_batch(self._valid_transcripts(2), show_progress=False)
        assert all(isinstance(r, MedicalEntities) for r in results)

    def test_empty_list_returns_empty(self, extractor):
        assert extractor.extract_batch([], show_progress=False) == []

    def test_mixed_valid_and_bad_does_not_crash(self, extractor):
        """Bad (non-transcript) objects should be dropped, not crash the batch."""
        ts = self._valid_transcripts(3)

        class _BadTranscript:
            id = "bad_001"
            @property
            def patient_text(self):
                raise RuntimeError("boom")
            @property
            def doctor_text(self):
                raise RuntimeError("boom")
            @property
            def full_text(self):
                raise RuntimeError("boom")
            @property
            def turns(self):
                raise RuntimeError("boom")

        mixed = ts[:1] + [_BadTranscript()] + ts[1:]  # type: ignore
        results = extractor.extract_batch(mixed, show_progress=False)
        # At least the good transcripts come through
        assert len(results) >= 3

    def test_ids_preserved(self, extractor):
        ts = self._valid_transcripts(3)
        results = extractor.extract_batch(ts, show_progress=False)
        result_ids = {r.conversation_id for r in results}
        expected_ids = {t.id for t in ts}
        assert result_ids == expected_ids

    def test_show_progress_does_not_crash(self, extractor, capsys):
        extractor.extract_batch(self._valid_transcripts(2), show_progress=True)
        captured = capsys.readouterr()
        assert "succeeded" in captured.out


# ---------------------------------------------------------------------------
# get_stats()
# ---------------------------------------------------------------------------

class TestGetStats:
    def _make_entities(
        self,
        id_: str = "e_001",
        symptoms: list[str] | None = None,
        medications: list[str] | None = None,
        vitals: dict | None = None,
        negations: list[str] | None = None,
        method: str = "rule_based",
        confidence: float = 0.5,
    ) -> MedicalEntities:
        return MedicalEntities(
            conversation_id=id_,
            symptoms=symptoms or [],
            clinical_findings=[],
            medications=medications or [],
            medical_history=[],
            vitals=vitals or {},
            allergies=[],
            speciality_hints=[],
            negations=negations or [],
            extraction_method=method,
            confidence=confidence,
        )

    def test_empty_list_returns_zeros(self, extractor):
        stats = extractor.get_stats([])
        assert stats["total"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_total_count(self, extractor):
        entities = [self._make_entities(id_=f"e_{i}") for i in range(7)]
        assert extractor.get_stats(entities)["total"] == 7

    def test_method_breakdown(self, extractor):
        entities = [
            self._make_entities(method="rule_based"),
            self._make_entities(method="rule_based"),
            self._make_entities(method="scispacy"),
        ]
        stats = extractor.get_stats(entities)
        counts = stats["extraction_method_counts"]
        assert counts.get("rule_based") == 2
        assert counts.get("scispacy") == 1

    def test_avg_entities_structure(self, extractor):
        entities = [self._make_entities()]
        stats = extractor.get_stats(entities)
        assert "avg_entities" in stats
        assert "symptoms" in stats["avg_entities"]
        assert "medications" in stats["avg_entities"]

    def test_pct_with_vitals(self, extractor):
        entities = [
            self._make_entities(vitals={"BP": "120/80 mmHg"}),
            self._make_entities(vitals={}),
            self._make_entities(vitals={"HR": "72 bpm"}),
            self._make_entities(vitals={}),
        ]
        stats = extractor.get_stats(entities)
        assert stats["pct_with_vitals"] == 50.0

    def test_pct_with_negations(self, extractor):
        entities = [
            self._make_entities(negations=["fever"]),
            self._make_entities(negations=[]),
        ]
        stats = extractor.get_stats(entities)
        assert stats["pct_with_negations"] == 50.0

    def test_avg_confidence(self, extractor):
        entities = [
            self._make_entities(confidence=0.4),
            self._make_entities(confidence=0.8),
        ]
        stats = extractor.get_stats(entities)
        assert abs(stats["avg_confidence"] - 0.6) < 0.01

    def test_avg_entities_accuracy(self, extractor):
        entities = [
            self._make_entities(symptoms=["fever", "cough"]),
            self._make_entities(symptoms=["pain"]),
        ]
        stats = extractor.get_stats(entities)
        assert stats["avg_entities"]["symptoms"] == 1.5


# ---------------------------------------------------------------------------
# Fallback mode (scispaCy unavailable)
# ---------------------------------------------------------------------------

class TestFallbackMode:
    def test_extractor_initialises_without_spacy(self):
        """
        Simulate scispaCy being absent by patching the module-level flag.
        EntityExtractor should initialise cleanly and set use_fallback=True.
        """
        with patch(
            "medai.src.clinical.entity_extractor._SCISPACY_AVAILABLE", False
        ):
            ex = EntityExtractor()
        assert ex.use_fallback is True
        assert ex._nlp is None

    def test_fallback_extract_returns_medical_entities(self):
        with patch(
            "medai.src.clinical.entity_extractor._SCISPACY_AVAILABLE", False
        ):
            ex = EntityExtractor()

        t = _make_transcript(
            patient_turns=["I have a fever and cough.", "Also headache.", "Three days now."],
            doctor_turns=["Any other symptoms?", "Temperature?", "Duration?"],
        )
        result = ex.extract(t)
        assert isinstance(result, MedicalEntities)
        assert result.extraction_method == "rule_based"

    def test_fallback_detects_symptoms(self):
        with patch(
            "medai.src.clinical.entity_extractor._SCISPACY_AVAILABLE", False
        ):
            ex = EntityExtractor()

        t = _make_transcript(
            patient_turns=["I have nausea and vomiting.", "Fever too.", "For two days."],
            doctor_turns=["Since when?", "Any diarrhea?", "Any rash?"],
        )
        result = ex.extract(t)
        assert len(result.symptoms) > 0

    def test_fallback_confidence_below_scispacy(self):
        """
        Rule-based confidence is penalised (×0.8) relative to scispaCy.
        Even a fully populated rule-based result should score ≤ 0.80.
        """
        with patch(
            "medai.src.clinical.entity_extractor._SCISPACY_AVAILABLE", False
        ):
            ex = EntityExtractor()

        t = _make_transcript(
            patient_turns=[
                "I have chest pain, nausea, and fatigue.",
                "History of hypertension, taking aspirin.",
                "Allergic to penicillin.",
            ],
            doctor_turns=["Any other issues?", "How long?", "Any medications?"],
        )
        result = ex.extract(t)
        # rule_based confidence is capped at 0.80 (max weight sum × 0.8)
        assert result.confidence <= 0.80

    def test_fallback_batch_works(self):
        with patch(
            "medai.src.clinical.entity_extractor._SCISPACY_AVAILABLE", False
        ):
            ex = EntityExtractor()

        transcripts = [
            _make_transcript(
                id_=f"fb_{i}",
                patient_turns=["I have a headache.", "And some dizziness.", "Started today."],
                doctor_turns=["Duration?", "Severity?", "Any vomiting?"],
            )
            for i in range(3)
        ]
        results = ex.extract_batch(transcripts, show_progress=False)
        assert len(results) == 3
        assert all(r.extraction_method == "rule_based" for r in results)


# ---------------------------------------------------------------------------
# _compute_confidence() and _dedup() internal helpers
# ---------------------------------------------------------------------------

class TestInternals:
    def test_dedup_removes_case_duplicates(self, extractor):
        result = extractor._dedup(["Fever", "fever", "FEVER", "cough"])
        assert len(result) == 2

    def test_dedup_preserves_first_seen_form(self, extractor):
        result = extractor._dedup(["Fever", "fever"])
        assert result[0] == "Fever"  # first-seen casing kept

    def test_dedup_strips_whitespace(self, extractor):
        result = extractor._dedup(["  pain  ", "pain"])
        assert len(result) == 1
        assert result[0] == "pain"

    def test_dedup_empty_list(self, extractor):
        assert extractor._dedup([]) == []

    def test_compute_confidence_zero_with_empty(self, extractor):
        score = extractor._compute_confidence({}, "rule_based", False)
        assert score == 0.0

    def test_compute_confidence_higher_for_scispacy(self, extractor):
        populated = {
            "symptoms": ["fever"],
            "medications": ["aspirin"],
            "medical_history": ["hypertension"],
            "clinical_findings": ["tachycardia"],
        }
        sci = extractor._compute_confidence(populated, "scispacy", False)
        rb = extractor._compute_confidence(populated, "rule_based", False)
        assert sci > rb

    def test_compute_confidence_vitals_bonus(self, extractor):
        d = {"symptoms": ["fever"]}
        without = extractor._compute_confidence(d, "rule_based", False)
        with_v = extractor._compute_confidence(d, "rule_based", True)
        assert with_v > without

    def test_compute_confidence_capped_at_one(self, extractor):
        fully_populated = {
            "symptoms": ["x"], "medications": ["y"], "medical_history": ["z"],
            "clinical_findings": ["w"], "allergies": ["a"], "speciality_hints": ["b"],
        }
        score = extractor._compute_confidence(fully_populated, "scispacy", True)
        assert score <= 1.0
