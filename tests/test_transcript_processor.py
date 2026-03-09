"""
tests/test_transcript_processor.py

Pytest suite for medai.src.voice.transcript_processor.

Run with:
    pytest tests/test_transcript_processor.py -v
"""

import sys
from pathlib import Path

import pytest

# Make the repo root importable regardless of where pytest is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medai.src.voice.transcript_processor import (
    ProcessedTranscript,
    TranscriptProcessor,
    Turn,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_RECORD = {
    "id": "mts_0001",
    "source_dataset": "mts_dialog",
    "language": "en",
    "speciality": "cardiology",
    "turns": [
        {"speaker": "DOCTOR", "text": "What brings you in today?"},
        {"speaker": "PATIENT", "text": "I have had chest pain for two days."},
        {"speaker": "DOCTOR", "text": "Where exactly is the pain located?"},
        {"speaker": "PATIENT", "text": "It is in the centre of my chest."},
        {"speaker": "DOCTOR", "text": "Does it radiate anywhere like your arm or jaw?"},
        {"speaker": "PATIENT", "text": "Yes, it goes down my left arm sometimes."},
    ],
    "reference_note": "SUBJECTIVE: Patient presents with two-day history of chest pain.",
    "is_synthetic": False,
    "turn_count": 6,
}


def _make_record(**overrides) -> dict:
    """Return a copy of VALID_RECORD with any overrides applied."""
    rec = {**VALID_RECORD, "turns": [dict(t) for t in VALID_RECORD["turns"]]}
    rec.update(overrides)
    return rec


@pytest.fixture
def processor() -> TranscriptProcessor:
    return TranscriptProcessor()


@pytest.fixture
def valid_transcript(processor) -> ProcessedTranscript:
    return processor.process(VALID_RECORD)


# ---------------------------------------------------------------------------
# process() — field types and basic correctness
# ---------------------------------------------------------------------------

class TestProcess:
    def test_returns_processed_transcript(self, valid_transcript):
        assert isinstance(valid_transcript, ProcessedTranscript)

    def test_id_preserved(self, valid_transcript):
        assert valid_transcript.id == "mts_0001"

    def test_source_dataset_preserved(self, valid_transcript):
        assert valid_transcript.source_dataset == "mts_dialog"

    def test_language_preserved(self, valid_transcript):
        assert valid_transcript.language == "en"

    def test_turns_is_list_of_turn(self, valid_transcript):
        assert isinstance(valid_transcript.turns, list)
        assert all(isinstance(t, Turn) for t in valid_transcript.turns)

    def test_turn_count(self, valid_transcript):
        assert valid_transcript.turn_count == 6

    def test_word_count_is_int_gt_zero(self, valid_transcript):
        assert isinstance(valid_transcript.word_count, int)
        assert valid_transcript.word_count > 0

    def test_has_reference_note_true(self, valid_transcript):
        assert valid_transcript.has_reference_note is True

    def test_reference_note_text(self, valid_transcript):
        assert "chest pain" in valid_transcript.reference_note

    def test_has_reference_note_false_when_absent(self, processor):
        rec = _make_record(reference_note=None)
        t = processor.process(rec)
        assert t.has_reference_note is False
        assert t.reference_note is None

    def test_doctor_text_contains_only_doctor_words(self, valid_transcript):
        assert "chest pain" not in valid_transcript.doctor_text
        assert "brings you in" in valid_transcript.doctor_text

    def test_patient_text_contains_only_patient_words(self, valid_transcript):
        assert "chest pain" in valid_transcript.patient_text
        assert "brings you in" not in valid_transcript.patient_text

    def test_full_text_has_speaker_labels(self, valid_transcript):
        assert "DOCTOR:" in valid_transcript.full_text
        assert "PATIENT:" in valid_transcript.full_text

    def test_flags_is_list(self, valid_transcript):
        assert isinstance(valid_transcript.flags, list)

    def test_turn_index_matches_position(self, valid_transcript):
        # Turn.index is the original raw index; they should be monotone
        indices = [t.index for t in valid_transcript.turns]
        assert indices == sorted(indices)

    def test_turn_word_count_accurate(self, processor):
        rec = _make_record(turns=[
            {"speaker": "DOCTOR", "text": "one two three"},
            {"speaker": "PATIENT", "text": "four five"},
            {"speaker": "DOCTOR", "text": "six seven eight nine"},
            {"speaker": "PATIENT", "text": "ten"},
            {"speaker": "DOCTOR", "text": "eleven twelve"},
            {"speaker": "PATIENT", "text": "thirteen fourteen fifteen"},
        ])
        t = processor.process(rec)
        assert t.turns[0].word_count == 3
        assert t.turns[1].word_count == 2

    def test_is_question_true_for_question_mark(self, valid_transcript):
        # "What brings you in today?" → True
        assert valid_transcript.turns[0].is_question is True

    def test_is_question_false_for_statement(self, valid_transcript):
        # "I have had chest pain for two days." → False
        assert valid_transcript.turns[1].is_question is False

    def test_is_question_true_for_question_word(self, processor):
        rec = _make_record(turns=[
            {"speaker": "DOCTOR", "text": "How long have you had this"},
            {"speaker": "PATIENT", "text": "About three days now"},
            {"speaker": "DOCTOR", "text": "Does it hurt more at night"},
            {"speaker": "PATIENT", "text": "Yes it does"},
            {"speaker": "DOCTOR", "text": "Can you point to where it hurts"},
            {"speaker": "PATIENT", "text": "Right here"},
        ])
        t = processor.process(rec)
        assert t.turns[0].is_question is True   # "How …"
        assert t.turns[2].is_question is True   # "Does …"
        assert t.turns[4].is_question is True   # "Can …"
        assert t.turns[1].is_question is False  # statement

    def test_consecutive_run_resets_on_speaker_change(self, valid_transcript):
        # DOCTOR, PATIENT, DOCTOR, PATIENT, DOCTOR, PATIENT — all runs of 1
        for t in valid_transcript.turns:
            assert t.consecutive_run == 1

    def test_consecutive_run_increments_for_same_speaker(self, processor):
        rec = _make_record(turns=[
            {"speaker": "DOCTOR", "text": "Tell me about your symptoms"},
            {"speaker": "DOCTOR", "text": "Go ahead I am listening"},
            {"speaker": "DOCTOR", "text": "Please continue"},
            {"speaker": "PATIENT", "text": "I have a headache"},
            {"speaker": "PATIENT", "text": "And some nausea"},
            {"speaker": "DOCTOR", "text": "How long has this been going on"},
        ])
        t = processor.process(rec)
        assert t.turns[0].consecutive_run == 1
        assert t.turns[1].consecutive_run == 2
        assert t.turns[2].consecutive_run == 3
        assert t.turns[3].consecutive_run == 1  # reset
        assert t.turns[4].consecutive_run == 2
        assert t.turns[5].consecutive_run == 1  # reset

    def test_missing_required_field_raises_value_error(self, processor):
        for field in ("id", "source_dataset", "language", "turns"):
            rec = _make_record()
            del rec[field]
            with pytest.raises(ValueError, match=field):
                processor.process(rec)

    def test_empty_turns_raises_value_error(self, processor):
        rec = _make_record(turns=[])
        with pytest.raises(ValueError):
            processor.process(rec)

    def test_empty_turns_after_cleaning_raises(self, processor):
        # Turns that become empty strings after cleaning
        rec = _make_record(turns=[
            {"speaker": "DOCTOR", "text": "   \x00 \t "},
            {"speaker": "PATIENT", "text": "\x01\x02"},
        ])
        with pytest.raises(ValueError):
            processor.process(rec)


# ---------------------------------------------------------------------------
# clean_text()
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_strips_leading_trailing_whitespace(self, processor):
        assert processor.clean_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self, processor):
        assert processor.clean_text("one  two   three") == "one two three"

    def test_removes_null_bytes(self, processor):
        assert processor.clean_text("hel\x00lo") == "hello"

    def test_removes_non_printable_control_chars(self, processor):
        # \x01–\x08, \x0e–\x1f should be stripped
        assert processor.clean_text("\x01te\x07xt\x1f") == "text"

    def test_preserves_newline_content_via_space(self, processor):
        # Tabs should become spaces (newlines are typically not in turn text,
        # but tabs should collapse)
        result = processor.clean_text("word\there")
        assert result == "word here"

    def test_empty_string_returns_empty(self, processor):
        assert processor.clean_text("") == ""

    def test_whitespace_only_returns_empty(self, processor):
        assert processor.clean_text("   \t  ") == ""

    def test_preserves_uppercase_abbreviations(self, processor):
        text = "Patient has ECG changes and elevated TROPONIN levels"
        assert "ECG" in processor.clean_text(text)
        assert "TROPONIN" in processor.clean_text(text)

    def test_preserves_punctuation(self, processor):
        text = "Symptoms: fever, chills, and night sweats."
        assert processor.clean_text(text) == text

    def test_mixed_null_bytes_and_spaces(self, processor):
        result = processor.clean_text("  \x00 hello \x00 world  ")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# flag_anomalies()
# ---------------------------------------------------------------------------

def _make_turns(speaker_text_pairs: list[tuple[str, str]]) -> list[Turn]:
    """Build a Turn list from (speaker, text) pairs."""
    turns = []
    run = 0
    prev = None
    for i, (speaker, text) in enumerate(speaker_text_pairs):
        if speaker == prev:
            run += 1
        else:
            run = 1
            prev = speaker
        turns.append(Turn(
            index=i,
            speaker=speaker,
            text=text,
            word_count=len(text.split()),
            is_question=text.strip().endswith("?"),
            consecutive_run=run,
        ))
    return turns


class TestFlagAnomalies:
    def test_no_flags_for_clean_conversation(self, processor):
        turns = _make_turns([
            ("DOCTOR", "How can I help you today?"),
            ("PATIENT", "I have had a headache for two days."),
            ("DOCTOR", "Is it on one side or both sides?"),
            ("PATIENT", "It is on the right side."),
            ("DOCTOR", "Any nausea or vomiting with it?"),
            ("PATIENT", "Yes I feel nauseous in the morning."),
        ])
        assert processor.flag_anomalies(turns) == []

    def test_flags_short_conversation(self, processor):
        turns = _make_turns([
            ("DOCTOR", "How can I help?"),
            ("PATIENT", "I have a cough."),
            ("DOCTOR", "How long have you had it?"),
            ("PATIENT", "About a week."),
        ])
        flags = processor.flag_anomalies(turns)
        assert "short_conversation" in flags

    def test_flags_patient_speaks_first(self, processor):
        turns = _make_turns([
            ("PATIENT", "Hello I need help."),
            ("DOCTOR", "What seems to be the problem?"),
            ("PATIENT", "I have severe back pain."),
            ("DOCTOR", "How long has this been going on?"),
            ("PATIENT", "About three days."),
            ("DOCTOR", "Did anything trigger it?"),
        ])
        flags = processor.flag_anomalies(turns)
        assert "patient_speaks_first" in flags

    def test_flags_long_turn(self, processor):
        long_text = " ".join(["word"] * 310)  # 310 words
        turns = _make_turns([
            ("DOCTOR", "How are you?"),
            ("PATIENT", long_text),
            ("DOCTOR", "I see, thank you."),
            ("PATIENT", "Yes that is right."),
            ("DOCTOR", "Any other symptoms?"),
            ("PATIENT", "Not really no."),
        ])
        flags = processor.flag_anomalies(turns)
        assert any("long_turn:index:1" in f for f in flags)

    def test_flags_consecutive_run_of_4(self, processor):
        turns = _make_turns([
            ("DOCTOR", "Tell me more about your pain."),
            ("DOCTOR", "Where is it located exactly?"),
            ("DOCTOR", "Does it radiate anywhere?"),
            ("DOCTOR", "How severe is it on a scale of one to ten?"),
            ("PATIENT", "It is about a seven."),
            ("DOCTOR", "Thank you for that information."),
        ])
        flags = processor.flag_anomalies(turns)
        assert any("consecutive_run:DOCTOR:0" in f for f in flags)

    def test_consecutive_run_of_3_not_flagged(self, processor):
        turns = _make_turns([
            ("DOCTOR", "Tell me about your symptoms."),
            ("DOCTOR", "What else is bothering you?"),
            ("DOCTOR", "Anything else to add?"),
            ("PATIENT", "Just some fatigue."),
            ("DOCTOR", "How long has that been going on?"),
            ("PATIENT", "A couple of weeks now."),
        ])
        flags = processor.flag_anomalies(turns)
        assert not any("consecutive_run" in f for f in flags)

    def test_long_turn_flag_contains_correct_index(self, processor):
        long_text = " ".join(["word"] * 350)
        turns = _make_turns([
            ("DOCTOR", "What brings you in?"),
            ("PATIENT", "Normal answer here."),
            ("DOCTOR", long_text),          # index 2 is long
            ("PATIENT", "Okay I understand."),
            ("DOCTOR", "Any questions for me?"),
            ("PATIENT", "No I think I am fine."),
        ])
        flags = processor.flag_anomalies(turns)
        assert "long_turn:index:2" in flags

    def test_multiple_flags_can_coexist(self, processor):
        long_text = " ".join(["word"] * 310)
        turns = _make_turns([
            ("PATIENT", "Hello doctor."),   # patient first
            ("DOCTOR", long_text),           # long turn
            ("DOCTOR", "And more."),
            ("PATIENT", "Okay."),
        ])
        # short_conversation (4 < 6) + patient_speaks_first + long_turn
        flags = processor.flag_anomalies(turns)
        assert "short_conversation" in flags
        assert "patient_speaks_first" in flags
        assert any("long_turn" in f for f in flags)

    def test_empty_turns_returns_short_conversation(self, processor):
        flags = processor.flag_anomalies([])
        assert "short_conversation" in flags


# ---------------------------------------------------------------------------
# process_batch()
# ---------------------------------------------------------------------------

class TestProcessBatch:
    def _make_valid(self, idx: int) -> dict:
        return {
            "id": f"rec_{idx:03d}",
            "source_dataset": "test",
            "language": "en",
            "turns": [
                {"speaker": "DOCTOR", "text": f"Question {idx}?"},
                {"speaker": "PATIENT", "text": f"Answer {idx}."},
                {"speaker": "DOCTOR", "text": "Tell me more please."},
                {"speaker": "PATIENT", "text": "Sure I will elaborate here."},
                {"speaker": "DOCTOR", "text": "Any other symptoms to mention?"},
                {"speaker": "PATIENT", "text": "No that is all for now."},
            ],
            "reference_note": None,
            "is_synthetic": True,
            "turn_count": 6,
        }

    def test_returns_list_of_processed_transcripts(self, processor):
        records = [self._make_valid(i) for i in range(5)]
        results = processor.process_batch(records, show_progress=False)
        assert isinstance(results, list)
        assert all(isinstance(r, ProcessedTranscript) for r in results)

    def test_all_valid_records_succeed(self, processor):
        records = [self._make_valid(i) for i in range(5)]
        results = processor.process_batch(records, show_progress=False)
        assert len(results) == 5

    def test_invalid_records_excluded_not_raised(self, processor):
        records = [
            self._make_valid(0),
            {"id": "bad_001", "turns": []},              # missing fields
            self._make_valid(2),
            {"source_dataset": "x", "language": "en", "turns": []},  # missing id
            self._make_valid(4),
        ]
        results = processor.process_batch(records, show_progress=False)
        assert len(results) == 3

    def test_empty_batch_returns_empty_list(self, processor):
        assert processor.process_batch([], show_progress=False) == []

    def test_all_invalid_returns_empty_list(self, processor):
        records = [{"bad": True}, {}, {"id": "x", "turns": []}]
        results = processor.process_batch(records, show_progress=False)
        assert results == []

    def test_successful_ids_are_correct(self, processor):
        records = [self._make_valid(i) for i in range(3)]
        results = processor.process_batch(records, show_progress=False)
        assert [r.id for r in results] == ["rec_000", "rec_001", "rec_002"]

    def test_show_progress_does_not_crash(self, processor, capsys):
        records = [self._make_valid(i) for i in range(3)]
        processor.process_batch(records, show_progress=True)
        captured = capsys.readouterr()
        assert "succeeded" in captured.out


# ---------------------------------------------------------------------------
# get_stats()
# ---------------------------------------------------------------------------

class TestGetStats:
    def _minimal_transcript(
        self, id_: str, turn_count: int, word_count: int, flags: list[str]
    ) -> ProcessedTranscript:
        turns = [
            Turn(
                index=i,
                speaker="DOCTOR" if i % 2 == 0 else "PATIENT",
                text="word " * (word_count // turn_count),
                word_count=word_count // turn_count,
                is_question=False,
                consecutive_run=1,
            )
            for i in range(turn_count)
        ]
        return ProcessedTranscript(
            id=id_,
            source_dataset="test",
            language="en",
            turns=turns,
            doctor_text="",
            patient_text="",
            full_text="",
            turn_count=turn_count,
            word_count=word_count,
            has_reference_note=False,
            reference_note=None,
            flags=flags,
        )

    def test_empty_list_returns_zeros(self, processor):
        stats = processor.get_stats([])
        assert stats["total"] == 0
        assert stats["avg_turns"] == 0.0
        assert stats["avg_words"] == 0.0
        assert stats["flagged_count"] == 0

    def test_total_count(self, processor):
        ts = [self._minimal_transcript(f"id_{i}", 6, 100, []) for i in range(7)]
        assert processor.get_stats(ts)["total"] == 7

    def test_avg_turns(self, processor):
        ts = [
            self._minimal_transcript("a", 4, 100, []),
            self._minimal_transcript("b", 8, 100, []),
        ]
        assert processor.get_stats(ts)["avg_turns"] == 6.0

    def test_avg_words(self, processor):
        ts = [
            self._minimal_transcript("a", 4, 60, []),
            self._minimal_transcript("b", 4, 100, []),
        ]
        assert processor.get_stats(ts)["avg_words"] == 80.0

    def test_flagged_count(self, processor):
        ts = [
            self._minimal_transcript("a", 4, 100, ["short_conversation"]),
            self._minimal_transcript("b", 6, 100, []),
            self._minimal_transcript("c", 4, 100, ["patient_speaks_first", "short_conversation"]),
        ]
        assert processor.get_stats(ts)["flagged_count"] == 2

    def test_flag_type_breakdown(self, processor):
        ts = [
            self._minimal_transcript("a", 4, 100, ["short_conversation"]),
            self._minimal_transcript("b", 4, 100, ["short_conversation", "patient_speaks_first"]),
            self._minimal_transcript("c", 6, 100, ["long_turn:index:2"]),
        ]
        breakdown = processor.get_stats(ts)["flag_type_breakdown"]
        assert breakdown.get("short_conversation") == 2
        assert breakdown.get("patient_speaks_first") == 1
        assert breakdown.get("long_turn") == 1

    def test_doctor_word_share_is_percentage(self, processor):
        # process a real record so doctor_text is populated
        p = TranscriptProcessor()
        ts = p.process_batch([VALID_RECORD], show_progress=False)
        stats = p.get_stats(ts)
        share = stats["doctor_word_share"]
        assert 0.0 <= share <= 100.0
