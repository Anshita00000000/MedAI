"""
transcript_processor.py

Takes a raw unified conversation record (from data/raw/unified/) and returns
a clean, enriched ProcessedTranscript ready for the clinical entity extraction
pipeline.

Responsibilities:
  - Validate the incoming record has all required fields
  - Clean turn text (whitespace, null bytes, non-printable chars)
  - Build Turn objects with derived fields (is_question, consecutive_run)
  - Assemble doctor_text / patient_text / full_text blocks
  - Detect and flag anomalies (long turns, speaker runs, etc.)
  - Process batches with progress reporting and isolated error handling

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single speaker turn within a conversation."""

    index: int
    speaker: str          # "DOCTOR" or "PATIENT"
    text: str
    word_count: int
    is_question: bool     # ends with "?" or starts with a question word
    consecutive_run: int  # how many same-speaker turns in a row up to this one

    def __str__(self) -> str:
        q_marker = "?" if self.is_question else " "
        return (
            f"  [{self.index:>2}] {self.speaker:<7}{q_marker} "
            f"({self.word_count:>3}w, run={self.consecutive_run}) "
            f"{self.text[:80]}{'…' if len(self.text) > 80 else ''}"
        )


@dataclass
class ProcessedTranscript:
    """
    Enriched, cleaned representation of one doctor-patient conversation,
    ready to be passed to EntityExtractor.
    """

    id: str
    source_dataset: str
    language: str
    turns: List[Turn]
    doctor_text: str        # all DOCTOR turns joined as one block
    patient_text: str       # all PATIENT turns joined as one block
    full_text: str          # all turns with "DOCTOR: …\nPATIENT: …" labels
    turn_count: int
    word_count: int
    has_reference_note: bool
    reference_note: str | None
    flags: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        flag_str = ", ".join(self.flags) if self.flags else "none"
        header = (
            f"ProcessedTranscript(id={self.id!r}, "
            f"dataset={self.source_dataset!r}, "
            f"lang={self.language!r}, "
            f"turns={self.turn_count}, words={self.word_count}, "
            f"has_note={self.has_reference_note})\n"
            f"  flags: {flag_str}\n"
        )
        turns_str = "\n".join(str(t) for t in self.turns)
        note_str = ""
        if self.reference_note:
            snippet = " ".join(self.reference_note.split()[:40])
            if len(self.reference_note.split()) > 40:
                snippet += " …"
            note_str = f"\n  [REFERENCE NOTE] {snippet}"
        return header + turns_str + note_str


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

# Question words that indicate a turn is likely interrogative even without "?"
_QUESTION_WORDS = re.compile(
    r"^(what|when|where|who|whom|whose|which|why|how|"
    r"do|does|did|is|are|was|were|have|has|had|"
    r"can|could|will|would|shall|should|may|might|must)\b",
    re.IGNORECASE,
)

# Characters we want to strip: null bytes and C0/C1 control codes except \t\n
_NON_PRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Normalise interior whitespace (but leave single spaces intact)
_MULTI_SPACE = re.compile(r"[ \t]+")

REQUIRED_FIELDS = ("id", "source_dataset", "language", "turns")

# Anomaly thresholds
_LONG_TURN_WORDS = 300
_CONSECUTIVE_RUN_THRESHOLD = 4
_SHORT_CONV_TURNS = 6


class TranscriptProcessor:
    """
    Converts raw unified-schema records into ProcessedTranscript objects.

    Usage::

        processor = TranscriptProcessor()
        transcript = processor.process(record)
        batch = processor.process_batch(records)
        stats = processor.get_stats(batch)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, record: dict) -> ProcessedTranscript:
        """
        Process a single raw record dict into a ProcessedTranscript.

        Parameters
        ----------
        record:
            A dict following the unified MedAI schema.

        Returns
        -------
        ProcessedTranscript

        Raises
        ------
        ValueError
            If any required field is absent or if ``turns`` is empty
            after cleaning.
        """
        self._validate(record)

        raw_turns: list[dict] = record["turns"]
        turns: list[Turn] = []

        # Build Turn objects with consecutive-run tracking
        run_count = 0
        prev_speaker: str | None = None

        for i, raw in enumerate(raw_turns):
            speaker = str(raw.get("speaker", "")).strip()
            text = self.clean_text(str(raw.get("text", "")))

            if not text:
                # Skip turns that reduce to nothing after cleaning
                continue

            if speaker == prev_speaker:
                run_count += 1
            else:
                run_count = 1
                prev_speaker = speaker

            wc = len(text.split())
            is_q = text.rstrip().endswith("?") or bool(
                _QUESTION_WORDS.match(text.split()[0]) if text.split() else False
            )

            turns.append(Turn(
                index=i,
                speaker=speaker,
                text=text,
                word_count=wc,
                is_question=is_q,
                consecutive_run=run_count,
            ))

        if not turns:
            raise ValueError(
                f"Record {record['id']!r} has no usable turns after cleaning."
            )

        doctor_text = " ".join(t.text for t in turns if t.speaker == "DOCTOR")
        patient_text = " ".join(t.text for t in turns if t.speaker == "PATIENT")
        full_text = "\n".join(f"{t.speaker}: {t.text}" for t in turns)
        total_words = sum(t.word_count for t in turns)
        ref_note = record.get("reference_note") or None

        flags = self.flag_anomalies(turns)

        return ProcessedTranscript(
            id=record["id"],
            source_dataset=record["source_dataset"],
            language=record.get("language", "en"),
            turns=turns,
            doctor_text=doctor_text,
            patient_text=patient_text,
            full_text=full_text,
            turn_count=len(turns),
            word_count=total_words,
            has_reference_note=bool(ref_note),
            reference_note=ref_note,
            flags=flags,
        )

    def clean_text(self, text: str) -> str:
        """
        Return a cleaned version of *text* suitable for NLP processing.

        Steps applied (in order):

        1. Strip leading/trailing whitespace.
        2. Remove null bytes and non-printable control characters
           (C0/C1 except tab and newline).
        3. Replace internal tab characters with a single space.
        4. Collapse runs of spaces/tabs to a single space.
        5. Strip again to remove any edge whitespace introduced above.

        Uppercase tokens (medical abbreviations such as "ECG", "GERD",
        "NSAID") are preserved exactly as-is.
        """
        if not text:
            return ""
        text = text.strip()
        text = _NON_PRINTABLE.sub("", text)
        text = text.replace("\t", " ")
        text = _MULTI_SPACE.sub(" ", text)
        return text.strip()

    def flag_anomalies(self, turns: list[Turn]) -> list[str]:
        """
        Inspect a list of Turn objects and return warning strings.

        Flag codes returned
        -------------------
        ``"long_turn:index:{n}"``
            Turn at position *n* exceeds 300 words.
        ``"consecutive_run:{speaker}:{start_index}"``
            *speaker* speaks 4 or more consecutive turns starting at
            *start_index*.
        ``"short_conversation"``
            The conversation has fewer than 6 turns.
        ``"patient_speaks_first"``
            The first turn belongs to the PATIENT.
        """
        flags: list[str] = []

        # Short conversation
        if len(turns) < _SHORT_CONV_TURNS:
            flags.append("short_conversation")

        # Patient speaks first
        if turns and turns[0].speaker == "PATIENT":
            flags.append("patient_speaks_first")

        # Long turns
        for t in turns:
            if t.word_count > _LONG_TURN_WORDS:
                flags.append(f"long_turn:index:{t.index}")

        # Consecutive same-speaker runs of 4+
        # We record the flag once per run (at the turn that tips it to 4)
        flagged_run_starts: set[int] = set()
        for t in turns:
            if t.consecutive_run >= _CONSECUTIVE_RUN_THRESHOLD:
                start = t.index - t.consecutive_run + 1
                if start not in flagged_run_starts:
                    flagged_run_starts.add(start)
                    flags.append(f"consecutive_run:{t.speaker}:{start}")

        return flags

    def process_batch(
        self,
        records: list[dict],
        show_progress: bool = True,
    ) -> list[ProcessedTranscript]:
        """
        Process a list of raw records, returning only the successful ones.

        Parameters
        ----------
        records:
            Raw record dicts in unified MedAI schema.
        show_progress:
            If True, print a progress line every 100 records and a final
            summary.

        Returns
        -------
        list[ProcessedTranscript]
            Successfully processed transcripts (failed records are logged
            but excluded).
        """
        results: list[ProcessedTranscript] = []
        failures: list[tuple[str, str]] = []  # (record_id_or_index, reason)

        for i, record in enumerate(records):
            record_label = record.get("id", f"index:{i}")
            try:
                results.append(self.process(record))
            except Exception as exc:  # noqa: BLE001
                failures.append((record_label, str(exc)))

            if show_progress and (i + 1) % 100 == 0:
                print(
                    f"  … processed {i + 1}/{len(records)} records "
                    f"({len(failures)} failures so far)"
                )

        if show_progress:
            print(
                f"  Batch complete: {len(results)} succeeded, "
                f"{len(failures)} failed out of {len(records)} records."
            )
            if failures:
                print("  Failed records:")
                for fid, reason in failures[:10]:
                    print(f"    {fid}: {reason}")
                if len(failures) > 10:
                    print(f"    … and {len(failures) - 10} more.")

        return results

    def get_stats(self, transcripts: list[ProcessedTranscript]) -> dict:
        """
        Compute summary statistics over a list of processed transcripts.

        Returns
        -------
        dict with keys:
            ``total``                — number of transcripts
            ``avg_turns``            — mean turn count
            ``avg_words``            — mean word count
            ``flagged_count``        — transcripts with at least one flag
            ``flag_type_breakdown``  — count per flag-type prefix
            ``doctor_word_share``    — doctor's % of total words spoken
        """
        if not transcripts:
            return {
                "total": 0,
                "avg_turns": 0.0,
                "avg_words": 0.0,
                "flagged_count": 0,
                "flag_type_breakdown": {},
                "doctor_word_share": 0.0,
            }

        total = len(transcripts)
        avg_turns = sum(t.turn_count for t in transcripts) / total
        avg_words = sum(t.word_count for t in transcripts) / total

        flagged_count = sum(1 for t in transcripts if t.flags)

        # Break flags down by type prefix (e.g. "long_turn", "consecutive_run")
        flag_type_counter: dict[str, int] = {}
        for transcript in transcripts:
            for flag in transcript.flags:
                prefix = flag.split(":")[0]
                flag_type_counter[prefix] = flag_type_counter.get(prefix, 0) + 1

        # Doctor word share
        total_doctor_words = sum(
            len(t.text.split())
            for transcript in transcripts
            for t in transcript.turns
            if t.speaker == "DOCTOR"
        )
        total_all_words = sum(t.word_count for t in transcripts)
        doctor_word_share = (
            round(100 * total_doctor_words / total_all_words, 1)
            if total_all_words
            else 0.0
        )

        return {
            "total": total,
            "avg_turns": round(avg_turns, 2),
            "avg_words": round(avg_words, 2),
            "flagged_count": flagged_count,
            "flag_type_breakdown": flag_type_counter,
            "doctor_word_share": doctor_word_share,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(record: dict) -> None:
        """
        Raise ValueError if *record* is missing any required field or
        if the turns field is not a non-empty list.
        """
        missing = [f for f in REQUIRED_FIELDS if f not in record]
        if missing:
            _id = record.get("id", "<unknown>")
            raise ValueError(
                f"Record {_id!r} is missing required fields: {missing}"
            )
        if not isinstance(record["turns"], list) or len(record["turns"]) == 0:
            raise ValueError(
                f"Record {record['id']!r} has an empty or non-list 'turns' field."
            )


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

def _print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def _load_records(path: Path, n: int) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= n:
                break
    return records


if __name__ == "__main__":
    DATA_PATH = Path(__file__).resolve().parents[3] / "data/raw/unified/en_mts.jsonl"

    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}")
        print("Run data/pipeline/setup.py first to download the datasets.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  TranscriptProcessor — standalone smoke-test")
    print("=" * 70)

    records = _load_records(DATA_PATH, 10)
    print(f"\nLoaded {len(records)} records from {DATA_PATH.name}\n")

    processor = TranscriptProcessor()

    # ── Batch processing ──────────────────────────────────────────────
    _print_separator("─")
    print("process_batch() output:")
    _print_separator("─")
    transcripts = processor.process_batch(records, show_progress=True)

    # ── Full display of first 2 transcripts ───────────────────────────
    print()
    _print_separator("═")
    print("  Full ProcessedTranscript objects (first 2)")
    _print_separator("═")
    for transcript in transcripts[:2]:
        print()
        print(str(transcript))
        _print_separator("─")

    # ── Stats ─────────────────────────────────────────────────────────
    print()
    _print_separator("═")
    print("  get_stats() output")
    _print_separator("═")
    stats = processor.get_stats(transcripts)
    for key, value in stats.items():
        print(f"  {key:<28} {value}")
    print()
