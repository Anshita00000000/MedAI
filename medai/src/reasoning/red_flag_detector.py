"""
red_flag_detector.py

Scans conversation transcripts, extracted entities, and SOAP notes for clinical
red flags — symptoms or patterns that warrant urgent escalation or specialist
referral.

Rules are loaded from configs/red_flags.yaml. A hardcoded fallback is used when
the file is missing or malformed so the detector always operates without external
files.

Pure stdlib + optional PyYAML — no LLM dependency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from medai.src.voice.transcript_processor import ProcessedTranscript
    from medai.src.clinical.entity_extractor import MedicalEntities
    from medai.src.clinical.soap_generator import SOAPNote

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity ordering (higher index = higher severity)
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"monitor": 1, "urgent": 2, "critical": 3, "none": 0}

# ---------------------------------------------------------------------------
# Hardcoded fallback rules (mirrors configs/red_flags.yaml)
# ---------------------------------------------------------------------------

_DEFAULT_RULES: dict[str, list[dict]] = {
    "critical": [
        {
            "term": "chest pain",
            "category": "cardiac",
            "keywords": ["chest pain", "chest pressure", "chest tightness", "crushing chest"],
            "description": "Possible acute coronary syndrome or cardiac emergency",
            "action": "Immediate ECG and cardiac enzyme panel",
        },
        {
            "term": "difficulty breathing",
            "category": "respiratory",
            "keywords": [
                "difficulty breathing", "can't breathe", "unable to breathe",
                "shortness of breath", "respiratory distress",
            ],
            "description": "Acute respiratory compromise",
            "action": "Assess airway, O2 saturation, consider emergency intervention",
        },
        {
            "term": "stroke symptoms",
            "category": "neurological",
            "keywords": [
                "facial droop", "arm weakness", "speech difficulty", "sudden numbness",
                "worst headache of my life", "thunderclap headache",
            ],
            "description": "Possible acute stroke — FAST criteria",
            "action": "Immediate CT head, stroke team activation",
        },
        {
            "term": "altered consciousness",
            "category": "neurological",
            "keywords": [
                "unconscious", "unresponsive", "altered consciousness",
                "confusion", "disoriented", "loss of consciousness",
            ],
            "description": "Altered mental status requiring urgent evaluation",
            "action": "Neurological assessment, blood glucose, CT if indicated",
        },
        {
            "term": "severe bleeding",
            "category": "haematological",
            "keywords": [
                "severe bleeding", "uncontrolled bleeding", "hemorrhage",
                "coughing up blood", "vomiting blood", "blood in stool",
            ],
            "description": "Significant haemorrhage risk",
            "action": "Immediate haemostasis assessment, IV access, cross-match",
        },
    ],
    "urgent": [
        {
            "term": "high fever",
            "category": "infectious",
            "keywords": [
                "high fever", "temperature above 39", "fever of 40",
                "rigors", "sepsis", "infected",
            ],
            "description": "Possible systemic infection or sepsis",
            "action": "Blood cultures, sepsis protocol if indicated",
        },
        {
            "term": "severe pain",
            "category": "pain",
            "keywords": [
                "severe pain", "excruciating", "pain scale 9", "pain scale 10",
                "unbearable pain", "worst pain",
            ],
            "description": "Severe pain requiring prompt assessment",
            "action": "Pain assessment tool, analgesia review",
        },
        {
            "term": "allergic reaction",
            "category": "immunological",
            "keywords": [
                "allergic reaction", "anaphylaxis", "hives",
                "throat swelling", "tongue swelling", "epipen",
            ],
            "description": "Possible allergic or anaphylactic reaction",
            "action": "Epinephrine if anaphylaxis, antihistamine, monitor",
        },
    ],
    "monitor": [
        {
            "term": "weight loss",
            "category": "systemic",
            "keywords": ["weight loss", "losing weight", "lost weight unintentionally"],
            "description": "Unexplained weight loss — may indicate systemic illness",
            "action": "Document weight trend, screen for malignancy if >5% in 6 months",
        },
        {
            "term": "persistent cough",
            "category": "respiratory",
            "keywords": ["persistent cough", "cough for weeks", "cough for months", "chronic cough"],
            "description": "Chronic cough — investigate underlying cause",
            "action": "CXR, spirometry if indicated",
        },
        {
            "term": "safeguarding concern",
            "category": "safeguarding",
            "keywords": [
                "bruising", "unexplained injury", "abuse", "neglect",
                "afraid at home", "feels unsafe",
            ],
            "description": "Possible safeguarding concern",
            "action": "Follow safeguarding protocol, document carefully",
        },
    ],
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RedFlag:
    """A single triggered red-flag rule."""

    term: str                   # canonical term from the rule (e.g. "chest pain")
    severity: str               # "critical" | "urgent" | "monitor"
    category: str               # e.g. "cardiac", "neurological"
    description: str            # plain-English clinical explanation
    matched_text: str           # exact text snippet that triggered the match
    recommended_action: str     # what the clinician should do

    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.term} ({self.category})\n"
            f"  Description : {self.description}\n"
            f"  Matched     : \"{self.matched_text}\"\n"
            f"  Action      : {self.recommended_action}"
        )


@dataclass
class RedFlagResult:
    """Aggregated red-flag scan result for one conversation."""

    conversation_id: str
    flags: List[RedFlag]
    highest_severity: str               # "critical" | "urgent" | "monitor" | "none"
    flag_count: int
    requires_immediate_action: bool     # True if any flag is "critical"
    summary: str                        # one-sentence summary

    def __str__(self) -> str:
        flag_lines = "\n".join(f"  {f}" for f in self.flags) or "  (none)"
        return (
            f"RedFlagResult(id={self.conversation_id!r}, "
            f"severity={self.highest_severity!r}, "
            f"flags={self.flag_count}, "
            f"immediate={self.requires_immediate_action})\n"
            f"  Summary: {self.summary}\n"
            f"{flag_lines}"
        )


# ---------------------------------------------------------------------------
# RedFlagDetector
# ---------------------------------------------------------------------------

class RedFlagDetector:
    """
    Detects clinical red flags in patient conversations, extracted entities,
    and SOAP notes.

    Rules are loaded from ``configs/red_flags.yaml`` at initialisation and
    compiled into word-boundary regex patterns for efficient reuse.  Falls
    back to built-in defaults when the YAML file is absent or malformed.

    Usage::

        detector = RedFlagDetector()
        result = detector.detect(transcript, entities, soap)
        batch  = detector.detect_batch(transcripts, entities_list, soap_notes)
    """

    def __init__(self, config_path: str = "configs/red_flags.yaml") -> None:
        """
        Initialise the detector by loading red-flag rules.

        Parameters
        ----------
        config_path:
            Path to the YAML rule file, resolved relative to the ``medai/``
            package root.  Falls back to hardcoded defaults on any error.
        """
        self._rules: dict[str, list[dict]] = self._load_rules(config_path)
        # Pre-compile one regex per keyword for performance
        # Structure: {severity: [(rule_dict, [(keyword, compiled_re), ...]), ...]}
        self._compiled: dict[str, list[tuple[dict, list[tuple[str, re.Pattern]]]]] = {}
        for severity, rules in self._rules.items():
            compiled_rules = []
            for rule in rules:
                kw_patterns = [
                    (kw, re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE))
                    for kw in rule.get("keywords", [])
                ]
                compiled_rules.append((rule, kw_patterns))
            self._compiled[severity] = compiled_rules

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        transcript: ProcessedTranscript,
        entities: MedicalEntities,
        soap: SOAPNote | None = None,
    ) -> RedFlagResult:
        """
        Scan a conversation for red flags across all available text sources.

        Search targets (in order):
        1. ``transcript.full_text`` — raw conversation turns
        2. ``entities.symptoms`` and ``entities.clinical_findings`` joined
        3. ``soap.subjective`` + ``soap.assessment`` if *soap* is provided

        The same term is only flagged once even if matched in multiple sources.

        Parameters
        ----------
        transcript:
            ProcessedTranscript from TranscriptProcessor.
        entities:
            MedicalEntities from EntityExtractor.
        soap:
            Optional SOAPNote from SOAPGenerator; its Subjective and
            Assessment sections are also scanned when provided.

        Returns
        -------
        RedFlagResult
        """
        # Build one combined search corpus
        parts = [transcript.full_text]
        if entities.symptoms:
            parts.append(" ".join(entities.symptoms))
        if entities.clinical_findings:
            parts.append(" ".join(entities.clinical_findings))
        if soap is not None:
            parts.append(soap.subjective)
            parts.append(soap.assessment)
        search_text = " ".join(parts)

        all_flags: list[RedFlag] = []
        seen_terms: set[str] = set()

        for severity in ("critical", "urgent", "monitor"):
            for flag in self._match_flags(search_text, severity,
                                          self._compiled.get(severity, [])):
                if flag.term not in seen_terms:
                    seen_terms.add(flag.term)
                    all_flags.append(flag)

        highest = _highest_severity(all_flags)
        immediate = any(f.severity == "critical" for f in all_flags)
        summary = _build_summary(transcript.id, all_flags, highest)

        return RedFlagResult(
            conversation_id=transcript.id,
            flags=all_flags,
            highest_severity=highest,
            flag_count=len(all_flags),
            requires_immediate_action=immediate,
            summary=summary,
        )

    def _match_flags(
        self,
        text: str,
        severity: str,
        compiled_rules: list[tuple[dict, list[tuple[str, re.Pattern]]]],
    ) -> list[RedFlag]:
        """
        Check *text* against all compiled rules for a given *severity* level.

        For each rule, the first keyword that matches produces one ``RedFlag``.
        The matched-text snippet includes up to 20 characters of context on
        each side of the match.

        Parameters
        ----------
        text:
            The full search string (case-insensitive matching is applied).
        severity:
            Severity label for the rule set being checked.
        compiled_rules:
            Pre-compiled list of ``(rule_dict, [(keyword, pattern), ...])``
            tuples built during ``__init__``.

        Returns
        -------
        list[RedFlag] — one entry per matched rule (not per keyword hit).
        """
        flags: list[RedFlag] = []

        for rule, kw_patterns in compiled_rules:
            for keyword, pattern in kw_patterns:
                m = pattern.search(text)
                if m:
                    start = max(0, m.start() - 20)
                    end = min(len(text), m.end() + 20)
                    snippet = text[start:end].strip()
                    flags.append(RedFlag(
                        term=rule["term"],
                        severity=severity,
                        category=rule.get("category", ""),
                        description=rule.get("description", ""),
                        matched_text=snippet,
                        recommended_action=rule.get("action", ""),
                    ))
                    break   # one flag per rule; stop after first keyword hit

        return flags

    def detect_batch(
        self,
        transcripts: list[ProcessedTranscript],
        entities_list: list[MedicalEntities],
        soap_notes: list[SOAPNote] | None = None,
        show_progress: bool = True,
    ) -> list[RedFlagResult]:
        """
        Run red-flag detection for a list of conversations.

        All three lists are matched by ``conversation_id``.  Transcripts with
        no matching entities are skipped.  Per-record failures are isolated
        and excluded from the result.

        Parameters
        ----------
        transcripts:
            ProcessedTranscript objects in any order.
        entities_list:
            MedicalEntities objects matched to transcripts by conversation_id.
        soap_notes:
            Optional list of SOAPNote objects; matched by conversation_id.
        show_progress:
            Print progress every 10 records and a final summary.

        Returns
        -------
        list[RedFlagResult]
        """
        entities_index = {e.conversation_id: e for e in entities_list}
        soap_index: dict[str, SOAPNote] = {}
        if soap_notes:
            soap_index = {s.conversation_id: s for s in soap_notes}

        results: list[RedFlagResult] = []
        failures: list[tuple[str, str]] = []

        for i, transcript in enumerate(transcripts):
            t_id = getattr(transcript, "id", f"index:{i}")
            entities = entities_index.get(t_id)

            if entities is None:
                log.warning("No entities for transcript %r — skipping.", t_id)
                failures.append((t_id, "no matching entities"))
                continue

            soap = soap_index.get(t_id)

            try:
                results.append(self.detect(transcript, entities, soap))
            except Exception as exc:  # noqa: BLE001
                log.error("detect_batch: failed for %r — %s", t_id, exc)
                failures.append((t_id, str(exc)))

            if show_progress and (i + 1) % 10 == 0:
                print(
                    f"  … {i + 1}/{len(transcripts)} scanned "
                    f"({len(failures)} failures so far)"
                )

        if show_progress:
            print(
                f"  Batch complete: {len(results)} succeeded, "
                f"{len(failures)} failed out of {len(transcripts)} records."
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rules(config_path: str) -> dict[str, list[dict]]:
        """
        Load red-flag rules from *config_path*.

        Resolves the path relative to the ``medai/`` package root
        (two levels above this file's directory).  Falls back to
        ``_DEFAULT_RULES`` on any error.

        Returns
        -------
        dict mapping severity labels to lists of rule dicts.
        """
        yaml_path = Path(__file__).resolve().parents[2] / config_path

        if not yaml_path.exists():
            log.debug("red_flags.yaml not found at %s — using defaults.", yaml_path)
            return _DEFAULT_RULES

        try:
            import yaml  # type: ignore  # noqa: PLC0415
            with open(yaml_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            rules = data.get("red_flags", {})
            if not isinstance(rules, dict) or not rules:
                raise ValueError("red_flags key missing or empty")
            log.debug("Loaded red-flag rules from %s", yaml_path)
            return rules
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Could not load red_flags.yaml (%s) — using defaults.", exc
            )
            return _DEFAULT_RULES


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _highest_severity(flags: list[RedFlag]) -> str:
    """Return the highest severity label among *flags*, or 'none'."""
    if not flags:
        return "none"
    return max(flags, key=lambda f: _SEVERITY_RANK.get(f.severity, 0)).severity


def _build_summary(conversation_id: str, flags: list[RedFlag], highest: str) -> str:
    """Build a one-sentence summary of red-flag findings."""
    if not flags:
        return f"No red flags identified for conversation {conversation_id!r}."
    categories = list(dict.fromkeys(f.category for f in flags))   # ordered dedup
    cat_str = ", ".join(categories[:3])
    if len(categories) > 3:
        cat_str += f" (+{len(categories) - 3} more)"
    count = len(flags)
    noun = "flag" if count == 1 else "flags"
    return (
        f"{count} red {noun} detected (highest: {highest}) "
        f"in categories: {cat_str}."
    )


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

def _print_sep(char: str = "─", width: int = 70) -> None:
    print(char * width)


if __name__ == "__main__":
    import json
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from medai.src.voice.transcript_processor import TranscriptProcessor   # noqa: PLC0415
    from medai.src.clinical.entity_extractor import EntityExtractor        # noqa: PLC0415
    from medai.src.clinical.soap_generator import SOAPGenerator            # noqa: PLC0415

    DATA_PATH = _REPO_ROOT / "data/raw/unified/en_mts.jsonl"
    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}")
        print("Run data/pipeline/setup.py first.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  RedFlagDetector — standalone smoke-test")
    print("=" * 70)

    records: list[dict] = []
    with open(DATA_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= 3:
                break

    transcripts = TranscriptProcessor().process_batch(records, show_progress=False)
    entities_list = EntityExtractor().extract_batch(transcripts, show_progress=False)
    soaps = SOAPGenerator().generate_batch(transcripts, entities_list, show_progress=False)

    detector = RedFlagDetector()
    for transcript, entities, soap in zip(transcripts, entities_list, soaps):
        result = detector.detect(transcript, entities, soap)
        _print_sep("─")
        print(str(result))
    print()
