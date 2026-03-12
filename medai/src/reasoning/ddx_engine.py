"""
ddx_engine.py

Differential Diagnosis (DDx) Engine — produces a ranked list of candidate
diagnoses from extracted clinical entities and conversation context.

Two modes:
  rule_based — keyword matching against configs/disease_mapping.yaml
  llm        — Gemini reranks and enriches rule-based candidates

The LLM pass is optional: if google-genai is unavailable or GOOGLE_API_KEY
is not set, the engine operates in rule-based mode only.

Runnable smoke-test::

    python src/reasoning/ddx_engine.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from medai.src.voice.transcript_processor import ProcessedTranscript
    from medai.src.clinical.entity_extractor import MedicalEntities
    from medai.src.clinical.soap_generator import SOAPNote
    from medai.src.reasoning.red_flag_detector import RedFlagResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional google-genai import
# ---------------------------------------------------------------------------

_GENAI_AVAILABLE = False
_genai = None

try:
    from google import genai as _genai  # type: ignore
    _GENAI_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_DIAGNOSES = 5
_MAX_TRANSCRIPT_WORDS = 1000       # words sent to LLM in DDx prompt
_TEMPERATURE = 0.2
_MAX_OUTPUT_TOKENS = 1024
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2
_BATCH_INTER_CALL_DELAY = 1.0

# Hardcoded fallback disease mappings (mirrors configs/disease_mapping.yaml)
_DEFAULT_DISEASE_MAPPING: dict[str, list[dict]] = {
    "respiratory": [
        {
            "name": "Community Acquired Pneumonia", "icd10": "J18.9",
            "keywords": ["fever", "cough", "shortness of breath", "chest pain", "sputum"],
            "min_matches": 3,
        },
        {
            "name": "Asthma Exacerbation", "icd10": "J45.901",
            "keywords": ["wheezing", "shortness of breath", "chest tightness", "inhaler"],
            "min_matches": 2,
        },
        {
            "name": "COPD Exacerbation", "icd10": "J44.1",
            "keywords": ["copd", "emphysema", "chronic bronchitis", "dyspnea", "inhaler"],
            "min_matches": 2,
        },
    ],
    "cardiac": [
        {
            "name": "Acute Coronary Syndrome", "icd10": "I24.9",
            "keywords": ["chest pain", "chest pressure", "radiating pain", "diaphoresis", "nausea"],
            "min_matches": 2,
        },
        {
            "name": "Heart Failure", "icd10": "I50.9",
            "keywords": ["shortness of breath", "edema", "fatigue", "orthopnea"],
            "min_matches": 2,
        },
    ],
    "neurological": [
        {
            "name": "Migraine", "icd10": "G43.909",
            "keywords": ["headache", "nausea", "light sensitivity", "visual aura", "blurry vision"],
            "min_matches": 2,
        },
        {
            "name": "Ischemic Stroke", "icd10": "I63.9",
            "keywords": ["facial droop", "arm weakness", "speech difficulty", "sudden headache"],
            "min_matches": 2,
        },
        {
            "name": "Tension Headache", "icd10": "G44.209",
            "keywords": ["headache", "stress", "bilateral", "pressure"],
            "min_matches": 2,
        },
    ],
    "gastrointestinal": [
        {
            "name": "Gastroenteritis", "icd10": "K59.1",
            "keywords": ["nausea", "vomiting", "diarrhea", "abdominal pain", "fever"],
            "min_matches": 3,
        },
        {
            "name": "GERD", "icd10": "K21.0",
            "keywords": ["heartburn", "acid reflux", "chest burning", "regurgitation"],
            "min_matches": 2,
        },
    ],
    "musculoskeletal": [
        {
            "name": "Osteoarthritis", "icd10": "M19.90",
            "keywords": ["joint pain", "stiffness", "swelling", "osteoarthritis", "knee pain"],
            "min_matches": 2,
        },
        {
            "name": "Lower Back Pain", "icd10": "M54.5",
            "keywords": ["back pain", "lower back", "lumbar", "sciatica"],
            "min_matches": 2,
        },
    ],
    "infectious": [
        {
            "name": "Upper Respiratory Tract Infection", "icd10": "J06.9",
            "keywords": ["sore throat", "runny nose", "congestion", "cough", "fever"],
            "min_matches": 3,
        },
        {
            "name": "Urinary Tract Infection", "icd10": "N39.0",
            "keywords": ["burning urination", "frequent urination", "dysuria", "cloudy urine"],
            "min_matches": 2,
        },
    ],
}

# Probability tier thresholds
_PROB_TIERS = [(0.66, "high"), (0.33, "moderate"), (0.0, "low")]

# Neurological / cardiac speciality keywords for follow-up question generation
_CARDIAC_NAMES = {"acute coronary syndrome", "heart failure"}
_NEURO_NAMES = {"migraine", "ischemic stroke", "tension headache"}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Diagnosis:
    """A single candidate diagnosis with supporting evidence."""

    name: str
    icd10_code: str
    probability: str                        # "high" | "moderate" | "low"
    probability_score: float                # 0.0–1.0
    supporting_evidence: List[str]          # matched keywords / entity values
    contradicting_evidence: List[str]
    reasoning: str

    def __str__(self) -> str:
        sup = "; ".join(self.supporting_evidence) or "none"
        con = "; ".join(self.contradicting_evidence) or "none"
        return (
            f"  [{self.probability.upper()}] {self.name} (ICD-10: {self.icd10_code}, "
            f"score={self.probability_score:.2f})\n"
            f"    Supporting  : {sup}\n"
            f"    Contradicting: {con}\n"
            f"    Reasoning   : {self.reasoning}"
        )


@dataclass
class DDxResult:
    """Aggregated differential diagnosis result for one conversation."""

    conversation_id: str
    top_diagnoses: List[Diagnosis]          # sorted by probability_score descending
    primary_diagnosis: Diagnosis            # highest-scored diagnosis
    follow_up_questions: List[str]
    recommended_investigations: List[str]
    red_flag_result: RedFlagResult
    generation_method: str                  # "llm" | "rule_based"
    speciality: str

    def __str__(self) -> str:
        dx_lines = "\n".join(str(d) for d in self.top_diagnoses) or "  (none)"
        fup = "\n".join(f"  - {q}" for q in self.follow_up_questions) or "  (none)"
        inv = "\n".join(f"  - {i}" for i in self.recommended_investigations) or "  (none)"
        return (
            f"DDxResult(id={self.conversation_id!r}, "
            f"method={self.generation_method!r}, "
            f"speciality={self.speciality!r})\n"
            f"  Primary: {self.primary_diagnosis.name} "
            f"({self.primary_diagnosis.probability.upper()})\n"
            f"\nTop diagnoses:\n{dx_lines}\n"
            f"\nFollow-up questions:\n{fup}\n"
            f"\nRecommended investigations:\n{inv}\n"
        )


# ---------------------------------------------------------------------------
# DDxEngine
# ---------------------------------------------------------------------------

class DDxEngine:
    """
    Differential Diagnosis Engine.

    Stage 1 (always): rule-based keyword matching against disease_mapping.yaml.
    Stage 2 (optional): Gemini LLM reranks and enriches candidates.

    Usage::

        engine = DDxEngine()
        result = engine.analyse(transcript, entities, soap, red_flag_result)
        batch  = engine.analyse_batch(transcripts, entities_list, soaps, rfrs)
    """

    def __init__(
        self,
        config_path: str = "configs/disease_mapping.yaml",
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ) -> None:
        """
        Initialise the DDx engine.

        Parameters
        ----------
        config_path:
            Path to disease_mapping.yaml, resolved relative to medai/ root.
        api_key:
            Google AI API key. Defaults to GOOGLE_API_KEY env var.
            If absent, the engine operates in rule-based mode only.
        model:
            Gemini model to use for LLM reranking.
        """
        self.model_name = model
        self.use_fallback: bool = True
        self._client = None
        self._follow_up_questions: list[str] = []
        self._recommended_investigations: list[str] = []

        self._disease_mapping = self._load_disease_mapping(config_path)

        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not resolved_key:
            log.warning(
                "GOOGLE_API_KEY not set — DDxEngine will use rule-based mode only."
            )

        if not _GENAI_AVAILABLE:
            log.warning(
                "google-genai is not installed — DDxEngine using rule-based mode. "
                "Install with: pip install google-genai>=1.0.0"
            )
        elif resolved_key:
            try:
                self._client = _genai.Client(api_key=resolved_key)
                self.use_fallback = False
                log.info("DDxEngine initialised with model %r", model)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Failed to initialise Gemini client (%s) — rule-based only.", exc
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        transcript: ProcessedTranscript,
        entities: MedicalEntities,
        soap: SOAPNote,
        red_flag_result: RedFlagResult,
    ) -> DDxResult:
        """
        Produce a ranked differential diagnosis for one conversation.

        Stage 1 always runs rule-based DDx.  Stage 2 optionally uses Gemini
        to rerank and add clinical reasoning.

        Parameters
        ----------
        transcript:
            Source conversation.
        entities:
            Extracted clinical entities.
        soap:
            Generated SOAP note (used to infer speciality).
        red_flag_result:
            Pre-computed red-flag result to embed in the output.

        Returns
        -------
        DDxResult
        """
        # Stage 1 — always
        candidates = self._rule_based_ddx(entities, transcript)

        follow_up: list[str] = []
        investigations: list[str] = []
        method = "rule_based"

        # Stage 2 — LLM reranking if available and there are candidates
        if not self.use_fallback and candidates:
            try:
                candidates, follow_up, investigations = self._llm_rank_ddx(
                    candidates, transcript, entities
                )
                method = "llm"
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "LLM reranking failed for %r (%s) — using rule-based only.",
                    transcript.id, exc,
                )

        # Rule-based follow-up questions if LLM path was skipped or failed
        if not follow_up:
            follow_up = self._generate_follow_up_questions(entities, candidates)

        # Default investigations from top diagnoses when LLM didn't provide them
        if not investigations:
            investigations = _default_investigations(candidates)

        primary = candidates[0] if candidates else _unknown_diagnosis()

        return DDxResult(
            conversation_id=transcript.id,
            top_diagnoses=candidates,
            primary_diagnosis=primary,
            follow_up_questions=follow_up,
            recommended_investigations=investigations,
            red_flag_result=red_flag_result,
            generation_method=method,
            speciality=soap.speciality or "General Medicine",
        )

    def _rule_based_ddx(
        self,
        entities: MedicalEntities,
        transcript: ProcessedTranscript,
    ) -> list[Diagnosis]:
        """
        Score candidate diagnoses using keyword matching against entity fields
        and the full conversation text.

        Algorithm
        ---------
        1. Build a lowercase search corpus from symptoms, findings, and
           ``transcript.full_text``.
        2. For each disease in ``_disease_mapping``, count how many of its
           keywords appear in the corpus.
        3. Include the disease only if the match count meets ``min_matches``.
        4. Compute ``score = matched / total_keywords`` (0.0–1.0).
        5. Return up to ``_MAX_DIAGNOSES`` diseases sorted by score descending.

        Parameters
        ----------
        entities:
            Extracted MedicalEntities.
        transcript:
            Source ProcessedTranscript (full_text is searched).

        Returns
        -------
        list[Diagnosis] — sorted descending by probability_score, max 5.
        """
        # Build search corpus (lowercase)
        corpus_parts = list(entities.symptoms) + list(entities.clinical_findings)
        corpus_parts.append(transcript.full_text)
        corpus = " ".join(corpus_parts).lower()

        scored: list[Diagnosis] = []

        for _speciality, diseases in self._disease_mapping.items():
            for disease in diseases:
                keywords: list[str] = disease.get("keywords", [])
                min_matches: int = disease.get("min_matches", 1)
                if not keywords:
                    continue

                matched = [kw for kw in keywords if kw.lower() in corpus]
                if len(matched) < min_matches:
                    continue

                score = len(matched) / len(keywords)
                probability = _score_to_tier(score)

                scored.append(Diagnosis(
                    name=disease["name"],
                    icd10_code=disease.get("icd10", ""),
                    probability=probability,
                    probability_score=round(score, 4),
                    supporting_evidence=matched,
                    contradicting_evidence=[],
                    reasoning=(
                        f"Matched {len(matched)}/{len(keywords)} keywords "
                        f"({', '.join(matched[:3])}"
                        f"{'...' if len(matched) > 3 else ''})."
                    ),
                ))

        scored.sort(key=lambda d: d.probability_score, reverse=True)
        return scored[:_MAX_DIAGNOSES]

    def _llm_rank_ddx(
        self,
        candidates: list[Diagnosis],
        transcript: ProcessedTranscript,
        entities: MedicalEntities,
    ) -> tuple[list[Diagnosis], list[str], list[str]]:
        """
        Send candidates to Gemini for reranking, reasoning enrichment, and
        generation of follow-up questions and recommended investigations.

        The prompt requests a JSON array so the response can be parsed without
        heuristics.  On any parse error, the original *candidates* list is
        returned unchanged with empty auxiliary lists.

        Parameters
        ----------
        candidates:
            Rule-based candidate diagnoses.
        transcript:
            Source conversation (first 1000 words sent in prompt).
        entities:
            Extracted entities for context.

        Returns
        -------
        tuple of:
        - list[Diagnosis] sorted by updated probability_score
        - list[str] follow_up_questions
        - list[str] recommended_investigations
        """
        candidate_summary = [
            {"name": d.name, "icd10": d.icd10_code, "score": d.probability_score}
            for d in candidates
        ]

        # Truncate transcript
        words = transcript.full_text.split()
        excerpt = " ".join(words[:_MAX_TRANSCRIPT_WORDS])
        if len(words) > _MAX_TRANSCRIPT_WORDS:
            excerpt += "\n[… transcript truncated …]"

        prompt = (
            "Given this clinical transcript and extracted entities, "
            "review these candidate diagnoses and return a JSON array "
            "with updated probability scores and reasoning.\n\n"
            f"Candidates: {json.dumps(candidate_summary)}\n"
            f"Symptoms: {entities.symptoms}\n"
            f"Clinical Findings: {entities.clinical_findings}\n"
            f"Medical History: {entities.medical_history}\n"
            f"Transcript excerpt:\n{excerpt}\n\n"
            "Return ONLY valid JSON — no markdown, no prose. Format:\n"
            '[{"name": str, "probability_score": float (0-1), '
            '"reasoning": str, "follow_up_questions": [str], '
            '"investigations": [str]}]'
        )

        raw = self._call_gemini(prompt)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Try extracting JSON array from the response
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                except json.JSONDecodeError:
                    log.warning("LLM returned unparseable JSON — using rule-based candidates.")
                    return candidates, [], []
            else:
                log.warning("LLM returned no JSON array — using rule-based candidates.")
                return candidates, [], []

        if not isinstance(parsed, list):
            log.warning("LLM JSON was not a list — using rule-based candidates.")
            return candidates, [], []

        # Build a lookup from the original candidates
        orig_by_name = {d.name.lower(): d for d in candidates}
        enriched: list[Diagnosis] = []
        follow_up: list[str] = []
        investigations: list[str] = []

        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            orig = orig_by_name.get(name.lower())
            score = float(item.get("probability_score", orig.probability_score if orig else 0.0))
            score = max(0.0, min(1.0, score))

            enriched.append(Diagnosis(
                name=name or (orig.name if orig else "Unknown"),
                icd10_code=orig.icd10_code if orig else "",
                probability=_score_to_tier(score),
                probability_score=round(score, 4),
                supporting_evidence=orig.supporting_evidence if orig else [],
                contradicting_evidence=[],
                reasoning=str(item.get("reasoning", orig.reasoning if orig else "")),
            ))

            # Collect follow-up questions and investigations from any item
            for q in item.get("follow_up_questions", []):
                if q and q not in follow_up:
                    follow_up.append(q)
            for inv in item.get("investigations", []):
                if inv and inv not in investigations:
                    investigations.append(inv)

        if not enriched:
            return candidates, follow_up, investigations

        enriched.sort(key=lambda d: d.probability_score, reverse=True)
        return enriched[:_MAX_DIAGNOSES], follow_up[:5], investigations[:5]

    @staticmethod
    def _generate_follow_up_questions(
        entities: MedicalEntities,
        diagnoses: list[Diagnosis],
    ) -> list[str]:
        """
        Generate 3–5 clinically relevant follow-up questions based on gaps
        in the extracted entities and the nature of the top diagnoses.

        Rules applied:
        - If no vitals recorded: ask for current vital signs.
        - If no medical history: ask for past medical history.
        - If a cardiac diagnosis appears in the top 3: ask about family history
          of heart disease.
        - If a neurological diagnosis appears in the top 3: ask about recent
          head trauma or similar episodes.
        - Always append a medication adherence question as a safe default.

        Parameters
        ----------
        entities:
            Extracted MedicalEntities (used to detect gaps).
        diagnoses:
            Rule-based or LLM-enriched candidates (top 3 examined).

        Returns
        -------
        list[str] — 3 to 5 questions.
        """
        questions: list[str] = []

        if not entities.vitals:
            questions.append("What are the patient's current vital signs?")

        if not entities.medical_history:
            questions.append(
                "Does the patient have any relevant past medical history or chronic conditions?"
            )

        top3_names = {d.name.lower() for d in diagnoses[:3]}
        if top3_names & _CARDIAC_NAMES:
            questions.append(
                "Is there a family history of heart disease or sudden cardiac death?"
            )

        if top3_names & _NEURO_NAMES:
            questions.append(
                "Has the patient experienced any recent head trauma or similar episodes previously?"
            )

        questions.append(
            "Is the patient currently taking any medications, including over-the-counter drugs?"
        )

        return questions[:5]

    def analyse_batch(
        self,
        transcripts: list[ProcessedTranscript],
        entities_list: list[MedicalEntities],
        soap_notes: list[SOAPNote],
        red_flag_results: list[RedFlagResult],
        show_progress: bool = True,
    ) -> list[DDxResult]:
        """
        Run DDx analysis for a list of conversations.

        All four lists are matched by ``conversation_id``.  Records with no
        matching entities, SOAP note, or red-flag result are skipped.
        Per-record failures are isolated.  A 1-second delay is applied between
        LLM calls when operating in LLM mode.

        Parameters
        ----------
        transcripts:
            ProcessedTranscript objects in any order.
        entities_list:
            MedicalEntities objects matched by conversation_id.
        soap_notes:
            SOAPNote objects matched by conversation_id.
        red_flag_results:
            RedFlagResult objects matched by conversation_id.
        show_progress:
            Print progress every 5 records and a final summary.

        Returns
        -------
        list[DDxResult]
        """
        entities_index = {e.conversation_id: e for e in entities_list}
        soap_index = {s.conversation_id: s for s in soap_notes}
        rfr_index = {r.conversation_id: r for r in red_flag_results}

        results: list[DDxResult] = []
        failures: list[tuple[str, str]] = []

        for i, transcript in enumerate(transcripts):
            t_id = getattr(transcript, "id", f"index:{i}")

            entities = entities_index.get(t_id)
            soap = soap_index.get(t_id)
            rfr = rfr_index.get(t_id)

            if entities is None or soap is None or rfr is None:
                missing = [
                    label for label, val in
                    [("entities", entities), ("soap", soap), ("red_flag_result", rfr)]
                    if val is None
                ]
                reason = f"missing: {', '.join(missing)}"
                log.warning("analyse_batch: skipping %r — %s", t_id, reason)
                failures.append((t_id, reason))
                continue

            try:
                results.append(self.analyse(transcript, entities, soap, rfr))
            except Exception as exc:  # noqa: BLE001
                log.error("analyse_batch: failed for %r — %s", t_id, exc)
                failures.append((t_id, str(exc)))

            if not self.use_fallback and i < len(transcripts) - 1:
                time.sleep(_BATCH_INTER_CALL_DELAY)

            if show_progress and (i + 1) % 5 == 0:
                print(
                    f"  … {i + 1}/{len(transcripts)} analysed "
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

    def _call_gemini(self, prompt: str) -> str:
        """
        Send *prompt* to Gemini with retry logic.

        Retries up to ``_MAX_RETRIES`` times with ``_RETRY_DELAY_SECONDS``
        delay between attempts.

        Parameters
        ----------
        prompt:
            Full prompt string.

        Returns
        -------
        str — raw model response text.

        Raises
        ------
        RuntimeError
            If all retries are exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=_genai.types.GenerateContentConfig(
                        temperature=_TEMPERATURE,
                        max_output_tokens=_MAX_OUTPUT_TOKENS,
                    ),
                )
                return response.text
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning(
                    "Gemini call failed (attempt %d/%d): %s",
                    attempt, _MAX_RETRIES, exc,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_DELAY_SECONDS)

        raise RuntimeError(
            f"Gemini API failed after {_MAX_RETRIES} attempts: {last_exc}"
        )

    @staticmethod
    def _load_disease_mapping(config_path: str) -> dict[str, list[dict]]:
        """
        Load disease keyword mappings from *config_path*.

        Resolved relative to the ``medai/`` package root (two levels above
        this file's directory).  Falls back to ``_DEFAULT_DISEASE_MAPPING``
        on any error.

        Returns
        -------
        dict mapping speciality labels to lists of disease dicts.
        """
        yaml_path = Path(__file__).resolve().parents[2] / config_path

        if not yaml_path.exists():
            log.debug("disease_mapping.yaml not found at %s — using defaults.", yaml_path)
            return _DEFAULT_DISEASE_MAPPING

        try:
            import yaml  # type: ignore  # noqa: PLC0415
            with open(yaml_path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            mapping = data.get("disease_mapping", {})
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("disease_mapping key missing or empty")
            log.debug("Loaded disease mapping from %s", yaml_path)
            return mapping
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Could not load disease_mapping.yaml (%s) — using defaults.", exc
            )
            return _DEFAULT_DISEASE_MAPPING


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _score_to_tier(score: float) -> str:
    """Convert a 0–1 score to a probability tier string."""
    for threshold, label in _PROB_TIERS:
        if score >= threshold:
            return label
    return "low"


def _unknown_diagnosis() -> Diagnosis:
    """Return a placeholder Diagnosis when no candidates were found."""
    return Diagnosis(
        name="Undetermined",
        icd10_code="",
        probability="low",
        probability_score=0.0,
        supporting_evidence=[],
        contradicting_evidence=[],
        reasoning="Insufficient evidence to determine a differential diagnosis.",
    )


def _default_investigations(diagnoses: list[Diagnosis]) -> list[str]:
    """Return a minimal investigation list based on primary diagnosis category."""
    if not diagnoses:
        return ["Full blood count", "Basic metabolic panel"]
    primary = diagnoses[0].name.lower()
    if any(k in primary for k in ("coronary", "heart", "cardiac")):
        return ["ECG", "Troponin", "Chest X-ray", "Full blood count"]
    if any(k in primary for k in ("pneumonia", "respiratory", "asthma", "copd")):
        return ["Chest X-ray", "Spirometry", "SpO2 monitoring", "Full blood count"]
    if any(k in primary for k in ("stroke", "migraine", "headache", "neurolog")):
        return ["CT head", "Blood glucose", "Full blood count", "Blood pressure monitoring"]
    return ["Full blood count", "Basic metabolic panel", "Urinalysis"]


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

def _print_sep(char: str = "─", width: int = 70) -> None:
    print(char * width)


if __name__ == "__main__":
    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from medai.src.voice.transcript_processor import TranscriptProcessor         # noqa: PLC0415
    from medai.src.clinical.entity_extractor import EntityExtractor              # noqa: PLC0415
    from medai.src.clinical.soap_generator import SOAPGenerator                  # noqa: PLC0415
    from medai.src.reasoning.red_flag_detector import RedFlagDetector            # noqa: PLC0415

    DATA_PATH = _REPO_ROOT / "data/raw/unified/en_mts.jsonl"
    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}")
        print("Run data/pipeline/setup.py first.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  DDxEngine — standalone smoke-test")
    print("=" * 70)

    records: list[dict] = []
    with open(DATA_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= 3:
                break

    tp = TranscriptProcessor()
    ee = EntityExtractor()
    sg = SOAPGenerator()
    rd = RedFlagDetector()
    ddx = DDxEngine()

    transcripts = tp.process_batch(records, show_progress=False)
    entities_list = ee.extract_batch(transcripts, show_progress=False)
    soaps = sg.generate_batch(transcripts, entities_list, show_progress=False)
    rfrs = rd.detect_batch(transcripts, entities_list, soaps, show_progress=False)

    print(f"\nGeneration mode: {'LLM (Gemini)' if not ddx.use_fallback else 'rule-based'}\n")

    _print_sep("─")
    print("analyse_batch() output:")
    _print_sep("─")
    ddx_results = ddx.analyse_batch(
        transcripts, entities_list, soaps, rfrs, show_progress=True
    )

    print()
    _print_sep("═")
    print("  Full results")
    _print_sep("═")
    for result in ddx_results:
        print()
        print(str(result))
        _print_sep("─")
    print()
