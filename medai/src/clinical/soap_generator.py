"""
soap_generator.py

Transforms a ProcessedTranscript and MedicalEntities into a structured
SOAP note using Google Gemini (gemini-1.5-flash).

Two generation modes:
  llm               — Gemini API call with structured prompt
  template_fallback — pure-stdlib deterministic fallback (no API key needed)

The fallback activates automatically when:
  - GOOGLE_API_KEY is not set
  - google-generativeai is not installed
  - Any Gemini API call fails after 3 retries

Standalone smoke-test::

    python src/clinical/soap_generator.py
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from medai.src.voice.transcript_processor import ProcessedTranscript
    from medai.src.clinical.entity_extractor import MedicalEntities

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional google-generativeai import
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

_RED_FLAG_TERMS: list[str] = [
    "chest pain",
    "shortness of breath",
    "difficulty breathing",
    "suicidal",
    "altered consciousness",
    "severe headache",
    "stroke",
    "seizure",
    "sepsis",
]

_SECTION_HEADERS = ("SUBJECTIVE", "OBJECTIVE", "ASSESSMENT", "PLAN")

_DEFAULT_PROMPT_TEMPLATE = """\
You are a clinical documentation assistant. Generate a structured SOAP note \
from the doctor-patient conversation and extracted entities provided. \
Be concise, clinically accurate, and use standard medical terminology.

EXTRACTED ENTITIES:
- Symptoms: {symptoms}
- Clinical Findings: {clinical_findings}
- Medications: {medications}
- Medical History: {medical_history}
- Vitals: {vitals}
- Negations (denied symptoms): {negations}
- Speciality hints: {speciality_hints}

CONVERSATION TRANSCRIPT:
{transcript}

INSTRUCTIONS:
Generate exactly four sections with these exact labels (no extra text before \
SUBJECTIVE or after PLAN):

SUBJECTIVE:
OBJECTIVE:
ASSESSMENT:
PLAN:

Each section should be 2–5 sentences. Use bullet points within sections where \
appropriate.\
"""

# Maximum words of transcript to include in prompt (cost/token control)
_MAX_TRANSCRIPT_WORDS = 2000

# Gemini generation parameters
_TEMPERATURE = 0.3
_MAX_OUTPUT_TOKENS = 1024
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2

# Delay between batch LLM calls to respect rate limits
_BATCH_INTER_CALL_DELAY = 1.0


# ---------------------------------------------------------------------------
# SOAPNote dataclass
# ---------------------------------------------------------------------------

@dataclass
class SOAPNote:
    """
    Structured SOAP note produced by SOAPGenerator.

    Fields
    ------
    conversation_id   : matches the source ProcessedTranscript id
    speciality        : inferred from speciality_hints or "General Medicine"
    subjective        : S — patient-reported symptoms and history
    objective         : O — vitals, exam findings, investigations
    assessment        : A — primary diagnosis / clinical impression
    plan              : P — investigations, medications, follow-up
    red_flag_mentions : red-flag terms found anywhere in the note
    generated_at      : ISO 8601 UTC timestamp of generation
    model_used        : Gemini model name or "template_fallback"
    prompt_tokens     : estimated token count sent to the API
    confidence        : inherited from MedicalEntities.confidence
    generation_method : "llm" or "template_fallback"
    """

    conversation_id: str
    speciality: str
    subjective: str
    objective: str
    assessment: str
    plan: str
    red_flag_mentions: List[str] = field(default_factory=list)
    generated_at: str = ""
    model_used: str = ""
    prompt_tokens: int = 0
    confidence: float = 0.0
    generation_method: str = "template_fallback"

    def __str__(self) -> str:
        flags = ", ".join(self.red_flag_mentions) or "none"
        return (
            f"SOAPNote(id={self.conversation_id!r}, "
            f"speciality={self.speciality!r}, "
            f"method={self.generation_method!r}, "
            f"model={self.model_used!r}, "
            f"confidence={self.confidence}, "
            f"red_flags=[{flags}])\n"
            f"\n[SUBJECTIVE]\n{self.subjective}\n"
            f"\n[OBJECTIVE]\n{self.objective}\n"
            f"\n[ASSESSMENT]\n{self.assessment}\n"
            f"\n[PLAN]\n{self.plan}\n"
        )


# ---------------------------------------------------------------------------
# SOAPGenerator
# ---------------------------------------------------------------------------

class SOAPGenerator:
    """
    Generates structured SOAP notes from a ProcessedTranscript and
    MedicalEntities using Google Gemini (gemini-1.5-flash).

    Falls back to a deterministic template when the API key is absent,
    the google-generativeai package is not installed, or API calls fail.

    Usage::

        gen = SOAPGenerator()                      # reads GOOGLE_API_KEY
        soap = gen.generate(transcript, entities)
        batch = gen.generate_batch(transcripts, entities_list)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
    ) -> None:
        """
        Parameters
        ----------
        api_key:
            Google AI API key.  Defaults to the ``GOOGLE_API_KEY`` environment
            variable.  If absent or empty the instance silently falls back to
            template generation.
        model:
            Gemini model identifier used for all LLM calls.
        """
        self.model_name = model
        self.use_fallback: bool = True
        self._client = None

        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not resolved_key:
            log.warning(
                "GOOGLE_API_KEY not set — SOAPGenerator will use template "
                "fallback.  Set the env-var to enable Gemini generation."
            )

        if not _GENAI_AVAILABLE:
            log.warning(
                "google-genai is not installed — using template fallback. "
                "Install with: pip install google-genai>=1.0.0"
            )
        elif resolved_key:
            try:
                self._client = _genai.Client(api_key=resolved_key)
                self.use_fallback = False
                log.info("SOAPGenerator initialised with model %r", model)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Failed to initialise Gemini client (%s) — using template fallback.",
                    exc,
                )

        self._prompt_template = self._load_prompt_template()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        transcript: ProcessedTranscript,
        entities: MedicalEntities,
    ) -> SOAPNote:
        """
        Generate a SOAP note for one conversation.

        Dispatches to the Gemini LLM or the template fallback.  On any LLM
        error the method transparently retries (up to _MAX_RETRIES) then
        falls back to the template — it never raises.

        Parameters
        ----------
        transcript:
            ProcessedTranscript from TranscriptProcessor.
        entities:
            MedicalEntities from EntityExtractor.

        Returns
        -------
        SOAPNote
        """
        speciality = self._infer_speciality(entities)
        generated_at = datetime.now(timezone.utc).isoformat()

        if not self.use_fallback:
            try:
                soap = self._generate_with_llm(transcript, entities)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "LLM generation failed for %r (%s) — using template fallback.",
                    transcript.id,
                    exc,
                )
                soap = self._generate_template_fallback(transcript, entities)
        else:
            soap = self._generate_template_fallback(transcript, entities)

        # Fill in fields that are always set here regardless of path
        soap.conversation_id = transcript.id
        soap.speciality = speciality
        soap.generated_at = generated_at
        soap.confidence = entities.confidence
        soap.red_flag_mentions = self._detect_red_flags(soap)

        return soap

    def generate_batch(
        self,
        transcripts: List[ProcessedTranscript],
        entities_list: List[MedicalEntities],
        show_progress: bool = True,
    ) -> List[SOAPNote]:
        """
        Generate SOAP notes for a list of transcripts, matching each
        transcript to its corresponding MedicalEntities by conversation_id.

        Transcripts with no matching entities are skipped and logged.
        Per-record failures are isolated and excluded from the result.
        A 1-second delay is inserted between LLM calls to respect rate limits.

        Parameters
        ----------
        transcripts:
            ProcessedTranscript objects in any order.
        entities_list:
            MedicalEntities objects; matched to transcripts by conversation_id.
        show_progress:
            Print progress every 10 records and a final summary.

        Returns
        -------
        list[SOAPNote]
        """
        # Index entities by conversation_id for O(1) lookup
        entities_index: dict[str, MedicalEntities] = {
            e.conversation_id: e for e in entities_list
        }

        results: list[SOAPNote] = []
        failures: list[tuple[str, str]] = []

        for i, transcript in enumerate(transcripts):
            t_id = getattr(transcript, "id", f"index:{i}")
            entities = entities_index.get(t_id)

            if entities is None:
                log.warning(
                    "No MedicalEntities found for transcript %r — skipping.", t_id
                )
                failures.append((t_id, "no matching entities"))
                continue

            try:
                soap = self.generate(transcript, entities)
                results.append(soap)
            except Exception as exc:  # noqa: BLE001
                log.error("generate_batch: failed for %r — %s", t_id, exc)
                failures.append((t_id, str(exc)))

            # Rate-limit guard between LLM calls
            if not self.use_fallback and i < len(transcripts) - 1:
                time.sleep(_BATCH_INTER_CALL_DELAY)

            if show_progress and (i + 1) % 10 == 0:
                print(
                    f"  … {i + 1}/{len(transcripts)} processed "
                    f"({len(failures)} failures so far)"
                )

        if show_progress:
            print(
                f"  Batch complete: {len(results)} succeeded, "
                f"{len(failures)} failed out of {len(transcripts)} records."
            )
            for fid, reason in failures[:5]:
                print(f"    {fid}: {reason}")
            if len(failures) > 5:
                print(f"    … and {len(failures) - 5} more.")

        return results

    def evaluate_against_reference(
        self, soap: SOAPNote, reference_note: str
    ) -> dict:
        """
        Compute simple text-overlap metrics between a generated SOAP note and
        a reference note (e.g. from MTS-Dialog).

        No external libraries — uses set operations on tokenised word sets.

        Metrics
        -------
        jaccard:
            ``|intersection| / |union|`` — overall vocabulary overlap.
        recall:
            ``|intersection| / |reference_words|`` — fraction of reference
            vocabulary present in the generated note.

        Parameters
        ----------
        soap:
            Generated SOAPNote whose four section texts are evaluated.
        reference_note:
            Ground-truth note text to compare against.

        Returns
        -------
        dict with keys "conversation_id", "jaccard", "recall"
        """
        generated_text = " ".join([
            soap.subjective, soap.objective, soap.assessment, soap.plan
        ])
        gen_words = _tokenise(generated_text)
        ref_words = _tokenise(reference_note)

        if not ref_words:
            return {"conversation_id": soap.conversation_id, "jaccard": 0.0, "recall": 0.0}

        intersection = gen_words & ref_words
        union = gen_words | ref_words

        jaccard = len(intersection) / len(union) if union else 0.0
        recall = len(intersection) / len(ref_words) if ref_words else 0.0

        return {
            "conversation_id": soap.conversation_id,
            "jaccard": round(jaccard, 4),
            "recall": round(recall, 4),
        }

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _generate_with_llm(
        self,
        transcript: ProcessedTranscript,
        entities: MedicalEntities,
    ) -> SOAPNote:
        """
        Build a prompt, call Gemini, and parse the response into a SOAPNote.

        Raises on all errors — callers must handle and fall back.
        """
        prompt = self._build_prompt(transcript, entities)
        token_estimate = len(prompt.split())

        raw_response = self._call_gemini(prompt)
        soap = self._parse_soap_response(raw_response, transcript.id)
        soap.model_used = self.model_name
        soap.prompt_tokens = token_estimate
        soap.generation_method = "llm"
        return soap

    def _build_prompt(
        self,
        transcript: ProcessedTranscript,
        entities: MedicalEntities,
    ) -> str:
        """
        Assemble the full LLM prompt from the prompt template, extracted
        entities, and transcript text.

        The transcript is truncated to _MAX_TRANSCRIPT_WORDS words to keep
        token counts predictable.

        Parameters
        ----------
        transcript:
            Source conversation.
        entities:
            Extracted clinical entities.

        Returns
        -------
        str — the complete prompt ready to send to Gemini.
        """
        # Truncate full_text to avoid exceeding context window
        transcript_words = transcript.full_text.split()
        if len(transcript_words) > _MAX_TRANSCRIPT_WORDS:
            truncated = " ".join(transcript_words[:_MAX_TRANSCRIPT_WORDS])
            truncated += "\n[… transcript truncated for length …]"
        else:
            truncated = transcript.full_text

        vitals_str = (
            ", ".join(f"{k}: {v}" for k, v in entities.vitals.items())
            if entities.vitals else "None recorded"
        )

        return self._prompt_template.format(
            symptoms=entities.symptoms or ["None identified"],
            clinical_findings=entities.clinical_findings or ["None identified"],
            medications=entities.medications or ["None identified"],
            medical_history=entities.medical_history or ["None identified"],
            vitals=vitals_str,
            negations=entities.negations or ["None"],
            speciality_hints=entities.speciality_hints or ["None"],
            transcript=truncated,
        )

    def _call_gemini(self, prompt: str) -> str:
        """
        Send *prompt* to the Gemini API and return the response text.

        Retries up to _MAX_RETRIES times with exponential back-off on failure.

        Parameters
        ----------
        prompt:
            Full prompt string to send.

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

    def _parse_soap_response(self, response: str, conversation_id: str) -> SOAPNote:
        """
        Extract the four SOAP sections from the raw LLM response text.

        Parsing strategy
        ----------------
        1. Search for each header (SUBJECTIVE: / OBJECTIVE: / ASSESSMENT: /
           PLAN:) case-insensitively.
        2. The text between one header and the next is that section's content.
        3. Any section not found in the response is filled with
           "Not documented".
        4. Leading/trailing whitespace is stripped from each section.

        Parameters
        ----------
        response:
            Raw string returned by _call_gemini().
        conversation_id:
            Used to populate the returned SOAPNote.

        Returns
        -------
        SOAPNote (generation_method and model_used are set by the caller)
        """
        sections: dict[str, str] = {}
        header_pat = re.compile(
            r"(?:^|\n)\s*(SUBJECTIVE|OBJECTIVE|ASSESSMENT|PLAN)\s*:\s*",
            re.IGNORECASE,
        )

        matches = list(header_pat.finditer(response))

        for i, m in enumerate(matches):
            header = m.group(1).upper()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(response)
            sections[header] = response[start:end].strip()

        return SOAPNote(
            conversation_id=conversation_id,
            speciality="",          # filled by generate()
            subjective=sections.get("SUBJECTIVE", "Not documented"),
            objective=sections.get("OBJECTIVE", "Not documented"),
            assessment=sections.get("ASSESSMENT", "Not documented"),
            plan=sections.get("PLAN", "Not documented"),
        )

    # ------------------------------------------------------------------
    # Template fallback path
    # ------------------------------------------------------------------

    def _generate_template_fallback(
        self,
        transcript: ProcessedTranscript,
        entities: MedicalEntities,
    ) -> SOAPNote:
        """
        Produce a deterministic SOAP note from the extracted entities and
        transcript text — no API calls, no external libraries.

        Used when the Gemini API is unavailable or all retries are exhausted.

        Subjective  — first 200 words of patient_text + symptom list
        Objective   — vitals dict + clinical_findings list
        Assessment  — speciality hints + boilerplate placeholder
        Plan        — static boilerplate placeholder
        """
        # ── Subjective ────────────────────────────────────────────────
        patient_words = transcript.patient_text.split()
        patient_excerpt = " ".join(patient_words[:200])
        if len(patient_words) > 200:
            patient_excerpt += " [...]"

        sym_str = (
            "; ".join(entities.symptoms)
            if entities.symptoms else "none identified"
        )
        subjective = (
            f"Patient reports: {patient_excerpt}\n"
            f"Identified symptoms: {sym_str}."
        )
        if entities.negations:
            subjective += f"\nDenied: {'; '.join(entities.negations)}."

        # ── Objective ────────────────────────────────────────────────
        if entities.vitals:
            vitals_str = "; ".join(f"{k} {v}" for k, v in entities.vitals.items())
        else:
            vitals_str = "No vital signs recorded in transcript."

        findings_str = (
            "; ".join(entities.clinical_findings)
            if entities.clinical_findings else "none documented"
        )
        objective = (
            f"Vitals: {vitals_str}.\n"
            f"Clinical findings: {findings_str}."
        )
        if entities.medications:
            objective += f"\nCurrent medications: {'; '.join(entities.medications)}."

        # ── Assessment ───────────────────────────────────────────────
        if entities.speciality_hints:
            hint_str = ", ".join(h.capitalize() for h in entities.speciality_hints)
            assessment = (
                f"Presentation is consistent with {hint_str} aetiology. "
                "Clinical assessment pending full review by attending physician."
            )
        else:
            assessment = (
                "Clinical assessment pending full review by attending physician. "
                "Differential diagnosis to be established following examination."
            )
        if entities.medical_history:
            assessment += (
                f"\nRelevant background: {'; '.join(entities.medical_history)}."
            )

        # ── Plan ─────────────────────────────────────────────────────
        plan = "Treatment plan to be determined by attending physician."

        return SOAPNote(
            conversation_id=transcript.id,
            speciality="",        # filled by generate()
            subjective=subjective,
            objective=objective,
            assessment=assessment,
            plan=plan,
            model_used="template_fallback",
            generation_method="template_fallback",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_speciality(entities: MedicalEntities) -> str:
        """
        Infer a speciality label from entity speciality_hints.

        Returns the most frequently appearing hint (capitalised), or
        "General Medicine" when no hints are present.
        """
        hints = entities.speciality_hints
        if not hints:
            return "General Medicine"
        counts: dict[str, int] = {}
        for h in hints:
            counts[h] = counts.get(h, 0) + 1
        best = max(counts, key=counts.__getitem__)
        return best.capitalize()

    @staticmethod
    def _detect_red_flags(soap: SOAPNote) -> list[str]:
        """
        Scan all four SOAP section texts for known red-flag terms.

        Matching is case-insensitive and word-boundary anchored.

        Returns
        -------
        list[str] — deduplicated red-flag terms found in the note.
        """
        combined = " ".join([
            soap.subjective, soap.objective, soap.assessment, soap.plan
        ]).lower()

        found: list[str] = []
        for term in _RED_FLAG_TERMS:
            if re.search(r"\b" + re.escape(term) + r"\b", combined):
                found.append(term)
        return found

    def _load_prompt_template(self) -> str:
        """
        Load the SOAP prompt template from configs/prompts.yaml.

        Falls back to _DEFAULT_PROMPT_TEMPLATE when:
          - PyYAML is not installed
          - The YAML file is missing
          - The expected keys are absent in the file

        Returns
        -------
        str — a format-string with {symptoms}, {transcript}, etc. placeholders.
        """
        yaml_path = Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml"

        if not yaml_path.exists():
            log.debug("prompts.yaml not found at %s — using default template.", yaml_path)
            return _DEFAULT_PROMPT_TEMPLATE

        try:
            import yaml  # type: ignore  # noqa: PLC0415
            with open(yaml_path, encoding="utf-8") as fh:
                config = yaml.safe_load(fh)

            section = config.get("soap_generation", {})
            system = (section.get("system") or "").strip()
            user_tmpl = (section.get("user_template") or "").strip()

            if system and user_tmpl and "{transcript}" in user_tmpl:
                return system + "\n\n" + user_tmpl
            log.debug(
                "prompts.yaml soap_generation section is incomplete — "
                "using default template."
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not load prompts.yaml (%s) — using default template.", exc)

        return _DEFAULT_PROMPT_TEMPLATE


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set[str]:
    """
    Lowercase and split *text* into a word set, stripping punctuation.

    Stop words are kept so that recall reflects full coverage rather than
    content-word coverage alone.
    """
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return {w for w in cleaned.split() if w}


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

def _print_sep(char: str = "─", width: int = 70) -> None:
    print(char * width)


if __name__ == "__main__":
    import json  # noqa: PLC0415
    import sys   # noqa: PLC0415

    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from medai.src.voice.transcript_processor import TranscriptProcessor   # noqa: PLC0415
    from medai.src.clinical.entity_extractor import EntityExtractor        # noqa: PLC0415

    DATA_PATH = _REPO_ROOT / "data/raw/unified/en_mts.jsonl"
    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}")
        print("Run data/pipeline/setup.py first.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  SOAPGenerator — standalone smoke-test")
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
    transcripts = tp.process_batch(records, show_progress=False)

    ee = EntityExtractor()
    entities_list = ee.extract_batch(transcripts, show_progress=False)

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    gen = SOAPGenerator(api_key=api_key or None)

    mode = "LLM (Gemini)" if not gen.use_fallback else "template fallback"
    print(f"\nGeneration mode: {mode}\n")

    _print_sep("─")
    print("generate_batch() output:")
    _print_sep("─")
    soaps = gen.generate_batch(transcripts, entities_list, show_progress=True)

    print()
    _print_sep("═")
    print("  Full SOAPNote objects")
    _print_sep("═")
    for soap in soaps:
        print()
        print(str(soap))
        _print_sep("─")

    print()
    _print_sep("═")
    print("  evaluate_against_reference() — Jaccard + Recall scores")
    _print_sep("═")
    for soap, transcript in zip(soaps, transcripts):
        if transcript.reference_note:
            scores = gen.evaluate_against_reference(soap, transcript.reference_note)
            print(
                f"  {soap.conversation_id:<20} "
                f"jaccard={scores['jaccard']:.4f}  "
                f"recall={scores['recall']:.4f}"
            )
        else:
            print(f"  {soap.conversation_id:<20} no reference note")
    print()
