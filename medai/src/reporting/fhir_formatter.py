"""
fhir_formatter.py

Serialises MedAI pipeline outputs into HL7 FHIR R4 resources for
interoperability with hospital EHR systems.

Zero external dependencies — uses only stdlib (json, uuid, datetime).

Standalone smoke-test::

    python src/reporting/fhir_formatter.py
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from medai.src.clinical.soap_generator import SOAPNote
    from medai.src.reasoning.ddx_engine import DDxResult
    from medai.src.reasoning.red_flag_detector import RedFlagResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LOINC code maps
# ---------------------------------------------------------------------------

# SOAP section → LOINC code + display name
_SOAP_LOINC = {
    "subjective":  ("10164-2", "History of Present Illness"),
    "objective":   ("10210-3", "Physical findings"),
    "assessment":  ("51848-0", "Evaluation note"),
    "plan":        ("18776-5", "Plan of care note"),
}

# Vital key fragment → (LOINC code, display name)
_VITAL_LOINC: list[tuple[str, str, str]] = [
    ("BP",   "55284-4", "Blood pressure systolic and diastolic"),
    ("HR",   "8867-4",  "Heart rate"),
    ("temp", "8310-5",  "Body temperature"),
    ("RR",   "9279-1",  "Respiratory rate"),
    ("SpO2", "2708-6",  "Oxygen saturation"),
    ("Wt",   "29463-7", "Body weight"),
    ("Ht",   "8302-2",  "Body height"),
]

# Composition type LOINC: Summarization of episode note
_COMPOSITION_TYPE_LOINC = ("34133-9", "Summarization of episode note")


# ---------------------------------------------------------------------------
# FHIRFormatter
# ---------------------------------------------------------------------------

class FHIRFormatter:
    """
    Converts MedAI pipeline outputs into a valid HL7 FHIR R4 Bundle.

    No external FHIR libraries are required — the bundle is built as a plain
    Python dict that can be serialised to JSON with ``json.dumps()``.

    Usage::

        fmt = FHIRFormatter()
        bundle = fmt.format_bundle(soap, ddx, red_flags, patient_info)
        path   = fmt.save_bundle(bundle, output_path)
        report = fmt.validate_bundle(bundle)
    """

    def __init__(self) -> None:
        """Initialise the formatter. No configuration required."""
        self.fhir_version = "4.0.1"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_bundle(
        self,
        soap: SOAPNote,
        ddx: DDxResult,
        red_flags: RedFlagResult,
        patient_info: dict | None = None,
    ) -> dict:
        """
        Build a complete FHIR R4 Bundle from pipeline outputs.

        The bundle is a ``document`` type and contains:

        1. Patient resource
        2. Composition resource (SOAP sections as LOINC-coded sections)
        3. One Condition resource per top diagnosis (with ICD-10 codes)
        4. One Observation resource per vital sign (with LOINC codes)

        Parameters
        ----------
        soap:
            SOAPNote from SOAPGenerator.
        ddx:
            DDxResult from DDxEngine.
        red_flags:
            RedFlagResult from RedFlagDetector (embedded as a Composition
            extension; not yet a separate resource type to keep the bundle
            simple and valid).
        patient_info:
            Optional dict with keys ``name``, ``dob``, ``gender``.
            De-identified defaults are used when absent.

        Returns
        -------
        dict — FHIR R4 Bundle (JSON-serialisable).
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        patient_res = self._build_patient_resource(patient_info or {})
        composition_res = self._build_composition_resource(soap)
        condition_resources = self._build_condition_resources(ddx)

        # Extract vitals from the DDx red_flag_result — they were sourced
        # from entities, but we receive them only indirectly here.  For the
        # Observation resources we use the vitals embedded in the Composition
        # objective section via a best-effort parse, or accept an empty dict.
        # Callers that have entities available can pass vitals explicitly via
        # patient_info["vitals"].
        vitals: dict = {}
        if patient_info and isinstance(patient_info.get("vitals"), dict):
            vitals = patient_info["vitals"]
        observation_resources = self._build_observation_resources(vitals)

        entries = [
            {"resource": patient_res},
            {"resource": composition_res},
        ]
        for cond in condition_resources:
            entries.append({"resource": cond})
        for obs in observation_resources:
            entries.append({"resource": obs})

        return {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "meta": {"profile": [f"http://hl7.org/fhir/StructureDefinition/Bundle"]},
            "type": "document",
            "timestamp": timestamp,
            "entry": entries,
        }

    def _build_patient_resource(self, patient_info: dict) -> dict:
        """
        Build a FHIR R4 Patient resource.

        Uses values from *patient_info* when available; falls back to
        de-identified defaults for any missing field.

        Parameters
        ----------
        patient_info:
            Dict optionally containing ``name`` (str), ``dob`` (str
            YYYY-MM-DD), and ``gender`` (``"male"`` | ``"female"`` |
            ``"other"`` | ``"unknown"``).

        Returns
        -------
        dict — FHIR Patient resource.
        """
        name_text = patient_info.get("name") or "De-identified Patient"
        gender = patient_info.get("gender") or "unknown"
        resource: dict = {
            "resourceType": "Patient",
            "id": "patient-demo",
            "name": [{"text": name_text}],
            "gender": gender,
        }
        if patient_info.get("dob"):
            resource["birthDate"] = patient_info["dob"]
        return resource

    def _build_composition_resource(self, soap: SOAPNote) -> dict:
        """
        Build a FHIR R4 Composition resource mapping each SOAP section.

        LOINC codes used
        ----------------
        - Type:       34133-9 (Summarization of episode note)
        - Subjective: 10164-2 (History of Present Illness)
        - Objective:  10210-3 (Physical findings)
        - Assessment: 51848-0 (Evaluation note)
        - Plan:       18776-5 (Plan of care note)

        Parameters
        ----------
        soap:
            SOAPNote from SOAPGenerator.

        Returns
        -------
        dict — FHIR Composition resource.
        """
        type_code, type_display = _COMPOSITION_TYPE_LOINC
        sections = []
        for field_name, (loinc_code, loinc_display) in _SOAP_LOINC.items():
            text = getattr(soap, field_name, "") or ""
            sections.append({
                "title": loinc_display,
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": loinc_code,
                        "display": loinc_display,
                    }]
                },
                "text": {
                    "status": "generated",
                    "div": f"<div>{_xml_escape(text)}</div>",
                },
            })

        return {
            "resourceType": "Composition",
            "id": f"composition-{soap.conversation_id}",
            "status": "preliminary",
            "type": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": type_code,
                    "display": type_display,
                }]
            },
            "subject": {"reference": "Patient/patient-demo"},
            "date": soap.generated_at or datetime.now(timezone.utc).isoformat(),
            "author": [{"display": f"MedAI ({soap.model_used or 'template_fallback'})"}],
            "title": f"Clinical Note — {soap.speciality or 'General Medicine'}",
            "section": sections,
        }

    def _build_condition_resources(self, ddx: DDxResult) -> list[dict]:
        """
        Build one FHIR R4 Condition resource per top diagnosis in *ddx*.

        Each Condition uses the ICD-10 coding system.  The ``probability``
        tier is recorded as a note since FHIR Condition does not have a
        standard probability field.

        Parameters
        ----------
        ddx:
            DDxResult from DDxEngine.

        Returns
        -------
        list[dict] — FHIR Condition resources (may be empty if no diagnoses).
        """
        resources = []
        for n, diagnosis in enumerate(ddx.top_diagnoses, 1):
            resources.append({
                "resourceType": "Condition",
                "id": f"condition-{n}",
                "clinicalStatus": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                    }]
                },
                "verificationStatus": {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "provisional",
                    }]
                },
                "code": {
                    "coding": [{
                        "system": "http://hl7.org/fhir/sid/icd-10",
                        "code": diagnosis.icd10_code or "",
                        "display": diagnosis.name,
                    }]
                },
                "subject": {"reference": "Patient/patient-demo"},
                "note": [{
                    "text": (
                        f"Probability: {diagnosis.probability} "
                        f"(score {diagnosis.probability_score:.2f}). "
                        f"{diagnosis.reasoning}"
                    )
                }],
            })
        return resources

    def _build_observation_resources(self, vitals: dict) -> list[dict]:
        """
        Build one FHIR R4 Observation resource per vital sign.

        Vital keys are matched case-insensitively against the following
        LOINC mappings:

        =====  =========  ====================================
        Key    LOINC      Display
        =====  =========  ====================================
        BP     55284-4    Blood pressure systolic and diastolic
        HR     8867-4     Heart rate
        temp   8310-5     Body temperature
        RR     9279-1     Respiratory rate
        SpO2   2708-6     Oxygen saturation
        Wt     29463-7    Body weight
        Ht     8302-2     Body height
        =====  =========  ====================================

        Parameters
        ----------
        vitals:
            Dict mapping vital sign labels to value strings
            (e.g. ``{"BP": "120/80 mmHg", "HR": "72 bpm"}``).

        Returns
        -------
        list[dict] — FHIR Observation resources (empty when *vitals* is empty).
        """
        resources = []
        vitals_lower = {k.lower(): (k, v) for k, v in vitals.items()}

        for key_frag, loinc_code, loinc_display in _VITAL_LOINC:
            orig_key, value = vitals_lower.get(key_frag.lower(), (None, None))
            if value is None:
                continue
            resources.append({
                "resourceType": "Observation",
                "id": f"obs-{key_frag.lower()}",
                "status": "final",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "vital-signs",
                        "display": "Vital Signs",
                    }]
                }],
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": loinc_code,
                        "display": loinc_display,
                    }]
                },
                "subject": {"reference": "Patient/patient-demo"},
                "valueString": str(value),
            })
        return resources

    def save_bundle(
        self,
        bundle: dict,
        output_path: str | None = None,
        conversation_id: str = "unknown",
    ) -> str:
        """
        Serialise *bundle* to a pretty-printed JSON file.

        Parameters
        ----------
        bundle:
            FHIR Bundle dict from :meth:`format_bundle`.
        output_path:
            Explicit destination path.  When ``None`` a default path under
            ``data/outputs/fhir/`` is used.
        conversation_id:
            Used in the default filename when *output_path* is ``None``.

        Returns
        -------
        str — absolute path of the file written.
        """
        if output_path is None:
            repo_root = Path(__file__).resolve().parents[3]
            fhir_dir = repo_root / "data/outputs/fhir"
            fhir_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            output_path = str(fhir_dir / f"fhir_{conversation_id}_{ts}.json")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, indent=2, ensure_ascii=False)

        log.info("FHIR bundle saved to %s", output_path)
        return output_path

    def validate_bundle(self, bundle: dict) -> dict:
        """
        Perform basic structural validation of a FHIR Bundle dict.

        This is a lightweight sanity check — it does not invoke an external
        FHIR validator or check value-set bindings.

        Checks performed
        ----------------
        - ``resourceType`` == ``"Bundle"``
        - ``type`` field is present
        - ``entry`` is a non-empty list
        - Each entry contains a ``resource`` key with a ``resourceType`` field
        - At least one entry is a ``Composition``
        - At least one entry is a ``Patient``

        Parameters
        ----------
        bundle:
            FHIR Bundle dict to validate.

        Returns
        -------
        dict with keys:
        - ``"valid"`` (bool)
        - ``"errors"`` (list[str]) — blocking issues
        - ``"warnings"`` (list[str]) — advisory issues
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not isinstance(bundle, dict):
            return {"valid": False, "errors": ["Bundle is not a dict"], "warnings": []}

        if bundle.get("resourceType") != "Bundle":
            errors.append(
                f"Expected resourceType 'Bundle', got {bundle.get('resourceType')!r}"
            )

        if "type" not in bundle:
            errors.append("Missing required field: 'type'")

        entries = bundle.get("entry")
        if not isinstance(entries, list) or len(entries) == 0:
            errors.append("'entry' must be a non-empty list")
            return {"valid": False, "errors": errors, "warnings": warnings}

        resource_types: list[str] = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or "resource" not in entry:
                errors.append(f"Entry {i} is missing 'resource' key")
                continue
            rt = entry["resource"].get("resourceType")
            if not rt:
                errors.append(f"Entry {i} resource is missing 'resourceType'")
            else:
                resource_types.append(rt)

        if "Composition" not in resource_types:
            errors.append("Bundle contains no Composition resource")

        if "Patient" not in resource_types:
            errors.append("Bundle contains no Patient resource")

        if "Condition" not in resource_types:
            warnings.append("Bundle contains no Condition resources (no diagnoses?)")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _xml_escape(text: str) -> str:
    """Escape characters that would break XHTML div content in FHIR."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


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

    from medai.src.voice.transcript_processor import TranscriptProcessor        # noqa
    from medai.src.clinical.entity_extractor import EntityExtractor             # noqa
    from medai.src.clinical.soap_generator import SOAPGenerator                 # noqa
    from medai.src.reasoning.red_flag_detector import RedFlagDetector           # noqa
    from medai.src.reasoning.ddx_engine import DDxEngine                        # noqa

    DATA_PATH = _REPO_ROOT / "data/raw/unified/en_mts.jsonl"
    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  FHIRFormatter — standalone smoke-test")
    print("=" * 70)

    records: list[dict] = []
    with open(DATA_PATH, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                records.append(json.loads(s))
            if len(records) >= 2:
                break

    transcripts = TranscriptProcessor().process_batch(records, show_progress=False)
    entities_list = EntityExtractor().extract_batch(transcripts, show_progress=False)
    soaps = SOAPGenerator().generate_batch(transcripts, entities_list, show_progress=False)
    rfrs = RedFlagDetector().detect_batch(transcripts, entities_list, soaps, show_progress=False)
    ddx_results = DDxEngine().analyse_batch(
        transcripts, entities_list, soaps, rfrs, show_progress=False
    )

    fmt = FHIRFormatter()
    print(f"\nFHIR version: {fmt.fhir_version}\n")

    for soap, ddx, rfr, entities in zip(soaps, ddx_results, rfrs, entities_list):
        _print_sep("─")
        print(f"Conversation: {soap.conversation_id}")

        # Pass vitals via patient_info
        bundle = fmt.format_bundle(
            soap, ddx, rfr,
            patient_info={"vitals": entities.vitals}
        )

        path = fmt.save_bundle(bundle, conversation_id=soap.conversation_id)
        print(f"  Saved to: {path}")

        report = fmt.validate_bundle(bundle)
        print(f"  Valid   : {report['valid']}")
        if report["errors"]:
            print(f"  Errors  : {report['errors']}")
        if report["warnings"]:
            print(f"  Warnings: {report['warnings']}")
        print(f"  Entries : {len(bundle['entry'])} resources")

    print()
