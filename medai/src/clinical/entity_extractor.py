"""
entity_extractor.py

Extracts structured clinical entities from a ProcessedTranscript and returns
a MedicalEntities object ready for SOAP note generation.

Two extraction modes:
  scispacy   — uses en_core_sci_md NER when the library is installed
  rule_based — pure-stdlib regex + keyword fallback (always available)

Vitals and negations always use dedicated regex passes regardless of mode,
as regex outperforms NER for those narrow, structured patterns.

Standalone smoke-test::

    python src/clinical/entity_extractor.py
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported for type-checker — not needed at runtime because
    # from __future__ import annotations makes all hints lazy strings.
    from medai.src.voice.transcript_processor import ProcessedTranscript, Turn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional scispaCy import
# ---------------------------------------------------------------------------

_SCISPACY_AVAILABLE = False
_spacy_mod = None

try:
    import spacy as _spacy_mod  # type: ignore
    _SCISPACY_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MedicalEntities:
    """
    Structured clinical entities extracted from one doctor-patient conversation.

    Fields
    ------
    conversation_id  : matches the source ProcessedTranscript id
    symptoms         : conditions / complaints reported by the patient
    clinical_findings: objective findings noted by the doctor
    medications      : drugs mentioned by either speaker
    medical_history  : past diagnoses, surgeries, or treatments
    vitals           : vital-sign key → value string (always regex-extracted)
    allergies        : substances the patient is allergic to
    speciality_hints : terms that suggest a medical speciality
    negations        : symptoms explicitly denied by the patient
    extraction_method: "scispacy" | "rule_based" | "failed"
    confidence       : 0.0–1.0 estimate of extraction quality
    raw_entities     : serialisable entity dicts for debugging / re-processing
    """

    conversation_id: str
    symptoms: List[str]
    clinical_findings: List[str]
    medications: List[str]
    medical_history: List[str]
    vitals: Dict[str, str]
    allergies: List[str]
    speciality_hints: List[str]
    negations: List[str]
    extraction_method: str
    confidence: float
    raw_entities: List[Dict] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"MedicalEntities(id={self.conversation_id!r}, "
            f"method={self.extraction_method!r}, confidence={self.confidence})",
            f"  symptoms        : {self.symptoms}",
            f"  clinical_findings: {self.clinical_findings}",
            f"  medications     : {self.medications}",
            f"  medical_history : {self.medical_history}",
            f"  vitals          : {self.vitals}",
            f"  allergies       : {self.allergies}",
            f"  speciality_hints: {self.speciality_hints}",
            f"  negations       : {self.negations}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rule-based vocabulary (module-level compiled objects for performance)
# ---------------------------------------------------------------------------

# ── Symptoms ────────────────────────────────────────────────────────────────

# Multi-word keywords listed first so the alternation tries them before
# their shorter sub-strings (the list is sorted by length descending).
_SYMPTOM_KEYWORDS: list[str] = sorted([
    # multi-word
    "shortness of breath", "difficulty breathing", "difficulty swallowing",
    "chest pain", "chest tightness", "chest pressure",
    "abdominal pain", "stomach pain", "belly pain",
    "back pain", "neck pain", "joint pain", "muscle pain",
    "loss of appetite", "loss of consciousness",
    "blurred vision", "double vision",
    "blood in urine", "painful urination", "urinary frequency",
    "night sweats", "weight loss", "rapid heartbeat",
    "ringing in ears",
    # single-word
    "pain", "ache", "aching", "sore", "soreness", "tender", "tenderness",
    "discomfort", "hurt", "hurting",
    "fever", "febrile", "pyrexia",
    "chills", "rigors",
    "nausea", "nauseous",
    "vomiting", "vomits",
    "diarrhea", "diarrhoea",
    "constipation", "bloating", "flatulence",
    "heartburn", "indigestion",
    "headache", "migraine",
    "dizziness", "dizzy", "vertigo", "lightheadedness",
    "syncope", "faintness",
    "fatigue", "tired", "tiredness", "exhaustion", "lethargy",
    "malaise", "weakness",
    "numbness", "tingling", "paraesthesia",
    "cough", "coughing",
    "wheezing", "wheeze", "stridor",
    "dyspnea", "breathlessness",
    "palpitations",
    "swelling", "swollen", "edema", "oedema",
    "rash", "itching", "pruritus", "hives", "urticaria", "jaundice",
    "anxiety", "depression", "insomnia",
    "confusion", "seizure", "tremor",
    "bleeding", "haemorrhage", "hemorrhage", "bruising", "pallor",
], key=len, reverse=True)

_SYMPTOM_RE = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in _SYMPTOM_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# ── Medications ─────────────────────────────────────────────────────────────

# Words ending in common drug-class suffixes
_MED_SUFFIX_RE = re.compile(
    r"\b[a-z]{3,}(?:"
    r"cillin|mab|statin|pril|olol|sartan|prazole|mycin|cycline"
    r"|floxacin|oxacin|azole|kinib|umab|zumab|tide|lukast|dipine|vir|nib"
    r")\b",
    re.IGNORECASE,
)

# High-frequency generic drug names not covered by suffix patterns
_COMMON_DRUGS_RE = re.compile(
    r"\b(?:aspirin|ibuprofen|paracetamol|acetaminophen|warfarin|heparin"
    r"|metformin|insulin|prednisone|prednisolone|methotrexate"
    r"|hydroxychloroquine|albuterol|salbutamol|morphine|codeine|tramadol"
    r"|gabapentin|pregabalin|sertraline|fluoxetine|citalopram|amitriptyline"
    r"|diazepam|lorazepam|omeprazole|ranitidine|furosemide"
    r"|hydrochlorothiazide|digoxin|amiodarone|lithium|haloperidol"
    r"|levothyroxine|thyroxine|bisoprolol|ramipril|amlodipine|simvastatin"
    r"|atorvastatin|clopidogrel|rivaroxaban|apixaban|doxycycline"
    r"|azithromycin|amoxicillin|trimethoprim|nitrofurantoin)\b",
    re.IGNORECASE,
)

# Context verbs that introduce a drug name
_MED_CONTEXT_RE = re.compile(
    r"\b(?:prescribed|taking|takes|took|started|stopped|on|given|using|uses"
    r"|administered)\s+([a-z][a-z\s\-]{1,30}?)"
    r"(?=\s+\d|\s+(?:mg|mcg|g\b|units?|ml|tablet|capsule)|[,\.\;\?\n]|$)",
    re.IGNORECASE,
)

# Dosage pattern: "metformin 500 mg" → captures "metformin"
_DOSAGE_RE = re.compile(
    r"\b([a-z][a-z\-]{3,25})\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g\b|units?|ml|IU)\b",
    re.IGNORECASE,
)

# ── Medical history ──────────────────────────────────────────────────────────

_HISTORY_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\bhistory\s+of\s+([a-z][a-z\s\-]{2,50}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdiagnosed\s+with\s+([a-z][a-z\s\-]{2,50}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpreviously\s+(?:had|diagnosed|treated)\s+(?:with\s+)?"
        r"([a-z][a-z\s\-]{2,50}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsuffered?\s+from\s+([a-z][a-z\s\-]{2,50}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bknown\s+(?:case\s+of\s+|to\s+have\s+)?"
        r"([a-z][a-z\s\-]{2,50}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btreated\s+for\s+([a-z][a-z\s\-]{2,50}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bunderwent\s+([a-z][a-z\s\-]{2,50}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bsurgery\s+for\s+([a-z][a-z\s\-]{2,50}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
]

# ── Allergies ────────────────────────────────────────────────────────────────

_ALLERGY_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\ballerg(?:ic\s+to|y\s+to)\s+([a-z][a-z\s\-]{1,40}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(r"\b([a-z][a-z\s\-]{1,30}?)\s+allergy\b", re.IGNORECASE),
    re.compile(
        r"\breaction\s+to\s+([a-z][a-z\s\-]{1,40}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcannot\s+take\s+([a-z][a-z\s\-]{1,30}?)\s+(?:due\s+to|because)",
        re.IGNORECASE,
    ),
]

# ── Negation patterns (PATIENT turns only) ───────────────────────────────────

# Trailing words that follow a negated term but are not part of it
# e.g. "No fever at all" → strip " at all" to get "fever"
_NEGATION_TRAILING_NOISE = re.compile(
    r"\s+(?:at|so|quite|very|a|an|the|all|any|else|too|either|really|just|well"
    r"|now|much|more|further|right|yet|ever|perhaps|maybe)\s*$",
    re.IGNORECASE,
)

_NEGATION_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\bno\s+(?:known\s+|history\s+of\s+|signs?\s+of\s+)?"
        r"([a-z][a-z\s\-]{2,40}?)(?=[,\.\;\?\)\n]|\s+(?:and|or|but)\b|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdon't\s+(?:really\s+)?have\s+(?:any\s+)?"
        r"([a-z][a-z\s\-]{2,35}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdidn't\s+have\s+(?:any\s+)?"
        r"([a-z][a-z\s\-]{2,35}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bhaven't\s+(?:been\s+having\s+|had\s+(?:any\s+)?)?"
        r"([a-z][a-z\s\-]{2,35}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdenies?\s+(?:any\s+)?"
        r"([a-z][a-z\s\-]{2,35}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwithout\s+(?:any\s+)?"
        r"([a-z][a-z\s\-]{2,35}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bnot\s+(?:been\s+)?(?:experiencing|having|feeling|suffering)\s+"
        r"(?:any\s+)?([a-z][a-z\s\-]{2,35}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bnone\s+of\s+(?:the\s+)?"
        r"([a-z][a-z\s\-]{2,35}?)(?=[,\.\;\?\n]|$)",
        re.IGNORECASE,
    ),
]

# ── Speciality hint terms ─────────────────────────────────────────────────────

# Maps speciality name → list of trigger terms (searched case-insensitively).
_SPECIALITY_TERMS: dict[str, list[str]] = {
    "cardiology": [
        "heart", "cardiac", "ECG", "EKG", "coronary", "angina", "arrhythmia",
        "palpitations", "myocardial", "atrial", "ventricular", "aorta",
        "stent", "catheter", "echocardiogram",
    ],
    "pulmonology": [
        "lungs", "pulmonary", "bronchial", "COPD", "asthma", "spirometry",
        "pleural", "pneumonia", "bronchitis", "inhaler", "nebulizer",
    ],
    "neurology": [
        "neurological", "seizure", "stroke", "TIA", "migraine", "CNS",
        "epilepsy", "neuropathy", "dementia", "Parkinson", "multiple sclerosis",
    ],
    "gastroenterology": [
        "GI", "gastro", "bowel", "intestinal", "colon", "hepatic", "liver",
        "pancreas", "endoscopy", "colonoscopy", "IBD", "Crohn", "bilirubin",
    ],
    "endocrinology": [
        "diabetes", "thyroid", "insulin", "glucose", "HbA1c", "A1c",
        "hypothyroid", "hyperthyroid", "adrenal", "cortisol",
    ],
    "orthopedics": [
        "fracture", "bone", "orthopedic", "spine", "tendon", "ligament",
        "osteoporosis", "arthritis", "cartilage",
    ],
    "psychiatry": [
        "mental health", "psychiatric", "bipolar", "schizophrenia", "PTSD",
        "OCD", "antidepressant", "hallucination",
    ],
    "nephrology": [
        "kidney", "renal", "creatinine", "dialysis", "GFR", "proteinuria",
    ],
    "oncology": [
        "cancer", "tumor", "tumour", "chemotherapy", "radiation", "malignant",
        "metastasis", "biopsy", "lymphoma", "leukemia", "carcinoma",
    ],
    "dermatology": [
        "eczema", "psoriasis", "dermatitis", "melanoma", "dermatology",
    ],
}

# ── Vital sign patterns ──────────────────────────────────────────────────────

# Blood pressure (requires a label OR an explicit mmHg unit)
_VITALS_BP_LABELED = re.compile(
    r"\b(?:BP|blood\s*pressure|b\.p\.)\s*(?:was|is|of|:)?\s*"
    r"(\d{2,3}\s*/\s*\d{2,3})\s*(?:mm\s*Hg|mmHg)?",
    re.IGNORECASE,
)
_VITALS_BP_UNIT = re.compile(
    r"\b(\d{2,3}\s*/\s*\d{2,3})\s*(?:mm\s*Hg|mmHg)\b",
    re.IGNORECASE,
)

# Temperature (with °C/°F symbol, or with label)
_VITALS_TEMP_UNIT = re.compile(
    r"(\d{2,3}(?:\.\d{1,2})?)\s*°?\s*([CF])\b",
    re.IGNORECASE,
)
_VITALS_TEMP_LABELED = re.compile(
    r"\b(?:temp(?:erature)?)\s*(?:was|is|of|:)?\s*"
    r"(\d{2,3}(?:\.\d{1,2})?)\s*(?:°?\s*[CF])?",
    re.IGNORECASE,
)

# Heart rate (label or bpm unit)
_VITALS_HR_LABELED = re.compile(
    r"\b(?:HR|heart\s*rate|pulse)\s*(?:was|is|of|:)?\s*"
    r"(\d{2,3})\s*(?:bpm|b\.p\.m\.)?",
    re.IGNORECASE,
)
_VITALS_HR_BPM = re.compile(
    r"\b(\d{2,3})\s*(?:bpm|beats?\s*/?\s*min)\b",
    re.IGNORECASE,
)

# Respiratory rate
_VITALS_RR = re.compile(
    r"\b(?:RR|resp(?:iratory)?\s*rate|respirations?)\s*(?:was|is|of|:)?\s*"
    r"(\d{1,2})\s*(?:breaths?(?:\s*/\s*min)?)?",
    re.IGNORECASE,
)
_VITALS_RR_UNIT = re.compile(
    r"\b(\d{1,2})\s*(?:breaths?)\s*/?\s*min\b",
    re.IGNORECASE,
)

# SpO2 / oxygen saturation
_VITALS_SPO2 = re.compile(
    r"\b(?:SpO2|spo2|O2\s*sat(?:uration)?|oxygen\s*sat(?:uration)?|sats?)\s*"
    r"(?:was|were|is|are|of|:)?\s*(\d{2,3})\s*%?",
    re.IGNORECASE,
)
_VITALS_SPO2_SATURATING = re.compile(
    r"\bsaturating\s+at\s+(\d{2,3})\s*%?",
    re.IGNORECASE,
)

# Weight
_VITALS_WEIGHT = re.compile(
    r"\b(?:weight|weighs?|wt\.?)\s*(?:was|is|of|:)?\s*"
    r"(\d{2,3}(?:\.\d{1,2})?)\s*(kg|lbs?|pounds?|kilograms?)",
    re.IGNORECASE,
)

# Height
_VITALS_HEIGHT = re.compile(
    r"\b(?:height|ht\.?)\s*(?:was|is|of|:)?\s*"
    r"(\d{1,3}(?:\.\d{1,2})?)\s*(cm|m\b|ft|feet|inches?)",
    re.IGNORECASE,
)

# Confidence weights per entity category (must sum to 1.0)
_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "symptoms": 0.35,
    "medications": 0.20,
    "medical_history": 0.20,
    "clinical_findings": 0.15,
    "allergies": 0.05,
    "speciality_hints": 0.05,
}

# scispaCy entity label → our category (patient context)
_SCI_LABEL_PATIENT: dict[str, str] = {
    "DISEASE": "symptoms",
    "ENTITY": "symptoms",     # en_core_sci_md generic label
    "CHEMICAL": "medications",
    "CHEBI": "medications",
    "GGP": "clinical_findings",
    "GENE_OR_GENE_PRODUCT": "clinical_findings",
}

# scispaCy entity label → our category (doctor context)
_SCI_LABEL_DOCTOR: dict[str, str] = {
    "DISEASE": "clinical_findings",
    "ENTITY": "clinical_findings",
    "CHEMICAL": "medications",
    "CHEBI": "medications",
    "GGP": "clinical_findings",
    "GENE_OR_GENE_PRODUCT": "clinical_findings",
}


# ---------------------------------------------------------------------------
# EntityExtractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    """
    Extracts structured clinical entities from a ProcessedTranscript.

    Tries to load a scispaCy NER model on construction.  If the model is
    unavailable the instance silently falls back to the regex/keyword engine.

    Usage::

        extractor = EntityExtractor()
        entities  = extractor.extract(transcript)
        batch     = extractor.extract_batch(transcripts)
        stats     = extractor.get_stats(batch)
    """

    def __init__(self, model: str = "en_core_sci_md", fallback: bool = True) -> None:
        """
        Parameters
        ----------
        model:
            Name of the scispaCy model to load (must be installed separately).
        fallback:
            If True, silently use rule-based extraction when the model is
            unavailable.  If False and the model fails to load, subsequent
            calls to extract() will still succeed via rule-based mode and log
            a warning rather than raising.
        """
        self.model_name = model
        self.fallback = fallback
        self.use_fallback: bool = True
        self._nlp = None

        if not _SCISPACY_AVAILABLE:
            log.warning(
                "spacy/scispaCy is not installed — using rule-based extraction. "
                "Install with: pip install scispacy"
            )
            return

        try:
            self._nlp = _spacy_mod.load(model)
            self.use_fallback = False
            log.info("Loaded scispaCy model %r", model)
        except OSError:
            if fallback:
                log.warning(
                    "scispaCy model %r not found — using rule-based extraction. "
                    "Install with: pip install %s",
                    model,
                    "https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/"
                    "releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz",
                )
            else:
                log.warning(
                    "scispaCy model %r not found; falling back to rule-based "
                    "extraction. Set fallback=False to suppress this warning.",
                    model,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, transcript: ProcessedTranscript) -> MedicalEntities:
        """
        Extract clinical entities from a single ProcessedTranscript.

        Dispatches to the scispaCy or rule-based engine depending on whether
        the model loaded successfully, then always applies dedicated regex
        passes for vitals and negations.  Never raises — returns a zeroed-out
        MedicalEntities with extraction_method="failed" on unexpected error.

        Parameters
        ----------
        transcript:
            A ProcessedTranscript produced by TranscriptProcessor.process().

        Returns
        -------
        MedicalEntities
        """
        try:
            if not self.use_fallback and self._nlp is not None:
                entities_dict = self._extract_scispacy(transcript)
                method = "scispacy"
            else:
                entities_dict = self._extract_rule_based(transcript)
                method = "rule_based"

            vitals = self._extract_vitals(transcript.full_text)
            negations = self._extract_negations(transcript.turns)
            confidence = self._compute_confidence(entities_dict, method, bool(vitals))

            return MedicalEntities(
                conversation_id=transcript.id,
                symptoms=entities_dict["symptoms"],
                clinical_findings=entities_dict["clinical_findings"],
                medications=entities_dict["medications"],
                medical_history=entities_dict["medical_history"],
                vitals=vitals,
                allergies=entities_dict["allergies"],
                speciality_hints=entities_dict["speciality_hints"],
                negations=negations,
                extraction_method=method,
                confidence=confidence,
                raw_entities=entities_dict["raw_entities"],
            )

        except Exception as exc:  # noqa: BLE001
            log.error(
                "EntityExtractor.extract() failed for %r: %s",
                getattr(transcript, "id", "?"),
                exc,
            )
            return MedicalEntities(
                conversation_id=getattr(transcript, "id", "unknown"),
                symptoms=[],
                clinical_findings=[],
                medications=[],
                medical_history=[],
                vitals={},
                allergies=[],
                speciality_hints=[],
                negations=[],
                extraction_method="failed",
                confidence=0.0,
                raw_entities=[],
            )

    def _extract_scispacy(self, transcript: ProcessedTranscript) -> dict:
        """
        Run scispaCy NER on patient and doctor text separately and map entity
        labels to our categories.

        Patient text entities → symptoms / medications
        Doctor text entities  → clinical_findings / medications

        History, allergies, and speciality hints are always supplemented from
        the rule-based engine since NER models are not trained for those
        patterns.

        Returns
        -------
        dict with keys matching _empty_entities_dict()
        """
        result = self._empty_entities_dict()

        def _process_doc(text: str, label_map: dict[str, str], source: str) -> None:
            if not text.strip():
                return
            doc = self._nlp(text)
            for ent in doc.ents:
                token = ent.text.strip()
                if not token:
                    continue
                category = label_map.get(ent.label_, "clinical_findings")
                result[category].append(token)
                result["raw_entities"].append({
                    "text": token,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "source": source,
                })

        _process_doc(transcript.patient_text, _SCI_LABEL_PATIENT, "patient")
        _process_doc(transcript.doctor_text, _SCI_LABEL_DOCTOR, "doctor")

        # Deduplicate the NER-extracted categories
        for key in ("symptoms", "clinical_findings", "medications"):
            result[key] = self._dedup(result[key])

        # Rule-based pass for categories NER doesn't cover well
        rb = self._extract_rule_based(transcript)
        result["medical_history"] = rb["medical_history"]
        result["allergies"] = rb["allergies"]
        result["speciality_hints"] = rb["speciality_hints"]
        # Merge rule-based medications in case NER missed common drug names
        result["medications"] = self._dedup(
            result["medications"] + rb["medications"]
        )

        return result

    def _extract_rule_based(self, transcript: ProcessedTranscript) -> dict:
        """
        Extract entities using regex patterns and keyword lists.

        Strategy
        --------
        patient_text → symptoms (keyword match) + history + allergies
        doctor_text  → clinical_findings (keyword match)
        full_text    → medications (suffix + common names + context + dosage)
        full_text    → speciality hints

        Returns
        -------
        dict with keys matching _empty_entities_dict()
        """
        result = self._empty_entities_dict()

        patient = transcript.patient_text
        doctor = transcript.doctor_text
        full = transcript.full_text

        # ── Symptoms from patient text ──────────────────────────────────
        for m in _SYMPTOM_RE.finditer(patient):
            result["symptoms"].append(m.group(1).lower())
            result["raw_entities"].append({
                "text": m.group(1),
                "label": "RULE_SYMPTOM",
                "pattern": "keyword",
                "source": "patient",
            })
        result["symptoms"] = self._dedup(result["symptoms"])

        # ── Clinical findings from doctor text ──────────────────────────
        for m in _SYMPTOM_RE.finditer(doctor):
            result["clinical_findings"].append(m.group(1).lower())
            result["raw_entities"].append({
                "text": m.group(1),
                "label": "RULE_FINDING",
                "pattern": "keyword",
                "source": "doctor",
            })
        result["clinical_findings"] = self._dedup(result["clinical_findings"])

        # ── Medications from full text ──────────────────────────────────
        meds: list[str] = []

        # 1. Drug-class suffixes
        for m in _MED_SUFFIX_RE.finditer(full):
            meds.append(m.group(0).lower())
            result["raw_entities"].append({
                "text": m.group(0),
                "label": "RULE_MED",
                "pattern": "suffix",
                "source": "full_text",
            })

        # 2. Known generic drug names
        for m in _COMMON_DRUGS_RE.finditer(full):
            meds.append(m.group(0).lower())
            result["raw_entities"].append({
                "text": m.group(0),
                "label": "RULE_MED",
                "pattern": "common_drug",
                "source": "full_text",
            })

        # 3. Context verbs ("prescribed X", "taking X", …)
        for m in _MED_CONTEXT_RE.finditer(full):
            candidate = m.group(1).strip().lower()
            if len(candidate.split()) <= 3:  # guard against greedy matches
                meds.append(candidate)
                result["raw_entities"].append({
                    "text": m.group(1).strip(),
                    "label": "RULE_MED",
                    "pattern": "context_verb",
                    "source": "full_text",
                })

        # 4. Dosage context ("metformin 500 mg")
        for m in _DOSAGE_RE.finditer(full):
            candidate = m.group(1).lower()
            # Only include if it looks drug-like (not a common word)
            if len(candidate) >= 4 and candidate not in {
                "take", "dose", "pain", "each", "once", "unit", "test", "this"
            }:
                meds.append(candidate)
                result["raw_entities"].append({
                    "text": m.group(1),
                    "label": "RULE_MED",
                    "pattern": "dosage",
                    "source": "full_text",
                })

        result["medications"] = self._dedup(meds)

        # ── Medical history from full text ──────────────────────────────
        for pattern in _HISTORY_PATTERNS:
            for m in pattern.finditer(full):
                result["medical_history"].append(m.group(1).strip().lower())
                result["raw_entities"].append({
                    "text": m.group(1).strip(),
                    "label": "RULE_HISTORY",
                    "pattern": pattern.pattern[:40],
                    "source": "full_text",
                })
        result["medical_history"] = self._dedup(result["medical_history"])

        # ── Allergies from full text ─────────────────────────────────────
        for pattern in _ALLERGY_PATTERNS:
            for m in pattern.finditer(full):
                result["allergies"].append(m.group(1).strip().lower())
                result["raw_entities"].append({
                    "text": m.group(1).strip(),
                    "label": "RULE_ALLERGY",
                    "pattern": pattern.pattern[:40],
                    "source": "full_text",
                })
        result["allergies"] = self._dedup(result["allergies"])

        # ── Speciality hints from full text ─────────────────────────────
        found_specialities: set[str] = set()
        for speciality, terms in _SPECIALITY_TERMS.items():
            for term in terms:
                if re.search(r"\b" + re.escape(term) + r"\b", full, re.IGNORECASE):
                    found_specialities.add(speciality)
                    break
        result["speciality_hints"] = sorted(found_specialities)

        return result

    def _extract_vitals(self, text: str) -> Dict[str, str]:
        """
        Extract vital sign values from *text* using regex patterns.

        Always called regardless of extraction mode (regex reliably outperforms
        NER on the tightly structured vital-sign format).

        Covered vitals
        --------------
        BP           blood pressure (requires a label or mmHg unit)
        temperature  temperature in °C or °F (requires unit or label)
        HR           heart rate (requires label or bpm unit)
        RR           respiratory rate
        SpO2         oxygen saturation
        weight       body weight in kg or lbs
        height       height in cm, m, or feet

        Returns
        -------
        dict mapping vital name → value string
        """
        vitals: Dict[str, str] = {}

        # Blood pressure
        m = _VITALS_BP_LABELED.search(text) or _VITALS_BP_UNIT.search(text)
        if m:
            raw = m.group(1).replace(" ", "")
            vitals["BP"] = f"{raw} mmHg"

        # Temperature — prefer the unit-confirmed match
        m = _VITALS_TEMP_UNIT.search(text)
        if m:
            unit = m.group(2).upper()
            vitals["temperature"] = f"{m.group(1)}°{unit}"
        else:
            m = _VITALS_TEMP_LABELED.search(text)
            if m:
                vitals["temperature"] = m.group(1)

        # Heart rate
        m = _VITALS_HR_LABELED.search(text) or _VITALS_HR_BPM.search(text)
        if m:
            vitals["HR"] = f"{m.group(1)} bpm"

        # Respiratory rate
        m = _VITALS_RR.search(text) or _VITALS_RR_UNIT.search(text)
        if m:
            vitals["RR"] = f"{m.group(1)} breaths/min"

        # SpO2
        m = (
            _VITALS_SPO2.search(text)
            or _VITALS_SPO2_SATURATING.search(text)
        )
        if m:
            vitals["SpO2"] = f"{m.group(1)}%"

        # Weight
        m = _VITALS_WEIGHT.search(text)
        if m:
            vitals["weight"] = f"{m.group(1)} {m.group(2)}"

        # Height
        m = _VITALS_HEIGHT.search(text)
        if m:
            vitals["height"] = f"{m.group(1)} {m.group(2)}"

        return vitals

    def _extract_negations(self, turns: List[Turn]) -> List[str]:
        """
        Scan PATIENT turns for explicitly negated symptoms or conditions.

        Patterns matched (examples)
        ---------------------------
        "no chest pain"                → "chest pain"
        "no history of diabetes"       → "diabetes"
        "I don't have any allergies"   → "allergies"
        "haven't had any fever"        → "fever"
        "denies shortness of breath"   → "shortness of breath"
        "without any nausea"           → "nausea"
        "not experiencing any pain"    → "pain"

        Only PATIENT speaker turns are checked; doctor reformulations ("denies
        X") are handled separately by the scispaCy pipeline.

        Returns
        -------
        list[str] — negated terms only (stripped of the negation phrase)
        """
        negated: list[str] = []
        for turn in turns:
            if turn.speaker != "PATIENT":
                continue
            for pattern in _NEGATION_PATTERNS:
                for m in pattern.finditer(turn.text):
                    term = m.group(1).strip().lower()
                    # Iteratively strip trailing noise so "fever at all" → "fever"
                    while True:
                        cleaned = _NEGATION_TRAILING_NOISE.sub("", term).strip()
                        if cleaned == term:
                            break
                        term = cleaned
                    # Drop very short or trivially long matches
                    if 2 < len(term) < 50:
                        negated.append(term)
        return self._dedup(negated)

    def extract_batch(
        self,
        transcripts: List[ProcessedTranscript],
        show_progress: bool = True,
    ) -> List[MedicalEntities]:
        """
        Extract entities from a list of transcripts, isolating per-record
        failures so a single bad record never aborts the batch.

        Parameters
        ----------
        transcripts:
            ProcessedTranscript objects from TranscriptProcessor.
        show_progress:
            Print a progress line every 50 records and a final summary.

        Returns
        -------
        list[MedicalEntities]
            Only successfully extracted records (method != "failed").
        """
        results: list[MedicalEntities] = []
        failures: list[tuple[str, str]] = []

        for i, transcript in enumerate(transcripts):
            record_id = getattr(transcript, "id", f"index:{i}")
            try:
                ent = self.extract(transcript)
                if ent.extraction_method == "failed":
                    failures.append((record_id, "extraction returned failed status"))
                else:
                    results.append(ent)
            except Exception as exc:  # noqa: BLE001
                failures.append((record_id, str(exc)))

            if show_progress and (i + 1) % 50 == 0:
                print(
                    f"  … {i + 1}/{len(transcripts)} records "
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

    def get_stats(self, entities_list: List[MedicalEntities]) -> dict:
        """
        Compute aggregate statistics over a list of MedicalEntities.

        Returns
        -------
        dict with keys:
            total                    — number of records
            extraction_method_counts — {"scispacy": n, "rule_based": n, ...}
            avg_entities             — per-category average entity count
            pct_with_vitals          — % of records with ≥1 vital extracted
            pct_with_negations       — % of records with ≥1 negation detected
            avg_confidence           — mean confidence score
        """
        if not entities_list:
            return {
                "total": 0,
                "extraction_method_counts": {},
                "avg_entities": {},
                "pct_with_vitals": 0.0,
                "pct_with_negations": 0.0,
                "avg_confidence": 0.0,
            }

        total = len(entities_list)

        # Method breakdown
        method_counts: dict[str, int] = {}
        for ent in entities_list:
            method_counts[ent.extraction_method] = (
                method_counts.get(ent.extraction_method, 0) + 1
            )

        # Average entities per category
        categories = (
            "symptoms", "clinical_findings", "medications",
            "medical_history", "allergies", "speciality_hints", "negations",
        )
        avg_entities = {
            cat: round(
                sum(len(getattr(e, cat)) for e in entities_list) / total, 2
            )
            for cat in categories
        }

        pct_vitals = round(
            100 * sum(1 for e in entities_list if e.vitals) / total, 1
        )
        pct_negations = round(
            100 * sum(1 for e in entities_list if e.negations) / total, 1
        )
        avg_confidence = round(
            sum(e.confidence for e in entities_list) / total, 3
        )

        return {
            "total": total,
            "extraction_method_counts": method_counts,
            "avg_entities": avg_entities,
            "pct_with_vitals": pct_vitals,
            "pct_with_negations": pct_negations,
            "avg_confidence": avg_confidence,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup(items: list[str]) -> list[str]:
        """
        Return *items* deduplicated case-insensitively, preserving first-seen
        order.
        """
        seen: set[str] = set()
        result: list[str] = []
        for item in items:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                result.append(item.strip())
        return result

    @staticmethod
    def _empty_entities_dict() -> dict:
        """Return a zeroed-out entity category dict."""
        return {
            "symptoms": [],
            "clinical_findings": [],
            "medications": [],
            "medical_history": [],
            "allergies": [],
            "speciality_hints": [],
            "raw_entities": [],
        }

    @staticmethod
    def _compute_confidence(
        entities_dict: dict, method: str, has_vitals: bool
    ) -> float:
        """
        Estimate extraction quality on a 0.0–1.0 scale.

        Score is the sum of per-category weights for each non-empty category,
        with a vitals bonus and a rule-based penalty (regex is less precise
        than NER).

        Parameters
        ----------
        entities_dict:
            Output of _extract_scispacy() or _extract_rule_based().
        method:
            "scispacy" or "rule_based".
        has_vitals:
            True if _extract_vitals() returned a non-empty dict.
        """
        score = sum(
            w
            for cat, w in _CONFIDENCE_WEIGHTS.items()
            if entities_dict.get(cat)
        )
        if has_vitals:
            score = min(score + 0.05, 1.0)
        if method == "rule_based":
            score *= 0.8
        return round(min(score, 1.0), 2)


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

def _print_sep(char: str = "─", width: int = 70) -> None:
    print(char * width)


if __name__ == "__main__":
    # Allow running as: python src/clinical/entity_extractor.py
    # from the medai/ directory, or as a module from the repo root.
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from medai.src.voice.transcript_processor import (  # noqa: PLC0415
        TranscriptProcessor,
    )

    import json  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    DATA_PATH = _REPO_ROOT / "data/raw/unified/en_mts.jsonl"
    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}")
        print("Run data/pipeline/setup.py first.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("  EntityExtractor — standalone smoke-test")
    print("=" * 70)

    # Load 5 records
    records = []
    with open(DATA_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if len(records) >= 5:
                break

    # Process through TranscriptProcessor then EntityExtractor
    tp = TranscriptProcessor()
    transcripts = tp.process_batch(records, show_progress=False)

    extractor = EntityExtractor()

    print(f"\nExtraction mode: {'rule_based (scispaCy unavailable)' if extractor.use_fallback else 'scispacy'}\n")

    _print_sep("─")
    print("extract_batch() output:")
    _print_sep("─")
    entities_list = extractor.extract_batch(transcripts, show_progress=True)

    # Print all 5 MedicalEntities objects
    print()
    _print_sep("═")
    print("  Full MedicalEntities objects")
    _print_sep("═")
    for ent in entities_list:
        print()
        print(str(ent))
        _print_sep("─")

    # Stats
    print()
    _print_sep("═")
    print("  get_stats() output")
    _print_sep("═")
    stats = extractor.get_stats(entities_list)
    for key, value in stats.items():
        print(f"  {key:<30} {value}")
    print()
