"""
pdf_generator.py

Renders a completed SOAP note, DDx result, and red-flag result into a
professional clinical PDF report (physician-facing) and a plain-language
patient summary.

Two rendering modes:
  weasyprint    — HTML → PDF via WeasyPrint (optional dependency)
  html_fallback — saves HTML file instead when WeasyPrint is unavailable

The soap_report.html template in configs/templates/ is used for the physician
report. Patient summaries use a self-contained inline HTML string.

Standalone smoke-test::

    python src/reporting/pdf_generator.py
"""

from __future__ import annotations

import html as html_lib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from medai.src.clinical.soap_generator import SOAPNote
    from medai.src.reasoning.ddx_engine import DDxResult
    from medai.src.reasoning.red_flag_detector import RedFlagResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional WeasyPrint import
# ---------------------------------------------------------------------------

_WEASYPRINT_AVAILABLE = False
_weasyprint_HTML = None

try:
    from weasyprint import HTML as _weasyprint_HTML  # type: ignore
    _WEASYPRINT_AVAILABLE = True
except Exception:   # ImportError or OSError (missing system libs)
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AI_DISCLAIMER = (
    "AI-Assisted Note: This report was generated with AI support and must be "
    "reviewed, verified, and signed by a licensed clinician before use in the "
    "patient record. It does not constitute a standalone medical opinion."
)

_SEVERITY_COLORS = {
    "critical": "#dc3545",
    "urgent": "#fd7e14",
    "monitor": "#ffc107",
}

_PROB_CLASSES = {
    "high": "prob-high",
    "moderate": "prob-moderate",
    "low": "prob-low",
}

_EMERGENCY_ADVICE = (
    "Call 999 (UK) or 911 (US) immediately, or go to your nearest "
    "Emergency Department, if you experience: chest pain or pressure, "
    "difficulty breathing, sudden severe headache, loss of consciousness, "
    "facial drooping or arm weakness, uncontrolled bleeding, or any other "
    "sudden worsening of your symptoms."
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReportOutput:
    """Paths and metadata for a generated report pair."""

    conversation_id: str
    physician_pdf_path: str     # path to physician PDF (or HTML fallback)
    patient_pdf_path: str       # path to patient summary PDF (or HTML fallback)
    generated_at: str           # ISO 8601 UTC timestamp
    page_count: int             # 0 when exact count is unknown (HTML mode)
    file_size_bytes: int        # size of physician report file
    generation_method: str      # "weasyprint" or "html_fallback"

    def __str__(self) -> str:
        return (
            f"ReportOutput(id={self.conversation_id!r}, "
            f"method={self.generation_method!r})\n"
            f"  Physician : {self.physician_pdf_path}\n"
            f"  Patient   : {self.patient_pdf_path}\n"
            f"  Generated : {self.generated_at}"
        )


# ---------------------------------------------------------------------------
# PDFGenerator
# ---------------------------------------------------------------------------

class PDFGenerator:
    """
    Converts pipeline outputs into physician and patient reports.

    Uses WeasyPrint for PDF rendering when available; falls back to saving
    HTML files so the class works without any optional dependencies.

    Usage::

        gen = PDFGenerator()
        output = gen.generate_physician_report(soap, ddx, red_flags)
        batch  = gen.generate_batch(soap_notes, ddx_results, rfr_list)
    """

    def __init__(
        self,
        output_dir: str = "data/outputs/reports",
        template_dir: str = "configs/templates",
    ) -> None:
        """
        Initialise the generator.

        Parameters
        ----------
        output_dir:
            Directory where generated files are saved.  Created if absent.
            Resolved relative to the repository root (3 levels above this file).
        template_dir:
            Directory containing ``soap_report.html``.  Resolved relative to
            the ``medai/`` package root (2 levels above this file).
        """
        self._repo_root = Path(__file__).resolve().parents[3]
        self._medai_root = Path(__file__).resolve().parents[2]

        self._output_dir = self._repo_root / output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self.use_fallback: bool = not _WEASYPRINT_AVAILABLE
        if self.use_fallback:
            log.warning(
                "WeasyPrint is not available — PDFGenerator will save HTML files. "
                "Install with: pip install weasyprint>=60.0"
            )

        self._template_html = self._load_template(
            self._medai_root / template_dir / "soap_report.html"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_physician_report(
        self,
        soap: SOAPNote,
        ddx: DDxResult,
        red_flags: RedFlagResult,
        patient_info: dict | None = None,
    ) -> ReportOutput:
        """
        Generate a complete physician-facing clinical report.

        Builds full HTML with SOAP sections, DDx table, red-flag alerts,
        follow-up questions, and recommended investigations, then converts
        to PDF (or saves as HTML if WeasyPrint is unavailable).

        Parameters
        ----------
        soap:
            SOAPNote from SOAPGenerator.
        ddx:
            DDxResult from DDxEngine.
        red_flags:
            RedFlagResult from RedFlagDetector.
        patient_info:
            Optional dict with keys ``name``, ``dob``, ``mrn``.
            Defaults to de-identified placeholder values.

        Returns
        -------
        ReportOutput
        """
        html_content = self._build_physician_html(soap, ddx, red_flags, patient_info or {})
        timestamp = _ts()
        ext = "html" if self.use_fallback else "pdf"
        filename = f"physician_{soap.conversation_id}_{timestamp}.{ext}"
        out_path = str(self._output_dir / filename)

        actual_path = self._html_to_pdf(html_content, out_path)
        file_size = Path(actual_path).stat().st_size if Path(actual_path).exists() else 0

        return ReportOutput(
            conversation_id=soap.conversation_id,
            physician_pdf_path=actual_path,
            patient_pdf_path="",   # filled by caller or generate_batch
            generated_at=datetime.now(timezone.utc).isoformat(),
            page_count=0,
            file_size_bytes=file_size,
            generation_method="html_fallback" if self.use_fallback else "weasyprint",
        )

    def generate_patient_summary(
        self,
        soap: SOAPNote,
        red_flags: RedFlagResult,
        patient_info: dict | None = None,
    ) -> ReportOutput:
        """
        Generate a patient-friendly simplified report in plain language.

        Avoids medical jargon. Covers: what the doctor found, urgent concerns,
        care plan, and a hardcoded emergency-care safety net.

        Parameters
        ----------
        soap:
            SOAPNote from SOAPGenerator.
        red_flags:
            RedFlagResult from RedFlagDetector.
        patient_info:
            Optional dict with keys ``name``, ``dob``, ``mrn``.

        Returns
        -------
        ReportOutput
        """
        html_content = self._build_patient_html(soap, red_flags, patient_info or {})
        timestamp = _ts()
        ext = "html" if self.use_fallback else "pdf"
        filename = f"patient_{soap.conversation_id}_{timestamp}.{ext}"
        out_path = str(self._output_dir / filename)

        actual_path = self._html_to_pdf(html_content, out_path)
        file_size = Path(actual_path).stat().st_size if Path(actual_path).exists() else 0

        return ReportOutput(
            conversation_id=soap.conversation_id,
            physician_pdf_path="",
            patient_pdf_path=actual_path,
            generated_at=datetime.now(timezone.utc).isoformat(),
            page_count=0,
            file_size_bytes=file_size,
            generation_method="html_fallback" if self.use_fallback else "weasyprint",
        )

    def generate_batch(
        self,
        soap_notes: list[SOAPNote],
        ddx_results: list[DDxResult],
        red_flag_results: list[RedFlagResult],
        show_progress: bool = True,
    ) -> list[ReportOutput]:
        """
        Generate physician and patient reports for a list of conversations.

        All three lists are matched by ``conversation_id``.  Records missing
        a DDx or red-flag result are skipped with a warning.  Per-record
        failures are isolated and excluded from the result.

        Parameters
        ----------
        soap_notes:
            SOAPNote objects in any order.
        ddx_results:
            DDxResult objects matched by conversation_id.
        red_flag_results:
            RedFlagResult objects matched by conversation_id.
        show_progress:
            Print progress every 5 records and a final summary.

        Returns
        -------
        list[ReportOutput] — one entry per successfully processed record,
        with both physician_pdf_path and patient_pdf_path set.
        """
        ddx_index = {d.conversation_id: d for d in ddx_results}
        rfr_index = {r.conversation_id: r for r in red_flag_results}

        results: list[ReportOutput] = []
        failures: list[tuple[str, str]] = []

        for i, soap in enumerate(soap_notes):
            cid = soap.conversation_id
            ddx = ddx_index.get(cid)
            rfr = rfr_index.get(cid)

            missing = [lbl for lbl, val in [("ddx", ddx), ("red_flags", rfr)] if val is None]
            if missing:
                reason = f"missing: {', '.join(missing)}"
                log.warning("generate_batch: skipping %r — %s", cid, reason)
                failures.append((cid, reason))
                continue

            try:
                physician = self.generate_physician_report(soap, ddx, rfr)
                patient = self.generate_patient_summary(soap, rfr)
                # Merge into a single ReportOutput
                physician.patient_pdf_path = patient.patient_pdf_path
                results.append(physician)
            except Exception as exc:  # noqa: BLE001
                log.error("generate_batch: failed for %r — %s", cid, exc)
                failures.append((cid, str(exc)))

            if show_progress and (i + 1) % 5 == 0:
                print(
                    f"  … {i + 1}/{len(soap_notes)} reports generated "
                    f"({len(failures)} failures so far)"
                )

        if show_progress:
            print(
                f"  Batch complete: {len(results)} succeeded, "
                f"{len(failures)} failed out of {len(soap_notes)} records."
            )

        return results

    # ------------------------------------------------------------------
    # HTML builders
    # ------------------------------------------------------------------

    def _build_physician_html(
        self,
        soap: SOAPNote,
        ddx: DDxResult,
        red_flags: RedFlagResult,
        patient_info: dict,
    ) -> str:
        """
        Assemble the complete physician-report HTML string.

        Uses the loaded ``soap_report.html`` template (string substitution
        with ``{{ placeholder }}`` slots).

        Structure
        ---------
        - Header with MedAI branding and report date
        - AI disclaimer banner
        - Patient banner (name / DOB / MRN from patient_info or defaults)
        - Red-flag alerts section (omitted when no flags present)
        - Four SOAP sections
        - DDx table: rank, name, ICD-10, probability badge, supporting evidence
        - Follow-up questions and recommended investigations as bullet lists
        - Footer

        Parameters
        ----------
        soap, ddx, red_flags:
            Pipeline outputs.
        patient_info:
            Keys ``name``, ``dob``, ``mrn`` (all optional).

        Returns
        -------
        str — complete HTML document.
        """
        report_date = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
        confidence_pct = f"{int(soap.confidence * 100)}%"

        # Patient info
        p_name = _esc(patient_info.get("name", "De-identified / Demo Patient"))
        p_dob = _esc(patient_info.get("dob", "—"))
        p_mrn = _esc(patient_info.get("mrn", "—"))

        # Red flags HTML
        red_flags_html = _build_red_flags_html(red_flags)

        # DDx table HTML
        ddx_table_html = _build_ddx_table_html(ddx)

        # Follow-up questions and investigations
        follow_up_html = _build_list_html(ddx.follow_up_questions)
        investigations_html = _build_list_html(ddx.recommended_investigations)

        # Simple {{ key }} substitution
        output = self._template_html
        replacements = {
            "report_date": report_date,
            "speciality": _esc(soap.speciality or "General Medicine"),
            "model_used": _esc(soap.model_used or "template_fallback"),
            "ai_disclaimer": _AI_DISCLAIMER,
            "patient_name": p_name,
            "patient_dob": p_dob,
            "patient_mrn": p_mrn,
            "confidence": confidence_pct,
            "red_flags_html": red_flags_html,
            "subjective": _esc(soap.subjective),
            "objective": _esc(soap.objective),
            "assessment": _esc(soap.assessment),
            "plan": _esc(soap.plan),
            "ddx_table_html": ddx_table_html,
            "follow_up_questions": follow_up_html,
            "investigations": investigations_html,
        }
        for key, value in replacements.items():
            output = output.replace("{{ " + key + " }}", value)

        return output

    def _build_patient_html(
        self,
        soap: SOAPNote,
        red_flags: RedFlagResult,
        patient_info: dict,
    ) -> str:
        """
        Build a plain-language patient summary HTML document.

        Deliberately avoids medical abbreviations and jargon.  Uses large
        fonts and a simple single-column layout for readability.

        Sections
        --------
        - What brought you in today (subjective, simplified)
        - What we found (objective findings)
        - Our assessment (assessment in plain terms)
        - Your care plan (plan section)
        - Important warnings (red flags in plain language, if any)
        - When to seek emergency care (hardcoded safety-net text)

        Parameters
        ----------
        soap:
            SOAPNote from SOAPGenerator.
        red_flags:
            RedFlagResult from RedFlagDetector.
        patient_info:
            Keys ``name``, ``dob``, ``mrn`` (all optional).

        Returns
        -------
        str — complete standalone HTML document.
        """
        p_name = _esc(patient_info.get("name", "Patient"))
        report_date = datetime.now(timezone.utc).strftime("%d %B %Y")

        # Urgent concerns block
        if red_flags.flags:
            concerns_items = "".join(
                f"<li>{_esc(f.description)} — <em>{_esc(f.recommended_action)}</em></li>"
                for f in red_flags.flags
            )
            concerns_html = (
                '<div class="warning-box">'
                "<strong>&#9888; Urgent concerns noted during your visit:</strong>"
                f"<ul>{concerns_items}</ul></div>"
            )
        else:
            concerns_html = ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Your Visit Summary — {report_date}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      font-size: 14pt;
      color: #222;
      background: #fff;
      max-width: 750px;
      margin: 0 auto;
      padding: 2em;
      line-height: 1.7;
    }}
    h1 {{ font-size: 22pt; color: #1a3a5c; margin-bottom: 0.2em; }}
    .subtitle {{ font-size: 11pt; color: #6b7280; margin-bottom: 1.5em; }}
    h2 {{
      font-size: 15pt;
      color: #1a3a5c;
      border-left: 4px solid #2d7a4f;
      padding-left: 0.6em;
      margin: 1.2em 0 0.4em 0;
    }}
    p {{ margin-bottom: 0.8em; white-space: pre-wrap; }}
    .warning-box {{
      background: #fff3cd;
      border: 2px solid #fd7e14;
      border-radius: 6px;
      padding: 0.8em 1.2em;
      margin: 1.2em 0;
      font-size: 13pt;
    }}
    .emergency-box {{
      background: #f8d7da;
      border: 2px solid #dc3545;
      border-radius: 6px;
      padding: 0.8em 1.2em;
      margin: 1.5em 0;
      font-size: 13pt;
    }}
    .emergency-box strong {{ color: #842029; font-size: 15pt; }}
    footer {{
      margin-top: 2em;
      border-top: 1px solid #dee2e6;
      padding-top: 0.6em;
      font-size: 10pt;
      color: #9ca3af;
    }}
  </style>
</head>
<body>

  <h1>Your Visit Summary</h1>
  <div class="subtitle">Prepared for: {p_name} &nbsp;|&nbsp; Date: {report_date}</div>

  <h2>What brought you in today</h2>
  <p>{_esc(soap.subjective)}</p>

  <h2>What we found</h2>
  <p>{_esc(soap.objective)}</p>

  <h2>Our assessment</h2>
  <p>{_esc(soap.assessment)}</p>

  <h2>Your care plan</h2>
  <p>{_esc(soap.plan)}</p>

  {concerns_html}

  <div class="emergency-box">
    <strong>&#128680; When to seek emergency care</strong><br/>
    {_EMERGENCY_ADVICE}
  </div>

  <footer>
    Generated by MedAI &nbsp;|&nbsp; This summary is for your information only.
    Always follow the advice of your doctor or healthcare provider.
  </footer>

</body>
</html>"""

    # ------------------------------------------------------------------
    # PDF / HTML output
    # ------------------------------------------------------------------

    def _html_to_pdf(self, html_content: str, output_path: str) -> str:
        """
        Convert *html_content* to PDF and write to *output_path*.

        Falls back to saving an HTML file when WeasyPrint is unavailable or
        raises an error.  The actual path used is always returned so callers
        can store it correctly regardless of the output format.

        Parameters
        ----------
        html_content:
            Complete HTML document string.
        output_path:
            Target file path.  Should end in ``.pdf``; if the HTML fallback
            is triggered the extension is changed to ``.html``.

        Returns
        -------
        str — path of the file actually written.
        """
        if not self.use_fallback:
            try:
                _weasyprint_HTML(string=html_content).write_pdf(output_path)
                return output_path
            except Exception as exc:  # noqa: BLE001
                log.error("WeasyPrint failed (%s) — saving HTML instead.", exc)

        # HTML fallback
        html_path = str(output_path).replace(".pdf", ".html")
        Path(html_path).write_text(html_content, encoding="utf-8")
        return html_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_template(path: Path) -> str:
        """
        Load the HTML template from *path*.

        Returns a minimal inline template string if the file does not exist.

        Parameters
        ----------
        path:
            Absolute path to ``soap_report.html``.

        Returns
        -------
        str — template content.
        """
        if path.exists():
            return path.read_text(encoding="utf-8")
        log.warning("Template not found at %s — using minimal fallback.", path)
        return _MINIMAL_TEMPLATE


# ---------------------------------------------------------------------------
# HTML fragment helpers
# ---------------------------------------------------------------------------

_MINIMAL_TEMPLATE = """\
<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>body{{font-family:Arial,sans-serif;padding:2em;}}
h2{{color:#1a3a5c;}} .soap-text{{white-space:pre-wrap;}}
</style></head><body>
<h1>MedAI Clinical Report</h1>
<p>Date: {{ report_date }} | Speciality: {{ speciality }}</p>
<p>{{ ai_disclaimer }}</p>
<p><strong>Patient:</strong> {{ patient_name }}</p>
{{ red_flags_html }}
<h2>Subjective</h2><div class="soap-text">{{ subjective }}</div>
<h2>Objective</h2><div class="soap-text">{{ objective }}</div>
<h2>Assessment</h2><div class="soap-text">{{ assessment }}</div>
<h2>Plan</h2><div class="soap-text">{{ plan }}</div>
{{ ddx_table_html }}
<h2>Follow-up Questions</h2>{{ follow_up_questions }}
<h2>Investigations</h2>{{ investigations }}
<p><em>Generated by MedAI | For clinical review only | Not a substitute for physician judgment</em></p>
</body></html>"""


def _esc(text: str | None) -> str:
    """HTML-escape *text*, returning an empty string for None."""
    return html_lib.escape(str(text or ""), quote=False)


def _build_red_flags_html(red_flags: RedFlagResult) -> str:
    """Build the red-flags HTML section, or empty string when no flags."""
    if not red_flags.flags:
        return ""

    items = ""
    for flag in red_flags.flags:
        color = _SEVERITY_COLORS.get(flag.severity, "#6b7280")
        items += (
            f'<div class="flag-item">'
            f'<span class="flag-badge {flag.severity}" '
            f'style="background:{color};color:#fff;">'
            f'{_esc(flag.severity.upper())}</span>'
            f'<span><strong>{_esc(flag.term)}</strong> — {_esc(flag.description)}</span>'
            f'</div>'
            f'<div class="flag-action" style="margin-left:5.5em;margin-bottom:0.6em;">'
            f'Action: {_esc(flag.recommended_action)}</div>'
        )

    return (
        '<div class="red-flags-section">'
        '<div class="section-heading red">&#9888; Red Flags / Alerts</div>'
        f'{items}'
        '</div>'
    )


def _build_ddx_table_html(ddx: DDxResult) -> str:
    """Build the DDx HTML table."""
    if not ddx.top_diagnoses:
        return "<p><em>No differential diagnoses identified.</em></p>"

    rows = ""
    for i, d in enumerate(ddx.top_diagnoses, 1):
        prob_cls = _PROB_CLASSES.get(d.probability, "prob-low")
        evidence = _esc(", ".join(d.supporting_evidence[:5]))
        rows += (
            f"<tr>"
            f"<td>{i}</td>"
            f"<td>{_esc(d.name)}</td>"
            f"<td>{_esc(d.icd10_code)}</td>"
            f'<td><span class="prob-badge {prob_cls}">{_esc(d.probability)}</span></td>'
            f"<td>{evidence}</td>"
            f"<td>{_esc(d.reasoning[:120])}</td>"
            f"</tr>"
        )

    return (
        '<table class="ddx-table">'
        "<thead><tr>"
        "<th>#</th><th>Diagnosis</th><th>ICD-10</th>"
        "<th>Probability</th><th>Supporting Evidence</th><th>Reasoning</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
    )


def _build_list_html(items: list[str]) -> str:
    """Build an HTML unordered list, or a placeholder when empty."""
    if not items:
        return "<p><em>None identified.</em></p>"
    lis = "".join(f"<li>{_esc(item)}</li>" for item in items)
    return f'<ul style="padding-left:1.5em;font-size:10pt;">{lis}</ul>'


def _ts() -> str:
    """Return a compact UTC timestamp string for use in filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


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
    print("  PDFGenerator — standalone smoke-test")
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

    gen = PDFGenerator()
    print(f"\nGeneration method: {'weasyprint' if not gen.use_fallback else 'html_fallback'}\n")

    _print_sep()
    outputs = gen.generate_batch(soaps, ddx_results, rfrs, show_progress=True)
    print()
    for out in outputs:
        print(str(out))
        _print_sep("─")
    print()
