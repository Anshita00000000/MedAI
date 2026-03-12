"""
main.py

FastAPI application for the MedAI Clinical Intelligence Platform.

Wraps the full pipeline chain as a REST API::

    TranscriptProcessor → EntityExtractor → SOAPGenerator
    → RedFlagDetector → DDxEngine → PDFGenerator + FHIRFormatter

Authentication
--------------
Set the ``MEDAI_API_KEY`` environment variable to enable API-key auth.
When set, all POST endpoints require the ``X-API-Key`` header.
GET endpoints (``/health``) are always public.

Running locally::

    python src/api/main.py
    # or
    uvicorn medai.src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("medai.api")

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TranscriptInput(BaseModel):
    """Raw conversation to be processed by the pipeline."""

    conversation_id: str = Field(..., description="Unique identifier for this conversation.")
    turns: List[Dict[str, str]] = Field(
        ...,
        min_length=1,
        description='List of turn dicts: [{"speaker": str, "text": str}, ...]',
    )
    source_dataset: str = Field("api_input", description="Source label for provenance tracking.")
    language: str = Field("en", description="BCP-47 language code.")
    reference_note: Optional[str] = Field(
        None, description="Optional gold-standard SOAP note for evaluation."
    )


class PipelineRequest(BaseModel):
    """Full pipeline run request."""

    transcript: TranscriptInput = Field(..., description="Conversation transcript.")
    patient_info: Optional[Dict[str, Any]] = Field(
        None, description="Optional patient metadata: name, dob, mrn, gender, vitals."
    )
    generate_pdf: bool = Field(True, description="Whether to generate PDF/HTML reports.")
    generate_fhir: bool = Field(True, description="Whether to generate FHIR R4 bundle.")
    use_llm: bool = Field(
        True, description="If False, force template/rule-based mode (no LLM calls)."
    )


class PipelineResponse(BaseModel):
    """Full pipeline run response."""

    conversation_id: str
    status: str = Field(..., description='"success" | "partial" | "failed"')
    soap_note: Dict[str, Any]
    red_flags: Dict[str, Any]
    ddx: Dict[str, Any]
    pdf_paths: Dict[str, str] = Field(
        ..., description='{"physician": path_or_empty, "patient": path_or_empty}'
    )
    fhir_bundle_path: str
    processing_time_seconds: float
    warnings: List[str]


class HealthResponse(BaseModel):
    """Service health and component status."""

    status: str
    version: str = "1.0.0"
    components: Dict[str, Any]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_API_KEY_ENV = os.getenv("MEDAI_API_KEY", "")
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _require_api_key(api_key: Optional[str] = Security(_API_KEY_HEADER)) -> None:
    """
    Dependency that enforces API key authentication on POST endpoints.

    When ``MEDAI_API_KEY`` is not set in the environment, auth is disabled
    and every request is allowed through.  When it is set, the ``X-API-Key``
    header must match exactly; otherwise HTTP 401 is returned.
    """
    if not _API_KEY_ENV:
        return  # auth disabled
    if api_key != _API_KEY_ENV:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


# ---------------------------------------------------------------------------
# Startup logic (lifespan)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(application: FastAPI):
    """
    Initialise all pipeline components at application startup.

    Each component is wrapped in try/except so a single failed import does
    not prevent the API from starting.  Status is stored in ``app.state``
    and reflected in ``GET /health``.
    """
    log.info("MedAI API starting up …")

    # TranscriptProcessor
    try:
        from medai.src.voice.transcript_processor import TranscriptProcessor
        application.state.processor = TranscriptProcessor()
        application.state.processor_ok = True
        log.info("  ✓ TranscriptProcessor")
    except Exception as exc:
        application.state.processor = None
        application.state.processor_ok = False
        log.warning("  ✗ TranscriptProcessor failed: %s", exc)

    # EntityExtractor
    try:
        from medai.src.clinical.entity_extractor import EntityExtractor
        application.state.extractor = EntityExtractor()
        application.state.extractor_ok = True
        mode = "scispacy" if not application.state.extractor.use_fallback else "rule_based"
        log.info("  ✓ EntityExtractor (%s)", mode)
    except Exception as exc:
        application.state.extractor = None
        application.state.extractor_ok = False
        log.warning("  ✗ EntityExtractor failed: %s", exc)

    # SOAPGenerator
    try:
        from medai.src.clinical.soap_generator import SOAPGenerator
        application.state.soap_gen = SOAPGenerator()
        application.state.soap_gen_ok = True
        llm = not application.state.soap_gen.use_fallback
        log.info("  ✓ SOAPGenerator (llm=%s)", llm)
    except Exception as exc:
        application.state.soap_gen = None
        application.state.soap_gen_ok = False
        log.warning("  ✗ SOAPGenerator failed: %s", exc)

    # RedFlagDetector
    try:
        from medai.src.reasoning.red_flag_detector import RedFlagDetector
        application.state.red_flag = RedFlagDetector()
        application.state.red_flag_ok = True
        log.info("  ✓ RedFlagDetector")
    except Exception as exc:
        application.state.red_flag = None
        application.state.red_flag_ok = False
        log.warning("  ✗ RedFlagDetector failed: %s", exc)

    # DDxEngine
    try:
        from medai.src.reasoning.ddx_engine import DDxEngine
        application.state.ddx_engine = DDxEngine()
        application.state.ddx_engine_ok = True
        log.info("  ✓ DDxEngine")
    except Exception as exc:
        application.state.ddx_engine = None
        application.state.ddx_engine_ok = False
        log.warning("  ✗ DDxEngine failed: %s", exc)

    # PDFGenerator
    try:
        from medai.src.reporting.pdf_generator import PDFGenerator
        application.state.pdf_gen = PDFGenerator()
        application.state.pdf_gen_ok = True
        pdf_mode = "html_fallback" if application.state.pdf_gen.use_fallback else "weasyprint"
        log.info("  ✓ PDFGenerator (%s)", pdf_mode)
    except Exception as exc:
        application.state.pdf_gen = None
        application.state.pdf_gen_ok = False
        log.warning("  ✗ PDFGenerator failed: %s", exc)

    # FHIRFormatter
    try:
        from medai.src.reporting.fhir_formatter import FHIRFormatter
        application.state.fhir = FHIRFormatter()
        application.state.fhir_ok = True
        log.info("  ✓ FHIRFormatter")
    except Exception as exc:
        application.state.fhir = None
        application.state.fhir_ok = False
        log.warning("  ✗ FHIRFormatter failed: %s", exc)

    if not _API_KEY_ENV:
        log.warning("MEDAI_API_KEY not set — API key authentication is DISABLED")
    else:
        log.info("API key authentication enabled")

    log.info("Startup complete.")
    yield
    # Shutdown — nothing to clean up currently


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MedAI Clinical Intelligence Platform",
    description="Automated clinical documentation and decision support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=_lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def _timing_middleware(request: Request, call_next):
    """Log method, path, and response time for every request."""
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info("%s %s completed in %.1fms", request.method, request.url.path, elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _dc_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass (or list/dict thereof) to a plain dict."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _dc_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_dc_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _dc_to_dict(v) for k, v in obj.items()}
    return obj


def _transcript_to_record(t: TranscriptInput) -> dict:
    """Convert a ``TranscriptInput`` Pydantic model to the raw record dict expected by TranscriptProcessor."""
    return {
        "id": t.conversation_id,
        "source_dataset": t.source_dataset,
        "language": t.language,
        "turns": t.turns,
        "reference_note": t.reference_note or "",
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["Monitoring"],
)
async def health(request: Request) -> HealthResponse:
    """
    Return the liveness status of the API and each pipeline component.

    This endpoint is always public (no API key required).  Each component
    is checked against the startup-cached instances; failures are reported
    as ``"offline"`` without crashing.
    """
    components: Dict[str, Any] = {}
    state = request.app.state

    ok = getattr(state, "processor_ok", False)
    components["transcript_processor"] = "online" if ok else "offline"

    ok = getattr(state, "extractor_ok", False)
    extractor = getattr(state, "extractor", None)
    components["entity_extractor"] = {
        "status": "online" if ok else "offline",
        "extraction_mode": (
            "scispacy" if (extractor and not extractor.use_fallback) else "rule_based"
        ),
    }

    ok = getattr(state, "soap_gen_ok", False)
    soap_gen = getattr(state, "soap_gen", None)
    components["soap_generator"] = {
        "status": "online" if ok else "offline",
        "llm_available": bool(soap_gen and not soap_gen.use_fallback),
    }

    ok = getattr(state, "red_flag_ok", False)
    components["red_flag_detector"] = "online" if ok else "offline"

    ok = getattr(state, "ddx_engine_ok", False)
    components["ddx_engine"] = "online" if ok else "offline"

    ok = getattr(state, "pdf_gen_ok", False)
    pdf_gen = getattr(state, "pdf_gen", None)
    components["pdf_generator"] = {
        "status": "online" if ok else "offline",
        "pdf_mode": (
            "weasyprint" if (pdf_gen and not pdf_gen.use_fallback) else "html_fallback"
        ),
    }

    ok = getattr(state, "fhir_ok", False)
    components["fhir_formatter"] = "online" if ok else "offline"

    all_online = all(
        (v == "online" or (isinstance(v, dict) and v.get("status") == "online"))
        for v in components.values()
    )
    return HealthResponse(
        status="healthy" if all_online else "degraded",
        components=components,
    )


@app.post(
    "/pipeline/run",
    response_model=PipelineResponse,
    summary="Run the full MedAI pipeline on a transcript",
    tags=["Pipeline"],
)
async def pipeline_run(
    body: PipelineRequest,
    request: Request,
    _: None = Depends(_require_api_key),
) -> PipelineResponse:
    """
    Execute the complete pipeline chain on the provided transcript.

    Steps executed in order:

    1. **TranscriptProcessor** — clean and structure the conversation turns.
    2. **EntityExtractor** — extract symptoms, medications, vitals, etc.
    3. **SOAPGenerator** — generate a structured SOAP note (LLM or template).
    4. **RedFlagDetector** — identify clinical red flags.
    5. **DDxEngine** — produce a ranked differential diagnosis.
    6. **PDFGenerator** (optional) — physician and patient reports.
    7. **FHIRFormatter** (optional) — FHIR R4 Bundle JSON.

    Partial failures in steps 3–7 are tolerated: the pipeline continues
    and warnings are collected.  HTTP 500 is only returned if step 1 or 2
    fails entirely.  The ``status`` field reflects the outcome:
    ``"success"``, ``"partial"``, or ``"failed"``.
    """
    state = request.app.state
    t_start = time.perf_counter()
    warnings: List[str] = []
    cid = body.transcript.conversation_id

    # ── Step 1: TranscriptProcessor ──────────────────────────────────────────
    if not getattr(state, "processor_ok", False):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="TranscriptProcessor is offline.",
        )
    try:
        record = _transcript_to_record(body.transcript)
        transcript = state.processor.process(record)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcript processing failed: {exc}",
        ) from exc

    # ── Step 2: EntityExtractor ───────────────────────────────────────────────
    if not getattr(state, "extractor_ok", False):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="EntityExtractor is offline.",
        )
    try:
        entities = state.extractor.extract(transcript)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {exc}",
        ) from exc

    # ── Step 3: SOAPGenerator ─────────────────────────────────────────────────
    soap = None
    if getattr(state, "soap_gen_ok", False):
        try:
            orig_fallback = state.soap_gen.use_fallback
            if not body.use_llm:
                state.soap_gen.use_fallback = True
            soap = state.soap_gen.generate(transcript, entities)
            state.soap_gen.use_fallback = orig_fallback
        except Exception as exc:
            warnings.append(f"SOAPGenerator failed: {exc}")
    else:
        warnings.append("SOAPGenerator is offline — SOAP note skipped.")

    soap_dict: Dict[str, Any] = _dc_to_dict(soap) if soap else {}

    # ── Step 4: RedFlagDetector ───────────────────────────────────────────────
    rfr = None
    if getattr(state, "red_flag_ok", False):
        try:
            rfr = state.red_flag.detect(transcript, entities, soap)
        except Exception as exc:
            warnings.append(f"RedFlagDetector failed: {exc}")
    else:
        warnings.append("RedFlagDetector is offline — red flags skipped.")

    rfr_dict: Dict[str, Any] = _dc_to_dict(rfr) if rfr else {}

    # ── Step 5: DDxEngine ─────────────────────────────────────────────────────
    ddx = None
    if getattr(state, "ddx_engine_ok", False) and soap is not None and rfr is not None:
        try:
            orig_fallback = state.ddx_engine.use_fallback
            if not body.use_llm:
                state.ddx_engine.use_fallback = True
            ddx = state.ddx_engine.analyse(transcript, entities, soap, rfr)
            state.ddx_engine.use_fallback = orig_fallback
        except Exception as exc:
            warnings.append(f"DDxEngine failed: {exc}")
    elif soap is None or rfr is None:
        warnings.append("DDxEngine skipped — SOAP or red-flag result unavailable.")
    else:
        warnings.append("DDxEngine is offline.")

    ddx_dict: Dict[str, Any] = _dc_to_dict(ddx) if ddx else {}

    # ── Step 6: PDFGenerator (optional) ──────────────────────────────────────
    pdf_paths: Dict[str, str] = {"physician": "", "patient": ""}
    if body.generate_pdf and ddx is not None and rfr is not None and soap is not None:
        if getattr(state, "pdf_gen_ok", False):
            try:
                report = state.pdf_gen.generate_physician_report(
                    soap, ddx, rfr, body.patient_info or {}
                )
                patient_report = state.pdf_gen.generate_patient_summary(
                    soap, rfr, body.patient_info or {}
                )
                pdf_paths["physician"] = report.physician_pdf_path
                pdf_paths["patient"] = patient_report.patient_pdf_path
            except Exception as exc:
                warnings.append(f"PDFGenerator failed: {exc}")
        else:
            warnings.append("PDFGenerator is offline — reports skipped.")
    elif body.generate_pdf:
        warnings.append("PDF generation skipped — upstream pipeline step(s) unavailable.")

    # ── Step 7: FHIRFormatter (optional) ──────────────────────────────────────
    fhir_path = ""
    if body.generate_fhir and ddx is not None and rfr is not None and soap is not None:
        if getattr(state, "fhir_ok", False):
            try:
                bundle = state.fhir.format_bundle(
                    soap, ddx, rfr, body.patient_info or {}
                )
                fhir_path = state.fhir.save_bundle(bundle, conversation_id=cid)
            except Exception as exc:
                warnings.append(f"FHIRFormatter failed: {exc}")
        else:
            warnings.append("FHIRFormatter is offline — FHIR bundle skipped.")
    elif body.generate_fhir:
        warnings.append("FHIR generation skipped — upstream pipeline step(s) unavailable.")

    elapsed = time.perf_counter() - t_start
    result_status = "success" if not warnings else "partial"

    return PipelineResponse(
        conversation_id=cid,
        status=result_status,
        soap_note=soap_dict,
        red_flags=rfr_dict,
        ddx=ddx_dict,
        pdf_paths=pdf_paths,
        fhir_bundle_path=fhir_path,
        processing_time_seconds=round(elapsed, 3),
        warnings=warnings,
    )


@app.post(
    "/transcripts/process",
    summary="Process transcript and extract entities",
    tags=["Pipeline"],
)
async def transcripts_process(
    body: TranscriptInput,
    request: Request,
    _: None = Depends(_require_api_key),
) -> dict:
    """
    Lightweight endpoint: **TranscriptProcessor** + **EntityExtractor** only.

    Useful for quick pre-processing or inspection without triggering SOAP
    generation or report production.  Returns the processed turn count,
    extracted entities dict, anomaly flags, and elapsed processing time.
    """
    state = request.app.state
    t_start = time.perf_counter()

    if not getattr(state, "processor_ok", False):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="TranscriptProcessor is offline.",
        )
    if not getattr(state, "extractor_ok", False):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="EntityExtractor is offline.",
        )

    try:
        record = _transcript_to_record(body)
        transcript = state.processor.process(record)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Transcript processing failed: {exc}",
        ) from exc

    try:
        entities = state.extractor.extract(transcript)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Entity extraction failed: {exc}",
        ) from exc

    elapsed = time.perf_counter() - t_start
    return {
        "conversation_id": body.conversation_id,
        "processed_turns": transcript.turn_count,
        "entities": _dc_to_dict(entities),
        "flags": transcript.flags,
        "processing_time_seconds": round(elapsed, 3),
    }


@app.post(
    "/soap/generate",
    summary="Generate a SOAP note from a transcript",
    tags=["Pipeline"],
)
async def soap_generate(
    body: TranscriptInput,
    request: Request,
    _: None = Depends(_require_api_key),
) -> dict:
    """
    Process a transcript and generate a structured SOAP note.

    Runs **TranscriptProcessor → EntityExtractor → SOAPGenerator**.
    Returns the SOAP note as a dict, the generation method
    (``"llm"`` or ``"template_fallback"``), and the confidence score.
    """
    state = request.app.state
    t_start = time.perf_counter()

    for comp, attr in [
        ("TranscriptProcessor", "processor_ok"),
        ("EntityExtractor", "extractor_ok"),
        ("SOAPGenerator", "soap_gen_ok"),
    ]:
        if not getattr(state, attr, False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{comp} is offline.",
            )

    try:
        record = _transcript_to_record(body)
        transcript = state.processor.process(record)
        entities = state.extractor.extract(transcript)
        soap = state.soap_gen.generate(transcript, entities)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SOAP generation failed: {exc}",
        ) from exc

    elapsed = time.perf_counter() - t_start
    return {
        "conversation_id": body.conversation_id,
        "soap_note": _dc_to_dict(soap),
        "generation_method": soap.generation_method,
        "confidence": soap.confidence,
        "processing_time_seconds": round(elapsed, 3),
    }


@app.post(
    "/reasoning/analyse",
    summary="Run red-flag detection and differential diagnosis",
    tags=["Pipeline"],
)
async def reasoning_analyse(
    body: TranscriptInput,
    request: Request,
    _: None = Depends(_require_api_key),
) -> dict:
    """
    Run the full reasoning pipeline on a transcript.

    Executes **TranscriptProcessor → EntityExtractor → SOAPGenerator →
    RedFlagDetector → DDxEngine** and returns the combined reasoning output.

    The ``requires_immediate_action`` flag is ``True`` when at least one
    ``"critical"`` red flag was detected.
    """
    state = request.app.state
    t_start = time.perf_counter()

    for comp, attr in [
        ("TranscriptProcessor", "processor_ok"),
        ("EntityExtractor", "extractor_ok"),
        ("SOAPGenerator", "soap_gen_ok"),
        ("RedFlagDetector", "red_flag_ok"),
        ("DDxEngine", "ddx_engine_ok"),
    ]:
        if not getattr(state, attr, False):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{comp} is offline.",
            )

    try:
        record = _transcript_to_record(body)
        transcript = state.processor.process(record)
        entities = state.extractor.extract(transcript)
        soap = state.soap_gen.generate(transcript, entities)
        rfr = state.red_flag.detect(transcript, entities, soap)
        ddx = state.ddx_engine.analyse(transcript, entities, soap, rfr)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reasoning analysis failed: {exc}",
        ) from exc

    elapsed = time.perf_counter() - t_start
    return {
        "conversation_id": body.conversation_id,
        "red_flags": _dc_to_dict(rfr),
        "ddx": _dc_to_dict(ddx),
        "requires_immediate_action": rfr.requires_immediate_action,
        "processing_time_seconds": round(elapsed, 3),
    }


@app.get(
    "/reports/{conversation_id}",
    summary="List generated report files for a conversation",
    tags=["Reports"],
)
async def reports_list(conversation_id: str) -> dict:
    """
    Return all report files on disk that match *conversation_id*.

    Searches ``data/outputs/reports/`` and ``data/outputs/fhir/`` for files
    whose names contain the given conversation ID.

    Returns HTTP 404 if no matching files are found.
    """
    search_dirs = [
        _REPO_ROOT / "data/outputs/reports",
        _REPO_ROOT / "data/outputs/fhir",
    ]
    files: List[Dict[str, Any]] = []

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for path in sorted(search_dir.iterdir()):
            if conversation_id in path.name and path.is_file():
                suffix = path.suffix.lstrip(".")
                file_type = (
                    "pdf" if suffix == "pdf"
                    else ("fhir" if suffix == "json" else "html")
                )
                files.append({
                    "type": file_type,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                })

    if not files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No report files found for conversation_id={conversation_id!r}",
        )

    return {"conversation_id": conversation_id, "files": files}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "medai.src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
