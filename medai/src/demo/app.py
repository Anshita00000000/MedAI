"""
app.py

Streamlit demo interface for the MedAI Clinical Intelligence Platform.

Sponsor-facing demonstration covering the full pipeline:
  Transcript -> SOAP Note -> Red Flags -> DDx -> PDF Report + FHIR Bundle

Tabs
----
1. Input   -- load a sample case or paste your own transcript, then run.
2. Results -- SOAP note, red-flag alerts, differential diagnosis.
3. Reports -- physician PDF / HTML preview and FHIR bundle viewer.

Run with::

    streamlit run src/demo/app.py

or via the convenience script::

    bash scripts/run_demo.sh
"""

from __future__ import annotations

import json
import time
import threading
from typing import Any

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config -- must be the first Streamlit call in the module
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MedAI -- Clinical Intelligence Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Hardcoded sample transcripts
# ---------------------------------------------------------------------------

SAMPLE_CASES: dict[str, str] = {
    "Case 1: Hypertension Follow-up": (
        "DOCTOR: Good morning. How have you been feeling since your last visit?\n"
        "PATIENT: Not too bad doctor. I've been having some headaches though.\n"
        "DOCTOR: How often are the headaches occurring?\n"
        "PATIENT: About three or four times a week. Mostly in the morning.\n"
        "DOCTOR: Are you taking your blood pressure medication regularly?\n"
        "PATIENT: I try to but I sometimes forget.\n"
        "DOCTOR: Let me check your blood pressure today. It reads 158 over 95.\n"
        "PATIENT: Is that high?\n"
        "DOCTOR: Yes it is elevated. Have you had any chest pain or shortness of breath?\n"
        "PATIENT: No chest pain. But I do get a little breathless climbing stairs.\n"
        "DOCTOR: Any swelling in your legs or ankles?\n"
        "PATIENT: A little bit in the evenings yes.\n"
        "DOCTOR: I am going to adjust your medication and refer you for an ECG.\n"
        "PATIENT: Okay thank you doctor."
    ),
    "Case 2: Migraine Presentation": (
        "DOCTOR: What brings you in today?\n"
        "PATIENT: I have the worst headache of my life. It started this morning suddenly.\n"
        "DOCTOR: On a scale of one to ten how severe is the pain?\n"
        "PATIENT: Definitely a nine. I also have blurry vision and feel nauseous.\n"
        "DOCTOR: Any sensitivity to light or sound?\n"
        "PATIENT: Yes both. I had to close the curtains at home.\n"
        "DOCTOR: Have you had headaches like this before?\n"
        "PATIENT: I get migraines sometimes but never this bad.\n"
        "DOCTOR: Any recent head injury or fever?\n"
        "PATIENT: No injury. No fever that I know of.\n"
        "DOCTOR: I am going to examine you and order a CT scan to rule out anything serious.\n"
        "PATIENT: Is it dangerous doctor?\n"
        "DOCTOR: We need to rule out some causes. The sudden severe onset is something we take seriously."
    ),
    "Case 3: Respiratory Complaint": (
        "DOCTOR: Hello, what seems to be the problem today?\n"
        "PATIENT: I have had a cough for about three weeks now and it is getting worse.\n"
        "DOCTOR: Is the cough dry or are you bringing up any phlegm?\n"
        "PATIENT: There is some yellow phlegm in the mornings.\n"
        "DOCTOR: Any fever or chills?\n"
        "PATIENT: I had a fever last week around 38.5. Chills too.\n"
        "DOCTOR: Any shortness of breath or chest pain?\n"
        "PATIENT: Some shortness of breath when I walk fast. No chest pain.\n"
        "DOCTOR: Do you smoke?\n"
        "PATIENT: I used to. Quit five years ago.\n"
        "DOCTOR: Let me listen to your chest. I can hear some crackles at the base.\n"
        "PATIENT: What does that mean?\n"
        "DOCTOR: It could indicate a chest infection. I want to get a chest X-ray and start you on antibiotics."
    ),
    "Case 4: Knee Pain (Orthopaedic)": (
        "DOCTOR: So you are here about your knee. Which knee is it?\n"
        "PATIENT: The right knee mainly but the left one is also starting to bother me.\n"
        "DOCTOR: How long have you had this pain?\n"
        "PATIENT: About two years but it has been much worse the last three months.\n"
        "DOCTOR: Does it hurt more going up stairs or sitting for long periods?\n"
        "PATIENT: Both. Stairs are very difficult. And after sitting I am very stiff.\n"
        "DOCTOR: Any swelling or redness around the joint?\n"
        "PATIENT: Some swelling yes especially in the evenings.\n"
        "DOCTOR: Have you tried any treatments so far?\n"
        "PATIENT: Paracetamol and ibuprofen. They help a little.\n"
        "DOCTOR: Based on your age and symptoms this sounds like osteoarthritis.\n"
        "PATIENT: Can it be treated?\n"
        "DOCTOR: Yes. We will start with physiotherapy and stronger anti-inflammatories."
    ),
    "Case 5: Urinary Symptoms": (
        "DOCTOR: What brings you in today?\n"
        "PATIENT: I have been having pain when I urinate for about four days.\n"
        "DOCTOR: Is the pain a burning sensation?\n"
        "PATIENT: Yes exactly. And I need to go very frequently.\n"
        "DOCTOR: Any blood in your urine?\n"
        "PATIENT: A little bit yesterday. It looked pinkish.\n"
        "DOCTOR: Any fever or back pain?\n"
        "PATIENT: Some lower back pain but no fever.\n"
        "DOCTOR: Are you sexually active?\n"
        "PATIENT: Yes.\n"
        "DOCTOR: I think this may be a urinary tract infection. I will do a urine dipstick test.\n"
        "PATIENT: How long will treatment take?\n"
        "DOCTOR: Usually three to five days of antibiotics clears it up completely."
    ),
}

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

_STATE_DEFAULTS: dict[str, Any] = {
    "pipeline_result": None,
    "transcript_text": "",
    "patient_name": "",
    "patient_dob": "",
    "patient_mrn": "",
    "generate_pdf": True,
    "generate_fhir": True,
    "last_api_url": "http://localhost:8000",
    "fhir_bundle": None,
    "reference_note": None,
}

for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_transcript(text: str) -> list[dict[str, str]]:
    """
    Parse a plain-text transcript into a list of turn dicts.

    Expected format (case-insensitive speaker labels)::

        DOCTOR: Hello, how are you feeling?
        PATIENT: Not great today, doctor.

    Lines that do not start with a recognised speaker label are appended
    to the previous turn.  Blank lines are ignored.

    Parameters
    ----------
    text:
        Raw multi-line transcript string.

    Returns
    -------
    list[dict] with keys ``"speaker"`` and ``"text"``.
    """
    turns: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("DOCTOR:"):
            turns.append({"speaker": "DOCTOR", "text": line[7:].strip()})
        elif upper.startswith("PATIENT:"):
            turns.append({"speaker": "PATIENT", "text": line[8:].strip()})
        elif turns:
            turns[-1]["text"] += " " + line
    return turns


def _api_headers(api_key: str) -> dict[str, str]:
    """Build request headers, adding X-API-Key only when non-empty."""
    h: dict[str, str] = {"Content-Type": "application/json"}
    if api_key.strip():
        h["X-API-Key"] = api_key.strip()
    return h


def _check_health(base_url: str, api_key: str) -> dict | None:
    """
    Call GET /health and return the parsed JSON dict.

    Returns ``None`` on any network or HTTP error.

    Parameters
    ----------
    base_url:
        API base URL.
    api_key:
        Optional API key for auth.

    Returns
    -------
    dict or None
    """
    try:
        r = requests.get(
            f"{base_url.rstrip('/')}/health",
            headers=_api_headers(api_key),
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _run_pipeline(
    base_url: str,
    api_key: str,
    turns: list[dict],
    patient_info: dict | None,
    generate_pdf: bool,
    generate_fhir: bool,
    use_llm: bool,
    conversation_id: str = "demo-session",
) -> tuple[dict | None, str]:
    """
    POST to /pipeline/run and return ``(result_dict, error_message)``.

    On success ``error_message`` is an empty string.
    On failure ``result_dict`` is ``None`` and ``error_message`` describes
    the problem in user-friendly terms.

    Parameters
    ----------
    base_url, api_key:
        Connection settings from the sidebar.
    turns:
        Parsed conversation turns from ``_parse_transcript()``.
    patient_info:
        Optional patient metadata dict (may be ``None``).
    generate_pdf, generate_fhir:
        Feature flags forwarded to the API.
    use_llm:
        Whether to allow LLM (Gemini) calls.
    conversation_id:
        Identifier attached to this pipeline run.

    Returns
    -------
    tuple[dict | None, str]
    """
    payload = {
        "transcript": {
            "conversation_id": conversation_id,
            "turns": turns,
            "source_dataset": "demo",
            "language": "en",
        },
        "patient_info": patient_info,
        "generate_pdf": generate_pdf,
        "generate_fhir": generate_fhir,
        "use_llm": use_llm,
    }
    try:
        r = requests.post(
            f"{base_url.rstrip('/')}/pipeline/run",
            json=payload,
            headers=_api_headers(api_key),
            timeout=120,
        )
        r.raise_for_status()
        return r.json(), ""
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API server. Is it running on the configured URL?"
    except requests.exceptions.Timeout:
        return None, "API request timed out (>120 s). The server may be overloaded."
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return None, f"API error {exc.response.status_code}: {detail}"
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


def _load_file_bytes(path: str) -> bytes | None:
    """
    Read a file from *path* and return its bytes.

    Returns ``None`` when the file cannot be accessed (the demo may run on
    a different host from the API server).
    """
    try:
        from pathlib import Path as _Path
        return _Path(path).read_bytes()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏥 MedAI Settings")
    st.divider()

    api_url: str = st.text_input(
        "API Base URL",
        value=st.session_state["last_api_url"],
        help="Base URL of the running MedAI FastAPI server.",
    )
    st.session_state["last_api_url"] = api_url

    api_key: str = st.text_input(
        "API Key (optional)",
        type="password",
        help="Required only when the server was started with MEDAI_API_KEY set.",
    )

    model_choice: str = st.radio(
        "Model Mode",
        ["LLM (Gemini)", "Template Fallback"],
        index=1,
        help=(
            "Template Fallback works without a Google API key. "
            "LLM mode requires GOOGLE_API_KEY to be set on the server."
        ),
    )
    use_llm: bool = model_choice == "LLM (Gemini)"

    st.divider()

    if st.button("🔍 Check API Status", use_container_width=True):
        with st.spinner("Checking…"):
            health = _check_health(api_url, api_key)

        if health is None:
            st.error("❌ API unreachable — check URL and that the server is running.")
        else:
            overall = health.get("status", "unknown")
            if overall == "healthy":
                st.success(f"✅ API status: **{overall}**")
            else:
                st.warning(f"⚠️ API status: **{overall}**")

            for comp_name, info in health.get("components", {}).items():
                label = comp_name.replace("_", " ").title()
                if isinstance(info, dict):
                    ok = info.get("status") == "online"
                    extras = "  ".join(
                        f"`{k}={v}`" for k, v in info.items() if k != "status"
                    )
                    st.markdown(f"{'✅' if ok else '❌'} **{label}** {extras}")
                else:
                    st.markdown(f"{'✅' if info == 'online' else '❌'} **{label}**")

    st.divider()
    st.caption(
        "MedAI v1.0 | AMPBA Capstone 2025W  \n"
        "Cloud Box Technologies LLC"
    )


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_input, tab_results, tab_reports = st.tabs(
    ["📋 Input", "🔬 Results", "📄 Reports"]
)


# ============================================================================
# TAB 1 -- INPUT
# ============================================================================

with tab_input:
    st.header("Clinical Transcript Input")

    col_sample, col_custom = st.columns(2, gap="large")

    # Left column -- sample cases
    with col_sample:
        st.subheader("Load Sample Transcript")
        selected_case: str = st.selectbox(
            "Select a sample case",
            options=list(SAMPLE_CASES.keys()),
            label_visibility="collapsed",
        )
        if st.button("📂 Load Sample", use_container_width=True):
            st.session_state["transcript_text"] = SAMPLE_CASES[selected_case]
            st.session_state["reference_note"] = None
            st.success(f"Loaded: **{selected_case}**")

        if st.session_state["transcript_text"]:
            preview = st.session_state["transcript_text"]
            st.markdown("**Preview:**")
            st.text(preview[:400] + ("\n…" if len(preview) > 400 else ""))

    # Right column -- manual input + patient info
    with col_custom:
        st.subheader("Upload Your Own Transcript")
        new_text: str = st.text_area(
            "Paste transcript",
            value=st.session_state["transcript_text"],
            height=260,
            placeholder=(
                "DOCTOR: What brings you in today?\n"
                "PATIENT: I have been having chest pain..."
            ),
            help='Format each line as "DOCTOR: ..." or "PATIENT: ..."',
            label_visibility="collapsed",
        )
        st.session_state["transcript_text"] = new_text

        st.markdown("**Patient Information** *(optional)*")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            p_name = st.text_input("Name", value=st.session_state["patient_name"])
            st.session_state["patient_name"] = p_name
        with pc2:
            p_dob = st.text_input(
                "DOB", value=st.session_state["patient_dob"], placeholder="YYYY-MM-DD"
            )
            st.session_state["patient_dob"] = p_dob
        with pc3:
            p_mrn = st.text_input("MRN", value=st.session_state["patient_mrn"])
            st.session_state["patient_mrn"] = p_mrn

        ck1, ck2 = st.columns(2)
        with ck1:
            gen_pdf = st.checkbox(
                "Generate PDF Report", value=st.session_state["generate_pdf"]
            )
            st.session_state["generate_pdf"] = gen_pdf
        with ck2:
            gen_fhir = st.checkbox(
                "Generate FHIR Output", value=st.session_state["generate_fhir"]
            )
            st.session_state["generate_fhir"] = gen_fhir

    st.divider()

    # Run button
    can_run = bool(st.session_state["transcript_text"].strip())
    run_clicked = st.button(
        "▶  Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
    )

    if not can_run:
        st.caption("Load or paste a transcript above, then click Run Pipeline.")

    if run_clicked:
        raw_text = st.session_state["transcript_text"].strip()
        turns = _parse_transcript(raw_text)

        if not turns:
            st.error(
                "Could not parse any DOCTOR/PATIENT turns. "
                "Ensure each line starts with 'DOCTOR:' or 'PATIENT:'."
            )
        else:
            # Build optional patient info
            patient_info: dict[str, str] = {}
            if st.session_state["patient_name"]:
                patient_info["name"] = st.session_state["patient_name"]
            if st.session_state["patient_dob"]:
                patient_info["dob"] = st.session_state["patient_dob"]
            if st.session_state["patient_mrn"]:
                patient_info["mrn"] = st.session_state["patient_mrn"]

            progress_bar = st.progress(0, text="Connecting to MedAI API…")
            status_placeholder = st.empty()

            # Advance progress bar in background while waiting for API
            _steps = [
                (12, "Processing transcript…"),
                (28, "Extracting clinical entities…"),
                (46, "Generating SOAP note…"),
                (64, "Detecting red flags…"),
                (80, "Running differential diagnosis…"),
                (92, "Generating reports…"),
            ]

            def _advance():
                for pct, msg in _steps:
                    time.sleep(0.45)
                    progress_bar.progress(pct, text=msg)

            _t = threading.Thread(target=_advance, daemon=True)
            _t.start()

            result, error = _run_pipeline(
                base_url=api_url,
                api_key=api_key,
                turns=turns,
                patient_info=patient_info if patient_info else None,
                generate_pdf=st.session_state["generate_pdf"],
                generate_fhir=st.session_state["generate_fhir"],
                use_llm=use_llm,
            )
            _t.join(timeout=2)
            progress_bar.progress(100, text="Done!")

            if error:
                st.error(f"**Pipeline failed:** {error}")
                progress_bar.empty()
            else:
                st.session_state["pipeline_result"] = result

                # Pre-load FHIR bundle from disk if path was returned
                fhir_path = result.get("fhir_bundle_path", "")
                if fhir_path:
                    raw_bytes = _load_file_bytes(fhir_path)
                    if raw_bytes:
                        try:
                            st.session_state["fhir_bundle"] = json.loads(raw_bytes.decode())
                        except Exception:
                            st.session_state["fhir_bundle"] = None
                else:
                    st.session_state["fhir_bundle"] = None

                elapsed = result.get("processing_time_seconds", 0.0)
                warnings = result.get("warnings", [])
                status_msg = (
                    f"Pipeline completed in **{elapsed:.2f}s** "
                    f"— status: **{result.get('status', '?')}**"
                )
                if warnings:
                    status_msg += f"  ⚠️ {len(warnings)} warning(s)"
                status_placeholder.success(status_msg)

                if warnings:
                    with st.expander(f"⚠️ {len(warnings)} pipeline warning(s)", expanded=False):
                        for w in warnings:
                            st.caption(f"• {w}")

                st.info("Results are ready — switch to the **🔬 Results** tab.")


# ============================================================================
# TAB 2 -- RESULTS
# ============================================================================

with tab_results:
    _result = st.session_state.get("pipeline_result")

    if _result is None:
        st.info("Run the pipeline on the **📋 Input** tab first.")
    else:
        soap = _result.get("soap_note", {})
        rfr = _result.get("red_flags", {})
        ddx = _result.get("ddx", {})

        # ── Section A: SOAP Note ─────────────────────────────────────────
        st.header("SOAP Note")

        m1, m2, m3 = st.columns(3)
        m1.metric("Speciality", soap.get("speciality") or "—")
        m2.metric("Confidence", f"{int(soap.get('confidence', 0) * 100)}%")
        m3.metric("Method", soap.get("generation_method") or "—")

        for _icon, _label, _field in [
            ("📝", "Subjective", "subjective"),
            ("🔍", "Objective", "objective"),
            ("🧠", "Assessment", "assessment"),
            ("📋", "Plan", "plan"),
        ]:
            with st.expander(f"{_icon} {_label}", expanded=True):
                _text = soap.get(_field, "")
                st.markdown(_text if _text else "*No content available.*")

        st.divider()

        # ── Section B: Red Flags + Differential Diagnosis ────────────────
        col_flags, col_ddx = st.columns(2, gap="large")

        with col_flags:
            st.subheader("🚨 Red Flag Alerts")
            flags = rfr.get("flags", [])

            if rfr.get("requires_immediate_action"):
                st.error("⚠️ **IMMEDIATE ACTION REQUIRED** — critical red flag detected.")

            if not flags:
                st.success("✅ No red flags detected.")
            else:
                for flag in flags:
                    sev = flag.get("severity", "monitor")
                    body = (
                        f"**{flag.get('term', '')}** — {flag.get('description', '')}\n\n"
                        f"*Recommended action: {flag.get('recommended_action', '')}*"
                    )
                    if sev == "critical":
                        st.error(f"🔴 CRITICAL | {body}")
                    elif sev == "urgent":
                        st.warning(f"🟠 URGENT | {body}")
                    else:
                        st.info(f"🟡 MONITOR | {body}")

        with col_ddx:
            st.subheader("🔬 Differential Diagnosis")
            diagnoses = ddx.get("top_diagnoses", [])[:5]

            if not diagnoses:
                st.info("No differential diagnoses available.")
            else:
                for i, dx in enumerate(diagnoses, 1):
                    prob_score = float(dx.get("probability_score", 0.0))
                    prob_label = dx.get("probability", "low")
                    icd = dx.get("icd10_code", "")
                    evidence = dx.get("supporting_evidence", [])

                    _prob_color = {
                        "high": "#065f46",
                        "moderate": "#92400e",
                        "low": "#6b7280",
                    }.get(prob_label, "#6b7280")

                    with st.container(border=True):
                        hdr_c, prob_c = st.columns([3, 1])
                        hdr_c.markdown(
                            f"**{i}. {dx.get('name', 'Unknown')}**  \n"
                            f"<span style='color:grey;font-size:0.85em'>{icd}</span>",
                            unsafe_allow_html=True,
                        )
                        prob_c.markdown(
                            f"<div style='text-align:right;font-weight:700;"
                            f"color:{_prob_color}'>{prob_label.upper()}</div>",
                            unsafe_allow_html=True,
                        )
                        st.progress(min(int(prob_score * 100), 100))
                        if evidence:
                            st.caption("Evidence: " + ", ".join(evidence[:4]))

        st.divider()

        # ── Section C: Follow-up guidance ───────────────────────────────
        st.header("Follow-up Guidance")
        col_fup, col_inv = st.columns(2, gap="large")

        with col_fup:
            st.subheader("❓ Suggested Follow-up Questions")
            for q in ddx.get("follow_up_questions", []):
                st.markdown(f"• {q}")
            if not ddx.get("follow_up_questions"):
                st.caption("None identified.")

        with col_inv:
            st.subheader("🔬 Recommended Investigations")
            for inv in ddx.get("recommended_investigations", []):
                st.markdown(f"• {inv}")
            if not ddx.get("recommended_investigations"):
                st.caption("None identified.")


# ============================================================================
# TAB 3 -- REPORTS
# ============================================================================

with tab_reports:
    _result = st.session_state.get("pipeline_result")

    if _result is None:
        st.info("Run the pipeline on the **📋 Input** tab first.")
    else:
        pdf_paths = _result.get("pdf_paths", {})
        fhir_path = _result.get("fhir_bundle_path", "")
        gen_method = _result.get("soap_note", {}).get("generation_method", "—")

        col_pdf, col_fhir = st.columns(2, gap="large")

        # ── Left: physician + patient reports ────────────────────────────
        with col_pdf:
            st.subheader("📋 Physician Report")

            for _label, _path in [
                ("Physician", pdf_paths.get("physician", "")),
                ("Patient Summary", pdf_paths.get("patient", "")),
            ]:
                if not _path:
                    st.caption(f"{_label} report: *not generated*")
                    continue

                file_bytes = _load_file_bytes(_path)
                is_pdf = _path.endswith(".pdf")
                mime = "application/pdf" if is_pdf else "text/html"
                fname = (
                    f"{_label.lower().replace(' ', '_')}_report"
                    f".{'pdf' if is_pdf else 'html'}"
                )

                if file_bytes:
                    kb = len(file_bytes) / 1024
                    st.caption(
                        f"**{_label}** — {kb:.1f} KB  "
                        f"({'PDF' if is_pdf else 'HTML — WeasyPrint unavailable'})"
                    )
                    st.download_button(
                        label=f"⬇️ Download {_label} Report",
                        data=file_bytes,
                        file_name=fname,
                        mime=mime,
                        use_container_width=True,
                    )
                    if not is_pdf:
                        with st.expander(f"🔍 Preview {_label} HTML"):
                            html_str = file_bytes.decode("utf-8", errors="replace")
                            st.components.v1.html(html_str, height=420, scrolling=True)
                else:
                    st.caption(
                        f"{_label} report generated at:\n`{_path}`\n"
                        "(not accessible from demo host)"
                    )

            st.caption(f"Generation method: `{gen_method}`")

        # ── Right: FHIR bundle ────────────────────────────────────────────
        with col_fhir:
            st.subheader("📦 FHIR R4 Bundle")

            bundle: dict | None = st.session_state.get("fhir_bundle")

            # Lazy-load from disk on first visit to this tab
            if bundle is None and fhir_path:
                raw = _load_file_bytes(fhir_path)
                if raw:
                    try:
                        bundle = json.loads(raw.decode())
                        st.session_state["fhir_bundle"] = bundle
                    except Exception:
                        bundle = None

            if bundle:
                # Inline validation
                try:
                    import sys
                    from pathlib import Path as _P
                    _root = str(_P(__file__).resolve().parents[3])
                    if _root not in sys.path:
                        sys.path.insert(0, _root)
                    from medai.src.reporting.fhir_formatter import FHIRFormatter
                    val = FHIRFormatter().validate_bundle(bundle)
                    if val["valid"]:
                        st.success("✅ Bundle validation passed")
                    else:
                        st.error("❌ Bundle validation failed")
                        for err in val["errors"]:
                            st.caption(f"• {err}")
                    if val.get("warnings"):
                        for w in val["warnings"]:
                            st.caption(f"⚠️ {w}")
                except Exception:
                    pass

                # Resource summary
                rt_counts: dict[str, int] = {}
                for entry in bundle.get("entry", []):
                    rt = entry.get("resource", {}).get("resourceType", "Unknown")
                    rt_counts[rt] = rt_counts.get(rt, 0) + 1
                st.caption(
                    f"**{len(bundle.get('entry', []))} resources**: "
                    + ", ".join(f"{v}× {k}" for k, v in rt_counts.items())
                )

                st.download_button(
                    label="⬇️ Download FHIR JSON",
                    data=json.dumps(bundle, indent=2).encode(),
                    file_name="fhir_bundle.json",
                    mime="application/json",
                    use_container_width=True,
                )
                with st.expander("🔍 View FHIR Bundle JSON"):
                    st.json(bundle)

            elif fhir_path:
                st.caption(
                    f"FHIR bundle saved to:\n`{fhir_path}`\n"
                    "(not accessible from demo host)"
                )
            else:
                st.caption("FHIR bundle not generated for this run.")

        # ── Evaluation section (shown only when reference note present) ───
        ref_note: str | None = st.session_state.get("reference_note")
        soap = _result.get("soap_note", {})

        if ref_note and soap:
            st.divider()
            st.subheader("📊 Evaluation vs Reference Note (MTS-Dialog)")

            generated = " ".join([
                soap.get("subjective", ""),
                soap.get("objective", ""),
                soap.get("assessment", ""),
                soap.get("plan", ""),
            ]).lower().split()

            ref_words = set(ref_note.lower().split())
            gen_words = set(generated)

            inter = ref_words & gen_words
            union = ref_words | gen_words

            jaccard = len(inter) / len(union) if union else 0.0
            recall = len(inter) / len(ref_words) if ref_words else 0.0

            ec1, ec2 = st.columns(2)
            ec1.metric(
                "Jaccard Similarity",
                f"{jaccard:.3f}",
                delta=f"{jaccard - 0.15:+.3f} vs 0.15 baseline",
            )
            ec2.metric(
                "Token Recall",
                f"{recall:.3f}",
                delta=f"{recall - 0.30:+.3f} vs 0.30 baseline",
            )
