"""
run_pipeline_test.py

End-to-end pipeline test script.

Loads up to 10 MTS-Dialog records from data/raw/unified/en_mts.jsonl,
runs the full pipeline chain on each, and prints a summary table plus
saves a JSON results file.

Usage::

    python scripts/run_pipeline_test.py
    python scripts/run_pipeline_test.py --n 5
    python scripts/run_pipeline_test.py --out data/outputs/my_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _jaccard(ref: str, gen: str) -> float:
    """Compute token-level Jaccard similarity between two strings."""
    ref_words = set(ref.lower().split())
    gen_words = set(gen.lower().split())
    if not ref_words and not gen_words:
        return 1.0
    inter = ref_words & gen_words
    union = ref_words | gen_words
    return len(inter) / len(union) if union else 0.0


def _col(text: str, width: int) -> str:
    """Left-pad/truncate *text* to *width* characters."""
    s = str(text)
    return s[:width].ljust(width)


def main(n: int = 10, out_path: str | None = None) -> None:
    """
    Run end-to-end pipeline on *n* MTS-Dialog records and print a summary.

    Parameters
    ----------
    n:
        Maximum number of records to process.
    out_path:
        Path to write the JSON results file.  If ``None``, uses
        ``data/outputs/e2e_test_results.json``.
    """
    data_path = _REPO_ROOT / "data/raw/unified/en_mts.jsonl"
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Run the data acquisition step first.")
        sys.exit(1)

    # ── Load records ─────────────────────────────────────────────────────────
    records: list[dict] = []
    with open(data_path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                records.append(json.loads(s))
            if len(records) >= n:
                break
    print(f"Loaded {len(records)} record(s) from {data_path.name}\n")

    # ── Import pipeline components ────────────────────────────────────────────
    print("Initialising pipeline components…")
    from medai.src.voice.transcript_processor import TranscriptProcessor
    from medai.src.clinical.entity_extractor import EntityExtractor
    from medai.src.clinical.soap_generator import SOAPGenerator
    from medai.src.reasoning.red_flag_detector import RedFlagDetector
    from medai.src.reasoning.ddx_engine import DDxEngine

    processor = TranscriptProcessor()
    extractor = EntityExtractor()
    soap_gen = SOAPGenerator()
    red_flag = RedFlagDetector()
    ddx_eng = DDxEngine()
    print("All components ready.\n")

    # ── Table header ─────────────────────────────────────────────────────────
    header = (
        f"{'ID':<22} {'Speciality':<20} {'Flags':<6} "
        f"{'DDx Top-1':<28} {'Jaccard':>8} {'Method':<16}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    results: list[dict] = []

    for record in records:
        rec_id = record.get("id", "?")
        t0 = time.perf_counter()

        try:
            transcript = processor.process(record)
            entities = extractor.extract(transcript)
            soap = soap_gen.generate(transcript, entities)
            rfr = red_flag.detect(transcript, entities, soap)
            ddx = ddx_eng.analyse(transcript, entities, soap, rfr)
        except Exception as exc:
            print(f"{'ERROR':<22} {rec_id}: {exc}")
            results.append({"id": rec_id, "error": str(exc)})
            continue

        elapsed = time.perf_counter() - t0

        # Evaluation metrics (Jaccard vs reference note)
        ref = transcript.reference_note or ""
        gen_text = " ".join([soap.subjective, soap.objective, soap.assessment, soap.plan])
        jac = _jaccard(ref, gen_text) if ref else float("nan")

        top1 = ddx.primary_diagnosis.name if ddx.primary_diagnosis else "—"
        flags = rfr.flag_count
        method = soap.generation_method

        print(
            f"{_col(rec_id, 22)} {_col(soap.speciality, 20)} {flags:<6} "
            f"{_col(top1, 28)} "
            f"{'N/A':>8}" if not ref else
            f"{_col(rec_id, 22)} {_col(soap.speciality, 20)} {flags:<6} "
            f"{_col(top1, 28)} {jac:>8.3f} {method:<16}",
            end="",
        )
        # Re-print cleanly
        print()

        results.append({
            "id": rec_id,
            "speciality": soap.speciality,
            "flag_count": flags,
            "highest_severity": rfr.highest_severity,
            "requires_immediate_action": rfr.requires_immediate_action,
            "primary_diagnosis": top1,
            "primary_icd10": ddx.primary_diagnosis.icd10_code if ddx.primary_diagnosis else "",
            "generation_method": method,
            "jaccard": round(jac, 4) if ref else None,
            "elapsed_seconds": round(elapsed, 3),
        })

    print(sep)
    print(f"\n{len(results)} records processed.\n")

    # ── Summary stats ─────────────────────────────────────────────────────────
    jac_vals = [r["jaccard"] for r in results if r.get("jaccard") is not None]
    if jac_vals:
        avg_jac = sum(jac_vals) / len(jac_vals)
        print(f"Average Jaccard: {avg_jac:.3f}  (over {len(jac_vals)} records with reference notes)")

    flag_records = sum(1 for r in results if r.get("flag_count", 0) > 0)
    print(f"Records with red flags: {flag_records}/{len(results)}")

    methods: dict[str, int] = {}
    for r in results:
        m = r.get("generation_method", "unknown")
        methods[m] = methods.get(m, 0) + 1
    print(f"Generation methods: {methods}")

    # ── Save results ──────────────────────────────────────────────────────────
    if out_path is None:
        out_dir = _REPO_ROOT / "data/outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / "e2e_test_results.json")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"records": results, "count": len(results)}, fh, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedAI end-to-end pipeline test")
    parser.add_argument(
        "--n", type=int, default=10, help="Number of records to process (default: 10)"
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Output JSON path (default: data/outputs/e2e_test_results.json)"
    )
    args = parser.parse_args()
    main(n=args.n, out_path=args.out)
