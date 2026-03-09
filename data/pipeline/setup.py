"""
MedAI — Step 0: Data Acquisition and Preparation Pipeline

Downloads and normalises doctor-patient conversation datasets into a unified
JSONL schema ready for the AI pipeline.

Usage:
    python data/pipeline/setup.py

Outputs:
    data/raw/unified/en_mts.jsonl
    data/raw/unified/en_notechat.jsonl
    data/raw/unified/ar_bimed.jsonl
    data/raw/unified/en_bimed.jsonl
    data/raw/samples/<dataset>_samples.jsonl
    data/processed/dataset_summary.json
"""

import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — adjust sample sizes and paths here
# ---------------------------------------------------------------------------

CONFIG = {
    # HuggingFace dataset identifiers
    "mts_dialog_hf_id": "har1/MTS_Dialogue-Clinical_Note",
    "notechat_hf_id": "akemiH/NoteChat",
    "bimed_hf_id": "BiMediX/BiMed1.3M",

    # Sample sizes
    "notechat_sample_size": 500,
    "bimed_ar_sample_size": 300,
    "bimed_en_sample_size": 200,

    # How many examples to save as samples for manual review
    "sample_preview_count": 5,

    # Minimum requirements for a conversation to be kept
    "min_turns": 4,
    "min_words_per_turn": 3,

    # Output paths (relative to repo root)
    "output_dir": "data/raw/unified",
    "samples_dir": "data/raw/samples",
    "processed_dir": "data/processed",

    # Random seed for reproducible sampling
    "random_seed": 42,
}

# ---------------------------------------------------------------------------
# Paths — resolve relative to repo root (two levels up from this file)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

OUTPUT_DIR = REPO_ROOT / CONFIG["output_dir"]
SAMPLES_DIR = REPO_ROOT / CONFIG["samples_dir"]
PROCESSED_DIR = REPO_ROOT / CONFIG["processed_dir"]

for _d in (OUTPUT_DIR, SAMPLES_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("medai.setup")

# ---------------------------------------------------------------------------
# Unified schema helpers
# ---------------------------------------------------------------------------

# Patterns that suggest a token is a DOCTOR label
_DOCTOR_TOKENS = re.compile(
    r"^(doctor|dr\.?|physician|clinician|provider|interviewer|"
    r"therapist|counselor|nurse|practitioner|دكتور|طبيب|physician)$",
    re.IGNORECASE,
)
# Patterns that suggest a token is a PATIENT label
_PATIENT_TOKENS = re.compile(
    r"^(patient|pt\.?|client|interviewee|user|subject|مريض|مريضة)$",
    re.IGNORECASE,
)


def normalise_speaker(raw_label: str) -> str | None:
    """Return 'DOCTOR', 'PATIENT', or None if unrecognisable."""
    cleaned = raw_label.strip().rstrip(":").strip()
    if _DOCTOR_TOKENS.match(cleaned):
        return "DOCTOR"
    if _PATIENT_TOKENS.match(cleaned):
        return "PATIENT"
    return None


def word_count(text: str) -> int:
    return len(text.split())


def clean_turns(raw_turns: list[dict]) -> list[dict]:
    """
    Filter out turns that are empty or have fewer than min_words_per_turn words.
    raw_turns items must already have keys 'speaker' and 'text'.
    """
    min_w = CONFIG["min_words_per_turn"]
    return [
        t for t in raw_turns
        if t.get("text", "").strip() and word_count(t["text"]) >= min_w
    ]


def make_record(
    idx: int,
    prefix: str,
    source_dataset: str,
    language: str,
    turns: list[dict],
    speciality: str | None = None,
    reference_note: str | None = None,
    is_synthetic: bool = True,
) -> dict:
    return {
        "id": f"{prefix}_{idx:04d}",
        "source_dataset": source_dataset,
        "language": language,
        "speciality": speciality,
        "turns": turns,
        "reference_note": reference_note,
        "is_synthetic": is_synthetic,
        "turn_count": len(turns),
    }


# ---------------------------------------------------------------------------
# Discard tracking
# ---------------------------------------------------------------------------

class DiscardTracker:
    def __init__(self):
        self.counts: dict[str, Counter] = defaultdict(Counter)

    def record(self, dataset: str, reason: str, n: int = 1):
        self.counts[dataset][reason] += n

    def total(self, dataset: str) -> int:
        return sum(self.counts[dataset].values())

    def report(self) -> dict:
        return {ds: dict(reasons) for ds, reasons in self.counts.items()}


discards = DiscardTracker()

# ---------------------------------------------------------------------------
# Dataset processors
# ---------------------------------------------------------------------------

def _load_hf_dataset(hf_id: str, **kwargs):
    """Load a HuggingFace dataset, returning None on failure."""
    try:
        from datasets import load_dataset  # noqa: PLC0415
        log.info("Downloading %s …", hf_id)
        ds = load_dataset(hf_id, **kwargs)
        log.info("Downloaded %s  — splits: %s", hf_id, list(ds.keys()) if hasattr(ds, "keys") else "N/A")
        return ds
    except Exception as exc:
        log.error("Failed to download %s: %s", hf_id, exc)
        return None


# ── MTS-Dialog ──────────────────────────────────────────────────────────────

def _parse_mts_dialogue(raw_text: str) -> list[dict]:
    """
    MTS-Dialog dialogues are formatted as interleaved lines like:
        Doctor: How can I help you today?
        Patient: I've had a headache for two days.
    Some lines may begin with 'D:' / 'P:' abbreviations.
    """
    turns = []
    abbrev_map = {"d": "DOCTOR", "p": "PATIENT"}

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Try "Label: text" pattern
        m = re.match(r"^([^:]{1,30}):\s*(.+)$", line)
        if m:
            raw_label, text = m.group(1).strip(), m.group(2).strip()
            speaker = normalise_speaker(raw_label)
            if speaker is None:
                # Try single-letter abbreviation
                speaker = abbrev_map.get(raw_label.lower())
            if speaker and text:
                turns.append({"speaker": speaker, "text": text})
        # Lines without a colon are appended to the last turn's text
        elif turns:
            turns[-1]["text"] += " " + line

    return turns


def process_mts_dialog() -> list[dict]:
    """Download and normalise MTS-Dialog."""
    ds = _load_hf_dataset(CONFIG["mts_dialog_hf_id"])
    if ds is None:
        return []

    records = []
    idx = 1
    dataset_name = "mts_dialog"

    # Collect all splits
    all_rows = []
    for split_name in ds.keys():
        all_rows.extend(ds[split_name])

    for row in all_rows:
        # Field names vary slightly across HF versions — try both
        dialogue_text = row.get("dialogue") or row.get("Dialogue") or row.get("conversation") or ""
        note_text = row.get("section_text") or row.get("note") or row.get("Note") or ""
        speciality = row.get("speciality") or row.get("specialty") or row.get("Specialty") or None
        if speciality:
            speciality = str(speciality).strip() or None

        if not dialogue_text:
            discards.record(dataset_name, "empty_dialogue")
            continue

        raw_turns = _parse_mts_dialogue(str(dialogue_text))
        raw_turns = clean_turns(raw_turns)

        if len(raw_turns) < CONFIG["min_turns"]:
            discards.record(dataset_name, "too_few_turns")
            continue

        records.append(make_record(
            idx=idx,
            prefix="mts",
            source_dataset=dataset_name,
            language="en",
            turns=raw_turns,
            speciality=speciality if speciality else None,
            reference_note=str(note_text).strip() if note_text else None,
            is_synthetic=False,
        ))
        idx += 1

    log.info("MTS-Dialog: %d records kept, %d discarded", len(records), discards.total(dataset_name))
    return records


# ── NoteChat ────────────────────────────────────────────────────────────────

def _parse_notechat_conversation(raw) -> list[dict]:
    """
    NoteChat conversations may be stored as:
      - A JSON string of list[dict] with 'role'/'content' keys
      - A list of dicts
      - A plain string
    """
    turns = []
    role_map = {
        "doctor": "DOCTOR",
        "physician": "DOCTOR",
        "assistant": "DOCTOR",
        "patient": "PATIENT",
        "user": "PATIENT",
    }

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            # Plain text — try line-by-line parsing
            for line in raw.splitlines():
                m = re.match(r"^([^:]{1,30}):\s*(.+)$", line.strip())
                if m:
                    speaker = normalise_speaker(m.group(1))
                    if speaker:
                        turns.append({"speaker": speaker, "text": m.group(2).strip()})
            return turns

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                role = str(item.get("role", item.get("speaker", ""))).lower().strip()
                text = str(item.get("content", item.get("text", ""))).strip()
                speaker = role_map.get(role) or normalise_speaker(role)
                if speaker and text:
                    turns.append({"speaker": speaker, "text": text})

    return turns


def process_notechat() -> list[dict]:
    """Download and normalise a sample of NoteChat."""
    ds = _load_hf_dataset(CONFIG["notechat_hf_id"])
    if ds is None:
        return []

    dataset_name = "notechat"
    random.seed(CONFIG["random_seed"])

    # Flatten splits
    all_rows = []
    for split_name in ds.keys():
        for row in ds[split_name]:
            all_rows.append(row)

    # Shuffle for diversity before sampling
    random.shuffle(all_rows)

    records = []
    idx = 1
    attempted = 0

    for row in all_rows:
        if len(records) >= CONFIG["notechat_sample_size"]:
            break
        attempted += 1

        conv_raw = (
            row.get("conversation")
            or row.get("dialogue")
            or row.get("messages")
            or row.get("chat")
            or row.get("text")
        )
        if conv_raw is None:
            discards.record(dataset_name, "no_conversation_field")
            continue

        speciality = row.get("speciality") or row.get("specialty") or row.get("department") or None
        if speciality:
            speciality = str(speciality).strip() or None

        raw_turns = _parse_notechat_conversation(conv_raw)
        raw_turns = clean_turns(raw_turns)

        if len(raw_turns) < CONFIG["min_turns"]:
            discards.record(dataset_name, "too_few_turns")
            continue

        records.append(make_record(
            idx=idx,
            prefix="notechat",
            source_dataset=dataset_name,
            language="en",
            turns=raw_turns,
            speciality=speciality,
            reference_note=None,
            is_synthetic=True,
        ))
        idx += 1

    log.info(
        "NoteChat: %d records kept from %d attempted, %d discarded",
        len(records), attempted, discards.total(dataset_name),
    )
    return records


# ── BiMed1.3M ───────────────────────────────────────────────────────────────

def _parse_bimed_conversation(raw) -> list[dict]:
    """
    BiMed1.3M stores conversations as a list of dicts with keys like:
      {"from": "human"/"gpt", "value": "..."}
    or with role/content keys.
    """
    turns = []
    from_map = {
        "human": "PATIENT",
        "user": "PATIENT",
        "patient": "PATIENT",
        "مريض": "PATIENT",
        "gpt": "DOCTOR",
        "assistant": "DOCTOR",
        "doctor": "DOCTOR",
        "physician": "DOCTOR",
        "طبيب": "DOCTOR",
        "دكتور": "DOCTOR",
    }

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return turns

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            from_val = str(item.get("from", item.get("role", item.get("speaker", "")))).lower().strip()
            text = str(item.get("value", item.get("content", item.get("text", "")))).strip()
            speaker = from_map.get(from_val) or normalise_speaker(from_val)
            if speaker and text:
                turns.append({"speaker": speaker, "text": text})

    return turns


def _detect_language(text: str) -> str:
    """Heuristic: if >15% of chars are Arabic script, label as 'ar'."""
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    return "ar" if arabic_chars / max(len(text), 1) > 0.15 else "en"


def process_bimed() -> tuple[list[dict], list[dict]]:
    """
    Download BiMed1.3M and return (arabic_records, english_records).
    We only use the 'chat' / conversation subset (not MCQA).
    """
    dataset_name = "bimed"
    random.seed(CONFIG["random_seed"])

    # Try loading the chat/conversation subset explicitly
    ds = None
    for subset in ("chat", "conversation", "dialogues", None):
        kwargs = {}
        if subset:
            kwargs["name"] = subset
        ds = _load_hf_dataset(CONFIG["bimed_hf_id"], **kwargs)
        if ds is not None:
            break

    if ds is None:
        log.error("Could not load BiMed1.3M — skipping.")
        return [], []

    # Flatten all splits
    all_rows = []
    for split_name in ds.keys():
        for row in ds[split_name]:
            all_rows.append(row)

    log.info("BiMed1.3M: %d total rows loaded (pre-filter)", len(all_rows))

    # Filter to conversation-type rows only (exclude MCQA)
    conversation_rows = []
    for row in all_rows:
        # MCQA rows typically have 'question'/'choices' keys
        if "choices" in row or "options" in row:
            continue
        # Must have some kind of conversation field
        conv_field = (
            row.get("conversations")
            or row.get("conversation")
            or row.get("dialogue")
            or row.get("messages")
            or row.get("chat")
        )
        if conv_field is not None:
            row["_conv_field"] = conv_field
            conversation_rows.append(row)

    log.info("BiMed1.3M: %d conversation rows after MCQA filter", len(conversation_rows))
    random.shuffle(conversation_rows)

    ar_records = []
    en_records = []
    ar_idx = en_idx = 1

    for row in conversation_rows:
        if (
            len(ar_records) >= CONFIG["bimed_ar_sample_size"]
            and len(en_records) >= CONFIG["bimed_en_sample_size"]
        ):
            break

        conv_raw = row["_conv_field"]
        raw_turns = _parse_bimed_conversation(conv_raw)
        raw_turns = clean_turns(raw_turns)

        if len(raw_turns) < CONFIG["min_turns"]:
            discards.record(dataset_name, "too_few_turns")
            continue

        # Detect language from concatenated turn texts
        all_text = " ".join(t["text"] for t in raw_turns)
        lang = row.get("language") or row.get("lang") or _detect_language(all_text)
        if isinstance(lang, str):
            lang = "ar" if lang.lower() in ("ar", "arabic", "عربي") else "en"

        speciality = row.get("speciality") or row.get("specialty") or row.get("category") or None
        if speciality:
            speciality = str(speciality).strip() or None

        if lang == "ar" and len(ar_records) < CONFIG["bimed_ar_sample_size"]:
            ar_records.append(make_record(
                idx=ar_idx,
                prefix="bimed_ar",
                source_dataset=dataset_name,
                language="ar",
                turns=raw_turns,
                speciality=speciality,
                reference_note=None,
                is_synthetic=True,
            ))
            ar_idx += 1
        elif lang == "en" and len(en_records) < CONFIG["bimed_en_sample_size"]:
            en_records.append(make_record(
                idx=en_idx,
                prefix="bimed_en",
                source_dataset=dataset_name,
                language="en",
                turns=raw_turns,
                speciality=speciality,
                reference_note=None,
                is_synthetic=True,
            ))
            en_idx += 1

    log.info(
        "BiMed1.3M: %d Arabic, %d English records kept, %d discarded",
        len(ar_records), len(en_records), discards.total(dataset_name),
    )
    return ar_records, en_records


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Saved %d records → %s", len(records), path.relative_to(REPO_ROOT))


def save_samples(records: list[dict], tag: str) -> None:
    """Save CONFIG['sample_preview_count'] random records and print them."""
    random.seed(CONFIG["random_seed"])
    n = min(CONFIG["sample_preview_count"], len(records))
    samples = random.sample(records, n)

    out_path = SAMPLES_DIR / f"{tag}_samples.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in samples:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Saved %d samples → %s", n, out_path.relative_to(REPO_ROOT))

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  SAMPLES — {tag.upper()}  ({n} of {len(records)} conversations)")
    print(sep)
    for i, rec in enumerate(samples, 1):
        print(f"\n--- Sample {i}/{n}  [{rec['id']}]  lang={rec['language']}"
              f"  speciality={rec['speciality']}  turns={rec['turn_count']} ---")
        for turn in rec["turns"][:6]:  # show first 6 turns only for readability
            print(f"  [{turn['speaker']}] {turn['text'][:200]}")
        if rec["turn_count"] > 6:
            print(f"  … (+{rec['turn_count'] - 6} more turns)")
        if rec["reference_note"]:
            snippet = rec["reference_note"][:300].replace("\n", " ")
            print(f"\n  [REFERENCE NOTE] {snippet}…")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def build_summary(
    mts: list[dict],
    notechat: list[dict],
    bimed_ar: list[dict],
    bimed_en: list[dict],
) -> dict:
    all_groups = {
        "mts_dialog": mts,
        "notechat": notechat,
        "bimed_ar": bimed_ar,
        "bimed_en": bimed_en,
    }

    per_source: dict[str, dict] = {}
    for name, recs in all_groups.items():
        if not recs:
            per_source[name] = {"count": 0}
            continue
        turn_counts = [r["turn_count"] for r in recs]
        with_note = sum(1 for r in recs if r.get("reference_note"))
        spec_counter: Counter = Counter()
        for r in recs:
            if r.get("speciality"):
                spec_counter[r["speciality"]] += 1
        per_source[name] = {
            "count": len(recs),
            "language": recs[0]["language"] if recs else None,
            "is_synthetic": recs[0]["is_synthetic"] if recs else None,
            "avg_turn_count": round(sum(turn_counts) / len(turn_counts), 2),
            "min_turn_count": min(turn_counts),
            "max_turn_count": max(turn_counts),
            "pct_with_reference_note": round(100 * with_note / len(recs), 1),
            "top_specialities": dict(spec_counter.most_common(10)),
        }

    by_language: dict[str, int] = Counter()
    for recs in all_groups.values():
        for r in recs:
            by_language[r["language"]] += 1

    total = sum(len(v) for v in all_groups.values())

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_conversations": total,
        "by_language": dict(by_language),
        "per_source": per_source,
        "discards": discards.report(),
    }


def print_summary(summary: dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  DATASET SUMMARY REPORT")
    print(sep)
    print(f"  Generated : {summary['generated_at']}")
    print(f"  Total     : {summary['total_conversations']} conversations")
    print(f"  Languages : {summary['by_language']}")
    print()

    for src, info in summary["per_source"].items():
        if info.get("count", 0) == 0:
            print(f"  [{src}]  NO DATA")
            continue
        print(f"  [{src.upper()}]")
        print(f"    Count          : {info['count']}")
        print(f"    Language       : {info['language']}")
        print(f"    Synthetic      : {info['is_synthetic']}")
        print(f"    Avg turns      : {info['avg_turn_count']}"
              f"  (min {info['min_turn_count']}, max {info['max_turn_count']})")
        print(f"    With note      : {info['pct_with_reference_note']}%")
        if info.get("top_specialities"):
            top = list(info["top_specialities"].items())[:5]
            top_str = ", ".join(f"{k}({v})" for k, v in top)
            print(f"    Top specialities: {top_str}")
        print()

    if summary.get("discards"):
        print("  DISCARDS")
        for ds_name, reasons in summary["discards"].items():
            for reason, count in reasons.items():
                print(f"    {ds_name:<20} {reason:<30} {count}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = time.time()
    log.info("MedAI — Data Acquisition Pipeline starting …")

    # ── MTS-Dialog ──────────────────────────────────────────────────────────
    log.info("── Processing MTS-Dialog ──")
    mts_records = process_mts_dialog()
    if mts_records:
        save_jsonl(mts_records, OUTPUT_DIR / "en_mts.jsonl")
        save_samples(mts_records, "en_mts")
    else:
        log.warning("No MTS-Dialog records — en_mts.jsonl will be empty.")
        save_jsonl([], OUTPUT_DIR / "en_mts.jsonl")

    # ── NoteChat ────────────────────────────────────────────────────────────
    log.info("── Processing NoteChat ──")
    notechat_records = process_notechat()
    if notechat_records:
        save_jsonl(notechat_records, OUTPUT_DIR / "en_notechat.jsonl")
        save_samples(notechat_records, "en_notechat")
    else:
        log.warning("No NoteChat records — en_notechat.jsonl will be empty.")
        save_jsonl([], OUTPUT_DIR / "en_notechat.jsonl")

    # ── BiMed1.3M ───────────────────────────────────────────────────────────
    log.info("── Processing BiMed1.3M ──")
    bimed_ar, bimed_en = process_bimed()
    if bimed_ar:
        save_jsonl(bimed_ar, OUTPUT_DIR / "ar_bimed.jsonl")
        save_samples(bimed_ar, "ar_bimed")
    else:
        log.warning("No BiMed Arabic records — ar_bimed.jsonl will be empty.")
        save_jsonl([], OUTPUT_DIR / "ar_bimed.jsonl")

    if bimed_en:
        save_jsonl(bimed_en, OUTPUT_DIR / "en_bimed.jsonl")
        save_samples(bimed_en, "en_bimed")
    else:
        log.warning("No BiMed English records — en_bimed.jsonl will be empty.")
        save_jsonl([], OUTPUT_DIR / "en_bimed.jsonl")

    # ── Summary ─────────────────────────────────────────────────────────────
    log.info("── Generating summary ──")
    summary = build_summary(mts_records, notechat_records, bimed_ar, bimed_en)

    summary_path = PROCESSED_DIR / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    log.info("Summary saved → %s", summary_path.relative_to(REPO_ROOT))

    print_summary(summary)

    elapsed = time.time() - start
    log.info("Pipeline complete in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
