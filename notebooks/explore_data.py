"""
explore_data.py — MedAI Dataset Exploration Script

Loads the Step 0 unified JSONL datasets and produces:
  - Schema validation report
  - Conversation statistics
  - Formatted content samples
  - Anomaly detection results
  - data/processed/exploration_report.json

Usage:
    python notebooks/explore_data.py
"""

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DATASETS = {
    "mts_dialog":  REPO_ROOT / "data/raw/unified/en_mts.jsonl",
    "notechat":    REPO_ROOT / "data/raw/unified/en_notechat.jsonl",
}

REPORT_PATH = REPO_ROOT / "data/processed/exploration_report.json"

RANDOM_SEED = 42
SAMPLES_PER_DATASET = 3
MAX_TURN_DISPLAY_WORDS = 60   # truncate long turns when printing samples
LONG_TURN_WORD_THRESHOLD = 500
CONSECUTIVE_SPEAKER_THRESHOLD = 3
MIN_TURNS = 4

# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

_WIDTH = 72

def section(title: str) -> None:
    print(f"\n{'=' * _WIDTH}")
    print(f"  {title}")
    print(f"{'=' * _WIDTH}")

def subsection(title: str) -> None:
    print(f"\n  ── {title} ──")

def bullet(label: str, value, indent: int = 4) -> None:
    pad = " " * indent
    print(f"{pad}{label:<40} {value}")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict] | None:
    """Return list of records, or None if the file doesn't exist / is empty."""
    if not path.exists():
        print(f"  [MISSING] {path.relative_to(REPO_ROOT)} — skipping.")
        return None
    records = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [WARN] {path.name} line {lineno}: JSON error — {exc}")
    if not records:
        print(f"  [EMPTY] {path.relative_to(REPO_ROOT)} — no records found.")
        return None
    return records

# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"id", "language", "source_dataset", "turns", "is_synthetic"}
VALID_SPEAKERS = {"DOCTOR", "PATIENT"}


def validate_schema(records: list[dict], dataset_name: str) -> dict:
    missing_field_counts: Counter = Counter()
    bad_speaker_convs: list[str] = []
    bad_speaker_total = 0

    for rec in records:
        # Check required top-level fields
        for field in REQUIRED_FIELDS:
            if field not in rec:
                missing_field_counts[field] += 1

        # Check speaker labels inside turns
        rec_id = rec.get("id", "?")
        turns = rec.get("turns", [])
        bad_in_this = [
            t["speaker"]
            for t in turns
            if isinstance(t, dict) and t.get("speaker") not in VALID_SPEAKERS
        ]
        if bad_in_this:
            bad_speaker_total += len(bad_in_this)
            bad_speaker_convs.append(rec_id)

    result = {
        "total": len(records),
        "records_missing_any_field": sum(
            1 for rec in records
            if any(f not in rec for f in REQUIRED_FIELDS)
        ),
        "missing_field_counts": dict(missing_field_counts),
        "conversations_with_bad_speakers": len(bad_speaker_convs),
        "total_bad_speaker_turns": bad_speaker_total,
        "example_bad_speaker_ids": bad_speaker_convs[:5],
    }

    subsection(f"Schema Validation — {dataset_name}")
    bullet("Total records", result["total"])
    bullet("Records missing ≥1 required field", result["records_missing_any_field"])
    if missing_field_counts:
        for field, cnt in missing_field_counts.items():
            bullet(f"  Missing '{field}'", cnt)
    else:
        bullet("  All required fields present", "✓")
    bullet("Conversations with non-standard speaker labels", result["conversations_with_bad_speakers"])
    if bad_speaker_total:
        bullet("  Total bad-speaker turns", bad_speaker_total)

    return result


# ---------------------------------------------------------------------------
# Conversation statistics
# ---------------------------------------------------------------------------

def _turn_bucket(n: int) -> str:
    if n <= 5:   return "1–5"
    if n <= 10:  return "6–10"
    if n <= 15:  return "11–15"
    if n <= 20:  return "16–20"
    return "20+"


def compute_stats(records: list[dict], dataset_name: str) -> dict:
    turn_counts = []
    doctor_first = 0
    has_reference_note = 0
    bucket_counter: Counter = Counter()

    for rec in records:
        turns = rec.get("turns", [])
        n = len(turns)
        turn_counts.append(n)
        bucket_counter[_turn_bucket(n)] += 1

        if turns and turns[0].get("speaker") == "DOCTOR":
            doctor_first += 1
        if rec.get("reference_note"):
            has_reference_note += 1

    total = len(records)
    avg = statistics.mean(turn_counts) if turn_counts else 0
    med = statistics.median(turn_counts) if turn_counts else 0

    result = {
        "avg_turn_count": round(avg, 2),
        "median_turn_count": med,
        "min_turn_count": min(turn_counts) if turn_counts else 0,
        "max_turn_count": max(turn_counts) if turn_counts else 0,
        "turn_count_distribution": dict(bucket_counter),
        "pct_doctor_speaks_first": round(100 * doctor_first / total, 1) if total else 0,
        "pct_with_reference_note": round(100 * has_reference_note / total, 1) if total else 0,
    }

    subsection(f"Conversation Statistics — {dataset_name}")
    bullet("Average turns", result["avg_turn_count"])
    bullet("Median turns", result["median_turn_count"])
    bullet("Min / Max turns", f"{result['min_turn_count']} / {result['max_turn_count']}")
    print()
    print("    Turn count distribution:")
    for bucket in ("1–5", "6–10", "11–15", "16–20", "20+"):
        count = bucket_counter.get(bucket, 0)
        pct = 100 * count / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"      {bucket:>5}  {bar:<30} {count:>5}  ({pct:.1f}%)")
    print()
    bullet("Doctor speaks first (%)", f"{result['pct_doctor_speaks_first']}%")
    bullet("With reference note (%)", f"{result['pct_with_reference_note']}%")

    return result


# ---------------------------------------------------------------------------
# Content samples
# ---------------------------------------------------------------------------

def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def print_samples(records: list[dict], dataset_name: str, n: int = SAMPLES_PER_DATASET) -> None:
    subsection(f"Sample Conversations — {dataset_name}")
    random.seed(RANDOM_SEED)
    samples = random.sample(records, min(n, len(records)))
    dash = "-" * 68

    for i, rec in enumerate(samples, 1):
        print(f"\n  [{i}/{n}] id={rec.get('id')}  "
              f"turns={rec.get('turn_count', len(rec.get('turns', [])))}  "
              f"speciality={rec.get('speciality') or 'N/A'}")
        print(f"  {dash}")
        for turn in rec.get("turns", []):
            speaker = turn.get("speaker", "?")
            text = _truncate_words(turn.get("text", ""), MAX_TURN_DISPLAY_WORDS)
            label = f"  [{speaker}]"
            # Indent continuation lines to align under the first word
            indent = " " * len(label)
            words = text.split()
            lines, line = [], []
            for word in words:
                line.append(word)
                if len(" ".join(line)) > 70:
                    lines.append(" ".join(line[:-1]))
                    line = [word]
            if line:
                lines.append(" ".join(line))
            print(f"{label} {lines[0]}" if lines else f"{label}")
            for cont in lines[1:]:
                print(f"{indent} {cont}")

        ref = rec.get("reference_note")
        if ref:
            snippet = " ".join(ref.split()[:60])
            if len(ref.split()) > 60:
                snippet += " …"
            print(f"\n  [REFERENCE NOTE]\n  {snippet}")
        print(f"  {dash}")


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(records: list[dict], dataset_name: str) -> dict:
    consecutive: list[dict] = []     # same speaker 3+ in a row
    long_turns: list[dict] = []      # any turn > LONG_TURN_WORD_THRESHOLD words
    short_convs: list[dict] = []     # fewer than MIN_TURNS turns

    for rec in records:
        turns = rec.get("turns", [])
        rec_id = rec.get("id", "?")

        # Fewer than MIN_TURNS
        if len(turns) < MIN_TURNS:
            short_convs.append({"id": rec_id, "turn_count": len(turns)})

        # Long turns
        for idx, turn in enumerate(turns):
            wc = len(turn.get("text", "").split())
            if wc > LONG_TURN_WORD_THRESHOLD:
                long_turns.append({
                    "id": rec_id,
                    "turn_index": idx,
                    "speaker": turn.get("speaker"),
                    "word_count": wc,
                    "snippet": " ".join(turn["text"].split()[:20]) + " …",
                })

        # Consecutive same-speaker turns
        run = 1
        for j in range(1, len(turns)):
            if turns[j].get("speaker") == turns[j - 1].get("speaker"):
                run += 1
                if run == CONSECUTIVE_SPEAKER_THRESHOLD:
                    consecutive.append({
                        "id": rec_id,
                        "speaker": turns[j].get("speaker"),
                        "run_length": run,
                        "starting_turn": j - run + 1,
                    })
            else:
                run = 1

    result = {
        "consecutive_speaker_count": len(consecutive),
        "long_turn_count": len(long_turns),
        "short_conv_count": len(short_convs),
        "consecutive_examples": consecutive[:2],
        "long_turn_examples": long_turns[:2],
        "short_conv_examples": short_convs[:2],
    }

    subsection(f"Anomaly Detection — {dataset_name}")

    bullet(f"Same speaker ≥{CONSECUTIVE_SPEAKER_THRESHOLD} consecutive turns",
           result["consecutive_speaker_count"])
    for ex in result["consecutive_examples"]:
        print(f"        id={ex['id']}  speaker={ex['speaker']}  "
              f"run={ex['run_length']}  starts at turn {ex['starting_turn']}")

    bullet(f"Turns exceeding {LONG_TURN_WORD_THRESHOLD} words",
           result["long_turn_count"])
    for ex in result["long_turn_examples"]:
        print(f"        id={ex['id']}  turn {ex['turn_index']}  "
              f"[{ex['speaker']}]  {ex['word_count']} words: {ex['snippet']}")

    bullet(f"Conversations with <{MIN_TURNS} turns",
           result["short_conv_count"])
    for ex in result["short_conv_examples"]:
        print(f"        id={ex['id']}  turn_count={ex['turn_count']}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * _WIDTH)
    print("  MedAI — DATA EXPLORATION REPORT")
    print("=" * _WIDTH)

    report: dict = {
        "datasets": {},
        "missing_files": [],
    }

    for ds_name, path in DATASETS.items():
        section(f"DATASET: {ds_name.upper()}")
        print(f"  File: {path.relative_to(REPO_ROOT)}")

        records = load_jsonl(path)
        if records is None:
            report["missing_files"].append(str(path.relative_to(REPO_ROOT)))
            report["datasets"][ds_name] = {"status": "file_missing_or_empty"}
            continue

        print(f"  Loaded {len(records)} records.\n")

        validation   = validate_schema(records, ds_name)
        stats        = compute_stats(records, ds_name)
        anomalies    = detect_anomalies(records, ds_name)
        print_samples(records, ds_name)

        report["datasets"][ds_name] = {
            "status": "ok",
            "file": str(path.relative_to(REPO_ROOT)),
            "validation": validation,
            "stats": stats,
            "anomalies": anomalies,
        }

    # ── Summary ──────────────────────────────────────────────────────────
    section("CROSS-DATASET SUMMARY")

    total_records = sum(
        d.get("validation", {}).get("total", 0)
        for d in report["datasets"].values()
        if d.get("status") == "ok"
    )
    print(f"\n  Total records across all loaded datasets: {total_records}")

    for ds_name, info in report["datasets"].items():
        if info.get("status") != "ok":
            print(f"\n  {ds_name}: NOT AVAILABLE")
            continue
        v = info["validation"]
        s = info["stats"]
        a = info["anomalies"]
        print(f"\n  {ds_name}:")
        print(f"    Records            : {v['total']}")
        print(f"    Schema issues      : {v['records_missing_any_field']} records missing fields, "
              f"{v['conversations_with_bad_speakers']} bad-speaker conversations")
        print(f"    Avg turns          : {s['avg_turn_count']}  "
              f"(range {s['min_turn_count']}–{s['max_turn_count']})")
        print(f"    Doctor first       : {s['pct_doctor_speaks_first']}%  "
              f"  Has reference note: {s['pct_with_reference_note']}%")
        print(f"    Anomalies          : {a['consecutive_speaker_count']} consecutive-speaker, "
              f"{a['long_turn_count']} long-turn, "
              f"{a['short_conv_count']} short-conv")

    if report["missing_files"]:
        print(f"\n  Missing / empty files: {report['missing_files']}")

    # ── Save report ───────────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"\n  Report saved → {REPORT_PATH.relative_to(REPO_ROOT)}")
    print(f"\n{'=' * _WIDTH}\n")


if __name__ == "__main__":
    main()
