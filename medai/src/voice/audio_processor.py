"""
audio_processor.py

Voice-to-transcript pipeline for MedAI clinical audio recordings.

Pipeline stages
---------------
1. Transcription  — openai-whisper (model="base") with word-level timestamps
2. Diarisation    — pyannote.audio with HF_TOKEN env var, auto num_speakers
3. Alignment      — merge Whisper words with pyannote segments → speaker turns
4. Role inference — Gemini identifies Doctor/Patient/etc. from transcript
                    content; falls back to SPEAKER_XX labels on error/quota
5. Output         — role_map dict + ProcessedTranscript compatible with pipeline

Usage::

    from medai.src.voice.audio_processor import AudioProcessor

    processor = AudioProcessor()
    role_map, transcript = processor.process("consultation.wav")
    # role_map = {"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient"}
    # transcript is a ProcessedTranscript ready for the clinical pipeline
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Role constants (exported for use by the demo UI)
# ---------------------------------------------------------------------------

#: Ordered list of valid clinical roles for the role-correction UI.
ROLE_OPTIONS: list[str] = ["Doctor", "Patient", "Caregiver", "Nurse", "Other"]

#: CSS colour per role for the demo's colour-coded transcript display.
ROLE_COLOURS: dict[str, str] = {
    "Doctor": "#1a6fad",
    "Patient": "#2d7a4f",
    "Caregiver": "#9b59b6",
    "Nurse": "#e67e22",
    "Other": "#7f8c8d",
}

#: Maps clinical role → pipeline speaker label accepted by TranscriptProcessor.
ROLE_TO_SPEAKER: dict[str, str] = {
    "Doctor": "DOCTOR",
    "Nurse": "DOCTOR",
    "Patient": "PATIENT",
    "Caregiver": "PATIENT",
    "Other": "DOCTOR",   # safe fallback for pipeline
}


# ---------------------------------------------------------------------------
# Lazy imports — keep startup fast; raise clearly on first use if not installed
# ---------------------------------------------------------------------------

def _import_whisper() -> Any:
    try:
        import whisper  # type: ignore[import]
        return whisper
    except ImportError as exc:
        raise ImportError(
            "openai-whisper is required for audio transcription. "
            "Install it with: pip install openai-whisper"
        ) from exc


def _import_pyannote_pipeline() -> Any:
    try:
        from pyannote.audio import Pipeline  # type: ignore[import]
        return Pipeline
    except ImportError as exc:
        raise ImportError(
            "pyannote.audio is required for speaker diarisation. "
            "Install it with: pip install pyannote.audio"
        ) from exc


# ---------------------------------------------------------------------------
# ProcessedTranscript import (lazy, to avoid circular imports at startup)
# ---------------------------------------------------------------------------

def _get_transcript_classes() -> tuple[Any, Any]:
    """Return (ProcessedTranscript, Turn) from the voice package."""
    _root = str(Path(__file__).resolve().parents[3])
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from medai.src.voice.transcript_processor import ProcessedTranscript, Turn  # noqa: PLC0415
    return ProcessedTranscript, Turn


# ---------------------------------------------------------------------------
# Question-word pattern (mirrors transcript_processor.py)
# ---------------------------------------------------------------------------

_QUESTION_WORDS = re.compile(
    r"^(what|when|where|who|whom|whose|which|why|how|"
    r"do|does|did|is|are|was|were|have|has|had|"
    r"can|could|will|would|shall|should|may|might|must)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core AudioProcessor
# ---------------------------------------------------------------------------

class AudioProcessor:
    """
    End-to-end voice → ProcessedTranscript pipeline.

    Parameters
    ----------
    whisper_model:
        Whisper model size.  Defaults to ``"base"`` (~75 MB, fast).
    hf_token:
        HuggingFace token for pyannote.audio model download.  When
        omitted the ``HF_TOKEN`` environment variable is used.
    """

    def __init__(
        self,
        whisper_model: str = "base",
        hf_token: str | None = None,
    ) -> None:
        self._whisper_model_name = whisper_model
        self._hf_token: str = hf_token or os.getenv("HF_TOKEN", "")
        self._whisper_model: Any = None    # lazy-loaded on first use
        self._pyannote_pipeline: Any = None  # lazy-loaded on first use

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        audio_path: str | Path,
        conversation_id: str | None = None,
    ) -> tuple[dict[str, str], Any]:
        """
        Full pipeline: audio file → ``(role_map, ProcessedTranscript)``.

        Parameters
        ----------
        audio_path:
            Path to the audio file (WAV, MP3, M4A, …).
        conversation_id:
            Optional ID embedded in the ProcessedTranscript.
            Auto-generated when omitted.

        Returns
        -------
        tuple[dict[str, str], ProcessedTranscript]
            *role_map* maps speaker labels to inferred roles,
            e.g. ``{"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient"}``.
            *ProcessedTranscript* is ready for the clinical pipeline.

        Raises
        ------
        FileNotFoundError
            If *audio_path* does not exist.
        ImportError
            If openai-whisper or pyannote.audio is not installed.
        EnvironmentError
            If ``HF_TOKEN`` is not set.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        conv_id = conversation_id or f"audio-{uuid.uuid4().hex[:8]}"

        words = self.transcribe(audio_path)       # Stage 1
        segments = self.diarise(audio_path)       # Stage 2
        raw_turns = self.align(words, segments)   # Stage 3
        role_map = self.infer_roles(raw_turns)    # Stage 4
        transcript = self._build_transcript(raw_turns, role_map, conv_id)  # Stage 5

        return role_map, transcript

    # -- Stage 1: Transcription --------------------------------------------

    def transcribe(self, audio_path: Path) -> list[dict]:
        """
        Run Whisper transcription with word-level timestamps.

        Parameters
        ----------
        audio_path:
            Path to the audio file.

        Returns
        -------
        list[dict]
            Each element has keys ``"word"``, ``"start"``, ``"end"``
            (float seconds).
        """
        model = self._load_whisper()
        result = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            verbose=False,
        )
        words: list[dict] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                word_text = w.get("word", "").strip()
                if word_text:
                    words.append({
                        "word": word_text,
                        "start": float(w.get("start", 0.0)),
                        "end": float(w.get("end", 0.0)),
                    })
        return words

    # -- Stage 2: Diarisation ----------------------------------------------

    def diarise(self, audio_path: Path) -> list[dict]:
        """
        Run pyannote.audio speaker diarisation (num_speakers inferred).

        Parameters
        ----------
        audio_path:
            Path to the audio file.

        Returns
        -------
        list[dict]
            Each element has keys ``"speaker"`` (e.g. ``"SPEAKER_00"``),
            ``"start"``, ``"end"`` (float seconds), sorted by start time.
        """
        pipeline = self._load_pyannote()
        diarization = pipeline(str(audio_path))

        segments: list[dict] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": float(turn.start),
                "end": float(turn.end),
            })

        segments.sort(key=lambda s: s["start"])
        return segments

    # -- Stage 3: Alignment ------------------------------------------------

    def align(
        self,
        words: list[dict],
        segments: list[dict],
    ) -> list[dict]:
        """
        Assign each Whisper word to its diarisation speaker, then merge
        consecutive same-speaker words into turn-level dicts.

        Parameters
        ----------
        words:
            Word-level output from :meth:`transcribe`.
        segments:
            Speaker segments from :meth:`diarise`.

        Returns
        -------
        list[dict]
            Each element has keys ``"speaker"``, ``"text"``, ``"start"``,
            ``"end"`` (float seconds).
        """
        if not words:
            return []

        def _speaker_for_word(w: dict) -> str:
            """Return the speaker label whose segment contains the word mid-point."""
            w_mid = (w["start"] + w["end"]) / 2.0
            best_label = segments[0]["speaker"] if segments else "SPEAKER_00"
            best_dist = float("inf")
            for seg in segments:
                if seg["start"] <= w_mid <= seg["end"]:
                    return seg["speaker"]
                dist = min(abs(w_mid - seg["start"]), abs(w_mid - seg["end"]))
                if dist < best_dist:
                    best_dist = dist
                    best_label = seg["speaker"]
            return best_label

        # Merge consecutive words with the same speaker into turns
        turns: list[dict] = []
        for word in words:
            spk = _speaker_for_word(word)
            text = word["word"]
            if not text.strip():
                continue
            if turns and turns[-1]["speaker"] == spk:
                turns[-1]["text"] += " " + text
                turns[-1]["end"] = word["end"]
            else:
                turns.append({
                    "speaker": spk,
                    "text": text,
                    "start": word["start"],
                    "end": word["end"],
                })

        # Normalise whitespace
        for t in turns:
            t["text"] = " ".join(t["text"].split())

        return [t for t in turns if t["text"]]

    # -- Stage 4: Role inference -------------------------------------------

    def infer_roles(self, turns: list[dict]) -> dict[str, str]:
        """
        Ask Gemini to map ``SPEAKER_XX`` labels → clinical roles.

        Falls back to identity mapping (``SPEAKER_XX`` → ``SPEAKER_XX``)
        when Gemini is unavailable, quota is exceeded, or the response
        cannot be parsed.

        Parameters
        ----------
        turns:
            Aligned speaker turns from :meth:`align`.

        Returns
        -------
        dict[str, str]
            e.g. ``{"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient"}``
        """
        speakers = list(dict.fromkeys(t["speaker"] for t in turns))

        # Build a short excerpt (first 10 turns, each truncated to 120 chars)
        preview_lines = [
            f"{t['speaker']}: {t['text'][:120]}"
            for t in turns[:10]
        ]
        preview = "\n".join(preview_lines)

        prompt = (
            "This is a medical consultation transcript with speakers labelled "
            "SPEAKER_00, SPEAKER_01 etc. Based on the content, identify each "
            "speaker's role: Doctor, Patient, Caregiver, Nurse, or Other. "
            "Return JSON only with no markdown fences: "
            '{"SPEAKER_00": "Doctor", "SPEAKER_01": "Patient", ...}\n\n'
            f"Transcript excerpt:\n{preview}"
        )

        try:
            role_map = self._call_gemini(prompt, speakers)
        except Exception:
            # Fallback: keep original SPEAKER_XX labels verbatim
            role_map = {spk: spk for spk in speakers}

        # Guarantee every speaker has an entry
        for spk in speakers:
            if spk not in role_map:
                role_map[spk] = spk

        return role_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_whisper(self) -> Any:
        if self._whisper_model is None:
            whisper = _import_whisper()
            self._whisper_model = whisper.load_model(self._whisper_model_name)
        return self._whisper_model

    def _load_pyannote(self) -> Any:
        if self._pyannote_pipeline is None:
            if not self._hf_token:
                raise EnvironmentError(
                    "HF_TOKEN environment variable is required for "
                    "pyannote.audio speaker diarisation. "
                    "Set HF_TOKEN before calling process()."
                )
            Pipeline = _import_pyannote_pipeline()
            self._pyannote_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self._hf_token,
            )
        return self._pyannote_pipeline

    def _call_gemini(self, prompt: str, speakers: list[str]) -> dict[str, str]:
        """
        Call Gemini to infer speaker roles.

        Raises on any network/quota/parse error so the caller can fall back.
        """
        try:
            from google import genai as _genai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("google-genai package not installed") from exc

        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set")

        client = _genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        raw = response.text.strip()

        # Strip markdown code fences if the model wrapped the JSON
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.splitlines()
                if not line.startswith("```")
            ).strip()

        role_map: dict[str, str] = json.loads(raw)

        valid_roles = set(ROLE_OPTIONS)
        return {
            k: (v if v in valid_roles else "Other")
            for k, v in role_map.items()
            if k in speakers
        }

    def _build_transcript(
        self,
        raw_turns: list[dict],
        role_map: dict[str, str],
        conv_id: str,
    ) -> Any:
        """
        Convert aligned speaker turns + role_map into a ProcessedTranscript.

        Speaker roles are normalised to pipeline-compatible labels:

        * Doctor / Nurse  → ``"DOCTOR"``
        * Patient / Caregiver → ``"PATIENT"``
        * Other → ``"DOCTOR"`` (safe fallback)
        """
        ProcessedTranscript, Turn = _get_transcript_classes()

        turns: list[Any] = []
        run_count = 0
        prev_speaker: str | None = None

        for i, raw in enumerate(raw_turns):
            role = role_map.get(raw["speaker"], raw["speaker"])
            speaker = ROLE_TO_SPEAKER.get(role, "DOCTOR")
            text = " ".join(raw["text"].split())
            if not text:
                continue

            if speaker == prev_speaker:
                run_count += 1
            else:
                run_count = 1
                prev_speaker = speaker

            wc = len(text.split())
            first_word = text.split()[0] if text.split() else ""
            is_q = text.rstrip().endswith("?") or bool(
                _QUESTION_WORDS.match(first_word)
            )

            turns.append(Turn(
                index=i,
                speaker=speaker,
                text=text,
                word_count=wc,
                is_question=is_q,
                consecutive_run=run_count,
            ))

        doctor_text = " ".join(t.text for t in turns if t.speaker == "DOCTOR")
        patient_text = " ".join(t.text for t in turns if t.speaker == "PATIENT")
        full_text = "\n".join(f"{t.speaker}: {t.text}" for t in turns)
        total_words = sum(t.word_count for t in turns)

        return ProcessedTranscript(
            id=conv_id,
            source_dataset="audio_upload",
            language="en",
            turns=turns,
            doctor_text=doctor_text,
            patient_text=patient_text,
            full_text=full_text,
            turn_count=len(turns),
            word_count=total_words,
            has_reference_note=False,
            reference_note=None,
            flags=[],
        )


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AudioProcessor — standalone smoke-test"
    )
    parser.add_argument("audio_file", help="Path to a WAV/MP3/M4A file")
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model size (default: base)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  AudioProcessor — smoke-test")
    print("=" * 70)

    ap = AudioProcessor(whisper_model=args.model)
    print(f"\nProcessing: {args.audio_file}")

    role_map, transcript = ap.process(args.audio_file)

    print(f"\nRole map: {json.dumps(role_map, indent=2)}")
    print(f"\nProcessedTranscript:\n{transcript}")
