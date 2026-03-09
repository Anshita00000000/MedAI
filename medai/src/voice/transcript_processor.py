"""
transcript_processor.py

Accepts raw audio or pre-transcribed text from a doctor-patient consultation
and returns a structured list of speaker-labelled turns ready for the
clinical pipeline.

Planned responsibilities:
  - Receive audio bytes or a raw transcript string
  - Run speaker diarisation to separate DOCTOR vs PATIENT turns
  - Clean artefacts (filler words, overlapping speech, timestamps)
  - Detect language (en / ar) and route to the appropriate downstream model
  - Return a list of {"speaker": "DOCTOR"|"PATIENT", "text": str} dicts

Future dependencies (uncomment when implementing):
  # import whisper                  # OpenAI Whisper for ASR
  # from pyannote.audio import Pipeline  # speaker diarisation
  # import langdetect               # language identification
"""


class TranscriptProcessor:
    pass
