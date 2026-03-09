"""
red_flag_detector.py

Scans conversation turns and extracted entities for clinical red flags —
symptoms or patterns that warrant urgent escalation or specialist referral.

Planned responsibilities:
  - Load red-flag rules from configs/red_flags.yaml
  - Match against entities and free-text turns using keyword + semantic rules
  - Assign severity levels (URGENT / WARNING / WATCH)
  - Produce a list of triggered flags with the supporting evidence snippet
  - Integrate with SOAPGenerator so flagged items surface in the Assessment

Future dependencies (uncomment when implementing):
  # import yaml                     # load red_flags.yaml rule set
  # import anthropic                # optional LLM pass for semantic matching
  # import re
"""


class RedFlagDetector:
    pass
