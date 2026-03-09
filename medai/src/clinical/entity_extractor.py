"""
entity_extractor.py

Extracts structured clinical entities from a normalised conversation transcript.

Planned responsibilities:
  - Identify symptoms, their onset, duration, and severity
  - Extract medications (name, dose, frequency) mentioned by either speaker
  - Tag anatomical locations, vital signs, and lab values
  - Detect negations ("no chest pain") and uncertainty ("possible fever")
  - Return a structured dict of entity categories for downstream SOAP assembly

Future dependencies (uncomment when implementing):
  # import anthropic                # Claude API for LLM-based extraction
  # import medspacy                 # clinical NLP pipeline
  # from negspacy.negation import Negex  # negation detection
"""


class EntityExtractor:
    pass
