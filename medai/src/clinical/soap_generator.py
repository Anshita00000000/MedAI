"""
soap_generator.py

Transforms extracted clinical entities and the raw transcript into a
structured SOAP note (Subjective, Objective, Assessment, Plan).

Planned responsibilities:
  - Accept turn list + EntityExtractor output
  - Prompt a Claude model with conversation context and entity dict
  - Parse model response into the four SOAP sections
  - Validate completeness (warn if Objective section is thin due to
    missing vitals/exam findings in a phone/text consultation)
  - Return a SOAPNote dataclass or dict with section text

Future dependencies (uncomment when implementing):
  # import anthropic                # Claude API for note generation
  # from dataclasses import dataclass, field
  # import yaml                     # load prompt templates from configs/prompts.yaml
"""


class SOAPGenerator:
    pass
