"""
main.py

FastAPI application entry point for the MedAI clinical intelligence platform.

Planned endpoints:
  POST /transcripts/process
      Accept raw transcript text or audio upload; return structured turns.

  POST /notes/generate
      Accept structured turns; return a SOAP note JSON + DDx list.

  POST /notes/export/pdf
      Accept a SOAP note dict; return a PDF byte stream.

  POST /notes/export/fhir
      Accept a SOAP note dict; return a FHIR R4 Bundle JSON.

  GET  /health
      Liveness probe returning service version and status.

Future dependencies (uncomment when implementing):
  # from fastapi import FastAPI, UploadFile, HTTPException
  # from fastapi.middleware.cors import CORSMiddleware
  # from pydantic import BaseModel
  # import uvicorn
"""

# app = FastAPI(title="MedAI", version="0.1.0")
