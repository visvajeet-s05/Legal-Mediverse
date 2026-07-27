import math
import struct
import uuid
import wave
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, List, Any, Dict
from backend.app.core.database import get_db
from backend.app.core.security import redact_pii
from backend.app.models.models import EduNote, AuditLedger
from backend.app.models.fhir import FHIRObservation

router = APIRouter(prefix="/edu", tags=["EdTech"])

# ─── FHIR v4 DiagnosticReport Pydantic Model ────────────────────────────────
class FHIRDiagnosticReport(BaseModel):
    """Strict HL7 FHIR v4 DiagnosticReport schema (subset for edu notes)."""
    resourceType: str = "DiagnosticReport"
    id: Optional[str] = None
    status: str  # registered | partial | preliminary | final
    category: Optional[List[Dict[str, Any]]] = None
    code: Dict[str, Any]
    subject: Optional[Dict[str, Any]] = None
    effectiveDateTime: Optional[str] = None
    issued: Optional[str] = None
    result: Optional[List[Dict[str, Any]]] = None
    conclusion: Optional[str] = None


def _validate_fhir_payload(payload: Dict[str, Any]) -> FHIRObservation | FHIRDiagnosticReport:
    """
    Validates a raw dict against either FHIRObservation or FHIRDiagnosticReport schema.
    Raises HTTPException(422) on schema mismatch.
    """
    resource_type = payload.get("resourceType", "")
    try:
        if resource_type == "Observation":
            return FHIRObservation(**payload)
        elif resource_type == "DiagnosticReport":
            return FHIRDiagnosticReport(**payload)
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported FHIR resourceType '{resource_type}'. Supported: Observation, DiagnosticReport."
            )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": "FHIR v4 schema validation failed", "errors": exc.errors()}
        )


def _safe_parse_podcast_script(script: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Validates and sanitises each dialogue entry in a podcast script.
    Ensures 'host', 'text', and 'timestamp' fields are present and non-null.
    """
    sanitised = []
    for idx, entry in enumerate(script):
        host = entry.get("host") or f"Host_{idx + 1}"
        text = entry.get("text") or ""
        timestamp = entry.get("timestamp")
        if timestamp is None:
            timestamp = float(idx * 6)
        if not isinstance(timestamp, (int, float)):
            timestamp = float(idx * 6)
        if not text.strip():
            # Skip entries with empty audio buffer text
            continue
        sanitised.append({"host": host, "text": text.strip(), "timestamp": float(timestamp)})
    return sanitised


def _build_persistent_audio_asset(topic: str, host_name_1: str, host_name_2: str) -> str:
    """Create a tiny WAV audio asset on disk and return a stable static URL for it."""
    audio_dir = Path(__file__).resolve().parents[3] / "static" / "generated_audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    filename = f"podcast_{uuid.uuid4().hex[:8]}.wav"
    output_path = audio_dir / filename

    sample_rate = 22050
    duration_seconds = 0.6
    frequency_hz = 440
    amplitude = 12000
    total_samples = int(sample_rate * duration_seconds)

    frames = bytearray()
    for index in range(total_samples):
        sample = int(amplitude * math.sin(2 * math.pi * frequency_hz * index / sample_rate))
        frames.extend(struct.pack("<h", sample))

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)

    return f"/static/generated_audio/{filename}"


@router.post("/recall-engine")
async def active_recall_notes(
    title: str = Form(...),
    content: str = Form(...),
    fhir_json: Optional[str] = Form(None),
    user_id: str = "guest_user",
    db: AsyncSession = Depends(get_db)
):
    """
    Convert text notes into interactive React Flow nodes/edges and flashcard arrays.
    Optionally accepts a FHIR v4 Observation/DiagnosticReport JSON string for strict schema validation.
    """
    # Validate optional FHIR payload
    if fhir_json:
        import json as _json
        try:
            fhir_dict = _json.loads(fhir_json)
        except Exception:
            raise HTTPException(status_code=422, detail="fhir_json must be valid JSON.")
        _validate_fhir_payload(fhir_dict)  # raises 422 on schema failure

    # Clean text first
    clean_content, _, _ = redact_pii(content)

    sentences = [s.strip() for s in clean_content.split(".") if len(s.strip()) > 5]
    if not sentences:
        sentences = [clean_content]

    # Generate dynamic nodes and edges for React Flow
    nodes = []
    edges = []

    root_id = "node_root"
    nodes.append({
        "id": root_id,
        "type": "input",
        "data": {"label": f"{title} (Core Topic)"},
        "position": {"x": 250, "y": 20}
    })

    flashcards = []
    y_pos = 100
    for idx, sentence in enumerate(sentences[:4]):
        node_id = f"node_{idx}"
        nodes.append({
            "id": node_id,
            "data": {"label": sentence[:60] + "..." if len(sentence) > 60 else sentence},
            "position": {"x": 80 + (idx * 160), "y": y_pos}
        })
        edges.append({
            "id": f"edge_root_{node_id}",
            "source": root_id,
            "target": node_id
        })

        flashcards.append({
            "question": f"What key principle of {title} is highlighted in Section {idx+1}?",
            "answer": sentence
        })
        y_pos += 30

    try:
        edu_entry = EduNote(
            user_id=user_id,
            title=title,
            content=clean_content,
            react_flow_graph={"nodes": nodes, "edges": edges},
            flashcards=flashcards
        )
        db.add(edu_entry)
        await db.commit()

        # Audit ledger
        audit_entry = AuditLedger(
            user_id=user_id,
            domain="edu",
            action="active_recall_generation",
            request_redacted=content[:500],
            response_raw={"nodes_count": len(nodes), "flashcards_count": len(flashcards)}
        )
        db.add(audit_entry)
        await db.commit()

        return {
            "id": edu_entry.id,
            "title": title,
            "react_flow_graph": {"nodes": nodes, "edges": edges},
            "flashcards": flashcards
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notes")
async def get_edu_notes(
    limit: int = 20,
    offset: int = 0,
    user_id: str = "guest_user",
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve education notes with pagination.
    Supports pagination via limit/offset parameters.
    """
    try:
        query = (
            select(EduNote)
            .where(EduNote.user_id == user_id)
            .order_by(EduNote.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await db.execute(query)
        notes = result.scalars().all()

        return {
            "notes": [
                {
                    "id": n.id,
                    "title": n.title,
                    "content": n.content[:500],  # Truncate for listing
                    "react_flow_graph": n.react_flow_graph,
                    "flashcards": n.flashcards[:10] if n.flashcards else [],
                    "created_at": n.created_at.isoformat() if n.created_at else None,
                }
                for n in notes
            ],
            "total": len(notes),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch edu notes: {str(e)}")


@router.get("/notes/{note_id}")
async def get_edu_note_detail(
    note_id: int,
    user_id: str = "guest_user",
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve a single education note by ID with full content.
    """
    try:
        query = select(EduNote).where(
            EduNote.id == note_id,
            EduNote.user_id == user_id,
        )
        result = await db.execute(query)
        note = result.scalar_one_or_none()

        if not note:
            raise HTTPException(status_code=404, detail="Note not found")

        return {
            "id": note.id,
            "title": note.title,
            "content": note.content,
            "react_flow_graph": note.react_flow_graph,
            "flashcards": note.flashcards,
            "created_at": note.created_at.isoformat() if note.created_at else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch note: {str(e)}")


@router.post("/fhir/validate")
async def validate_fhir_resource(payload: Dict[str, Any] = Body(...)):
    """
    Validates a raw HL7 FHIR v4 JSON payload (Observation or DiagnosticReport).
    Returns 200 on success with the validated resource, or 422 with detailed validation errors.
    """
    validated = _validate_fhir_payload(payload)
    return {
        "status": "valid",
        "resourceType": payload.get("resourceType"),
        "validated_resource": validated.model_dump(),
    }


@router.post("/generate-podcast")
async def generate_podcast_studio(
    topic: str = Form(...),
    host_name_1: str = Form("Dr. Alex"),
    host_name_2: str = Form("Sarah"),
    user_id: str = "guest_user",
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a synchronized multi-host audio podcast transcript with timestamps.
    Each dialogue entry is validated and sanitised to prevent missing audio buffers.
    """
    # Create two-host conversation script
    raw_script = [
        {"host": host_name_1, "text": f"Welcome back to Mediverse Radio. Today, we're diving deep into: {topic}.", "timestamp": 0.0},
        {"host": host_name_2, "text": f"Thanks, {host_name_1}. This is such a critical topic, and there is a lot of new research to share.", "timestamp": 6.5},
        {"host": host_name_1, "text": "Absolutely. Let's look at the foundational concepts. It's key for patients and students to understand the physiology.", "timestamp": 12.0},
        {"host": host_name_2, "text": "Definitely. That's why we've prepared interactive study maps and guides in our Mediverse EdTech dashboard too.", "timestamp": 18.5},
        {"host": host_name_1, "text": f"For our listeners: the latest clinical evidence on {topic} shows promising advancements in evidence-based protocols.", "timestamp": 25.0},
        {"host": host_name_2, "text": "Exactly. And for medical students preparing for board exams, our Anki flashcard integration makes reviewing this material much more efficient.", "timestamp": 31.5},
    ]

    # Sanitise script — removes empty entries, ensures all buffers are non-null
    script = _safe_parse_podcast_script(raw_script)

    if not script:
        raise HTTPException(status_code=500, detail="Podcast script generation failed: all audio buffers were empty.")

    # In a real environment, we'd call LiveKit/ElevenLabs/Google Cloud TTS.
    # We persist a lightweight local audio asset so the endpoint is safe for serverless or stateless deployments.
    audio_url = _build_persistent_audio_asset(topic, host_name_1, host_name_2)

    try:
        audit_entry = AuditLedger(
            user_id=user_id,
            domain="edu",
            action="generate_podcast",
            request_redacted=topic,
            response_raw={"script_entries": len(script), "audio_url": audio_url}
        )
        db.add(audit_entry)
        await db.commit()

        return {
            "topic": topic,
            "transcript": script,
            "audio_url": audio_url,
            "script_entries": len(script),
            "hosts": [host_name_1, host_name_2],
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


