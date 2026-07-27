import json
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from backend.app.core.database import get_db
from backend.app.core.config import settings
from backend.app.core.security import decode_access_token, create_guest_session
from backend.app.models.models import AuditLedger, HealthRecord
from backend.app.models.fhir import create_fhir_steps_observation, create_fhir_sleep_observation, create_fhir_nutrition_observation
from backend.app.agents.router import router_agent

from sqlalchemy import select as sa_select  # Avoid name clash

router = APIRouter(prefix="/health", tags=["Health"])

MAX_ALLOWED_LAG = 100  # Mark unhealthy if >100 blocks behind


@router.get("/indexer")
async def get_indexer_health(db: AsyncSession = Depends(get_db)):
    """
    Health check endpoint for the Escrow Indexer service.
    Used by Kubernetes/Docker liveness probes and uptime monitoring.
    Returns:
        - 200: healthy (lag within acceptable range)
        - 503: degraded (lag exceeds threshold)
        - 500: health check itself failed
    """
    try:
        from web3 import Web3
        from backend.app.models.models import IndexerState, STATE_KEY_LAST_BLOCK

        # Use sync Web3 for health check (lightweight)
        rpc_url = getattr(settings, "POLYGON_AMOY_RPC_URL") or getattr(settings, "WEB3_PROVIDER_URL")
        if not rpc_url:
            return {"status": "unconfigured", "detail": "No RPC URL configured"}

        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            return {"status": "unhealthy", "detail": "Cannot connect to RPC endpoint"}

        latest_block = w3.eth.block_number

        result = await db.execute(
            sa_select(IndexerState).where(IndexerState.key == STATE_KEY_LAST_BLOCK)
        )
        state = result.scalars().first()
        last_processed = state.value if state else 0

        lag = latest_block - last_processed
        status = "healthy" if lag <= MAX_ALLOWED_LAG else "degraded"

        health_data = {
            "status": status,
            "service": "escrow_indexer",
            "latest_chain_block": latest_block,
            "last_processed_block": last_processed,
            "lag_blocks": lag,
            "max_allowed_lag": MAX_ALLOWED_LAG,
        }

        if lag > MAX_ALLOWED_LAG:
            from fastapi import HTTPException as FastAPIHTTPException
            raise FastAPIHTTPException(status_code=503, detail=health_data)

        return health_data

    except Exception as e:
        from fastapi import HTTPException as FastAPIHTTPException
        raise FastAPIHTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}",
        )

async def get_user_id_from_auth(authorization: Optional[str] = None) -> str:
    """
    Decodes bearer token or returns a guest user session ID.
    """
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        payload = decode_access_token(token)
        if payload:
            return payload.get("sub", "guest_unknown")
    # Fallback to a temporary guest ID
    return "guest_user"

@router.post("/triage")
async def clinical_triage(
    description: str = Form(...),
    image: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id_from_auth)
):
    """
    Triage clinical patient query, perform image vision analysis, and route events across domains if severe.
    """
    image_bytes = None
    if image:
        try:
            image_bytes = await image.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read image file: {e}")

    try:
        # Run central router dispatcher
        result_state = await router_agent.route_event(description, image_bytes)
        
        # Save transaction log in the Audit Ledger
        audit_entry = AuditLedger(
            user_id=user_id,
            domain="health",
            action="clinical_triage",
            request_redacted=description[:1000],
            response_raw=result_state,
            confidence_score=85.0 if result_state.get("severity") != "mild" else 70.0
        )
        db.add(audit_entry)
        await db.commit()

        # Build normalized risk level
        severity = result_state.get("severity", "mild")
        risk_level = result_state.get("risk_level")
        if not risk_level:
            risk_level_map = {"mild": "Low", "moderate": "Moderate", "severe": "Urgent", "critical": "Urgent"}
            risk_level = risk_level_map.get(severity, "Low")

        # Format diagnoses list if not directly in state
        diagnoses = result_state.get("diagnoses")
        if not diagnoses:
            diagnoses = []
            for d in result_state.get("differential_diagnoses", []):
                diagnoses.append({
                    "condition": d.get("condition"),
                    "match_percentage": d.get("match_percentage", f"{int(d.get('confidence_score', 80))}%"),
                    "icd10_code": d.get("icd10_code"),
                    "description": d.get("reasoning"),
                    "source": d.get("citation")
                })

        summary = result_state.get("summary") or result_state.get("treatment") or "Clinical assessment complete."

        return {
            "primary_concern": result_state.get("primary_concern", "Medical Assessment"),
            "icd_10_code": result_state.get("icd_10_code", "M79.89"),
            "confidence_score": result_state.get("confidence_score", 1.0),
            "citations": result_state.get("citations", result_state.get("sources", [])),
            "sources": result_state.get("sources", []),
            "risk_level": risk_level,
            "severity": severity,
            "summary": summary,
            "phi_elements_scrubbed_count": result_state.get("phi_elements_scrubbed_count", 0),
            "diagnoses": diagnoses,
            "diagnosis": result_state.get("diagnosis", ""),
            "treatment": result_state.get("treatment", ""),
            "recommended_immediate_treatment": result_state.get("treatment", ""),
            "requires_appeal": result_state.get("requires_appeal", False),
            "appeal_letter": result_state.get("appeal_letter", ""),
            "campaign_prefilled": result_state.get("campaign_prefilled", {}),
            "recovery_guide": result_state.get("recovery_guide", {}),
            "differential_diagnoses": result_state.get("differential_diagnoses", [])
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Triage process failed: {str(e)}")


@router.post("/log")
async def log_health_telemetry(
    record_type: str, # nutrition, steps, sleep
    value: float,
    protein: Optional[float] = 0,
    carbs: Optional[float] = 0,
    fat: Optional[float] = 0,
    date_str: Optional[str] = "2026-07-23",
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id_from_auth)
):
    """
    Save health logs formatted to HL7 FHIR v4 schemas.
    """
    try:
        if record_type == "steps":
            fhir_obs = create_fhir_steps_observation(user_id, int(value), date_str)
        elif record_type == "sleep":
            fhir_obs = create_fhir_sleep_observation(user_id, value, date_str)
        elif record_type == "nutrition":
            fhir_obs = create_fhir_nutrition_observation(user_id, value, protein, carbs, fat, date_str)
        else:
            raise HTTPException(status_code=400, detail="Invalid record type. Must be nutrition, steps, or sleep.")
        
        record = HealthRecord(
            user_id=user_id,
            record_type=record_type,
            fhir_observation=fhir_obs.model_dump()
        )
        db.add(record)
        await db.commit()
        
        return {"message": f"Successfully logged {record_type} as FHIR v4 Observation", "data": fhir_obs}
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fhir/observation")
async def log_fhir_observation(
    payload: dict,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id_from_auth)
):
    """
    Direct endpoint for posting structured HL7 FHIR v4 Observation JSON objects.
    """
    try:
        import uuid
        from datetime import datetime, timezone
        
        obs_id = payload.get("id", f"obs-{uuid.uuid4().hex[:8]}")
        timestamp = datetime.now(timezone.utc).isoformat()
        
        payload["id"] = obs_id
        payload["effectiveDateTime"] = timestamp
        
        record = HealthRecord(
            user_id=user_id,
            record_type="fhir_observation",
            fhir_observation=payload
        )
        db.add(record)
        await db.commit()
        
        return {
            "status": "success",
            "message": "FHIR Observation successfully created",
            "observation_id": obs_id,
            "timestamp": timestamp,
            "data": payload
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to log FHIR Observation: {str(e)}")


@router.get("/records")
async def get_health_records(
    record_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_user_id_from_auth),
):
    """
    Retrieve health records for the authenticated user.
    Supports filtering by record_type and pagination via limit/offset.
    """
    try:
        query = sa_select(HealthRecord).where(HealthRecord.user_id == user_id)

        if record_type:
            query = query.where(HealthRecord.record_type == record_type)

        query = query.order_by(HealthRecord.created_at.desc()).offset(offset).limit(limit)

        result = await db.execute(query)
        records = result.scalars().all()

        return {
            "records": [
                {
                    "id": r.id,
                    "user_id": r.user_id,
                    "record_type": r.record_type,
                    "fhir_observation": r.fhir_observation,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in records
            ],
            "total": len(records),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch health records: {str(e)}")
