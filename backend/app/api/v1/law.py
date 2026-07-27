from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
from backend.app.core.database import get_db
from backend.app.models.models import LegalCase, AuditLedger
from backend.app.agents.legal_agent import LegalAppealAgent
from backend.app.core.security import RoleChecker, UserRole
import io

# ─── ReportLab (formal legal PDF generation) ────────────────────────────────
_reportlab_available = False
try:
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib import colors
    _reportlab_available = True
except ImportError:
    pass


def generate_legal_pdf(
    patient_name: str,
    policy_id: str,
    insurance_provider: str,
    claim_id: Optional[str],
    denial_reason: str,
    applicable_statute: str,
    appeal_letter: str,
) -> bytes:
    """
    Generates a formal legal-grade PDF appeal document.
    Layout: 1-inch margins, formal header, body text, auto-numbered statutory footnotes.
    Statutory Citations Enforced:
        [1] ACA Section 2719 (45 CFR § 147.136)
        [2] ERISA 29 U.S.C. § 1133
        [3] HIPAA Right of Access 45 CFR § 164.524
    Returns PDF bytes (or plain-text bytes if ReportLab unavailable).
    """
    if not _reportlab_available:
        # Graceful plaintext fallback
        return appeal_letter.encode("utf-8")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )

    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "LegalTitle",
        parent=styles["Heading1"],
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=6,
        textColor=colors.HexColor("#1a1a2e"),
        fontName="Helvetica-Bold",
    )
    style_subtitle = ParagraphStyle(
        "LegalSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        alignment=TA_CENTER,
        spaceAfter=4,
        textColor=colors.HexColor("#333333"),
    )
    style_section = ParagraphStyle(
        "LegalSection",
        parent=styles["Heading2"],
        fontSize=11,
        spaceAfter=4,
        spaceBefore=10,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#1a1a2e"),
    )
    style_body = ParagraphStyle(
        "LegalBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    style_footnote = ParagraphStyle(
        "LegalFootnote",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#555555"),
        leftIndent=0,
    )

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("LEGAL MEDIVERSE ADVOCACY PLATFORM", style_title))
    story.append(Paragraph("FORMAL NOTICE OF APPEAL — HEALTHCARE CLAIM DENIAL", style_title))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 0.15 * inch))

    # ── Case Metadata ────────────────────────────────────────────────────────
    story.append(Paragraph("CASE IDENTIFICATION", style_section))
    meta_lines = [
        f"<b>Patient Name:</b> {patient_name}",
        f"<b>Policy / Member ID:</b> {policy_id}",
        f"<b>Insurance Carrier:</b> {insurance_provider}",
        f"<b>Claim ID:</b> {claim_id or policy_id}",
        f"<b>Denial Summary:</b> {denial_reason[:300]}",
        f"<b>Applicable Statutes:</b> {applicable_statute}",
    ]
    for line in meta_lines:
        story.append(Paragraph(line, style_body))
    story.append(Spacer(1, 0.1 * inch))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#999999")))
    story.append(Spacer(1, 0.1 * inch))

    # ── Appeal Letter Body ────────────────────────────────────────────────────
    story.append(Paragraph("FORMAL APPEAL LETTER", style_section))
    for paragraph in appeal_letter.split("\n"):
        para_text = paragraph.strip()
        if para_text:
            # Replace statutory references with superscript footnote markers
            para_text = para_text.replace("ACA Section 2719", "ACA Section 2719<super>[1]</super>")
            para_text = para_text.replace("45 CFR § 147.136", "45 CFR § 147.136<super>[1]</super>")
            para_text = para_text.replace("29 U.S.C. § 1133", "29 U.S.C. § 1133<super>[2]</super>")
            para_text = para_text.replace("ERISA", "ERISA<super>[2]</super>")
            para_text = para_text.replace("45 CFR § 164.524", "45 CFR § 164.524<super>[3]</super>")
            story.append(Paragraph(para_text, style_body))

    story.append(Spacer(1, 0.2 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 0.1 * inch))

    # ── Statutory Footnotes ────────────────────────────────────────────────────
    story.append(Paragraph("STATUTORY REFERENCES", style_section))
    footnotes = [
        "[1] ACA Section 2719 (45 CFR § 147.136) — Internal Claims & Appeals. Guarantees a full and fair review process for all adverse benefit determinations.",
        "[2] ERISA 29 U.S.C. § 1133 — Employee Retirement Income Security Act. Mandates plan participants receive written notice of denial with specific reasons and the opportunity to request a full and fair review.",
        "[3] HIPAA Right of Access 45 CFR § 164.524 — Grants individuals the right to access their own protected health information within 30 days of a written request.",
    ]
    for fn in footnotes:
        story.append(Paragraph(fn, style_footnote))

    doc.build(story)
    return buffer.getvalue()


router = APIRouter(prefix="/law", tags=["Legal Services"])

# Instantiate legal agent
legal_agent = LegalAppealAgent()

@router.post("/appeal")
async def generate_appeal_letter(
    denial_letter: str = Form(...),
    patient_name: str = Form(...),
    policy_id: str = Form(...),
    insurance_provider: Optional[str] = Form("Insurance Carrier"),
    claim_id: Optional[str] = Form(None),
    denial_code: Optional[str] = Form(None),
    is_urgent: Optional[bool] = Form(False),
    user_id: str = "guest_user",
    db: AsyncSession = Depends(get_db),
    user_payload: dict = Depends(RoleChecker([UserRole.LEGAL_COUNSEL]))
):
    """
    Parse insurance denial details and draft a formal appeal citing ACA Section 2719 (45 CFR § 147.136).
    """
    try:
        appeal_data = await legal_agent.generate_appeal(
            denial_text=denial_letter,
            patient_name=patient_name,
            policy_id=policy_id,
            insurance_provider=insurance_provider,
            claim_id=claim_id,
            denial_code=denial_code,
            is_urgent=is_urgent
        )
        
        # Save case in DB
        case = LegalCase(
            user_id=user_id,
            doc_type="denial_appeal",
            raw_text=denial_letter,
            appeal_letter=appeal_data.get("appeal_letter")
        )
        db.add(case)
        await db.commit()
        try:
            await db.refresh(case)
        except Exception:
            pass

        # Audit ledger log
        audit_entry = AuditLedger(
            user_id=user_id,
            domain="law",
            action="generate_appeal_letter",
            request_redacted=denial_letter[:500],
            response_raw=appeal_data
        )
        db.add(audit_entry)
        await db.commit()

        return {
            "id": getattr(case, "id", 1),
            "denial_reason": appeal_data.get("denial_reason"),
            "applicable_statute": appeal_data.get("applicable_statute"),
            "appeal_letter": case.appeal_letter,
            "citations": appeal_data.get("citations", [])
        }
    except Exception as e:
        import traceback
        print(f"DEBUG EXCEPTION IN APPEAL: {e}\n{traceback.format_exc()}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/appeal/pdf", response_class=Response)
async def generate_appeal_pdf(
    denial_letter: str = Form(...),
    patient_name: str = Form(...),
    policy_id: str = Form(...),
    insurance_provider: Optional[str] = Form("Insurance Carrier"),
    claim_id: Optional[str] = Form(None),
    denial_code: Optional[str] = Form(None),
    is_urgent: Optional[bool] = Form(False),
    db: AsyncSession = Depends(get_db),
    user_payload: dict = Depends(RoleChecker([UserRole.LEGAL_COUNSEL]))
):
    """
    Generates a formal legal PDF (1-inch margins, statutory footnotes) for the insurance appeal.
    Enforces ACA § 2719, ERISA § 1133, and HIPAA § 164.524 citations.
    Returns a downloadable PDF binary.
    """
    try:
        appeal_data = await legal_agent.generate_appeal(
            denial_text=denial_letter,
            patient_name=patient_name,
            policy_id=policy_id,
            insurance_provider=insurance_provider,
            claim_id=claim_id,
            denial_code=denial_code,
            is_urgent=is_urgent,
        )

        pdf_bytes = generate_legal_pdf(
            patient_name=patient_name,
            policy_id=policy_id,
            insurance_provider=insurance_provider or "Insurance Carrier",
            claim_id=claim_id,
            denial_reason=appeal_data.get("denial_reason", denial_letter),
            applicable_statute=appeal_data.get("applicable_statute", "ACA Section 2719 (45 CFR § 147.136) / ERISA 29 U.S.C. § 1133 / HIPAA 45 CFR § 164.524"),
            appeal_letter=appeal_data.get("appeal_letter", ""),
        )

        filename = f"appeal_{patient_name.replace(' ', '_')}_{policy_id}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/redline")
async def redline_contract(
    contract_text: str = Form(...),
    user_id: str = "guest_user",
    db: AsyncSession = Depends(get_db),
    user_payload: dict = Depends(RoleChecker([UserRole.LEGAL_COUNSEL]))
):
    """
    Scrub contract, highlight predatory liability/arbitration clauses side-by-side, and suggest revisions.
    """
    try:
        analysis_data = await legal_agent.analyze_contract(contract_text)
        
        # Save case in DB
        case = LegalCase(
            user_id=user_id,
            doc_type="contract_redline",
            raw_text=contract_text,
            parsed_clauses=analysis_data
        )
        db.add(case)
        await db.commit()

        # Audit ledger log
        audit_entry = AuditLedger(
            user_id=user_id,
            domain="law",
            action="redline_contract",
            request_redacted=contract_text[:500],
            response_raw=analysis_data
        )
        db.add(audit_entry)
        await db.commit()

        return {
            "id": case.id,
            "overall_risk_score": analysis_data.get("overall_risk_score"),
            "predatory_clauses": analysis_data.get("predatory_clauses")
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hipaa-request")
async def generate_hipaa_letter(
    patient_name: str = Form(...),
    dob: str = Form(...),
    provider_name: str = Form(...),
    date_range: Optional[str] = Form("All Available Records"),
    target_recipient: Optional[str] = Form("Self / Designated Advocate"),
    user_id: str = "guest_user",
    db: AsyncSession = Depends(get_db)
):
    """
    Generate an automated, legally binding HIPAA medical records release request (45 CFR § 164.508).
    """
    try:
        letter_content = legal_agent.generate_hipaa_request(
            patient_name=patient_name,
            dob=dob,
            provider_name=provider_name,
            date_range=date_range,
            target_recipient=target_recipient
        )
        
        # Save case in DB
        case = LegalCase(
            user_id=user_id,
            doc_type="hipaa_request",
            raw_text=f"HIPAA Request for {provider_name}",
            appeal_letter=letter_content
        )
        db.add(case)
        await db.commit()

        # Audit ledger log
        audit_entry = AuditLedger(
            user_id=user_id,
            domain="law",
            action="generate_hipaa_request",
            request_redacted=f"Patient: {patient_name}, Provider: {provider_name}",
            response_raw={"status": "generated"}
        )
        db.add(audit_entry)
        await db.commit()

        return {
            "id": case.id,
            "hipaa_letter": letter_content
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
