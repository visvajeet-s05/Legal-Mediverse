from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import Optional, List
from decimal import Decimal
from backend.app.core.database import get_db
from backend.app.models.models import CrowdfundCampaign, AuditLedger, Donation
from backend.app.agents.fraud_agent import FraudVerificationAgent
from backend.app.core.security import RoleChecker, UserRole
from backend.app.core.config import settings

router = APIRouter(prefix="/community", tags=["Crowdfunding"])

# Instantiate fraud verification agent
fraud_agent = FraudVerificationAgent()

MAX_BILL_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_BILL_CONTENT_TYPES = {"application/pdf", "image/jpeg", "image/png"}
PDF_ACTIVE_CONTENT_MARKERS = (b"/javascript", b"/js", b"/launch", b"/embeddedfile")


def _reject_simulated_escrow_response() -> None:
    """Block mock transaction hashes outside local development and tests."""
    if settings.requires_live_escrow:
        raise HTTPException(
            status_code=501,
            detail=(
                "Live escrow transaction submission is not enabled in this API. "
                "Submit the transaction through the configured wallet flow and "
                "synchronize the verified on-chain receipt instead."
            ),
        )


def _validate_bill_upload(upload: UploadFile, content: bytes) -> None:
    """Validate untrusted invoice uploads before they reach the OCR pipeline."""
    if upload.content_type not in ALLOWED_BILL_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only PDF, PNG, and JPEG hospital invoices are accepted.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Bill upload is empty.")
    if len(content) > MAX_BILL_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Bill upload exceeds the 10 MB limit.")

    normalized_content_type = upload.content_type.lower()
    if normalized_content_type == "application/pdf":
        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="Uploaded PDF has an invalid file signature.")
        lowered_content = content.lower()
        if any(marker in lowered_content for marker in PDF_ACTIVE_CONTENT_MARKERS):
            raise HTTPException(
                status_code=400,
                detail="Uploaded PDF contains unsupported active content.",
            )
    elif normalized_content_type == "image/png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(status_code=400, detail="Uploaded PNG has an invalid file signature.")
    elif normalized_content_type == "image/jpeg" and not content.startswith(b"\xff\xd8\xff"):
        raise HTTPException(status_code=400, detail="Uploaded JPEG has an invalid file signature.")


@router.post("/campaigns")
async def create_campaign(
    creator_id: int = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    target_amount: float = Form(...),
    escrow_address: Optional[str] = Form(None),
    on_chain_campaign_id: Optional[int] = Form(None),
    on_chain_tx_hash: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Creates a new crowdfunding campaign mapped to a smart contract escrow address.
    Accepts on_chain_campaign_id from the contract creation receipt and
    on_chain_tx_hash for event sync.
    """
    try:
        campaign = CrowdfundCampaign(
            creator_id=creator_id,
            title=title,
            description=description,
            target_amount=Decimal(str(target_amount)),
            escrow_address=escrow_address,
            on_chain_campaign_id=on_chain_campaign_id,
            bill_verification_status="pending",
            total_bill_amount=0.0
        )
        db.add(campaign)
        await db.commit()
        await db.refresh(campaign)

        # Audit ledger log
        audit_entry = AuditLedger(
            user_id=str(creator_id),
            domain="community",
            action="create_campaign",
            request_redacted=f"Title: {title}",
            response_raw={
                "campaign_id": campaign.id,
                "on_chain_campaign_id": on_chain_campaign_id,
                "escrow_address": escrow_address,
                "on_chain_tx_hash": on_chain_tx_hash
            }
        )
        db.add(audit_entry)
        await db.commit()

        return campaign
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/campaigns/{campaign_id}")
async def get_campaign_detail(
    campaign_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve a single campaign by ID with full details including donations.
    """
    try:
        stmt = select(CrowdfundCampaign).where(CrowdfundCampaign.id == campaign_id)
        result = await db.execute(stmt)
        campaign = result.scalar_one_or_none()

        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        # Fetch donations for this campaign
        donations_stmt = select(Donation).where(Donation.campaign_id == campaign_id)
        donations_result = await db.execute(donations_stmt)
        donations = donations_result.scalars().all()

        return {
            "id": campaign.id,
            "creator_id": campaign.creator_id,
            "title": campaign.title,
            "description": campaign.description,
            "target_amount": str(campaign.target_amount),
            "current_amount": str(campaign.current_amount),
            "escrow_address": campaign.escrow_address,
            "on_chain_campaign_id": campaign.on_chain_campaign_id,
            "bill_verification_status": campaign.bill_verification_status,
            "total_bill_amount": str(campaign.total_bill_amount),
            "fraud_risk_score": str(campaign.fraud_risk_score),
            "is_released": campaign.is_released,
            "created_at": campaign.created_at.isoformat() if campaign.created_at else None,
            "donations": [
                {
                    "id": d.id,
                    "donor_address": d.donor_address,
                    "amount": str(d.amount),
                    "tx_hash": d.tx_hash,
                    "block_number": d.block_number,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                }
                for d in donations
            ],
            "donations_count": len(donations),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/campaigns/{campaign_id}/verify-bill")
async def verify_campaign_bill(
    campaign_id: int,
    bill_image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
    user_payload: dict = Depends(RoleChecker([UserRole.CLINICIAN]))
):
    """
    Accepts bill PDF/Image, processes it with Fraud Agent OCR, and matches total to release escrow.
    """
    # Fetch campaign
    stmt = select(CrowdfundCampaign).where(CrowdfundCampaign.id == campaign_id)
    result = await db.execute(stmt)
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    uploaded_file = bill_image or file
    if uploaded_file is None:
        raise HTTPException(status_code=400, detail="Bill image file is required")

    try:
        bill_bytes = await uploaded_file.read()
        _validate_bill_upload(uploaded_file, bill_bytes)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=f"Failed to read bill: {e}")

    try:
        # Run Fraud OCR Verification Agent
        verification_result = await fraud_agent.verify_bill(
            bill_image_bytes=bill_bytes,
            campaign_target=float(campaign.target_amount)
        )

        is_verified = verification_result.get("is_verified", False)
        total_extracted = verification_result.get("total_amount_extracted", 0.0)
        fraud_risk_score = float(verification_result.get("fraud_risk_score", 0.99))

        # Escrow assertion: reject verification if fraud risk is too high
        if fraud_risk_score >= 0.10:
            is_verified = False
            verification_result["is_verified"] = False
            verification_result["verification_reason"] = (
                f"Escrow release BLOCKED: fraud_risk_score {fraud_risk_score:.2f} >= 0.10 threshold. "
                "Manual review required before funds can be released."
            )

        # Update campaign verification state
        campaign.bill_verification_status = "verified" if is_verified else "failed"
        campaign.total_bill_amount = Decimal(str(total_extracted))
        # Store fraud score on campaign for release-milestone guard
        campaign.fraud_risk_score = Decimal(str(fraud_risk_score))

        await db.commit()

        # Audit Log
        audit_entry = AuditLedger(
            user_id=str(campaign.creator_id),
            domain="community",
            action="verify_bill",
            request_redacted=f"Verify bill for campaign {campaign_id}",
            response_raw=verification_result
        )
        db.add(audit_entry)
        await db.commit()

        return {
            "campaign_id": campaign_id,
            "verification_status": campaign.bill_verification_status,
            "provider_name": verification_result.get("provider_name") or verification_result.get("hospital_name", "Unknown Provider"),
            "total_due": total_extracted,
            "total_extracted": total_extracted,
            "itemized_breakdown": verification_result.get("itemized_breakdown", []),
            "fraud_risk_score": fraud_risk_score,
            "anomalies": verification_result.get("detected_anomalies", []),
            "reason": verification_result.get("verification_reason", ""),
            "ocr_verification": verification_result
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"OCR Verification failed: {str(e)}")


@router.post("/campaigns/{campaign_id}/donate")
async def donate_to_campaign(
    campaign_id: int,
    amount: float = Form(...),
    tx_hash: Optional[str] = Form(None),
    donor_address: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Records an on-chain donation to a campaign.
    Accepts the transaction hash from the client-side Wagmi writeContract call.
    Also accepts the donor wallet address for Donation record tracking.
    In live testnet/production mode, a valid tx_hash is required.
    """
    _reject_simulated_escrow_response()
    stmt = select(CrowdfundCampaign).where(CrowdfundCampaign.id == campaign_id)
    result = await db.execute(stmt)
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # In live mode, require a real tx_hash from the client
    if settings.requires_live_escrow and not tx_hash:
        raise HTTPException(
            status_code=400,
            detail="tx_hash is required in live mode. Submit the donation via MetaMask first."
        )

    try:
        current = float(campaign.current_amount or 0.0)
        new_total = current + amount
        campaign.current_amount = Decimal(str(new_total))

        # Use client-provided tx_hash or generate a mock one for development
        if tx_hash:
            resolved_tx_hash = tx_hash
        else:
            import random as _random
            resolved_tx_hash = "0x" + "".join(_random.choices("0123456789abcdef", k=40))

        # Create Donation record in DB with tx_hash and donor address
        from backend.app.models.models import Donation
        donation = Donation(
            campaign_id=campaign_id,
            donor_address=donor_address or "0xUnknown",
            amount=Decimal(str(amount)),
            tx_hash=resolved_tx_hash,
        )
        db.add(donation)

        audit_entry = AuditLedger(
            user_id="donor_user",
            domain="community",
            action="donate_to_campaign",
            request_redacted=f"Donated ${amount} to campaign {campaign_id}",
            response_raw={"tx_hash": resolved_tx_hash, "new_total": new_total, "donor_address": donor_address}
        )
        db.add(audit_entry)

        return {
            "campaign_id": campaign_id,
            "amount_donated": amount,
            "new_total": new_total,
            "tx_hash": resolved_tx_hash,
            "donor_address": donor_address,
            "escrow_vault": campaign.escrow_address or "0x7a89B310c141A1B903E5dB82103f191D5F76A529"
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/campaigns/{campaign_id}/release-milestone")
async def release_milestone_funds(
    campaign_id: int,
    tx_hash: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Releases verified escrow funds from Web3 Vault to hospital provider address.
    Accepts on-chain transaction hash from Wagmi writeContract.
    """
    _reject_simulated_escrow_response()
    stmt = select(CrowdfundCampaign).where(CrowdfundCampaign.id == campaign_id)
    result = await db.execute(stmt)
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # In live mode, require a real tx_hash
    if settings.requires_live_escrow and not tx_hash:
        raise HTTPException(
            status_code=400,
            detail="tx_hash is required in live mode. Submit the release via MetaMask first."
        )

    try:
        # Fraud Risk Guard
        stored_fraud_score = float(getattr(campaign, "fraud_risk_score", None) or 0.0)
        if stored_fraud_score >= 0.10:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Escrow release DENIED: bill_verification_status='{campaign.bill_verification_status}', "
                    f"fraud_risk_score={stored_fraud_score:.2f}. "
                    "Funds can only be released to a verified provider with fraud_risk_score < 0.10."
                )
            )

        campaign.bill_verification_status = "verified"
        campaign.is_released = True

        # Use client-provided tx_hash or mock for development
        if tx_hash:
            resolved_tx_hash = tx_hash
        else:
            import random as _random2
            resolved_tx_hash = "0x" + "".join(_random2.choices("0123456789abcdef", k=40))

        return {
            "campaign_id": campaign_id,
            "status": "verified",
            "is_released": True,
            "funds_released": float(campaign.current_amount),
            "fraud_risk_score": stored_fraud_score,
            "hospital_address": campaign.escrow_address or "0x7a89B310c141A1B903E5dB82103f191D5F76A529",
            "release_tx_hash": resolved_tx_hash
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/campaigns/{campaign_id}/claim-refund")
async def claim_refund(
    campaign_id: int,
    tx_hash: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Records a donor refund claim for a campaign with failed verification status.
    Accepts on-chain transaction hash from Wagmi writeContract.
    """
    _reject_simulated_escrow_response()
    stmt = select(CrowdfundCampaign).where(CrowdfundCampaign.id == campaign_id)
    result = await db.execute(stmt)
    campaign = result.scalar_one_or_none()
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Only allow refund if bill verification failed
    if campaign.bill_verification_status != "failed":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Refund not available. Bill verification status is "
                f"'{campaign.bill_verification_status}', must be 'failed'."
            )
        )

    if settings.requires_live_escrow and not tx_hash:
        raise HTTPException(
            status_code=400,
            detail="tx_hash is required in live mode. Submit the refund via MetaMask first."
        )

    try:
        if tx_hash:
            resolved_tx_hash = tx_hash
        else:
            import random as _random3
            resolved_tx_hash = "0x" + "".join(_random3.choices("0123456789abcdef", k=40))

        audit_entry = AuditLedger(
            user_id="donor_user",
            domain="community",
            action="claim_refund",
            request_redacted=f"Refund claimed for campaign {campaign_id}",
            response_raw={"tx_hash": resolved_tx_hash}
        )
        db.add(audit_entry)

        return {
            "campaign_id": campaign_id,
            "status": "refund_claimed",
            "tx_hash": resolved_tx_hash
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/campaigns")
async def list_campaigns(
    skip: int = 0,
    limit: int = 20,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve all current verified crowdfunding campaigns with pagination.
    """
    try:
        stmt = (
            select(CrowdfundCampaign)
            .order_by(CrowdfundCampaign.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await db.execute(stmt)
        if hasattr(result, "scalars"):
            campaigns = result.scalars().all()
        else:
            campaigns = result.all()

        if not campaigns:
            # Seed 2 initial realistic campaigns
            seed_c1 = CrowdfundCampaign(
                creator_id=1,
                title="Pediatric Surgery & Cardiology Fund",
                description="Emergency pediatric cardiac repair for 6-year-old patient at St. Jude Medical Center. All funds held in Polygon Escrow Vault.",
                target_amount=Decimal("15000.00"),
                current_amount=Decimal("12400.00"),
                escrow_address="0x7a89B310c141A1B903E5dB82103f191D5F76A529",
                bill_verification_status="verified",
                total_bill_amount=Decimal("15000.00")
            )
            seed_c2 = CrowdfundCampaign(
                creator_id=2,
                title="Emergency Oncology Support & Chemotherapy",
                description="Targeted chemotherapy and immunotherapy protocol funding for stage 3 lymphoma treatment. Direct hospital vault release upon bill verification.",
                target_amount=Decimal("25000.00"),
                current_amount=Decimal("18200.00"),
                escrow_address="0x3f5cE410B222a014902D8B90141a5B5158A4091A",
                bill_verification_status="pending",
                total_bill_amount=Decimal("0.00")
            )
            db.add(seed_c1)
            db.add(seed_c2)

            result = await db.execute(
                select(CrowdfundCampaign)
                .order_by(CrowdfundCampaign.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            if hasattr(result, "scalars"):
                campaigns = result.scalars().all()
            else:
                campaigns = result.all()

        # Format campaign data
        result_campaigns = []
        for c in campaigns:
            result_campaigns.append({
                "id": c.id,
                "creator_id": c.creator_id,
                "title": c.title,
                "description": c.description,
                "target_amount": str(c.target_amount),
                "current_amount": str(c.current_amount),
                "escrow_address": c.escrow_address,
                "on_chain_campaign_id": c.on_chain_campaign_id,
                "bill_verification_status": c.bill_verification_status,
                "total_bill_amount": str(c.total_bill_amount),
                "fraud_risk_score": str(c.fraud_risk_score),
                "is_released": c.is_released,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            })

        return result_campaigns
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))