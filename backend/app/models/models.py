from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Numeric, Boolean
from sqlalchemy.orm import relationship
from backend.app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    campaigns = relationship("CrowdfundCampaign", back_populates="creator")


class AuditLedger(Base):
    """
    Immutable Audit Ledger table tracking prompts, model outputs, confidence, and action.
    """
    __tablename__ = "audit_ledger"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(50), nullable=True, index=True)  # supports guest_xxxx or integer user.id
    domain = Column(String(50), nullable=False, index=True)  # health, edu, community, law
    action = Column(String(100), nullable=False)
    request_redacted = Column(Text, nullable=False)
    response_raw = Column(JSON, nullable=False)
    confidence_score = Column(Numeric(5, 2), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)


class HealthRecord(Base):
    """
    Stores FHIR-compliant observations for nutrition, sleep, and physical activity.
    """
    __tablename__ = "health_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(50), nullable=False, index=True)
    record_type = Column(String(50), nullable=False)  # nutrition, steps, sleep
    fhir_observation = Column(JSON, nullable=False)  # Full FHIR JSON
    created_at = Column(DateTime, default=datetime.utcnow)


class Donation(Base):
    """
    Tracks individual on-chain donations synced from MedicalEscrow events.
    """
    __tablename__ = "donations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    campaign_id = Column(Integer, ForeignKey("crowdfund_campaigns.id", ondelete="CASCADE"), nullable=False, index=True)
    donor_address = Column(String(42), nullable=False, index=True)
    amount = Column(Numeric(18, 4), nullable=False)
    tx_hash = Column(String(66), nullable=False, unique=True, index=True)
    block_number = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    campaign = relationship("CrowdfundCampaign", back_populates="donations")


class CrowdfundCampaign(Base):
    """
    Crowdfunding campaign data with verification details and Web3 smart contract reference.
    """
    __tablename__ = "crowdfund_campaigns"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    creator_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    target_amount = Column(Numeric(18, 4), nullable=False)
    current_amount = Column(Numeric(18, 4), default=0.0)
    escrow_address = Column(String(42), nullable=True)  # Ethereum address of Escrow contract
    on_chain_campaign_id = Column(Integer, nullable=True, index=True)  # Mapped on-chain campaign id
    bill_verification_status = Column(String(50), default="pending")  # pending, verified, failed
    total_bill_amount = Column(Numeric(18, 2), default=0.0)
    fraud_risk_score = Column(Numeric(5, 4), default=0.0)
    is_released = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    creator = relationship("User", back_populates="campaigns")
    donations = relationship("Donation", back_populates="campaign", cascade="all, delete-orphan")


class IndexerState(Base):
    """
    Persistent key-value state storage for background indexers.
    Used by EscrowIndexer to survive service restarts without rescanning
    or missing events.
    """
    __tablename__ = "indexer_state"

    key = Column(String(50), primary_key=True, index=True)
    value = Column(Integer, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Well-known state keys
STATE_KEY_LAST_BLOCK = "escrow_last_processed_block"


class LegalCase(Base):
    """
    Saves document analysis, regulatory appeals, and contract reviews.
    """
    __tablename__ = "legal_cases"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(50), nullable=False, index=True)
    doc_type = Column(String(50), nullable=False)  # denial_appeal, contract_redline, hipaa_request
    raw_text = Column(Text, nullable=False)
    appeal_letter = Column(Text, nullable=True)
    parsed_clauses = Column(JSON, nullable=True)  # highlight predatory clauses in contracts
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to user if logged in (guest is stored as string user_id)
    user = relationship("User", foreign_keys=[user_id], primaryjoin="User.id == cast(LegalCase.user_id, Integer)", uselist=False, viewonly=True)


class EduNote(Base):
    """
    Saves flashcards and React Flow graph representations of user notes.
    """
    __tablename__ = "edu_notes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(50), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    react_flow_graph = Column(JSON, nullable=True)
    flashcards = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", foreign_keys=[user_id], primaryjoin="User.id == cast(EduNote.user_id, Integer)", uselist=False, viewonly=True)
