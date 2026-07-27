import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.security import redact_pii, create_access_token
from backend.app.core.database import get_db
from decimal import Decimal
from backend.app.models.models import CrowdfundCampaign

# Simplified Mock database session for non-interactive tests
class MockAsyncSession:
    def add(self, instance):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def refresh(self, instance):
        if hasattr(instance, "id") and instance.id is None:
            instance.id = 1

    async def execute(self, statement):
        class MockResult:
            def scalars(self):
                class MockScalars:
                    def all(self):
                        return [
                            CrowdfundCampaign(
                                id=1,
                                creator_id=1,
                                title="Help Jane's Knee Surgery",
                                description="Jane needs assistance for her meniscus repair surgery.",
                                target_amount=Decimal("5000.00"),
                                current_amount=Decimal("250.00"),
                                escrow_address="0xMockEscrowAddress",
                                bill_verification_status="pending"
                            )
                        ]
                return MockScalars()
            
            def scalar_one_or_none(self):
                return CrowdfundCampaign(
                    id=1,
                    creator_id=1,
                    title="Help Jane's Knee Surgery",
                    description="Jane needs assistance for her meniscus repair surgery.",
                    target_amount=Decimal("5000.00"),
                    current_amount=Decimal("250.00"),
                    escrow_address="0xMockEscrowAddress",
                    bill_verification_status="pending"
                )
        return MockResult()

    async def close(self):
        pass

async def override_get_db():
    yield MockAsyncSession()

# Apply dependency overrides to bypass local MySQL connection requirement in tests
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# Pre-generate role tokens for tests
token_counsel = create_access_token({"sub": "advocate_user", "role": "LEGAL_COUNSEL"})
token_clinician = create_access_token({"sub": "doctor_user", "role": "CLINICIAN"})
token_patient = create_access_token({"sub": "patient_user", "role": "PATIENT"})

def test_root_and_health():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_phi_redaction():
    text_with_pii = "Patient John Doe (SSN: 123-45-6789, Email: john@example.com, Phone: 555-123-4567) has pain."
    redacted, scrubbed_count, scrubbed_types = redact_pii(text_with_pii)
    
    assert "John Doe" not in redacted
    assert "123-45-6789" not in redacted
    assert "john@example.com" not in redacted
    assert "555-123-4567" not in redacted
    assert "<SSN>" in redacted or "SSN" in redacted
    assert "<EMAIL_ADDRESS>" in redacted or "Email" in redacted
    assert scrubbed_count > 0


def test_health_triage_endpoint():
    response = client.post(
        "/api/v1/health/triage",
        data={"description": "My foot is swollen after playing soccer."}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "diagnosis" in json_data
    assert "severity" in json_data
    assert "treatment" in json_data
    assert "requires_appeal" in json_data


def test_health_log_endpoint():
    response = client.post(
        "/api/v1/health/log",
        params={"record_type": "steps", "value": 8000, "date_str": "2026-07-23"}
    )
    assert response.status_code == 200
    assert "Successfully logged steps" in response.json()["message"]


def test_edu_recall_engine():
    response = client.post(
        "/api/v1/edu/recall-engine",
        data={
            "title": "Grade 2 Ankle Sprains",
            "content": "This represents a partial tear of the lateral ligament. Patients show swelling and bruising. Conservative treatment involves the RICE protocol."
        }
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "react_flow_graph" in json_data
    assert "flashcards" in json_data
    assert len(json_data["flashcards"]) > 0


def test_edu_generate_podcast():
    response = client.post(
        "/api/v1/edu/generate-podcast",
        data={"topic": "Grade 3 Sprain Pathology"}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "transcript" in json_data
    assert "audio_url" in json_data


def test_community_campaigns():
    # Create Campaign
    response = client.post(
        "/api/v1/community/campaigns",
        data={
            "creator_id": 1,
            "title": "Help Jane's Knee Surgery",
            "description": "Jane needs assistance for her surgery.",
            "target_amount": 5000.00,
            "escrow_address": "0xMockEscrowAddress"
        }
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Help Jane's Knee Surgery"

    # List Campaigns
    response = client.get("/api/v1/community/campaigns")
    assert response.status_code == 200
    assert len(response.json()) > 0


def test_community_verify_bill_rbac():
    jpeg_fixture = b"\xff\xd8\xffmockedpixelbytes"

    # Attempt without auth token
    response = client.post(
        "/api/v1/community/campaigns/1/verify-bill",
        files={"bill_image": ("bill.jpg", jpeg_fixture, "image/jpeg")}
    )
    assert response.status_code == 401

    # Attempt with incorrect role (PATIENT)
    response = client.post(
        "/api/v1/community/campaigns/1/verify-bill",
        files={"bill_image": ("bill.jpg", jpeg_fixture, "image/jpeg")},
        headers={"Authorization": f"Bearer {token_patient}"}
    )
    assert response.status_code == 403

    # Attempt with correct role (CLINICIAN)
    response = client.post(
        "/api/v1/community/campaigns/1/verify-bill",
        files={"bill_image": ("bill.jpg", jpeg_fixture, "image/jpeg")},
        headers={"Authorization": f"Bearer {token_clinician}"}
    )
    assert response.status_code == 200
    assert response.json()["verification_status"] == "verified"


def test_community_verify_bill_rejects_invalid_file_signature():
    response = client.post(
        "/api/v1/community/campaigns/1/verify-bill",
        files={"bill_image": ("bill.jpg", b"not-a-jpeg", "image/jpeg")},
        headers={"Authorization": f"Bearer {token_clinician}"},
    )

    assert response.status_code == 400
    assert "invalid file signature" in response.json()["detail"]


def test_law_appeal_letter():
    # Post with LEGAL_COUNSEL credentials
    response = client.post(
        "/api/v1/law/appeal",
        data={
            "denial_letter": "Claim is denied because MRI scan was out of network and not medically required.",
            "patient_name": "Jane Miller",
            "policy_id": "POL-101-INS"
        },
        headers={"Authorization": f"Bearer {token_counsel}"}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "appeal_letter" in json_data
    assert "Jane Miller" in json_data["appeal_letter"]


def test_law_contract_redliner():
    response = client.post(
        "/api/v1/law/redline",
        data={"contract_text": "Patient agrees to waive jury trials and settle all disputes via binding arbitration."},
        headers={"Authorization": f"Bearer {token_counsel}"}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "overall_risk_score" in json_data
    assert "predatory_clauses" in json_data
    assert len(json_data["predatory_clauses"]) > 0


def test_law_hipaa_request():
    response = client.post(
        "/api/v1/law/hipaa-request",
        data={
            "patient_name": "Alice Cooper",
            "dob": "10/12/1985",
            "provider_name": "Valley Care Clinic"
        }
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "hipaa_letter" in json_data
    assert "Alice Cooper" in json_data["hipaa_letter"]
    assert "Valley Care Clinic" in json_data["hipaa_letter"]
