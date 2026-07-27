import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.security import redact_pii
from backend.app.core.database import get_db

from decimal import Decimal
from backend.app.models.models import CrowdfundCampaign

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

app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

def test_emergency_triage_escalation():
    response = client.post(
        "/api/v1/health/triage",
        data={"description": "Patient experiencing severe chest pain and shortness of breath."}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["risk_level"] == "Urgent"
    assert json_data["severity"] in ["severe", "critical"]

def test_phi_redaction_pipeline():
    raw_text = "John Doe (SSN: 999-00-1234, email: test@mediverse.com) has severe foot pain."
    redacted_text, scrubbed_count, scrubbed_types = redact_pii(raw_text)
    assert "John Doe" not in redacted_text
    assert "999-00-1234" not in redacted_text
    assert "test@mediverse.com" not in redacted_text
    assert scrubbed_count >= 3
    assert "SSN" in scrubbed_types or "EMAIL_ADDRESS" in scrubbed_types

def test_triage_icd10_grounding():
    response = client.post(
        "/api/v1/health/triage",
        data={"description": "Patient experiencing severe chest pain and shortness of breath."}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["risk_level"] == "Urgent"
    assert "primary_concern" in json_data
    assert "icd_10_code" in json_data
    assert json_data["confidence_score"] == 1.0
    assert json_data["citations"]
    assert any("ICD-10" in citation for citation in json_data["citations"])


def test_phi_hybrid_scrubber():
    raw_text = "John Doe (SSN: 999-00-1234, email: test@mediverse.com) lives at 123 Main St."
    redacted_text, scrubbed_count, scrubbed_types = redact_pii(raw_text)
    assert redacted_text != raw_text
    assert scrubbed_count >= 4
    assert "NAME" in scrubbed_types or "EMAIL_ADDRESS" in scrubbed_types
    assert "SSN" in scrubbed_types
    assert "ADDRESS" in scrubbed_types


def test_fhir_observation_logging():
    payload = {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": "55423-8", "display": "Number of steps in unspecified time Pedometer"}]},
        "valueQuantity": {"value": 8000, "unit": "steps"}
    }
    response = client.post("/api/v1/health/fhir/observation", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert "observation_id" in json_data
    assert json_data["data"]["valueQuantity"]["value"] == 8000
